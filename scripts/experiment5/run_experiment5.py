"""Experiment 5 - selection under a transformation family the training never saw.

Every selector, with and without augmentation, is fitted once and scored on the
held-out block under each of five transformation families. With
``--require-cached`` the tables and figures are rebuilt from the stage store alone.

uv run python scripts/experiment5/run_experiment5.py --require-cached
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.assemble_paired_views import check_source_agreement
from aas_shortcut.augment_training_samples import DEPLOYED_REP
from aas_shortcut.compare_selection_runs import (
    FAMILY_COLUMN,
    PROBLEM_COLUMNS,
    compare_selection_representations,
    compare_selection_trainings,
    problem_levels,
    selection_levels,
    selection_runs,
)
from aas_shortcut.compare_training_runs import CONDITIONS
from aas_shortcut.create_figures.define_figure_vocabulary import (
    VARIANT_ORDER,
    feature_set_stem,
)
from aas_shortcut.create_figures.draw_selection_split_figure import (
    AUGMENTATION_SPLIT,
    AUGMENTATION_SPLIT_STEM,
    PROBLEM_KEYS,
    figure_selection_split_pooled,
)
from aas_shortcut.define_paths import (
    FEATURE_SETS,
    ORACLE_VARIANT,
    TRANSFORMS,
    RunPaths,
    representation_path,
)
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_UNSEEN,
    SCOPES,
    UNSEEN_FAMILIES,
    measurable_scopes,
    scope_mean,
)
from aas_shortcut.define_warp_audits import AUDITS
from aas_shortcut.measure_selection_transfer import TRAINING_CONDITIONS
from aas_shortcut.resolve_portfolio_blocks import (
    SelectionHeldOut,
    SelectionTraining,
    resolve_portfolio_bindings,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
)
from aas_shortcut.resolve_selection_cells import (
    FamilySelection,
    resolve_family_selection,
)
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_learner_option,
    add_overrides,
    add_store_options,
    build_context,
    build_parser,
    compose_config,
    record_run,
    resolve_run_paths,
    store_policy,
)
from aas_shortcut.split_bbob_instances import TEST_INSTANCES

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# Holds both representation sets, in rows of their own and never pooled.
GAIN_TABLE = Path("augmentation_gain.csv")
# `oracle` is a column, not a row: it is the ceiling a row is read against, not a
# fourth representation.
GAIN_COLUMNS: tuple[str, ...] = (
    "feature_set",
    SCOPE_COLUMN,
    "representation",
    "augmentation",
    "superiority_vs_sbs_mean",
    "oracle",
)
# The seed-level tests, named as Experiment 4's.
REPRESENTATION_TEST_TABLE = Path("representation_tests.csv")
TRAINING_TEST_TABLE = Path("training_tests.csv")

# The family is part of a problem's key because the violins pool the four unseen
# families' problems.
SPLIT_COLUMNS: tuple[str, ...] = (
    "feature_set",
    "feature_variant",
    "training",
    FAMILY_COLUMN,
    *PROBLEM_COLUMNS,
    "superiority_vs_sbs",
)

# Relative tolerance the untransformed anchor is checked at.
ANCHOR_TOLERANCE = 1e-9


def check_anchor(training: SelectionTraining, block: SelectionHeldOut) -> int:
    """Count the untransformed features of a held-out block that match training.

    That half does not depend on the family; if it moved, the audit would measure a
    different problem rather than a different warp.
    """
    return check_source_agreement(
        cast(
            pd.DataFrame,
            block.feature_table[block.feature_table["rep"] == DEPLOYED_REP],
        ),
        training.feature_table,
        TEST_INSTANCES,
        ANCHOR_TOLERANCE,
        "the training block",
    )


def oracle_ceilings(
    oracle: pd.DataFrame, labels: Sequence[str], scopes: Sequence[str]
) -> pd.DataFrame:
    """Return the oracle's level over each range, per representation set.

    The oracle reads no feature, so its rows repeat per training condition; they
    are averaged rather than one picked.
    """
    rows = []
    for label in labels:
        family = oracle[
            (oracle[FAMILY_COLUMN].astype(str) == label)
            & (oracle["feature_variant"].astype(str) == ORACLE_VARIANT)
            & (oracle["transform"].astype(str) == TRANSFORMS[1])
        ]
        means = (
            family.groupby(["feature_set", "training"], observed=True)[
                "superiority_vs_sbs"
            ]
            .mean()
            .groupby("feature_set", observed=True)
            .mean()
        )
        rows += [
            {"feature_set": str(name), FAMILY_COLUMN: label, "value": float(value)}
            for name, value in means.items()
        ]
    families = pd.DataFrame(rows)
    return pd.concat(
        [
            scope_mean(families, ["feature_set"], scope, FAMILY_COLUMN).assign(
                **{SCOPE_COLUMN: scope}
            )
            for scope in scopes
        ],
        ignore_index=True,
    )


def augmentation_gain(
    levels: pd.DataFrame,
    ceilings: pd.DataFrame,
    scopes: Sequence[str] = tuple(SCOPES),
    variants: Sequence[str] = VARIANT_ORDER,
    conditions: Sequence[str] = TRAINING_CONDITIONS,
) -> pd.DataFrame:
    """Lay each range's cell levels out as the gain table, with the oracle's."""
    ceiling = {
        (str(row["feature_set"]), str(row[SCOPE_COLUMN])): float(row["value"])
        for _, row in ceilings.iterrows()
    }
    keyed = {
        (
            str(row["feature_set"]),
            str(row[SCOPE_COLUMN]),
            str(row["feature_variant"]),
            str(row["training"]),
        ): float(row["value"])
        for _, row in levels.iterrows()
    }
    rows = []
    for feature_set in dict.fromkeys(str(name) for name in levels["feature_set"]):
        held = {str(name) for name in levels[SCOPE_COLUMN]}
        for scope in (name for name in scopes if name in held):
            for variant in variants:
                for training in conditions:
                    level = keyed.get((feature_set, scope, variant, training))
                    if level is None:
                        continue
                    top = ceiling.get((feature_set, scope), float("nan"))
                    rows.append(
                        {
                            "feature_set": feature_set,
                            SCOPE_COLUMN: scope,
                            "representation": variant,
                            "augmentation": training,
                            "superiority_vs_sbs_mean": level,
                            "oracle": top,
                        }
                    )
    return pd.DataFrame(rows, columns=list(GAIN_COLUMNS))


def report_gain(
    records: pd.DataFrame,
    oracle: pd.DataFrame,
    labels: Sequence[str],
    summary_paths: RunPaths,
) -> list[Path]:
    """Write what augmentation buys over each range."""
    summary_paths.mkdirs()
    scopes, lacking = measurable_scopes(labels)
    for scope, missing in lacking.items():
        LOGGER.warning(
            "the %r range needs %s, which this run did not audit", scope, missing
        )
    levels = selection_levels(records, scopes)
    ceilings = oracle_ceilings(oracle, labels, scopes)
    gain_target = summary_paths.root / GAIN_TABLE
    gain = augmentation_gain(levels, ceilings)
    gain.to_csv(gain_target, index=False)
    return [gain_target]


def report_split(
    units: FamilySelection,
    oracle: pd.DataFrame,
    gain: pd.DataFrame,
    summary_paths: RunPaths,
) -> list[Path]:
    """Draw both training conditions as split violins over the unseen families.

    A violin pools the 960 transformed problems; its rule is the level in ``gain``.
    Nothing is drawn unless every unseen family was audited.
    """
    if not set(UNSEEN_FAMILIES) <= set(units.labels):
        return []
    post = TRANSFORMS[1]
    # The cells' problem tables hold both blocks; keep the transformed one.
    problems = pd.concat(
        [
            problem_levels(_on_block(units.problems([label]), post))
            for label in UNSEEN_FAMILIES
        ],
        ignore_index=True,
    )
    ceiling = _on_block(
        cast(pd.DataFrame, oracle[oracle[FAMILY_COLUMN].isin(UNSEEN_FAMILIES)]), post
    )
    table = cast(
        pd.DataFrame,
        pd.concat(
            [ceiling[list(SPLIT_COLUMNS)], problems[list(SPLIT_COLUMNS)]],
            ignore_index=True,
        ),
    )
    written: list[Path] = []
    for name in FEATURE_SETS:
        rows = gain[
            (gain["feature_set"] == name) & (gain[SCOPE_COLUMN] == SCOPE_UNSEEN)
        ]
        if rows.empty:
            continue
        levels = {
            (str(representation), str(condition)): float(value)
            for representation, condition, value in zip(
                rows["representation"],
                rows["augmentation"],
                rows["superiority_vs_sbs_mean"],
                strict=True,
            )
        }
        written += figure_selection_split_pooled(
            table,
            summary_paths,
            stem=feature_set_stem(AUGMENTATION_SPLIT_STEM, name),
            feature_set=name,
            levels=levels,
            split="training",
            blocks=AUGMENTATION_SPLIT,
            keys=(*PROBLEM_KEYS, FAMILY_COLUMN),
        )
    return written


def _on_block(rows: pd.DataFrame, transform: str) -> pd.DataFrame:
    return cast(pd.DataFrame, rows[rows["transform"].astype(str) == transform])


def report_seed_tests(records: pd.DataFrame, summary_paths: RunPaths) -> list[Path]:
    """Run and write the seed-level tests, one representation set at a time.

    A range whose families the run did not audit is left out with a warning.
    """
    scopes, lacking = measurable_scopes(records[FAMILY_COLUMN].astype(str))
    for scope, missing in lacking.items():
        LOGGER.warning(
            "the %r range needs %s, which this run did not audit", scope, missing
        )
    written: list[Path] = []
    for name, rows in records.groupby("feature_set", observed=True):
        trainings = set(rows["training"].astype(str))
        tables: dict[Path, list[pd.DataFrame]] = {
            REPRESENTATION_TEST_TABLE: [],
            TRAINING_TEST_TABLE: [],
        }
        for scope in scopes:
            runs = selection_runs(rows, scope)
            tables[REPRESENTATION_TEST_TABLE].append(
                compare_selection_representations(runs, scope)
            )
            if set(CONDITIONS) <= trainings:
                tables[TRAINING_TEST_TABLE].append(
                    compare_selection_trainings(runs, scope)
                )
        for table_name, parts in tables.items():
            if not parts:
                continue
            table = pd.concat(parts, ignore_index=True)
            target = summary_paths.root / representation_path(table_name, str(name))
            table.to_csv(target, index=False)
            written.append(target)
    return written


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Score both training conditions on a held-out block warped by a "
            "transformation family the training never saw. Trailing arguments "
            "are passed to Hydra as overrides."
        ),
        epilog="example: run_experiment5.py task.n_jobs=26",
    )
    add_store_options(parser)
    add_learner_option(parser)
    return add_overrides(parser).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    spec = ExperimentSpec.load(SPEC_PATH)
    # Every family in one run: the families a selector is scored on enter its unit.
    audits = list(AUDITS)

    cfg = compose_config(
        spec.hydra_overrides(args.preset, args.overrides, learner=args.learner)
    )
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)
    invocation = {
        "audits": [audit.name for audit in audits],
        "argv": list(argv or sys.argv[1:]),
    }

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        bindings = resolve_portfolio_bindings(store, context)
        training = resolve_selection_training(store, context, bindings)
        blocks = resolve_selection_held_out_in_range(store, context, bindings, audits)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1
    if not blocks:
        print("no audit produced an order-preserving warp", file=sys.stderr)
        return 1

    for block in blocks:
        LOGGER.info(
            "anchor: %d untransformed features of %s agree with the training block",
            check_anchor(training, block),
            block.label,
        )
    run_paths.mkdirs()
    try:
        units = resolve_family_selection(store, context, training, blocks)
        records, oracle = units.records(), units.oracle()
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    labels = [block.label for block in blocks]
    for path in report_gain(records, oracle, labels, run_paths):
        print(f"written: {path}")
    for path in report_seed_tests(records, run_paths):
        print(f"written: {path}")
    # The figure is drawn from the table just written.
    gain = pd.read_csv(run_paths.root / GAIN_TABLE)
    for path in report_split(units, oracle, gain, run_paths):
        print(f"written: {path}")
    store.write_list(run_paths.root)
    record_run(spec, cfg, run_paths, extra=invocation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
