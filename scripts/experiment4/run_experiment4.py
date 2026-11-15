"""Experiment 4 - the superiority regression, augmented, under held-out families.

Per algorithm group: three representations by two training conditions (with and
without transformed copies), each fitted once and scored on the held-out block
under each of five transformation families. With ``--require-cached`` the tables
are rebuilt from the stage store alone.

uv run python scripts/experiment4/run_experiment4.py --require-cached
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.assemble_paired_views import check_source_agreement
from aas_shortcut.augment_training_samples import DEPLOYED_REP
from aas_shortcut.compare_training_runs import (
    compare_representations_by_scope,
    compare_training_conditions_by_scope,
)
from aas_shortcut.create_figures.draw_hero_figures import (
    AUGMENTATION_BLOCKS,
    HERO_FORMATS,
    figure_representation_hero,
)
from aas_shortcut.define_paths import (
    TRANSFORMS,
    RunPaths,
    representation_path,
)
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_UNSEEN,
    measurable_scopes,
    scope_families,
)
from aas_shortcut.define_warp_audits import AUDITS
from aas_shortcut.measure_augmented_regression import (
    AugmentedRegressionConfig,
    board_levels,
    group_board,
    group_spread,
    seed_levels,
)
from aas_shortcut.measure_representation_transfer import (
    PREMISE_TABLE,
    describe_target_invariance,
    invariance_holds,
    premise_table,
)
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_regression_cells import resolve_family_deployments
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.resolve_tuning_stages import resolve_pair_binding
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_learner_option,
    add_overrides,
    add_store_options,
    build_context,
    build_parser,
    chosen_feature_sets,
    compose_config,
    record_run,
    resolve_run_paths,
    store_policy,
)
from aas_shortcut.sample_monotone_transform import (
    WARP_STRENGTH_TABLE,
    warp_strengths,
)
from aas_shortcut.split_bbob_instances import TEST_INSTANCES

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# The step whose pair tuning every cell binds from.
TUNING_STEP = "E2-1"

# Three groups by six cells, per held-out block and range. On the unseen families'
# range a cell is their mean with their spread beside it.
LEVELS_TABLE = Path("group_regression_levels.csv")

# The two planned comparisons: a direction and a p-value each, never a gap.
REPRESENTATION_TEST_TABLE = Path("representation_tests.csv")
TRAINING_TEST_TABLE = Path("training_tests.csv")

# Relative tolerance the untransformed anchor is checked at.
ANCHOR_TOLERANCE = 1e-9

TRAINING_BLOCK = "training block"


def check_premises(superiorities: pd.DataFrame, block: str) -> bool:
    """Check that G1's superiority did not move on a block.

    G1's members reproduce each trial bit for bit under a strictly increasing map,
    so any difference is a broken measurement, not a small effect.
    """
    if invariance_holds(superiorities):
        return True
    LOGGER.error(
        "the G1 superiority moved on the %s; its members search by comparison "
        "alone, so any difference is a broken measurement",
        block,
    )
    return False


def report_audits(
    table: pd.DataFrame,
    deployments: pd.DataFrame,
    config: AugmentedRegressionConfig,
    summary_paths: RunPaths,
) -> list[Path]:
    """Write one feature set's levels and tests.

    The spread is read from ``table`` (per-seed levels); the levels and the tests
    from ``deployments``, the per-deployment rows it was reduced from.
    """
    name = config.feature_set
    written: list[Path] = []
    target = summary_paths.root / representation_path(LEVELS_TABLE, name)
    target.parent.mkdir(parents=True, exist_ok=True)

    # A cell is the median over eleven training runs of each run's mean over the
    # range's families, with the spread of the families' own medians beside it
    # (never averaged in). The known family's range has one family, so no spread.
    scopes, lacking = measurable_scopes(config.audits)
    for scope, missing in lacking.items():
        LOGGER.warning(
            "the %r range needs %s, which this run did not audit", scope, missing
        )
    numbers: list[pd.DataFrame] = []
    for transform in TRANSFORMS:
        for scope in scopes:
            families = scope_families(scope)
            spread = (
                group_spread(table, scope, transform) if len(families) > 1 else None
            )
            mean = group_board(deployments, scope, transform)
            numbers.append(board_levels(mean, spread, scope, transform))
    pd.concat(numbers, ignore_index=True).to_csv(target, index=False)
    written.append(target)

    written += _report_tests(deployments, config, summary_paths)
    return written


def draw_conditions(
    tables: Mapping[str, pd.DataFrame], summary_paths: RunPaths
) -> list[Path]:
    """Draw both training conditions on the transformed block over unseen families.

    Read back from the levels table just written, so a bar is the reported number.
    """
    rows = []
    for feature_set in tables:
        levels = pd.read_csv(
            summary_paths.root / representation_path(LEVELS_TABLE, feature_set)
        )
        drawn = levels[
            (levels[SCOPE_COLUMN] == SCOPE_UNSEEN)
            & (levels["transform"] == TRANSFORMS[1])
        ]
        rows.append(
            pd.DataFrame(
                {
                    "feature_set": feature_set,
                    "group": drawn["group"],
                    "feature_variant": drawn["feature_variant"],
                    "metric": drawn["training"],
                    "value": drawn["level"],
                }
            )
        )
    report = pd.concat(rows, ignore_index=True)
    if report.empty:
        return []
    return figure_representation_hero(
        report, summary_paths, HERO_FORMATS, blocks=AUGMENTATION_BLOCKS
    )


def _report_tests(
    deployments: pd.DataFrame,
    config: AugmentedRegressionConfig,
    summary_paths: RunPaths,
) -> list[Path]:
    """Run the two planned comparisons and write their tables.

    The unit is the training run (eleven of them), so this reads the
    per-deployment rows rather than the levels.
    """
    name = config.feature_set
    written: list[Path] = []
    for tested, table_name in (
        # Which representation is above which, paired by seed, over the unseen
        # families' mean and over the known family alone.
        (
            compare_representations_by_scope(deployments, config.audits),
            REPRESENTATION_TEST_TABLE,
        ),
        # Augmentation against its absence, unpaired, over the same two ranges.
        (
            compare_training_conditions_by_scope(deployments, config.audits),
            TRAINING_TEST_TABLE,
        ),
    ):
        if tested.empty:
            LOGGER.warning("no range is measurable from the families this run audited")
            continue
        target = summary_paths.root / representation_path(table_name, name)
        tested.to_csv(target, index=False)
        written.append(target)
    return written


def check_anchor(training: TrainingBlock, block: HeldOutBlock) -> int:
    """Count the untransformed features of a held-out block that match training.

    That half does not depend on the family; if it moved, the block would score a
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


def write_premises(
    training: TrainingBlock,
    blocks: Sequence[HeldOutBlock],
    anchors: dict[str, int],
    context,
    run_paths: RunPaths,
) -> list[Path]:
    """Write the premise checks' counts and the augmentation maps' strengths.

    A strength is how far a training problem's augmentation map departs from the
    identity.
    """
    movements = {TRAINING_BLOCK: describe_target_invariance(training.superiorities)}
    movements |= {
        block.label: describe_target_invariance(block.superiorities) for block in blocks
    }
    premises = run_paths.root / PREMISE_TABLE
    premise_table(movements, anchors, ANCHOR_TOLERANCE).to_csv(premises, index=False)
    strengths = run_paths.root / WARP_STRENGTH_TABLE
    warp_strengths(
        training.superiorities, int(context.seed), context.dim, context.warp
    ).to_csv(strengths, index=False)
    return [premises, strengths]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run Experiment 4: the superiority regression for all three algorithm "
            "groups and three representations, fitted with and without transform "
            "augmentation, scored on the held-out block under five transformation "
            "families. Trailing arguments are passed to Hydra as overrides."
        ),
        epilog="example: run_experiment4.py task.n_jobs=26",
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
    # Every family in one run: the families a cell is scored on enter its unit.
    audits = list(AUDITS)

    cfg = compose_config(
        spec.hydra_overrides(args.preset, args.overrides, learner=args.learner)
    )
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)
    feature_sets = chosen_feature_sets(context.feature_sets)
    invocation = {
        "audits": [audit.name for audit in audits],
        "feature_sets": feature_sets,
        "argv": list(argv or sys.argv[1:]),
    }

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
        training = resolve_training_block(store, context, binding, tuning, TUNING_STEP)
        blocks = resolve_held_out_in_range(
            store, context, binding, tuning, TUNING_STEP, audits
        )
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1
    if not blocks:
        print("no audit produced an order-preserving warp", file=sys.stderr)
        return 1

    anchors = {block.label: check_anchor(training, block) for block in blocks}
    for label, agreed in anchors.items():
        LOGGER.info(
            "anchor: %d untransformed features of %s agree with the training block",
            agreed,
            label,
        )
    if not check_premises(training.superiorities, TRAINING_BLOCK):
        LOGGER.error("experiment 4: a premise failed; the result is not readable")
        return 1
    for block in blocks:
        if not check_premises(block.superiorities, block.label):
            LOGGER.error("experiment 4: a premise failed under %s", block.label)
            return 1
    run_paths.mkdirs()
    for path in write_premises(training, blocks, anchors, context, run_paths):
        LOGGER.info("written: %s", path)

    scored = [audit for audit in audits if audit.label in {b.label for b in blocks}]
    labels = tuple(audit.label for audit in scored)
    candidates = tuple(cfg.task.model_selection.candidates)
    deployments: dict[str, pd.DataFrame] = {}
    try:
        for feature_set in feature_sets:
            deployments[feature_set] = resolve_family_deployments(
                store, context, feature_set, candidates, training, blocks, scored
            )
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    tables = {
        feature_set: seed_levels(rows) for feature_set, rows in deployments.items()
    }
    for feature_set, table in tables.items():
        config = AugmentedRegressionConfig(feature_set=feature_set, audits=labels)
        for path in report_audits(table, deployments[feature_set], config, run_paths):
            print(f"written: {path}")
    for path in draw_conditions(tables, run_paths):
        print(f"written: {path}")

    store.write_list(run_paths.root)
    record_run(spec, cfg, run_paths, extra=invocation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
