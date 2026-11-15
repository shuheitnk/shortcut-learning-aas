"""Appendix - checks of the paper's statements from the stored results.

Reads Experiments 4 and 5's units (refitting Experiment 4's cells for their
per-problem predictions, as the bootstrap appendix does) and writes:

* the selection's Θ on the untransformed test block, with and without
  augmentation, and their Mann-Whitney tests;
* G1's mean |Ĵ(p̃) − Ĵ(p)| with and without augmentation, and its tests;
* the target's variance per group and view, the regression's MSE beside its R²,
  and G3's members' mean superiorities;
* every cell's level on each family;
* the features the non-finite rule drops when it reads the training block alone;
* whether G1's runs are the same before and after the transformation.

uv run python scripts/appendix_checks/run_appendix_checks.py \
    paths.output_root=outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.compare_predictor_sets import ALL_PREDICTORS, REPORTED_PAIR
from aas_shortcut.compare_selection_runs import (
    CELL_COLUMNS as SELECTION_CELL,
)
from aas_shortcut.compare_selection_runs import (
    FAMILY_COLUMN,
    compare_selection_trainings,
    seed_values,
    view_levels,
)
from aas_shortcut.compare_training_runs import (
    compare_training_conditions,
    median_over_seeds,
    per_seed,
)
from aas_shortcut.compare_view_performance import trajectory_identity
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_feature_columns import EXCLUDED_COLUMNS
from aas_shortcut.define_paths import MANIFEST_SUFFIX, representation_path
from aas_shortcut.define_reporting_scopes import SCOPES, scope_families
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.describe_target_spread import member_means, target_spread
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.find_nonfinite_features import excluded_by, nonfinite_observations
from aas_shortcut.measure_augmented_regression import seed_levels
from aas_shortcut.measure_prediction_change import (
    CHANGE_METRIC,
    ERROR_METRICS,
    prediction_change,
    squared_errors,
)
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.resolve_appendix_inputs import (
    PREDICTORS,
    RegressionTables,
    SelectionTables,
    resolve_regression_tables,
    resolve_selection_tables,
)
from aas_shortcut.resolve_portfolio_blocks import CONDITIONS
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageArtifact,
    StageStore,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
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
from aas_shortcut.split_bbob_instances import TRAIN_INSTANCES

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# The untransformed test block: the same for every family.
UNTRANSFORMED = "untransformed test"

G1_MEMBERS = (REPORTED_PAIR.first, REPORTED_PAIR.second)
G3_PAIR = next(pair for pair in FOCUS_PAIRS if group_of(pair) == "G3")

# The regression's binding is the pair tuning; the selection's two are the roles.
REGRESSION_BINDING = "regression"
TRAINING_BLOCK = "training"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run the appendix's checks of the paper's statements on the stored "
            "results of Experiments 4 and 5. Trailing arguments are passed to Hydra "
            "as overrides."
        ),
        epilog="example: run_appendix_checks.py task.n_jobs=26",
    )
    add_store_options(parser)
    return add_overrides(parser).parse_args(argv)


def untransformed_selection(records: pd.DataFrame) -> pd.DataFrame:
    """Every selection seed's Θ on the untransformed block.

    The block is shared by every family, so each family must hold the same values.
    """
    keys = [*SELECTION_CELL, "seed", "rep"]
    pre = records[records["transform"].astype(str) == "pre"]
    wide = pre.pivot_table(
        index=keys, columns=FAMILY_COLUMN, values="superiority_vs_sbs"
    )
    if not (wide.to_numpy() == wide.to_numpy()[:, :1]).all():
        raise ValueError("the untransformed block's Θ differs between families")
    control = cast(pd.DataFrame, pre[pre[FAMILY_COLUMN] == CONTROL.label])
    return seed_values(control, view="pre")


def selection_on_untransformed(
    records: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The levels on the untransformed block and the augmentation tests there."""
    runs = untransformed_selection(records)
    levels = median_over_seeds(runs, SELECTION_CELL).assign(scope=UNTRANSFORMED)
    return levels, compare_selection_trainings(runs, UNTRANSFORMED)


def g1_prediction_change(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """G1's mean prediction change per range, and augmentation's tests on it.

    Rank-encoded cells move by nothing when the ranks do; they are reported, not
    tested.
    """
    rows = predictions[
        (predictions[PREDICTORS] == ALL_PREDICTORS)
        & (predictions["group"].astype(str) == group_of(REPORTED_PAIR))
    ]
    change = prediction_change(cast(pd.DataFrame, rows))
    levels, tests = [], []
    for scope in SCOPES:
        runs = per_seed(change, scope, CHANGE_METRIC)
        levels.append(median_over_seeds(runs).assign(scope=scope))
        tested = runs[runs["feature_variant"].isin(["value", "concat"])]
        tests.append(compare_training_conditions(cast(pd.DataFrame, tested), scope))
    return pd.concat(levels, ignore_index=True), pd.concat(tests, ignore_index=True)


def error_levels(regression: RegressionTables, spreads: pd.DataFrame) -> pd.DataFrame:
    """R², MSE and the target's variance of every cell, per range and view."""
    cells = regression.predictions[regression.predictions[PREDICTORS] == ALL_PREDICTORS]
    errors = squared_errors(cast(pd.DataFrame, cells))
    scored = regression.deployments[
        regression.deployments[PREDICTORS] == ALL_PREDICTORS
    ]
    frames = []
    for scope in SCOPES:
        families = scope_families(scope)
        for view in ("pre", "post"):
            r2 = median_over_seeds(
                per_seed(cast(pd.DataFrame, scored), scope, TEST_METRICS[view])
            ).rename(columns={"value": "r2"})
            mse = median_over_seeds(
                per_seed(errors, scope, ERROR_METRICS[view])
            ).rename(columns={"value": "mse"})
            variance = (
                spreads[
                    spreads["audit"].isin(list(families))
                    & (spreads["transform"] == view)
                ]
                .groupby("group")["variance"]
                .mean()
                .reset_index(name="target_variance")
            )
            frames.append(
                r2.merge(
                    mse, on=["feature_set", "group", "feature_variant", "training"]
                )
                .merge(variance, on="group")
                .assign(scope=scope, transform=view)
            )
    return pd.concat(frames, ignore_index=True)


def spread_tables(
    regression: RegressionTables, selection: SelectionTables
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The target's spread per family and view, and G3's members' means."""
    spreads = pd.concat(
        [
            target_spread(block.superiorities, block.label)
            for block in regression.blocks
        ],
        ignore_index=True,
    )
    members = [
        member_means(block.superiorities, block.label, G3_PAIR.slug).assign(
            measure="superiority over the other G3 member (regression binding)"
        )
        for block in regression.blocks
    ]
    for block in selection.blocks:
        table = block.mean_superiorities(CONDITIONS[0])
        table = table[table["algorithm"].isin([G3_PAIR.first, G3_PAIR.second])]
        means = (
            table.groupby(["transform", "algorithm"], observed=True)["mean_superiority"]
            .mean()
            .reset_index(name="mean")
        )
        members.append(
            means.assign(
                audit=block.label,
                measure="mean superiority over the portfolio (selection binding)",
            )
        )
    return spreads, pd.concat(members, ignore_index=True)


def family_levels(
    regression: RegressionTables, records: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every cell's level on each family and view, regression and selection."""
    scored = regression.deployments[
        regression.deployments[PREDICTORS] == ALL_PREDICTORS
    ]
    return seed_levels(cast(pd.DataFrame, scored)), view_levels(records)


def manifest_of(artifact: StageArtifact) -> dict:
    """The one shard manifest a feature unit holds."""
    names = [
        name for name in artifact.record["files"] if name.endswith(MANIFEST_SUFFIX)
    ]
    if len(names) != 1:
        raise ValueError(f"{artifact.path} holds {len(names)} shard manifest(s)")
    return json.loads(artifact.file(names[0]).read_text(encoding="utf-8"))


def exclusion_rules(regression: RegressionTables) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every non-finite observation, and the columns each reading of the rule drops."""
    observations = pd.concat(
        [
            nonfinite_observations(
                manifest_of(regression.training.features),
                TRAINING_BLOCK,
                TRAIN_INSTANCES,
            )
        ]
        + [
            nonfinite_observations(manifest_of(block.features), block.label)
            for block in regression.blocks
        ],
        ignore_index=True,
    )
    blocks = [TRAINING_BLOCK, *(block.label for block in regression.blocks)]
    rules = pd.DataFrame(
        [
            {
                "rule": "declared (every block the paper reads)",
                "excluded": ";".join(sorted(EXCLUDED_COLUMNS)),
            },
            {
                "rule": "recomputed over every block the paper reads",
                "excluded": ";".join(excluded_by(observations, blocks)),
            },
            {
                "rule": "training block alone (both views, instances 11-110)",
                "excluded": ";".join(excluded_by(observations, [TRAINING_BLOCK])),
            },
        ]
    )
    counts = (
        observations.groupby(["feature", "block", "transform"], observed=True)
        .size()
        .reset_index(name="observations")
    )
    return rules, counts


def g1_identity(
    regression: RegressionTables, selection: SelectionTables
) -> pd.DataFrame:
    """G1's runs before against after the transformation, every transformed block."""
    blocks: list[tuple[str, str, pd.DataFrame]] = [
        (
            REGRESSION_BINDING,
            TRAINING_BLOCK,
            pd.concat(
                [unit.performance for unit in regression.training.units.values()],
                ignore_index=True,
            ),
        )
    ]
    blocks += [
        (
            REGRESSION_BINDING,
            block.label,
            pd.concat(
                [block.pre.performance, block.post.performance], ignore_index=True
            ),
        )
        for block in regression.blocks
    ]
    for condition, unit in selection.training.post.items():
        blocks.append(
            (
                condition,
                TRAINING_BLOCK,
                pd.concat(
                    [selection.training.pre[condition].performance, unit.performance],
                    ignore_index=True,
                ),
            )
        )
    for block in selection.blocks:
        for condition in CONDITIONS:
            blocks.append((condition, block.label, block.performance(condition)))
    frames = []
    for binding, label, performance in blocks:
        g1 = performance[performance["algorithm"].isin(list(G1_MEMBERS))]
        frames.append(
            trajectory_identity(cast(pd.DataFrame, g1)).assign(
                binding=binding, block=label
            )
        )
    table = pd.concat(frames, ignore_index=True)
    return table[
        ["binding", "block", *(c for c in table if c not in ("binding", "block"))]
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the appendix and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    spec = ExperimentSpec.load(SPEC_PATH)
    cfg = compose_config(spec.hydra_overrides(args.preset, args.overrides))
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)
    feature_sets = chosen_feature_sets(context.feature_sets)
    candidates = list(cfg.task.model_selection.candidates)
    if len(candidates) != 1:
        print(f"stopped: one learner expected, got {candidates}", file=sys.stderr)
        return 1

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        regression = resolve_regression_tables(store, context, candidates, feature_sets)
        selection = resolve_selection_tables(store, context)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    records = selection.units.records()
    spreads, members = spread_tables(regression, selection)
    selection_levels, selection_tests = selection_on_untransformed(records)
    change_levels, change_tests = g1_prediction_change(regression.predictions)
    errors = error_levels(regression, spreads)
    regression_families, selection_families = family_levels(regression, records)
    rules, counts = exclusion_rules(regression)
    identity = g1_identity(regression, selection)

    run_paths.mkdirs()
    per_feature_set = {
        "untransformed_selection_levels.csv": selection_levels,
        "untransformed_selection_tests.csv": selection_tests,
        "g1_prediction_change_levels.csv": change_levels,
        "g1_prediction_change_tests.csv": change_tests,
        "regression_error_levels.csv": errors,
        "regression_family_levels.csv": regression_families,
        "selection_family_levels.csv": selection_families,
    }
    for name, table in per_feature_set.items():
        for feature_set in feature_sets:
            target = run_paths.root / representation_path(Path(name), feature_set)
            table[table["feature_set"].astype(str) == feature_set].to_csv(
                target, index=False
            )
            print(f"written: {target}")
    for name, table in {
        "target_spread.csv": spreads,
        "g3_member_superiority.csv": members,
        "feature_exclusion_rules.csv": rules,
        "nonfinite_observations.csv": counts,
        "g1_trajectory_identity.csv": identity,
    }.items():
        target = run_paths.root / name
        table.to_csv(target, index=False)
        print(f"written: {target}")

    declared, training_only = rules["excluded"].iloc[0], rules["excluded"].iloc[2]
    if declared != training_only:
        LOGGER.warning(
            "the training block alone drops %s, not the declared %s",
            training_only,
            declared,
        )
    differing = int(np.sum(identity["runs_differ"]))
    LOGGER.info("G1 runs that differ between the views: %d", differing)
    store.write_list(run_paths.root)
    record_run(spec, cfg, run_paths, extra={"feature_sets": feature_sets})
    return 0


if __name__ == "__main__":
    sys.exit(main())
