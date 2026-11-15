"""Appendix - plain-4800: the plain training condition at the augmented one's size.

The plain cells are fitted again on instances 11-210, all untransformed, with the
augmented condition's update budget; the members run on the further instances
111-210 under the pair and selection bindings. The SBS stays the one named on the
paper's training block. The cells are scored on the five families' test blocks and
set beside the stored plain and augmented cells.

uv run python scripts/appendix_doubled/run_appendix_doubled.py \
    paths.output_root=outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.compare_selection_runs import (
    CELL_COLUMNS as SELECTION_CELL,
)
from aas_shortcut.compare_selection_runs import (
    FAMILY_COLUMN,
    compare_selection_trainings,
    seed_values,
    selection_runs,
    view_levels,
)
from aas_shortcut.compare_training_runs import (
    compare_training_conditions,
    median_over_seeds,
    per_seed,
)
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import representation_path
from aas_shortcut.define_reporting_scopes import SCOPE_CONTROL, measurable_scopes
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_augmented_regression import (
    DEPLOYMENT_SHARD,
    seed_levels,
)
from aas_shortcut.measure_augmented_regression import (
    training_matrix as regression_matrix,
)
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
)
from aas_shortcut.resolve_appendix_inputs import TUNING_STEP
from aas_shortcut.resolve_further_training import (
    PLAIN_4800_INSTANCES,
    TRAINING_PLAIN_4800,
    check_same_matrix,
    check_shared_features,
    plain_4800_model,
    plain_4800_regression_data,
    plain_4800_selection_data,
    relabel,
    resolve_further_features,
    resolve_further_pairs,
    resolve_further_portfolio,
)
from aas_shortcut.resolve_pair_blocks import (
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_portfolio_blocks import (
    resolve_portfolio_bindings,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
)
from aas_shortcut.resolve_regression_cells import (
    cell_table,
    cells_of,
    family_regression_config,
    regression_data,
    resolve_regression_cells,
)
from aas_shortcut.resolve_selection_cells import (
    SelectionSetting,
    assemble_records,
    cell_config,
    resolve_selection_cells,
    selection_data,
)
from aas_shortcut.resolve_selection_cells import cells_of as selection_cells_of
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.resolve_tuning_stages import resolve_pair_binding
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
from aas_shortcut.select_algorithms import training_matrix as selection_matrix
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS, TRAIN_INSTANCES

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# Augmentation against the plain condition of the same size.
CONDITIONS = (TRAINING_AUGMENTED, TRAINING_PLAIN_4800)
UNTRANSFORMED = "untransformed test"
PRE, POST = TEST_METRICS["pre"], TEST_METRICS["post"]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run the appendix's plain-4800 training condition (instances 11-210, "
            "untransformed). Trailing arguments are passed to Hydra as overrides."
        ),
        epilog="example: run_appendix_doubled.py task.n_jobs=26",
    )
    add_store_options(parser)
    return add_overrides(parser).parse_args(argv)


def untransformed_runs(deployments: pd.DataFrame) -> pd.DataFrame:
    """Every regression seed's R² on the untransformed block, shared by the families."""
    pre = deployments[deployments["metric"] == PRE]
    keys = ["feature_set", "feature_variant", "training", "pair", "seed", "rep"]
    wide = pre.pivot_table(index=keys, columns="audit", values="value")
    if not (wide.to_numpy() == wide.to_numpy()[:, :1]).all():
        raise ValueError("the untransformed block's R² differs between families")
    control = cast(pd.DataFrame, pre[pre["audit"] == CONTROL.label])
    return per_seed(control, SCOPE_CONTROL, metric=PRE)


def regression_reports(
    deployments: pd.DataFrame, families: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-family levels, levels over each range, and the augmentation tests."""
    scopes, _ = measurable_scopes(families)
    untransformed = untransformed_runs(deployments)
    levels = [median_over_seeds(untransformed).assign(scope=UNTRANSFORMED)]
    tests = [compare_training_conditions(untransformed, UNTRANSFORMED, CONDITIONS)]
    for scope in scopes:
        runs = per_seed(deployments, scope, metric=POST)
        levels.append(median_over_seeds(runs).assign(scope=scope))
        tests.append(compare_training_conditions(runs, scope, CONDITIONS))
    return (
        seed_levels(deployments),
        pd.concat(levels, ignore_index=True),
        pd.concat(tests, ignore_index=True),
    )


def selection_reports(
    records: pd.DataFrame, families: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-family levels, levels over each range, and the augmentation tests."""
    scopes, _ = measurable_scopes(families)
    pre = records[records["transform"].astype(str) == "pre"]
    wide = pre.pivot_table(
        index=[*SELECTION_CELL, "seed", "rep"],
        columns=FAMILY_COLUMN,
        values="superiority_vs_sbs",
    )
    if not (wide.to_numpy() == wide.to_numpy()[:, :1]).all():
        raise ValueError("the untransformed block's Θ differs between families")
    untransformed = seed_values(
        cast(pd.DataFrame, pre[pre[FAMILY_COLUMN] == CONTROL.label]), view="pre"
    )
    levels = [
        median_over_seeds(untransformed, SELECTION_CELL).assign(scope=UNTRANSFORMED)
    ]
    tests = [compare_selection_trainings(untransformed, UNTRANSFORMED, CONDITIONS)]
    for scope in scopes:
        runs = selection_runs(records, scope)
        levels.append(median_over_seeds(runs, SELECTION_CELL).assign(scope=scope))
        tests.append(compare_selection_trainings(runs, scope, CONDITIONS))
    return (
        view_levels(records),
        pd.concat(levels, ignore_index=True),
        pd.concat(tests, ignore_index=True),
    )


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

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
        training = resolve_training_block(store, context, binding, tuning, TUNING_STEP)
        blocks = resolve_held_out_in_range(store, context, binding, tuning, TUNING_STEP)
        bindings = resolve_portfolio_bindings(store, context)
        selection_training = resolve_selection_training(store, context, bindings)
        selection_blocks = resolve_selection_held_out_in_range(store, context, bindings)
        further = resolve_further_features(store, context)
        pairs = resolve_further_pairs(store, context, binding, tuning, TUNING_STEP)
        portfolio = resolve_further_portfolio(store, context, bindings)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1
    shared = check_shared_features(further.feature_table, training.feature_table)
    LOGGER.info("the wider feature unit holds the paper's %d value(s)", shared)
    families = [block.label for block in blocks]
    plain_training = replace(training, post=None)
    plain_selection = replace(selection_training, post={})

    deployments: list[pd.DataFrame] = []
    try:
        for feature_set in feature_sets:
            stored_config = family_regression_config(context, feature_set, candidates)
            stored = resolve_regression_cells(
                store,
                stored_config,
                cells_of(stored_config),
                regression_data(training, blocks, stored_config.features),
                context.scale,
            )
            config = replace(
                stored_config,
                trainings=(TRAINING_PLAIN,),
                train_instances=PLAIN_4800_INSTANCES,
                model=plain_4800_model(stored_config.model),
            )
            data = plain_4800_regression_data(
                further, plain_training, pairs, blocks, config.features
            )
            paper = regression_data(plain_training, blocks, config.features)
            for cell in cells_of(config):
                check_same_matrix(
                    regression_matrix(
                        data.train_features,
                        data.train_superiorities,
                        cell.variant,
                        cell.training,
                        cell.pair.slug,
                        replace(config, train_instances=TRAIN_INSTANCES),
                    ),
                    regression_matrix(
                        paper.train_features,
                        paper.train_superiorities,
                        cell.variant,
                        cell.training,
                        cell.pair.slug,
                        stored_config,
                    ),
                )
            refit = resolve_regression_cells(
                store, config, cells_of(config), data, context.scale
            )
            deployments += [
                artifact.read_table(cell_table(DEPLOYMENT_SHARD))
                for artifact in stored.values()
            ]
            deployments += [
                relabel(artifact.read_table(cell_table(DEPLOYMENT_SHARD)))
                for artifact in refit.values()
            ]

        stored_config = context.selection_config()
        features = paper_columns(context.feature_sets, context.scale)
        setting = SelectionSetting(
            features=tuple(features),
            dim=context.dim,
            base_seed=context.seed,
            sample_size=context.feature_config(PROBLEM_CLASS).sample_size,
        )
        stored_units = resolve_selection_cells(
            store,
            stored_config,
            selection_cells_of(stored_config),
            selection_data(selection_training, selection_blocks, features),
            setting,
            context.scale,
        )
        config = replace(stored_config, model=plain_4800_model(stored_config.model))
        cells = selection_cells_of(config, (TRAINING_PLAIN,))
        data = plain_4800_selection_data(
            further, plain_selection, portfolio, selection_blocks, features
        )
        paper = selection_data(plain_selection, selection_blocks, features)
        on_paper = data.features["instance_id"].isin(TRAIN_INSTANCES)
        for cell in cells:
            check_same_matrix(
                selection_matrix(
                    cast(pd.DataFrame, data.features[on_paper]),
                    cast(
                        pd.DataFrame,
                        data.mean_superiorities[
                            data.mean_superiorities["instance_id"].isin(TRAIN_INSTANCES)
                        ],
                    ),
                    cell.variant,
                    cell.feature_set,
                    cell_config(config, cell),
                )[0],
                selection_matrix(
                    paper.features,
                    paper.mean_superiorities,
                    cell.variant,
                    cell.feature_set,
                    cell_config(stored_config, cell),
                )[0],
            )
        refit_units = resolve_selection_cells(
            store, config, cells, data, setting, context.scale
        )
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    labels = [block.label for block in selection_blocks]
    records = pd.concat(
        [
            assemble_records(stored_units, labels),
            relabel(assemble_records(refit_units, labels)),
        ],
        ignore_index=True,
    )
    regression = pd.concat(deployments, ignore_index=True)

    run_paths.mkdirs()
    for kind, reports in (
        ("regression", regression_reports(regression, families)),
        ("selection", selection_reports(records, labels)),
    ):
        for name, table in zip(("levels", "scopes", "tests"), reports, strict=True):
            for feature_set in feature_sets:
                target = run_paths.root / representation_path(
                    Path(f"plain4800_{kind}_{name}.csv"), feature_set
                )
                table[table["feature_set"].astype(str) == feature_set].to_csv(
                    target, index=False
                )
                print(f"written: {target}")
    store.write_list(run_paths.root)
    record_run(
        spec,
        cfg,
        run_paths,
        extra={"train_instances": [PLAIN_4800_INSTANCES[0], PLAIN_4800_INSTANCES[-1]]},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
