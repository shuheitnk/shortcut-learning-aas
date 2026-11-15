"""Appendix - further test blocks of the control family: the identity map and redraws.

``spline-alpha0`` maps the rescaled objective by the identity, so only the
rescaling to [0, 1] remains; ``spline-draw1`` to ``spline-draw4`` draw the control
family's maps again. Every portfolio member runs on each block under each
binding. The experiments' cells are fitted again, unchanged, and scored on the
five families and the new blocks together; the run stops unless every score on
the five families is the stored one bit for bit.

uv run python scripts/appendix_blocks/run_appendix_blocks.py \
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

import numpy as np
import pandas as pd

from aas_shortcut.compare_predictor_sets import (
    ALL_PREDICTORS,
    INVARIANT_PREDICTORS,
    REPORTED_PAIR,
    REPORTED_VARIANT,
    invariant_cell_config,
    invariant_columns,
)
from aas_shortcut.compare_selection_runs import FAMILY_COLUMN, view_levels
from aas_shortcut.compare_view_performance import trajectory_identity
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import ELA_FEATURE_SET, representation_path
from aas_shortcut.define_warp_audits import APPENDIX_BLOCKS, AUDITS, CONTROL, REDRAWS
from aas_shortcut.describe_target_spread import member_means
from aas_shortcut.measure_augmented_regression import DEPLOYMENT_SHARD, seed_levels
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.resolve_appendix_inputs import TUNING_STEP
from aas_shortcut.resolve_pair_blocks import (
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_portfolio_blocks import (
    CONDITIONS,
    resolve_portfolio_bindings,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
)
from aas_shortcut.resolve_predictor_cells import invariant_cell_data
from aas_shortcut.resolve_regression_cells import (
    RegressionCell,
    RegressionData,
    cell_table,
    cells_of,
    family_regression_config,
    regression_data,
    resolve_regression_cells,
)
from aas_shortcut.resolve_selection_cells import (
    SelectionSetting,
    assemble_records,
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
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS
from aas_shortcut.summarize_block_shift import superiority_shift
from aas_shortcut.summarize_draw_spread import draw_spread

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

G1_MEMBERS = (REPORTED_PAIR.first, REPORTED_PAIR.second)
G3_PAIR = next(pair for pair in FOCUS_PAIRS if group_of(pair) == "G3")
REGRESSION_BINDING = "regression"


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run the appendix's further test blocks (the identity map and four "
            "redraws of the control family). Trailing arguments are passed to Hydra "
            "as overrides."
        ),
        epilog="example: run_appendix_blocks.py task.n_jobs=26",
    )
    add_store_options(parser)
    return add_overrides(parser).parse_args(argv)


def check_reproduced(
    refit: pd.DataFrame, stored: pd.DataFrame, keys: Sequence[str], value: str
) -> int:
    """Stop unless the refit holds every stored row with the same value, bit for bit."""
    joined = stored[[*keys, value]].merge(
        refit[[*keys, value]], on=list(keys), how="left", suffixes=("", "_refit")
    )
    held = joined[value].to_numpy(dtype=np.float64)
    again = joined[f"{value}_refit"].to_numpy(dtype=np.float64)
    if np.isnan(again).any() or not np.array_equal(held, again):
        differ = int(np.sum(~(held == again)))
        raise ValueError(
            f"{differ} of {len(joined)} stored value(s) are not reproduced by the refit"
        )
    return len(joined)


def _keyed(table: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    out = table.copy()
    for key in keys:
        out[key] = out[key].astype(str)
    return out


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
    every = (*AUDITS, *APPENDIX_BLOCKS)
    new_labels = [audit.label for audit in APPENDIX_BLOCKS]
    old_labels = [audit.label for audit in AUDITS]

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
        training = resolve_training_block(store, context, binding, tuning, TUNING_STEP)
        blocks = resolve_held_out_in_range(
            store, context, binding, tuning, TUNING_STEP, every
        )
        bindings = resolve_portfolio_bindings(store, context)
        selection_training = resolve_selection_training(store, context, bindings)
        selection_blocks = resolve_selection_held_out_in_range(
            store, context, bindings, list(every)
        )
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1
    held = [block.label for block in blocks]
    if held != [audit.label for audit in every]:
        print(f"stopped: blocks out of range, resolved {held}", file=sys.stderr)
        return 1

    deployment_keys = [
        "feature_set",
        "feature_variant",
        "training",
        "pair",
        "seed",
        "rep",
        "audit",
        "metric",
    ]
    deployments: list[pd.DataFrame] = []
    records: list[pd.DataFrame] = []
    try:
        # Regression: every cell on the ten blocks, against its own five-family unit.
        for feature_set in feature_sets:
            stored_config = family_regression_config(context, feature_set, candidates)
            config = family_regression_config(context, feature_set, candidates, every)
            cells = cells_of(config)
            stored = resolve_regression_cells(
                store,
                stored_config,
                cells,
                regression_data(
                    training, blocks[: len(AUDITS)], stored_config.features
                ),
                context.scale,
            )
            refit = resolve_regression_cells(
                store,
                config,
                cells,
                regression_data(training, blocks, config.features),
                context.scale,
            )
            for cell in cells:
                table = refit[cell].read_table(cell_table(DEPLOYMENT_SHARD))
                check_reproduced(
                    _keyed(table, deployment_keys),
                    _keyed(
                        stored[cell].read_table(cell_table(DEPLOYMENT_SHARD)),
                        deployment_keys,
                    ),
                    deployment_keys,
                    "value",
                )
                deployments.append(table.assign(predictors=ALL_PREDICTORS))
        if ELA_FEATURE_SET in feature_sets:
            columns = invariant_columns(context.scale)
            cell = RegressionCell(
                REPORTED_PAIR, REPORTED_VARIANT, TRAINING_PLAIN, candidates[0]
            )
            controls = [b for b in blocks if b.label in (CONTROL.label, *new_labels)]
            stored_config = invariant_cell_config(
                context.model, candidates, columns, context.n_jobs
            )
            config = replace(stored_config, audits=tuple(b.label for b in controls))
            stored = resolve_regression_cells(
                store,
                stored_config,
                [cell],
                invariant_cell_data(training, blocks[0], columns),
                context.scale,
            )[cell]
            data = invariant_cell_data(training, blocks[0], columns)
            data = RegressionData(
                train_features=data.train_features,
                train_superiorities=data.train_superiorities,
                test_features={
                    b.label: invariant_cell_data(training, b, columns).test_features[
                        b.label
                    ]
                    for b in controls
                },
                test_superiorities={b.label: b.superiorities for b in controls},
                inputs={
                    k: v
                    for b in controls
                    for k, v in invariant_cell_data(training, b, columns).inputs.items()
                },
            )
            table = resolve_regression_cells(
                store, config, [cell], data, context.scale
            )[cell].read_table(cell_table(DEPLOYMENT_SHARD))
            check_reproduced(
                _keyed(table, deployment_keys),
                _keyed(
                    stored.read_table(cell_table(DEPLOYMENT_SHARD)), deployment_keys
                ),
                deployment_keys,
                "value",
            )
            deployments.append(table.assign(predictors=INVARIANT_PREDICTORS))

        # Selection: every cell on the ten blocks, against its five-family unit.
        selection_config = context.selection_config()
        features = paper_columns(context.feature_sets, context.scale)
        setting = SelectionSetting(
            features=tuple(features),
            dim=context.dim,
            base_seed=context.seed,
            sample_size=context.feature_config(PROBLEM_CLASS).sample_size,
        )
        selection_cells = selection_cells_of(selection_config)
        stored_units = resolve_selection_cells(
            store,
            selection_config,
            selection_cells,
            selection_data(
                selection_training, selection_blocks[: len(AUDITS)], features
            ),
            setting,
            context.scale,
        )
        refit_units = resolve_selection_cells(
            store,
            selection_config,
            selection_cells,
            selection_data(selection_training, selection_blocks, features),
            setting,
            context.scale,
        )
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    selection_keys = [
        "feature_set",
        "feature_variant",
        "training",
        "transform",
        "seed",
        "rep",
        FAMILY_COLUMN,
    ]
    refit_records = assemble_records(refit_units, [b.label for b in selection_blocks])
    stored_records = assemble_records(stored_units, old_labels)
    check_reproduced(
        _keyed(refit_records, selection_keys),
        _keyed(stored_records, selection_keys),
        selection_keys,
        "superiority_vs_sbs",
    )
    records.append(refit_records)
    regression = pd.concat(deployments, ignore_index=True)
    selection = pd.concat(records, ignore_index=True)

    # What each block did to the target and to G1's runs.
    shift = pd.concat(
        [superiority_shift(block.superiorities, block.label) for block in blocks],
        ignore_index=True,
    )
    members = pd.concat(
        [
            member_means(block.superiorities, block.label, G3_PAIR.slug)
            for block in blocks
        ],
        ignore_index=True,
    )
    identity_frames = []
    for block in blocks:
        performance = pd.concat(
            [block.pre.performance, block.post.performance], ignore_index=True
        )
        identity_frames.append(
            trajectory_identity(
                cast(
                    pd.DataFrame,
                    performance[performance["algorithm"].isin(list(G1_MEMBERS))],
                )
            ).assign(binding=REGRESSION_BINDING, block=block.label)
        )
    for block in selection_blocks:
        for condition in CONDITIONS:
            performance = block.performance(condition)
            identity_frames.append(
                trajectory_identity(
                    cast(
                        pd.DataFrame,
                        performance[performance["algorithm"].isin(list(G1_MEMBERS))],
                    )
                ).assign(binding=condition, block=block.label)
            )
    identity = pd.concat(identity_frames, ignore_index=True)

    regression_levels = pd.concat(
        [
            seed_levels(
                cast(pd.DataFrame, regression[regression["predictors"] == name])
            ).assign(predictors=name)
            for name in (ALL_PREDICTORS, INVARIANT_PREDICTORS)
            if (regression["predictors"] == name).any()
        ],
        ignore_index=True,
    )
    selection_levels = view_levels(selection)
    draws = [CONTROL.label, *(audit.label for audit in REDRAWS)]

    run_paths.mkdirs()
    for name, table in (
        ("block_regression_levels.csv", regression_levels),
        ("block_selection_levels.csv", selection_levels),
        (
            "block_regression_draw_spread.csv",
            draw_spread(regression_levels, "audit", draws),
        ),
        (
            "block_selection_draw_spread.csv",
            draw_spread(selection_levels, FAMILY_COLUMN, draws),
        ),
    ):
        for feature_set in feature_sets:
            target = run_paths.root / representation_path(Path(name), feature_set)
            table[table["feature_set"].astype(str) == feature_set].to_csv(
                target, index=False
            )
            print(f"written: {target}")
    for name, table in (
        ("block_target_shift.csv", shift),
        ("block_g3_members.csv", members),
        ("block_g1_identity.csv", identity),
    ):
        target = run_paths.root / name
        table.to_csv(target, index=False)
        print(f"written: {target}")
    store.write_list(run_paths.root)
    record_run(spec, cfg, run_paths, extra={"blocks": new_labels})
    return 0


if __name__ == "__main__":
    sys.exit(main())
