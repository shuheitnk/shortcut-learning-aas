"""The units the closing appendices read, resolved as Experiments 1, 4 and 5 do.

Every regression cell of Experiment 4 (and Experiment 1's `invariant` cell) with its
per-problem predictions, and every selection cell of Experiment 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from aas_shortcut.compare_predictor_sets import (
    ALL_PREDICTORS,
    INVARIANT_PREDICTORS,
    REPORTED_PAIR,
    REPORTED_VARIANT,
    invariant_cell_config,
    invariant_columns,
)
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import ELA_FEATURE_SET
from aas_shortcut.define_warp_audits import AUDITS, CONTROL
from aas_shortcut.measure_augmented_regression import DEPLOYMENT_SHARD
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_portfolio_blocks import (
    SelectionHeldOut,
    SelectionTraining,
    resolve_portfolio_bindings,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
)
from aas_shortcut.resolve_prediction_cells import (
    read_predictions,
    resolve_prediction_cells,
)
from aas_shortcut.resolve_predictor_cells import invariant_cell_data
from aas_shortcut.resolve_regression_cells import (
    RegressionCell,
    cell_table,
    cells_of,
    family_regression_config,
    regression_data,
    resolve_regression_cells,
)
from aas_shortcut.resolve_selection_cells import (
    FamilySelection,
    resolve_family_selection,
)
from aas_shortcut.resolve_stage_artifacts import StageStore
from aas_shortcut.resolve_tuning_stages import resolve_pair_binding

# Experiment 4's pair tuning, which every regression cell binds from.
TUNING_STEP = "E2-1"

# Which predictor set a regression row was fitted on.
PREDICTORS = "predictors"


@dataclass(frozen=True)
class RegressionTables:
    """Experiment 4's blocks, and every cell's predictions and stored scores."""

    training: TrainingBlock
    blocks: tuple[HeldOutBlock, ...]
    predictions: pd.DataFrame
    deployments: pd.DataFrame


@dataclass(frozen=True)
class SelectionTables:
    """Experiment 5's blocks and units."""

    training: SelectionTraining
    blocks: tuple[SelectionHeldOut, ...]
    units: FamilySelection


def resolve_regression_tables(
    store: StageStore,
    context: ExperimentContext,
    candidates: Sequence[str],
    feature_sets: Sequence[str],
) -> RegressionTables:
    """Resolve every regression cell's predictions; refit the missing ones."""
    binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
    training = resolve_training_block(store, context, binding, tuning, TUNING_STEP)
    blocks = resolve_held_out_in_range(
        store, context, binding, tuning, TUNING_STEP, AUDITS
    )
    predictions: list[pd.DataFrame] = []
    deployments: list[pd.DataFrame] = []
    for feature_set in feature_sets:
        config = family_regression_config(context, feature_set, candidates)
        cells = cells_of(config)
        data = regression_data(training, blocks, config.features)
        sources = resolve_regression_cells(store, config, cells, data, context.scale)
        predicted = resolve_prediction_cells(store, config, cells, data, context.scale)
        predictions.append(
            read_predictions(predicted).assign(**{PREDICTORS: ALL_PREDICTORS})
        )
        deployments += [
            sources[cell]
            .read_table(cell_table(DEPLOYMENT_SHARD))
            .assign(**{PREDICTORS: ALL_PREDICTORS})
            for cell in cells
        ]
    if ELA_FEATURE_SET in feature_sets:
        control = next(block for block in blocks if block.label == CONTROL.label)
        columns = invariant_columns(context.scale)
        config = invariant_cell_config(
            context.model, candidates, columns, context.n_jobs
        )
        cell = RegressionCell(
            REPORTED_PAIR, REPORTED_VARIANT, TRAINING_PLAIN, candidates[0]
        )
        data = invariant_cell_data(training, control, columns)
        source = resolve_regression_cells(store, config, [cell], data, context.scale)
        predicted = resolve_prediction_cells(store, config, [cell], data, context.scale)
        predictions.append(
            read_predictions(predicted).assign(**{PREDICTORS: INVARIANT_PREDICTORS})
        )
        deployments.append(
            source[cell]
            .read_table(cell_table(DEPLOYMENT_SHARD))
            .assign(**{PREDICTORS: INVARIANT_PREDICTORS})
        )
    return RegressionTables(
        training,
        tuple(blocks),
        pd.concat(predictions, ignore_index=True),
        pd.concat(deployments, ignore_index=True),
    )


def resolve_selection_tables(
    store: StageStore, context: ExperimentContext
) -> SelectionTables:
    """Resolve Experiment 5's selection cells and oracles."""
    bindings = resolve_portfolio_bindings(store, context)
    training = resolve_selection_training(store, context, bindings)
    blocks = resolve_selection_held_out_in_range(store, context, bindings, list(AUDITS))
    units = resolve_family_selection(store, context, training, blocks)
    return SelectionTables(training, tuple(blocks), units)
