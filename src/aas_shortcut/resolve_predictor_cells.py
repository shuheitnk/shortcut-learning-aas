"""Experiment 1's two cells: the `invariant` cell and Experiment 4's `all` cell.

The `invariant` cell (G1 x ``value`` x `plain`) is fitted on the structurally
invariant ELA columns. The `all` cell is the very unit Experiment 4 resolves, so
neither run fits it if the other has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.compare_predictor_sets import (
    ALL_PREDICTORS,
    INVARIANT_PREDICTORS,
    REPORTED_PAIR,
    REPORTED_VARIANT,
    Cell,
    SourceCells,
    invariant_cell_config,
    invariant_columns,
)
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import ELA_FEATURE_SET
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_augmented_regression import DEPLOYMENT_SHARD
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    resolve_held_out_blocks,
    resolve_training_block,
)
from aas_shortcut.resolve_regression_cells import (
    RegressionCell,
    RegressionData,
    family_regression_config,
    regression_data,
    resolve_regression_cells,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore
from aas_shortcut.resolve_tuning_stages import resolve_pair_binding
from aas_shortcut.split_bbob_instances import TRAIN_INSTANCES


@dataclass(frozen=True)
class PredictorBlocks:
    """The training block and every family's held-out block, resolved."""

    training: TrainingBlock
    held_out: dict[str, HeldOutBlock]

    @property
    def control(self) -> HeldOutBlock:
        """The block the comparison is measured on."""
        return self.held_out[CONTROL.label]


@dataclass(frozen=True)
class PredictorCells:
    """The two halves of the comparison, as units of the store."""

    invariant: StageArtifact
    all: StageArtifact
    columns: tuple[str, ...]


def resolve_predictor_blocks(
    store: StageStore, context: ExperimentContext, step: str
) -> PredictorBlocks:
    """Resolve the untransformed training block and the five held-out blocks.

    All five, because the `all` cell is named by every family's units. Both cells are
    plain, so the transformed training benchmark is not needed.
    """
    binding, tuning = resolve_pair_binding(store, context, step)
    training = resolve_training_block(
        store, context, binding, tuning, step, trainings=(TRAINING_PLAIN,)
    )
    held_out = resolve_held_out_blocks(store, context, binding, tuning, step)
    return PredictorBlocks(training, held_out)


def ela_values(features: pd.DataFrame) -> pd.DataFrame:
    """Keep the representation and the encoding the comparison is defined on."""
    return features[
        (features["feature_set"] == ELA_FEATURE_SET)
        & (features["feature_variant"] == REPORTED_VARIANT)
    ]


def _restrict(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Keep only the rows of the predictors the cell is fitted on."""
    return features[features["feature"].isin(list(columns))]


def invariant_cell_data(
    training: TrainingBlock, control: HeldOutBlock, columns: Sequence[str]
) -> RegressionData:
    """Return the `invariant` cell's training and test data: ELA values only."""
    source = ela_values(training.feature_table)
    return RegressionData(
        train_features=_restrict(
            cast(pd.DataFrame, source[source["instance_id"].isin(TRAIN_INSTANCES)]),
            columns,
        ),
        train_superiorities=training.superiorities,
        test_features={
            control.label: _restrict(ela_values(control.feature_table), columns)
        },
        test_superiorities={control.label: control.superiorities},
        inputs={**training.inputs, **control.inputs},
    )


def resolve_predictor_cells(
    store: StageStore,
    candidates: Sequence[str],
    context: ExperimentContext,
    blocks: PredictorBlocks,
) -> PredictorCells:
    """Resolve the two cells, fitting what is missing with ``candidates[0]``."""
    columns = invariant_columns(context.scale)
    config = invariant_cell_config(context.model, candidates, columns, context.n_jobs)
    cell = RegressionCell(
        REPORTED_PAIR, REPORTED_VARIANT, TRAINING_PLAIN, candidates[0]
    )
    invariant = resolve_regression_cells(
        store,
        config,
        [cell],
        invariant_cell_data(blocks.training, blocks.control, columns),
        context.scale,
    )[cell]
    family = family_regression_config(context, ELA_FEATURE_SET, candidates)
    every = resolve_regression_cells(
        store,
        family,
        [cell],
        regression_data(
            blocks.training, list(blocks.held_out.values()), family.features
        ),
        context.scale,
    )[cell]
    return PredictorCells(invariant, every, columns)


def _cell(predictors: str, artifact: StageArtifact, output_root: Path) -> Cell:
    """Read one half from its unit, which holds the stage's own table names."""
    # The store hands out absolute paths; the output root may be given relative.
    return Cell(
        predictors,
        str(artifact.path.resolve().relative_to(output_root.resolve())),
        DEPLOYMENT_SHARD,
    )


def comparison_sources(cells: PredictorCells, output_root: Path) -> SourceCells:
    """Return where the two halves of the comparison are read from."""
    return SourceCells(
        all_cell=_cell(ALL_PREDICTORS, cells.all, output_root),
        invariant_cell=_cell(INVARIANT_PREDICTORS, cells.invariant, output_root),
    )
