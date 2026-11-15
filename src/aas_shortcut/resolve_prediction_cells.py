"""The appendix's per-problem predictions as units of the stage store.

A unit is named by its regression cell's own settings and inputs, plus that cell's
unit, whose stored R² the predictions must reproduce before the unit is written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from aas_shortcut.define_paths import RunPaths
from aas_shortcut.measure_augmented_regression import (
    DEPLOYMENT_SHARD,
    AugmentedRegressionConfig,
)
from aas_shortcut.measure_regression_predictions import (
    PREDICTION_SCHEMA,
    PREDICTION_TABLE,
    STAGE,
    check_reproduced,
    measure_predictions,
)
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.resolve_regression_cells import (
    RegressionCell,
    RegressionData,
    cell_table,
    regression_cell_unit,
    resolve_regression_cells,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit

# The input role naming the regression unit a prediction unit reproduces.
SOURCE_ROLE = "regression_cell"


def prediction_unit(
    config: AugmentedRegressionConfig,
    cell: RegressionCell,
    inputs: Mapping[str, str],
) -> StageUnit:
    """Return the unit holding one regression cell's predictions."""
    source = regression_cell_unit(config, cell, inputs)
    return StageUnit(
        STAGE,
        {**source.settings, "records": "per-problem predictions"},
        inputs={**source.inputs, SOURCE_ROLE: source.fingerprint},
    )


def resolve_prediction_cells(
    store: StageStore,
    config: AugmentedRegressionConfig,
    cells: Sequence[RegressionCell],
    data: RegressionData,
    scale: str,
) -> dict[RegressionCell, StageArtifact]:
    """Return every cell's predictions, refitting the missing cells together.

    The regression units are resolved first: a prediction unit is written only once
    it reproduces its cell's stored scores.
    """
    sources = resolve_regression_cells(store, config, cells, data, scale)
    units = {cell: prediction_unit(config, cell, data.inputs) for cell in cells}

    def compute(
        missing: Sequence[StageUnit], scratches: Mapping[str, RunPaths]
    ) -> None:
        wanted = [cell for cell, unit in units.items() if unit in missing]
        for cell, predictions in measure_predictions(config, wanted, data).items():
            deployments = sources[cell].read_table(cell_table(DEPLOYMENT_SHARD))
            check_reproduced(predictions, deployments)
            unit = units[cell]
            write_table(
                predictions.assign(scale=scale, config_hash=unit.fingerprint),
                scratches[unit.fingerprint].root / PREDICTION_TABLE,
                PREDICTION_SCHEMA,
            )

    artifacts = store.resolve_together(list(units.values()), compute)
    return {cell: artifacts[unit.fingerprint] for cell, unit in units.items()}


def read_predictions(
    artifacts: Mapping[RegressionCell, StageArtifact],
) -> pd.DataFrame:
    """Every cell's predictions in one table."""
    return pd.concat(
        [artifact.read_table(PREDICTION_TABLE) for artifact in artifacts.values()],
        ignore_index=True,
    )
