"""The appendix's LOFO cells as units of the stage store, one unit per cell.

A unit holds a cell's out-of-fold predictions and its pooled scores. Its inputs
are the training block's views its condition fits on and the untransformed test
block, so an interrupted run resumes cell by cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from aas_shortcut.define_paths import RunPaths
from aas_shortcut.measure_lofo_regression import (
    PREDICTION_SCHEMA,
    PREDICTION_TABLE,
    SCORE_SCHEMA,
    SCORE_TABLE,
    STAGE,
    LofoRegressionCell,
    measure_cells,
)
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.resolve_pair_blocks import HeldOutBlock, TrainingBlock, block_inputs
from aas_shortcut.resolve_regression_cells import cell_inputs
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit

# The test block's transformed view, which no LOFO cell reads.
_UNREAD_TEST_ROLE = "test_superiority_post:"


def lofo_inputs(
    training: TrainingBlock, test: HeldOutBlock, condition: str
) -> dict[str, str]:
    """Return the fingerprints a cell of one training condition reads."""
    inputs = cell_inputs(block_inputs(training, [test]), condition)
    return {
        role: value
        for role, value in inputs.items()
        if not role.startswith(_UNREAD_TEST_ROLE)
    }


def lofo_regression_unit(
    cell: LofoRegressionCell, training: TrainingBlock, test: HeldOutBlock
) -> StageUnit:
    """Return the unit of one LOFO regression cell."""
    return StageUnit(
        STAGE, cell.as_dict(), inputs=lofo_inputs(training, test, cell.training)
    )


def resolve_lofo_regression(
    store: StageStore,
    cells: Sequence[LofoRegressionCell],
    training: TrainingBlock,
    test: HeldOutBlock,
    scale: str,
    n_jobs: int,
) -> dict[LofoRegressionCell, StageArtifact]:
    """Return every cell, computing the missing ones together."""
    units = {cell: lofo_regression_unit(cell, training, test) for cell in cells}

    def compute(
        missing: Sequence[StageUnit], scratches: Mapping[str, RunPaths]
    ) -> None:
        wanted = [cell for cell, unit in units.items() if unit in missing]
        results = measure_cells(
            wanted,
            training.feature_table,
            training.superiorities,
            test.feature_table,
            test.superiorities,
            n_jobs,
        )
        for cell, (predictions, scores) in results.items():
            unit = units[cell]
            root = scratches[unit.fingerprint].root
            for table, path, schema in (
                (predictions, PREDICTION_TABLE, PREDICTION_SCHEMA),
                (scores, SCORE_TABLE, SCORE_SCHEMA),
            ):
                write_table(
                    table.assign(scale=scale, config_hash=unit.fingerprint),
                    root / path,
                    schema,
                )

    artifacts = store.resolve_together(list(units.values()), compute)
    return {cell: artifacts[unit.fingerprint] for cell, unit in units.items()}
