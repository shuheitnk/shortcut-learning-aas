"""The appendix's LOFO cells as units: what each reads."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.measure_lofo_regression import STAGE, LofoRegressionCell
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN
from aas_shortcut.resolve_lofo_cells import lofo_inputs, lofo_regression_unit


def _blocks() -> tuple[Any, Any]:
    training = SimpleNamespace(
        inputs={
            "train_features": "f",
            "train_superiority_pre": "p",
            "train_superiority_post": "q",
        }
    )
    test = SimpleNamespace(
        label="spline/warpheld",
        inputs={
            "test_features:spline/warpheld": "t",
            "test_superiority_pre": "u",
            "test_superiority_post:spline/warpheld": "v",
        },
    )
    return cast(Any, training), cast(Any, test)


class TestWhatACellReads:
    def test_a_plain_cell_reads_neither_transformed_superiority(self):
        assert lofo_inputs(*_blocks(), TRAINING_PLAIN) == {
            "train_features": "f",
            "train_superiority_pre": "p",
            "test_features:spline/warpheld": "t",
            "test_superiority_pre": "u",
        }

    def test_an_augmented_cell_reads_the_transformed_training_runs(self):
        inputs = lofo_inputs(*_blocks(), TRAINING_AUGMENTED)
        assert inputs["train_superiority_post"] == "q"
        assert "test_superiority_post:spline/warpheld" not in inputs

    def test_the_unit_is_a_stage_of_its_own(self):
        cell = LofoRegressionCell(
            "ela", "value", FOCUS_PAIRS[0], TRAINING_PLAIN, ("a",)
        )
        unit = lofo_regression_unit(cell, *_blocks())
        assert unit.stage == STAGE == "lofo_regression"
        assert unit.settings == cell.as_dict()
