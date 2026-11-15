"""The appendix's prediction units: named by the regression cell they reproduce."""

from __future__ import annotations

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.measure_augmented_regression import AugmentedRegressionConfig
from aas_shortcut.measure_regression_predictions import STAGE
from aas_shortcut.resolve_prediction_cells import SOURCE_ROLE, prediction_unit
from aas_shortcut.resolve_regression_cells import RegressionCell, regression_cell_unit

INPUTS = {
    "train_features": "a",
    "train_superiority_pre": "b",
    "test_superiority_pre": "c",
}


def _config() -> AugmentedRegressionConfig:
    return AugmentedRegressionConfig(
        audits=("spline/warpheld",),
        model=ModelConfig(kind="random-forest"),
        candidates=("random-forest",),
    )


def test_the_unit_names_the_regression_unit_it_reproduces():
    cell = RegressionCell(FOCUS_PAIRS[0], "value", "plain", "random-forest")
    source = regression_cell_unit(_config(), cell, INPUTS)
    unit = prediction_unit(_config(), cell, INPUTS)
    assert unit.stage == STAGE
    assert unit.inputs[SOURCE_ROLE] == source.fingerprint
    assert unit.fingerprint != source.fingerprint


def test_cells_differ_in_their_units():
    plain = RegressionCell(FOCUS_PAIRS[0], "value", "plain", "random-forest")
    rank = RegressionCell(FOCUS_PAIRS[0], "rank", "plain", "random-forest")
    assert (
        prediction_unit(_config(), plain, INPUTS).fingerprint
        != prediction_unit(_config(), rank, INPUTS).fingerprint
    )
