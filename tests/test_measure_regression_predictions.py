"""The appendix's per-problem predictions: a refit that reproduces the stored R²."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.evaluate_two_view_cell import SINGLE_SAMPLE
from aas_shortcut.measure_augmented_regression import (
    AugmentedRegressionConfig,
    run_augmented_regression_stage,
)
from aas_shortcut.measure_regression_predictions import (
    PREDICTION_SCHEMA,
    check_reproduced,
    deployment_rows,
    measure_predictions,
    rescore,
)
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.resolve_regression_cells import RegressionCell, RegressionData
from tests.conftest import make_features
from tests.test_measure_representation_transfer import make_view_superiorities

TRAIN = tuple(range(3, 15))
TEST = (1, 2)
INSTANCES = len(TRAIN) + len(TEST)
DEPLOYMENTS = 3
LABELS = ("spline/warpheld", "power/warpheld")


@pytest.fixture
def data() -> RegressionData:
    superiorities = make_view_superiorities(n_instances=INSTANCES)
    return RegressionData(
        train_features=make_features(
            problem_class="bbob", function_id=1, n_instances=INSTANCES, n_reps=1
        ),
        train_superiorities=superiorities,
        test_features={
            label: make_features(
                problem_class="bbob",
                function_id=1,
                n_instances=INSTANCES,
                n_reps=DEPLOYMENTS,
                value_shift=shift,
            )
            for label, shift in zip(LABELS, (1.5, 4.0), strict=True)
        },
        test_superiorities=dict.fromkeys(LABELS, superiorities),
        inputs={},
    )


@pytest.fixture
def config() -> AugmentedRegressionConfig:
    return AugmentedRegressionConfig(
        audits=LABELS,
        train_instances=TRAIN,
        test_instances=TEST,
        candidates=("random-forest",),
        model=ModelConfig(kind="random-forest"),
        n_seeds=2,
    )


def _cells() -> list[RegressionCell]:
    return [
        RegressionCell(FOCUS_PAIRS[2], "value", training, "random-forest")
        for training in (TRAINING_PLAIN, TRAINING_AUGMENTED)
    ]


def _rows(stored: pd.DataFrame, training: str) -> pd.DataFrame:
    return cast(pd.DataFrame, stored[stored["training"] == training])


def _stored(config, data, tmp_path) -> pd.DataFrame:
    return run_augmented_regression_stage(
        data.train_features,
        data.train_superiorities,
        data.test_features,
        data.test_superiorities,
        RunPaths.from_root(tmp_path / "stage"),
        "debug",
        "0123456789abcdef",
        config,
        cells=[(c.pair, c.variant, c.training, c.kind) for c in _cells()],
    )


class TestPredictions:
    def test_the_refit_reproduces_every_stored_score_bit_for_bit(
        self, config, data, tmp_path
    ):
        stored = _stored(config, data, tmp_path)
        for cell, predictions in measure_predictions(config, _cells(), data).items():
            rows = _rows(stored, cell.training)
            assert check_reproduced(predictions, rows) == len(rows)

    def test_every_problem_is_predicted_per_seed_family_view_and_deployment(
        self, config, data, tmp_path
    ):
        cell = _cells()[0]
        predictions = measure_predictions(config, [cell], data)[cell]
        counted = predictions.groupby(["seed", "audit", "transform", "rep"]).size()
        assert len(counted) == 2 * len(LABELS) * 2 * DEPLOYMENTS
        assert (counted == len(TEST)).all()
        write_table(
            predictions.assign(scale="debug", config_hash="x"),
            tmp_path / "predictions.parquet",
            PREDICTION_SCHEMA,
        )

    def test_a_changed_prediction_is_refused(self, config, data, tmp_path):
        stored = _stored(config, data, tmp_path)
        cell = _cells()[0]
        predictions = measure_predictions(config, [cell], data)[cell]
        predictions.loc[predictions.index[0], "predicted"] += 1e-9
        with pytest.raises(ValueError, match="differ from the stored"):
            check_reproduced(predictions, _rows(stored, cell.training))

    def test_a_missing_score_is_refused(self, config, data, tmp_path):
        stored = _stored(config, data, tmp_path)
        cell = _cells()[0]
        predictions = measure_predictions(config, [cell], data)[cell]
        rows = _rows(stored, cell.training)
        with pytest.raises(ValueError, match="one side only"):
            check_reproduced(predictions, rows.iloc[1:])


class TestDeploymentRows:
    def test_a_block_with_one_sample_is_one_group(self):
        block = pd.DataFrame({"rep": [1, 1], "x": [0.0, 1.0]})
        assert [rep for rep, _ in deployment_rows(block)] == [SINGLE_SAMPLE]

    def test_samples_are_split_in_order(self):
        block = pd.DataFrame({"rep": [2, 1, 2], "x": [0.0, 1.0, 2.0]})
        assert [rep for rep, _ in deployment_rows(block)] == [1, 2]


def test_rescore_names_each_view_s_metric():
    predictions = pd.DataFrame(
        {
            "seed": 42,
            "audit": "a",
            "transform": ["pre", "pre", "post", "post"],
            "rep": 1,
            "observed": [0.0, 1.0, 0.0, 1.0],
            "predicted": [0.0, 1.0, 0.5, 0.5],
        }
    )
    scores = rescore(predictions).set_index("metric")["value"]
    assert scores["r2_test_pre"] == 1.0
    assert scores["r2_test_post"] == 0.0
