"""The appendix's LOFO regression: folds by function, R² over pooled predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.measure_lofo_regression import (
    METRIC,
    PREDICTION_SCHEMA,
    SCORE_SCHEMA,
    LofoRegressionCell,
    measure_cells,
    pooled_scores,
)
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN
from aas_shortcut.predict_superiority import r2_score
from aas_shortcut.read_write_tables import write_table
from tests.conftest import FEATURE_NAMES_BY_SET, make_features
from tests.test_measure_representation_transfer import make_view_superiorities

FUNCTIONS = (1, 2, 3)
TRAIN = tuple(range(3, 9))
TEST = (1, 2)
DEPLOYMENTS = 2
COLUMNS = tuple(FEATURE_NAMES_BY_SET["ela"])


def _per_function(frames) -> pd.DataFrame:
    return pd.concat(
        [
            frame.assign(function_id=f)
            for f, frame in zip(FUNCTIONS, frames, strict=True)
        ],
        ignore_index=True,
    )


@pytest.fixture
def data():
    n = len(TRAIN) + len(TEST)
    train = _per_function(
        make_features(n_instances=n, n_reps=1, seed=f) for f in FUNCTIONS
    )
    test = _per_function(
        make_features(n_instances=n, n_reps=DEPLOYMENTS, seed=10 + f) for f in FUNCTIONS
    )
    superiorities = _per_function(
        make_view_superiorities(n_instances=n) for _ in FUNCTIONS
    )
    return train, superiorities, test, superiorities


def _cell(training: str = TRAINING_PLAIN) -> LofoRegressionCell:
    return LofoRegressionCell(
        feature_set="ela",
        variant="value",
        pair=FOCUS_PAIRS[2],
        training=training,
        features=COLUMNS,
        model=ModelConfig(kind="random-forest"),
        functions=FUNCTIONS,
        train_instances=TRAIN,
        test_instances=TEST,
        n_seeds=2,
    )


class TestCell:
    def test_the_fingerprint_names_the_protocol_and_the_condition(self):
        plain, augmented = _cell().as_dict(), _cell(TRAINING_AUGMENTED).as_dict()
        assert plain["protocol"] == "leave-one-function-out"
        assert plain["functions"] == list(FUNCTIONS)
        assert plain != augmented

    def test_the_augmented_condition_doubles_the_training_rows(self, data):
        plain, _ = _cell().matrices(*data)
        augmented, _ = _cell(TRAINING_AUGMENTED).matrices(*data)
        assert len(augmented) == 2 * len(plain) == 2 * len(FUNCTIONS) * len(TRAIN)

    def test_the_test_block_is_untransformed_with_every_deployment(self, data):
        _, test = _cell().matrices(*data)
        assert set(test["transform"]) == {"pre"}
        assert len(test) == len(FUNCTIONS) * len(TEST) * DEPLOYMENTS


class TestMeasure:
    def test_every_problem_is_predicted_once_per_seed_and_deployment(
        self, data, tmp_path
    ):
        cell = _cell()
        predictions, scores = measure_cells([cell], *data, n_jobs=1)[cell]
        counted = predictions.groupby(["seed", "rep"]).size()
        assert (counted == len(FUNCTIONS) * len(TEST)).all()
        assert len(scores) == cell.n_seeds * DEPLOYMENTS
        for table, schema in ((predictions, PREDICTION_SCHEMA), (scores, SCORE_SCHEMA)):
            write_table(
                table.assign(scale="debug", config_hash="x"),
                tmp_path / f"{schema.name}.parquet",
                schema,
            )

    def test_a_fold_never_trains_on_its_own_function(self, data, monkeypatch):
        import aas_shortcut.measure_lofo_regression as module

        seen = []
        original = module.predict_fold

        def spy(train, test, *rest):
            seen.append((set(train["function_id"]), set(test["function_id"])))
            return original(train, test, *rest)

        monkeypatch.setattr(module, "predict_fold", spy)
        monkeypatch.setattr(
            module,
            "run_jobs",
            lambda _, tasks, worker, n_jobs: (
                (i, worker(*task)) for i, task in enumerate(tasks)
            ),
        )
        module.measure_cells([_cell()], *data, n_jobs=1)
        for trained, held in seen:
            assert len(held) == 1 and not trained & held


class TestPooledScores:
    def test_r2_is_taken_over_the_pooled_predictions_not_per_fold(self):
        rng = np.random.default_rng(0)
        observed = np.repeat([0.1, 0.5, 0.9], 4) + rng.normal(0, 0.01, 12)
        predicted = np.repeat([0.2, 0.5, 0.8], 4)
        predictions = pd.DataFrame(
            {
                "seed": 42,
                "rep": 1,
                "function_id": np.repeat([1, 2, 3], 4),
                "observed": observed,
                "predicted": predicted,
            }
        )
        scores = pooled_scores(predictions, 12)
        assert scores["metric"].tolist() == [METRIC]
        assert scores["value"].iloc[0] == r2_score(observed, predicted)
        per_fold = np.mean(
            [
                r2_score(observed[i : i + 4], predicted[i : i + 4])
                for i in range(0, 12, 4)
            ]
        )
        assert scores["value"].iloc[0] > 0.9 > per_fold

    def test_a_deployment_short_of_predictions_is_refused(self):
        predictions = pd.DataFrame(
            {"seed": [42], "rep": [1], "observed": [0.1], "predicted": [0.2]}
        )
        with pytest.raises(ValueError, match="expected 2"):
            pooled_scores(predictions, 2)
