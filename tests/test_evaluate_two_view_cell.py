"""Tests for scoring one regression fit against every deployment of a held-out block.

Every training run is scored on every deployment; none is chosen to report.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

import aas_shortcut.evaluate_two_view_cell as cell_module
from aas_shortcut.augment_training_samples import REP_COLUMN
from aas_shortcut.build_feature_matrix import INSTANCE_COLUMNS
from aas_shortcut.build_regression_model import ModelConfig, fit_model
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.predict_superiority import TARGET, score_on

INSTANCES = tuple(range(1, 21))


def make_matrix(seed: int = 0, reps: int = 0) -> pd.DataFrame:
    """A matrix with a learnable target, and ``reps`` deployments if asked."""
    rng = np.random.default_rng(seed)
    rows = 8 * len(INSTANCES)
    frame = pd.DataFrame({f"ela_x{i}": rng.normal(size=rows) for i in range(4)})
    frame["instance_id"] = np.repeat(INSTANCES, 8)
    frame["problem_class"] = "bbob"
    frame["function_id"] = np.tile(np.arange(8) + 1, len(INSTANCES))
    frame[TARGET] = frame["ela_x0"] * 0.7 + rng.normal(scale=0.1, size=rows)
    if reps:
        frame[REP_COLUMN] = np.tile(np.arange(1, reps + 1), rows // reps)
    return frame


def predictors(matrix: pd.DataFrame) -> list[str]:
    """The matrix's predictor columns."""
    return [column for column in matrix.columns if column.startswith("ela_")]


@pytest.fixture
def model():
    """A small fit to score blocks with."""
    train = make_matrix()
    return fit_model(
        train,
        predictors(train),
        ModelConfig(kind="random-forest"),
        TARGET,
        group_columns=cell_module.GROUP_COLUMNS,
    )


class TestDeploymentScores:
    def test_one_score_per_deployment_keyed_by_it(self, model):
        block = make_matrix(seed=1, reps=4)
        scored = cell_module.deployment_scores(model, block, predictors(block))
        assert sorted(scored) == [1, 2, 3, 4]

    def test_a_deployment_is_scored_on_its_own_rows(self, model):
        """Pooling the deployments would report a problem sampled four times."""
        block = make_matrix(seed=1, reps=4)
        columns = predictors(block)
        scored = cell_module.deployment_scores(model, block, columns)
        for rep, rows in block.groupby(REP_COLUMN):
            assert scored[int(cast(int, rep))] == score_on(model, rows, columns)

    def test_a_block_without_deployments_is_one_score(self, model):
        block = make_matrix(seed=1)
        columns = predictors(block)
        scored = cell_module.deployment_scores(model, block, columns)
        assert scored == {cell_module.SINGLE_SAMPLE: score_on(model, block, columns)}


class TestNames:
    def test_one_test_metric_per_view(self):
        assert {
            view: f"r2_test_{view}" for view in TRANSFORMS
        } == cell_module.TEST_METRICS

    def test_the_group_columns_identify_a_problem(self):
        """Not `instance_id` alone, which holds one row per function."""
        assert cell_module.GROUP_COLUMNS == INSTANCE_COLUMNS
        assert "function_id" in cell_module.GROUP_COLUMNS
