"""The plain-4800 condition: its instances, update budget and checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.match_update_budget import matched_config, total_updates
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_VIEWS
from aas_shortcut.resolve_further_training import (
    FURTHER_INSTANCES,
    PLAIN_4800_INSTANCES,
    TRAINING_PLAIN_4800,
    check_same_matrix,
    check_shared_features,
    further_role,
    plain_4800_model,
    relabel,
)
from aas_shortcut.split_bbob_instances import TEST_INSTANCES, TRAIN_INSTANCES


def test_the_further_block_follows_the_training_block():
    assert tuple(range(111, 211)) == FURTHER_INSTANCES
    assert tuple(range(11, 211)) == PLAIN_4800_INSTANCES
    assert not set(PLAIN_4800_INSTANCES) & set(TEST_INSTANCES)
    assert len(PLAIN_4800_INSTANCES) == 2 * len(TRAIN_INSTANCES)


def test_the_model_spends_the_augmented_update_budget():
    model = ModelConfig()
    halved = plain_4800_model(model)
    assert halved.epochs == model.epochs // 2
    augmented = matched_config(model, len(TRAINING_VIEWS[TRAINING_AUGMENTED]))
    assert total_updates(4800, halved) == total_updates(4800, augmented)


def test_the_further_role_names_its_block():
    assert further_role("train_superiority_pre") == "train_superiority_pre:further"


def make_features(instances: range, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "function_id": 1,
            "instance_id": list(instances),
            "transform": "pre",
            "feature": "ela_meta.lin_simple.adj_r2",
            "value": values,
            "config_hash": "x",
        }
    )


def test_shared_features_pass_when_equal_and_stop_otherwise():
    paper = make_features(range(1, 4), np.array([0.1, np.nan, 0.3]))
    wider = make_features(range(1, 7), np.array([0.1, np.nan, 0.3, 1, 2, 3]))
    assert check_shared_features(wider.assign(config_hash="y"), paper) == 3
    wider.loc[0, "value"] = 0.2
    with pytest.raises(ValueError, match="differ"):
        check_shared_features(wider, paper)


def test_a_different_matrix_stops():
    matrix = pd.DataFrame({"a": [1.0, 2.0]})
    assert check_same_matrix(matrix, matrix.copy()) == 2
    with pytest.raises(AssertionError):
        check_same_matrix(matrix, matrix + 1e-15)


def test_relabel_takes_plain_cells_only():
    rows = pd.DataFrame({"training": ["plain", "plain"]})
    assert set(relabel(rows)["training"]) == {TRAINING_PLAIN_4800}
    with pytest.raises(ValueError, match="plain"):
        relabel(pd.DataFrame({"training": ["augmented"]}))
