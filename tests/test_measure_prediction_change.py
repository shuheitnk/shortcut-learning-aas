"""The appendix's per-deployment summaries of the regression's predictions."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.measure_prediction_change import (
    CHANGE_METRIC,
    prediction_change,
    squared_errors,
)


def _predictions(post_shift: float) -> pd.DataFrame:
    rows = []
    for transform in ("pre", "post"):
        for instance_id, predicted in ((1, 0.2), (2, 0.6)):
            rows.append(
                {
                    "feature_set": "ela",
                    "group": "G1",
                    "feature_variant": "value",
                    "training": "plain",
                    "audit": "spline/warpheld",
                    "seed": 42,
                    "rep": 1,
                    "function_id": 1,
                    "instance_id": instance_id,
                    "transform": transform,
                    "observed": 0.5,
                    "predicted": predicted
                    + (post_shift * instance_id if transform == "post" else 0.0),
                }
            )
    return pd.DataFrame(rows)


def test_the_change_is_the_mean_absolute_move_of_a_problem_s_prediction():
    change = prediction_change(_predictions(0.1))
    assert change["metric"].tolist() == [CHANGE_METRIC]
    assert change["value"].iloc[0] == pytest.approx((0.1 + 0.2) / 2)


def test_predictions_that_do_not_move_change_by_nothing():
    assert prediction_change(_predictions(0.0))["value"].iloc[0] == 0.0


def test_a_problem_predicted_in_one_view_only_is_refused():
    with pytest.raises(ValueError, match="same predictions"):
        prediction_change(_predictions(0.1).iloc[1:])


def test_the_squared_error_is_per_view():
    errors = squared_errors(_predictions(0.1)).set_index("metric")["value"]
    assert errors["mse_test_pre"] == pytest.approx((0.3**2 + 0.1**2) / 2)
    assert errors["mse_test_post"] == pytest.approx((0.2**2 + 0.3**2) / 2)
