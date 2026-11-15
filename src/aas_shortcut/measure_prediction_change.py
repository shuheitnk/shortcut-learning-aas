"""Appendix: per-deployment summaries of the regression's per-problem predictions.

Each summary is written as a deployment table (one value per cell, family, seed and
deployment, under a metric name), so the report's own reduction
(`compare_training_runs.per_seed`, `median_over_seeds`) turns it into a level.
"""

from __future__ import annotations

import pandas as pd

# How far a problem's prediction moves when only the objective is transformed.
CHANGE_METRIC = "abs_prediction_change"

# The mean squared error, which R² divides by the target's variance.
ERROR_METRICS: dict[str, str] = {"pre": "mse_test_pre", "post": "mse_test_post"}

CELL_KEYS: tuple[str, ...] = (
    "feature_set",
    "group",
    "feature_variant",
    "training",
    "audit",
    "seed",
    "rep",
)

PROBLEM: tuple[str, ...] = ("function_id", "instance_id")


def prediction_change(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean |Ĵ(p̃) − Ĵ(p)| over a deployment's problems, the same sample in both views.

    The untransformed prediction is the same model's on the same deployment, so the
    difference is due to the transformation of the objective alone.
    """
    keys = [*CELL_KEYS, *PROBLEM]
    views = {
        view: predictions[predictions["transform"].astype(str) == view]
        .set_index(keys)["predicted"]
        .sort_index()
        for view in ("pre", "post")
    }
    if not views["pre"].index.equals(views["post"].index):
        raise ValueError("the two views do not hold the same predictions")
    change = (views["post"] - views["pre"]).abs().rename("value").reset_index()
    return (
        change.groupby(list(CELL_KEYS), observed=True)["value"]
        .mean()
        .reset_index()
        .assign(metric=CHANGE_METRIC)
    )


def squared_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """The mean squared error of every deployment, per view."""
    error = (predictions["observed"] - predictions["predicted"]) ** 2
    table = predictions.assign(value=error)
    rows = (
        table.groupby([*CELL_KEYS, "transform"], observed=True)["value"]
        .mean()
        .reset_index()
    )
    rows["metric"] = rows["transform"].astype(str).map(ERROR_METRICS)
    return rows.drop(columns="transform")
