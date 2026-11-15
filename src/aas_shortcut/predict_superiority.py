"""Predicting a pair's superiority from a landscape representation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

TARGET = "superiority"


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination, defined at zero variance."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0:
        return 0.0

    # Detected on the spread, not on the sum of squares: the mean of a constant
    # array is not always bit-exactly that constant.
    scale = float(np.max(np.abs(y_true))) or 1.0
    if float(np.ptp(y_true)) <= 1e-12 * scale:
        return 0.0

    ss_total = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_residual = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_residual / ss_total


def score_on(
    model, frame: pd.DataFrame, columns: list[str], target: str = TARGET
) -> float:
    """Return one fitted model's ``R^2`` on one evaluation set."""
    return float(r2_score(*observed_and_predicted(model, frame, columns, target)))


def observed_and_predicted(
    model, frame: pd.DataFrame, columns: list[str], target: str = TARGET
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` for one evaluation set.

    Shared with :func:`score_on`, so stored predictions are the ones the score used.
    """
    X = frame[columns].to_numpy(dtype=np.float64)
    return frame[target].to_numpy(dtype=np.float64), np.asarray(model.predict(X))


def restrict_to_common_features(
    features_by_class: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Keep only the features every problem class has."""
    per_class = {
        problem_class: set(table["feature"].unique())
        for problem_class, table in features_by_class.items()
    }
    if not per_class:
        return features_by_class

    common = set.intersection(*per_class.values())
    dropped = {
        problem_class: sorted(names - common)
        for problem_class, names in per_class.items()
        if names - common
    }
    if not dropped:
        return features_by_class

    LOGGER.warning(
        "restricting the prediction experiment to the %d feature(s) shared by "
        "every suite; dropped %s",
        len(common),
        dropped,
    )
    return {
        problem_class: table[table["feature"].isin(sorted(common))]
        for problem_class, table in features_by_class.items()
    }
