"""Instance-wise algorithm ranking prediction."""

from __future__ import annotations

import logging

import pandas as pd

from aas_shortcut.build_feature_matrix import INSTANCE_COLUMNS
from aas_shortcut.define_algorithm_portfolio import ALGORITHM_SLUGS, ALGORITHMS

LOGGER = logging.getLogger(__name__)

# Prefix of the target column carrying one member's mean superiority.
TARGET_PREFIX = "mean_superiority__"


def target_column(algorithm: str) -> str:
    """Return the matrix column holding ``algorithm``'s mean superiority."""
    return f"{TARGET_PREFIX}{ALGORITHM_SLUGS[algorithm]}"


def join_mean_superiorities(
    matrix: pd.DataFrame,
    mean_superiorities: pd.DataFrame,
    algorithms: tuple[str, ...] = ALGORITHMS,
    *,
    validate: str = "1:1",
) -> pd.DataFrame:
    """Attach one target column per portfolio member, keeping complete instances."""
    wide = mean_superiorities.pivot_table(
        index=list(INSTANCE_COLUMNS),
        columns="algorithm",
        values="mean_superiority",
        aggfunc="mean",
    ).reset_index()

    missing = [algorithm for algorithm in algorithms if algorithm not in wide.columns]
    if missing:
        raise ValueError(
            f"no mean superiorities for {missing}; the ranking task needs the whole "
            "portfolio"
        )

    wide = wide[[*INSTANCE_COLUMNS, *algorithms]].rename(
        columns={algorithm: target_column(algorithm) for algorithm in algorithms}
    )
    joined = matrix.merge(
        wide, on=list(INSTANCE_COLUMNS), how="inner", validate=validate
    )
    dropped = len(matrix) - len(joined)
    if dropped:
        LOGGER.debug("dropped %d instance(s) without a complete ranking", dropped)
    return joined
