"""Appendix: the spread of the regression's target on a test block, per view.

R² divides a cell's squared error by the target's variance over the block, so a
view whose superiorities spread less is scored against a smaller denominator.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.measure_representation_transfer import group_of


def target_spread(superiorities: pd.DataFrame, family: str) -> pd.DataFrame:
    """Mean and variance (the R² denominator over n) of each pair's target, per view."""
    groups = {pair.slug: group_of(pair) for pair in FOCUS_PAIRS}
    rows = superiorities[superiorities["pair"].astype(str).isin(list(groups))]
    summary = (
        rows.groupby(["pair", "transform"], observed=True)["superiority"]
        .agg(
            n="size",
            mean="mean",
            variance=lambda values: float(values.var(ddof=0)),
        )
        .reset_index()
    )
    summary.insert(0, "group", summary["pair"].astype(str).map(groups))
    summary.insert(0, "audit", family)
    return summary


def member_means(superiorities: pd.DataFrame, family: str, pair: str) -> pd.DataFrame:
    """Each member's mean superiority over the other on a block, per view.

    The pair's target is the first member's superiority, so the second's is its
    complement.
    """
    rows = superiorities[superiorities["pair"].astype(str) == pair]
    first = rows.groupby("transform", observed=True)["superiority"].mean()
    names = cast(
        pd.DataFrame,
        cast(pd.DataFrame, rows[["algorithm_a", "algorithm_b"]]).drop_duplicates(),
    )
    if len(names) != 1:
        raise ValueError(f"the pair {pair} names {len(names)} member orders")
    a, b = (str(name) for name in names.iloc[0])
    return pd.DataFrame(
        [
            {"audit": family, "transform": str(view), "algorithm": name, "mean": value}
            for view, mean in first.items()
            for name, value in ((a, float(mean)), (b, 1.0 - float(mean)))
        ]
    )
