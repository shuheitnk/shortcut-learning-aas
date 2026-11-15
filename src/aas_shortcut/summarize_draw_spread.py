"""Appendix: how far a level moves over the control family's draws of its maps."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def draw_spread(
    levels: pd.DataFrame, block: str, draws: Sequence[str], value: str = "value"
) -> pd.DataFrame:
    """Minimum, median and maximum of each level over the draws' blocks."""
    rows = levels[levels[block].astype(str).isin(list(draws))]
    found = set(rows[block].astype(str))
    if found != set(draws):
        raise ValueError(f"draws missing from the levels: {sorted(set(draws) - found)}")
    keys = [column for column in levels.columns if column not in (block, value)]
    return (
        rows.groupby(keys, observed=True, dropna=False)[value]
        .agg(n_draws="size", min="min", median="median", max="max")
        .reset_index()
    )
