"""Joining a run's untransformed half to a transformed half measured elsewhere."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.define_paths import TRANSFORMS

# Columns that name one feature observation.
FEATURE_IDENTITY = (
    "function_id",
    "instance_id",
    "feature_set",
    "feature_variant",
    "feature",
)


def departures(
    measured: pd.Series, reference: pd.Series, tolerance: float
) -> dict[str, int]:
    """Count the cells of each feature that disagree, by relative difference."""
    apart = ~np.isclose(
        measured.to_numpy(dtype=float),
        reference.to_numpy(dtype=float),
        rtol=tolerance,
        atol=0.0,
        equal_nan=True,
    )
    names = measured.index.get_level_values("feature")[apart]
    return {
        str(name): int(count) for name, count in pd.Series(names).value_counts().items()
    }


def describe_departures(moved: dict[str, int]) -> str:
    """Name the features that disagree and how widely, for an error message."""
    return ", ".join(f"{name} ({count} cell(s))" for name, count in moved.items())


def check_source_agreement(
    features: pd.DataFrame,
    source: pd.DataFrame,
    instances: tuple[int, ...],
    tolerance: float,
    source_name: str,
) -> int:
    """Check the untransformed features agree with the source run's; count the cells."""
    key = list(FEATURE_IDENTITY)
    mine = cast(
        pd.Series,
        features[features["transform"] == TRANSFORMS[0]].set_index(key)["value"],
    )
    theirs = cast(
        pd.Series,
        source[
            (source["transform"] == TRANSFORMS[0])
            & (source["instance_id"].isin(instances))
        ].set_index(key)["value"],
    )
    shared = cast(pd.Index, mine.index.intersection(theirs.index))
    if shared.empty:
        raise ValueError("the two feature shards share no row to compare")

    moved = departures(
        cast(pd.Series, mine[shared]), cast(pd.Series, theirs[shared]), tolerance
    )
    if moved:
        raise ValueError(
            f"{sum(moved.values())} of {len(shared)} untransformed feature(s) "
            f"disagree with {source_name}'s, so the settings are not sharing a "
            f"training block: {describe_departures(moved)}"
        )
    return len(shared)
