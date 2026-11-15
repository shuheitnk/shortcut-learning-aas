"""Training-time sampling repetitions: many samples per instance, one at test time."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pandas as pd

from aas_shortcut.build_feature_matrix import INSTANCE_COLUMNS, build_feature_matrix
from aas_shortcut.define_paths import ELA_FEATURE_SET

REP_COLUMN = "rep"

# An instance *and* which of its samples.
AUGMENTED_ID_COLUMNS: tuple[str, ...] = (*INSTANCE_COLUMNS, REP_COLUMN)

# The one repetition a test instance uses. Repetitions come from independent seeds,
# so any one would do; what matters is that it is always the same one.
DEPLOYED_REP = 1

# Keep each repetition as its own training row (the augmentation).
REDUCTION_ROWS = "rows"

# Average the repetitions into one row per instance: the control for the
# augmentation. Same feature budget, but a cleaner representation than deployment
# gives, so it separates "shown the noise" from "shown more rows".
REDUCTION_MEAN = "mean"

REDUCTIONS: tuple[str, ...] = (REDUCTION_ROWS, REDUCTION_MEAN)


def validate_reduction(reduction: str) -> str:
    """Return ``reduction`` unchanged after checking it is one of :data:`REDUCTIONS`."""
    if reduction not in REDUCTIONS:
        raise ValueError(
            f"unknown reduction {reduction!r}; expected one of {list(REDUCTIONS)}"
        )
    return reduction


def available_reps(features: pd.DataFrame) -> tuple[int, ...]:
    """Return the sampling repetitions a feature shard holds, in order."""
    return tuple(sorted(int(rep) for rep in features[REP_COLUMN].unique()))


def build_matrix(
    features: pd.DataFrame,
    transform: str,
    variant: str,
    *,
    feature_set: str = ELA_FEATURE_SET,
    instances: Sequence[int] | None = None,
    reps: Sequence[int] | None = None,
    reduction: str = REDUCTION_ROWS,
) -> pd.DataFrame:
    """Build a feature matrix from a chosen set of instances and repetitions.

    Under :data:`REDUCTION_MEAN` the ``rep`` column is dropped.
    """
    selected = features
    if instances is not None:
        selected = cast(
            pd.DataFrame, selected[selected["instance_id"].isin(list(instances))]
        )
    if reps is not None:
        selected = cast(
            pd.DataFrame, selected[selected[REP_COLUMN].isin([int(r) for r in reps])]
        )
    if selected.empty:
        raise ValueError(
            f"the feature shard holds no row for instances={list(instances or [])} "
            f"and reps={list(reps or [])}; it covers instances "
            f"{sorted(features['instance_id'].unique())} and reps "
            f"{list(available_reps(features))}"
        )

    validate_reduction(reduction)
    id_columns = (
        AUGMENTED_ID_COLUMNS if reduction == REDUCTION_ROWS else INSTANCE_COLUMNS
    )
    return build_feature_matrix(
        selected,
        transform,
        variant,
        feature_set=feature_set,
        id_columns=id_columns,
    )
