"""Building the wide feature matrix a model consumes."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    FEATURE_ENCODINGS,
    validate_feature_set,
    validate_feature_variant,
    validate_transform,
)

LOGGER = logging.getLogger(__name__)

# Columns identifying one problem instance in a feature matrix.
INSTANCE_COLUMNS = ("problem_class", "function_id", "instance_id")

# Encodings that make up each feature variant.
VARIANT_ENCODINGS: dict[str, tuple[str, ...]] = {
    "value": ("value",),
    "rank": ("rank",),
    "concat": FEATURE_ENCODINGS,
}


def _pivot_encoding(
    features: pd.DataFrame,
    transform: str,
    encoding: str,
    feature_set: str,
    id_columns: Sequence[str],
) -> pd.DataFrame:
    """Pivot one ``(set, transform, encoding)`` slice into a wide matrix."""
    selected = features[
        (features["transform"] == transform)
        & (features["feature_variant"] == encoding)
        & (features["feature_set"] == feature_set)
    ]
    if selected.empty:
        raise ValueError(
            f"no features found for transform={transform!r} "
            f"encoding={encoding!r} feature_set={feature_set!r}"
        )

    wide = selected.pivot_table(
        index=list(id_columns), columns="feature", values="value", aggfunc="mean"
    )
    wide.columns = [f"{encoding}__{name}" for name in wide.columns]
    return wide.reset_index().rename_axis(None, axis=1)


def build_feature_matrix(
    features: pd.DataFrame,
    transform: str,
    variant: str,
    *,
    feature_set: str = ELA_FEATURE_SET,
    id_columns: Sequence[str] = INSTANCE_COLUMNS,
) -> pd.DataFrame:
    """Return the wide feature matrix for one transform, set and variant.

    Feature columns are named ``<encoding>__<feature>``.
    """
    validate_transform(transform)
    validate_feature_variant(variant)
    validate_feature_set(feature_set)

    matrices = [
        _pivot_encoding(features, transform, encoding, feature_set, id_columns)
        for encoding in VARIANT_ENCODINGS[variant]
    ]
    matrix = matrices[0]
    for other in matrices[1:]:
        matrix = matrix.merge(other, on=list(id_columns), how="inner", validate="1:1")
    return matrix


def feature_columns(
    matrix: pd.DataFrame, id_columns: Sequence[str] = INSTANCE_COLUMNS
) -> list[str]:
    """Return the feature columns of a matrix, excluding the identifiers."""
    ids = set(id_columns)
    return [column for column in matrix.columns if column not in ids]


def join_targets(
    matrix: pd.DataFrame,
    superiorities: pd.DataFrame,
    pair: str,
    *,
    id_columns: Sequence[str] = INSTANCE_COLUMNS,
    target_name: str = "superiority",
    validate: str = "1:1",
) -> pd.DataFrame:
    """Attach one pair's superiority to a feature matrix, keeping matched rows only."""
    targets = superiorities[superiorities["pair"] == pair][
        [*id_columns, "superiority"]
    ].rename(columns={"superiority": target_name})

    joined = matrix.merge(targets, on=list(id_columns), how="inner", validate=validate)
    dropped = len(matrix) - len(joined)
    if dropped:
        LOGGER.debug(
            "dropped %d instance(s) without a superiority for %s", dropped, pair
        )
    return joined
