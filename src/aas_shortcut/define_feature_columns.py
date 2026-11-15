"""The feature columns the experiments read, fixed once per scale.

A feature that is NaN or inf in any block the paper uses is left out of every
experiment. The result is declared here so that no experiment derives its own column
set from whichever shards it reads; :func:`check_feature_unit` stops a run if a shard
loses a column outside the declared exclusions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from aas_shortcut.define_paths import (
    DEEP_ELA_FEATURE_SET,
    ELA_FEATURE_SET,
    validate_feature_set,
)

# The 59 ELA features, sorted by name.
ELA_COLUMNS: tuple[str, ...] = (
    "disp.diff_mean_02",
    "disp.diff_mean_05",
    "disp.diff_mean_10",
    "disp.diff_mean_25",
    "disp.diff_median_02",
    "disp.diff_median_05",
    "disp.diff_median_10",
    "disp.diff_median_25",
    "disp.ratio_mean_02",
    "disp.ratio_mean_05",
    "disp.ratio_mean_10",
    "disp.ratio_mean_25",
    "disp.ratio_median_02",
    "disp.ratio_median_05",
    "disp.ratio_median_10",
    "disp.ratio_median_25",
    "ela_distr.kurtosis",
    "ela_distr.number_of_peaks",
    "ela_distr.skewness",
    "ela_level.lda_qda_10",
    "ela_level.lda_qda_25",
    "ela_level.lda_qda_50",
    "ela_level.mmce_lda_10",
    "ela_level.mmce_lda_25",
    "ela_level.mmce_lda_50",
    "ela_level.mmce_qda_10",
    "ela_level.mmce_qda_25",
    "ela_level.mmce_qda_50",
    "ela_meta.lin_simple.adj_r2",
    "ela_meta.lin_simple.coef.max",
    "ela_meta.lin_simple.coef.max_by_min",
    "ela_meta.lin_simple.coef.min",
    "ela_meta.lin_simple.intercept",
    "ela_meta.lin_w_interact.adj_r2",
    "ela_meta.quad_simple.adj_r2",
    "ela_meta.quad_w_interact.adj_r2",
    "fitness_distance.distance_mean",
    "fitness_distance.distance_std",
    "fitness_distance.fd_correlation",
    "fitness_distance.fd_cov",
    "fitness_distance.fitness_mean",
    "fitness_distance.fitness_std",
    "ic.eps_max",
    "ic.eps_s",
    "ic.h_max",
    "ic.m0",
    "nbc.dist_ratio.coeff_var",
    "nbc.nb_fitness.cor",
    "nbc.nn_nb.cor",
    "nbc.nn_nb.mean_ratio",
    "nbc.nn_nb.sd_ratio",
    "pca.expl_var.cor_init",
    "pca.expl_var.cor_x",
    "pca.expl_var.cov_init",
    "pca.expl_var.cov_x",
    "pca.expl_var_PC1.cor_init",
    "pca.expl_var_PC1.cor_x",
    "pca.expl_var_PC1.cov_init",
    "pca.expl_var_PC1.cov_x",
)

# The ELA features the calculators produce but no experiment reads, with the reason.
EXCLUDED_COLUMNS: dict[str, str] = {
    "ela_meta.quad_simple.cond": (
        "non-finite in every shard the paper reads: the condition number of the "
        "simple quadratic model has observations whose denominator is zero"
    ),
    "ic.eps_ratio": (
        "non-finite in the power family's held-out shard only; why there and "
        "nowhere else has not been established"
    ),
}

# The scale the paper reports, which :data:`ELA_COLUMNS` is declared for.
PAPER_SCALE = "full"

# Debug scale only (100 points in 2-D), never used for a result of the paper: the
# same rule gives a different exclusion list on these small samples.
DEBUG_EXCLUDED_COLUMNS: dict[str, str] = {
    "ela_level.lda_qda_10": (
        "non-finite in every held-out block at the debug scale: a QDA error of zero "
        "in the denominator of the ratio, 4 of 2640 observations"
    ),
    "ela_level.lda_qda_25": (
        "non-finite in the training block at the debug scale (12 of 2640 "
        "observations), for the same reason"
    ),
    "ela_level.lda_qda_50": (
        "non-finite in every block at the debug scale (8 to 24 observations)"
    ),
    "ic.eps_ratio": (
        "non-finite in the power family's held-out block at the debug scale too "
        "(4 observations)"
    ),
}

# A scale with no entry here cannot be run.
EXCLUDED_BY_SCALE: dict[str, dict[str, str]] = {
    PAPER_SCALE: EXCLUDED_COLUMNS,
    "debug": DEBUG_EXCLUDED_COLUMNS,
}

DEEP_ELA_MODEL = "large_50d_v1"
DEEP_ELA_WIDTH = 48

# Deep-ELA's 48 coordinates, sorted by name. No shard loses any.
DEEP_ELA_COLUMNS: tuple[str, ...] = tuple(
    sorted(f"deep_ela.{DEEP_ELA_MODEL}.X{index}" for index in range(DEEP_ELA_WIDTH))
)


# Every column the calculators produce, before any rule excludes one.
_PRODUCED: dict[str, frozenset[str]] = {
    ELA_FEATURE_SET: frozenset(ELA_COLUMNS) | frozenset(EXCLUDED_COLUMNS),
    DEEP_ELA_FEATURE_SET: frozenset(DEEP_ELA_COLUMNS),
}


class UndeclaredColumnLoss(RuntimeError):
    """Raised when a shard lost a column the experiments read."""


def excluded_columns(scale: str) -> dict[str, str]:
    """Return the columns a scale's rule excludes, with the reasons."""
    if scale not in EXCLUDED_BY_SCALE:
        raise UndeclaredColumnLoss(
            f"no column list is declared for the {scale!r} scale; declared: "
            f"{sorted(EXCLUDED_BY_SCALE)} (define_feature_columns)"
        )
    return EXCLUDED_BY_SCALE[scale]


def paper_columns(
    feature_sets: Iterable[str], scale: str = PAPER_SCALE
) -> tuple[str, ...]:
    """Return the columns of the given representations at a scale, sorted by name.

    The column order is part of the result: it decides which input a network's
    initial weights meet.
    """
    excluded = set(excluded_columns(scale))
    names: set[str] = set()
    for feature_set in feature_sets:
        names.update(_PRODUCED[validate_feature_set(feature_set)] - excluded)
    return tuple(sorted(names))


def check_feature_unit(
    table: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    where: str,
    scale: str = PAPER_SCALE,
) -> None:
    """Stop if a shard dropped or lacks a column the experiments read.

    ``diagnostics`` is the shard manifest's record of the features it dropped.
    """
    excluded = excluded_columns(scale)
    dropped = diagnostics.get("dropped_features", {})
    lost = sorted(
        (set(dropped.get("nonfinite", ())) | set(dropped.get("unbalanced", ())))
        - set(excluded)
    )
    if lost:
        raise UndeclaredColumnLoss(
            f"{where}: the feature stage dropped {lost}, which the experiments read "
            f"at the {scale!r} scale; only {sorted(excluded)} may be lost "
            "(define_feature_columns)"
        )
    held = set(table["feature"].astype(str).unique())
    for feature_set in sorted(set(table["feature_set"].astype(str).unique())):
        missing = sorted(set(paper_columns([feature_set], scale)) - held)
        if missing:
            raise UndeclaredColumnLoss(
                f"{where}: the {feature_set} shard does not carry {missing}"
            )
