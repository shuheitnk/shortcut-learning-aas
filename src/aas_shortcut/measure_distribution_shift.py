"""The preanalysis's metric: the standardized shift, for features and for superiorities.

Measured deployment by deployment, and reported as the median over the deployments.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.calculate_delta_z import (
    PRIMARY_STANDARDIZATION,
    STATISTIC,
    pre_scale,
)
from aas_shortcut.define_algorithm_portfolio import ALGORITHM_GROUP
from aas_shortcut.define_paths import (
    FEATURE_QUANTITY,
    NOT_A_FEATURE,
    SUPERIORITY_QUANTITY,
    RunPaths,
    problem_slug,
)
from aas_shortcut.define_table_schema import SHIFT_SUMMARY_SCHEMA
from aas_shortcut.order_feature_labels import feature_class
from aas_shortcut.read_write_tables import write_table

LOGGER = logging.getLogger(__name__)


# Columns identifying one measured quantity of one instance in one deployment.
SHIFT_KEYS = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
    "rep",
    "quantity_kind",
    "feature_set",
    "feature_variant",
    "quantity",
    "quantity_class",
)

# Columns identifying a quantity across the instances of one deployment: the group
# ``sigma_{j,r}`` is computed within.
QUANTITY_KEYS = tuple(
    name for name in SHIFT_KEYS if name not in ("function_id", "instance_id")
)

SUMMARY_KEYS = (*QUANTITY_KEYS, "standardization")

# Each deployment is reduced over its instances on its own; the report is the
# median over deployments, as for the study's other numbers.
DEPLOYMENT_KEY = "rep"

REPORT_KEYS = tuple(name for name in SUMMARY_KEYS if name != DEPLOYMENT_KEY)

# The shape :func:`measure_shift` consumes. Each quantity has a single replicate per
# deployment; the column keeps the two views paired.
OBSERVATION_COLUMNS = (*SHIFT_KEYS, "transform", "replicate", "value")

SINGLE_REPLICATE = 1

# Tolerances below which a quantity's ``pre`` values count as constant across the
# instances: ``np.std`` of bit-identical values can be ~4e-16, and dividing a real
# shift by it would give ~1e14 "standard deviations". Orders match
# ``RANK_INVARIANCE_RTOL``.
SPREAD_RTOL = 1e-9
SPREAD_ATOL = 1e-12


def instance_values(observations: pd.DataFrame) -> pd.DataFrame:
    """Reduce each instance's repetitions to their mean, with the count."""
    missing = [name for name in OBSERVATION_COLUMNS if name not in observations.columns]
    if missing:
        raise ValueError(f"the observations are missing column(s) {missing}")
    index = [*SHIFT_KEYS, "transform"]
    return (
        observations.groupby(index, observed=True, sort=True)["value"]
        .agg(value="mean", n_replicates="size")
        .reset_index()
    )


def _paired(values: pd.DataFrame) -> pd.DataFrame:
    """Put the two views side by side, one row per instance and quantity.

    Both views must hold the same repetition count, or the difference would also be
    one between sample sizes.
    """
    keys = list(SHIFT_KEYS)
    wide = values.pivot_table(
        index=keys, columns="transform", values="value", aggfunc="mean"
    ).reset_index()
    missing = [name for name in ("pre", "post") if name not in wide.columns]
    if missing:
        raise ValueError(f"the observations are missing transform(s) {missing}")

    counts = values.groupby(keys, observed=True, sort=True)["n_replicates"].agg(
        ["min", "max"]
    )
    if not counts["min"].equals(counts["max"]):
        raise ValueError(
            "the two views hold different numbers of repetitions for at least one "
            "quantity; they must be measured on the same repetitions"
        )
    return wide.merge(
        cast(pd.Series, counts["min"]).rename("n_replicates").reset_index(),
        on=keys,
        validate="1:1",
    )


def _standardize(paired: pd.DataFrame) -> pd.DataFrame:
    """Attach ``sigma_pre``, ``delta_z`` and ``abs_delta_z`` to a paired table."""
    out = paired.copy()
    keys = list(QUANTITY_KEYS)
    grouped = out.groupby(keys, observed=True)["pre"]
    out["mu_pre"] = grouped.transform("mean")
    out["sigma_pre"] = grouped.transform(pre_scale)
    # The range, not the std: it is zero exactly when every instance agreed.
    spread = grouped.transform("max") - grouped.transform("min")
    out["no_spread"] = spread <= SPREAD_RTOL * out["mu_pre"].abs() + SPREAD_ATOL

    difference = (out["post"] - out["pre"]).to_numpy(dtype=np.float64)
    sigma = out["sigma_pre"].to_numpy(dtype=np.float64)
    flat = out["no_spread"].to_numpy(dtype=bool)

    # No difference is no shift whatever the scale; a quantity constant across the
    # instances that still moved has no unit, so its shift is undefined.
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(
            difference == 0.0,
            0.0,
            np.where(flat, np.nan, difference / sigma),
        )
    out["delta_z"] = delta
    out["abs_delta_z"] = np.abs(delta)
    out["abs_difference"] = np.abs(difference)
    return out


def measure_shift(observations: pd.DataFrame) -> pd.DataFrame:
    """Measure every quantity's per-instance standardized shift."""
    paired = _paired(instance_values(observations))
    shift = _standardize(paired).assign(standardization=PRIMARY_STANDARDIZATION)

    undefined = int(shift["abs_delta_z"].isna().sum())
    if undefined:
        LOGGER.warning(
            "shift: %d observation(s) have an undefined shift: the quantity is "
            "constant across the instances under pre but moved under post",
            undefined,
        )
    return shift


def _median(values: pd.Series) -> float:
    """Return the median over the deployments, NaN if any deployment is NaN.

    `np.median`, not pandas, which would skip a NaN and report a subset's median.
    """
    return float(np.median(values.to_numpy(dtype=np.float64)))


def summarize_shift(shift: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-instance shifts over the instances, then the deployments.

    ``p_moved`` reads the unstandardized difference, so an instance with an undefined
    shift still counts as moved. An even number of deployments raises, since the
    median would fall between two of them.
    """
    deployments = _summarize_deployments(shift)
    keys = list(REPORT_KEYS)
    rows: list[dict] = []
    for key, group in deployments.groupby(keys, observed=True, sort=True):
        identity = dict(zip(keys, cast(tuple, key), strict=True))
        if group[DEPLOYMENT_KEY].nunique() % 2 == 0:
            raise ValueError(
                f"{identity['quantity']}: {group[DEPLOYMENT_KEY].nunique()} "
                "deployment(s); an even number has no median among them"
            )
        counts = group["n_instances"].to_numpy()
        if (counts != counts[0]).any():
            raise ValueError(
                f"{identity['quantity']}: the deployments hold different numbers "
                "of instances"
            )
        rows.append(
            {
                **identity,
                "p_moved": _median(cast(pd.Series, group["p_moved"])),
                "n_instances": int(group["n_instances"].iloc[0]),
                "n_undefined": int(_median(cast(pd.Series, group["n_undefined"]))),
                "statistic": STATISTIC,
                "value": _median(cast(pd.Series, group["value"])),
                "value_raw": _median(cast(pd.Series, group["value_raw"])),
            }
        )
    return pd.DataFrame(rows)


def _summarize_deployments(shift: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-instance shifts over the instances, deployment by deployment."""
    keys = list(SUMMARY_KEYS)
    rows: list[dict] = []
    for key, group in shift.groupby(keys, observed=True, sort=True):
        identity = dict(zip(keys, cast(tuple, key), strict=True))
        values = group["abs_delta_z"].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        raw = group["abs_difference"].to_numpy(dtype=np.float64)
        rows.append(
            {
                **identity,
                "p_moved": float(np.mean(raw > 0.0)),
                "n_instances": len(group),
                "n_undefined": int(values.size - finite.size),
                # Mean over the defined shifts; `n_undefined` counts the rest.
                "value": float(np.mean(finite)) if finite.size else float("nan"),
                "value_raw": float(np.mean(raw)),
            }
        )
    return pd.DataFrame(rows)


def feature_observations(features: pd.DataFrame) -> pd.DataFrame:
    """Reshape the feature shard into what :func:`measure_shift` consumes."""
    return pd.DataFrame(
        {
            "problem_class": features["problem_class"],
            "function_id": features["function_id"],
            "instance_id": features["instance_id"],
            "dimension": features["dimension"],
            "rep": features["rep"],
            "quantity_kind": FEATURE_QUANTITY,
            "feature_set": features["feature_set"],
            "feature_variant": features["feature_variant"],
            "quantity": features["feature"],
            "quantity_class": features["feature"].map(feature_class),
            "transform": features["transform"],
            "replicate": SINGLE_REPLICATE,
            "value": features["value"],
        }
    )


def view_superiority_observations(
    superiorities: pd.DataFrame, reps: Sequence[int]
) -> pd.DataFrame:
    """Reshape the superiority shard into what :func:`measure_shift` consumes.

    The shard is the regression's own target, so the shift is measured on what the
    regression predicts. It does not depend on the feature sample, so every
    deployment in ``reps`` reads the same value.
    """
    rates = pd.DataFrame(
        {
            "problem_class": superiorities["problem_class"],
            "function_id": superiorities["function_id"],
            "instance_id": superiorities["instance_id"],
            "dimension": superiorities["dimension"],
            "quantity_kind": SUPERIORITY_QUANTITY,
            "feature_set": NOT_A_FEATURE,
            "feature_variant": NOT_A_FEATURE,
            "quantity": superiorities["pair"],
            "quantity_class": superiorities["algorithm_a"].map(ALGORITHM_GROUP),
            "transform": superiorities["transform"],
            "replicate": SINGLE_REPLICATE,
            "value": superiorities["superiority"],
        }
    )
    return pd.concat(
        [rates.assign(rep=rep) for rep in reps], ignore_index=True
    ).reindex(columns=list(OBSERVATION_COLUMNS))


def _write_summary(
    summary: pd.DataFrame, problem_class: str, dim: int, run_paths: RunPaths
) -> pd.DataFrame:
    """Write a summary table, warning about rows with no standardized estimate."""
    # Kept, not dropped: a missing row would read as a quantity never measured.
    undefined = int(summary["value"].isna().sum())
    if undefined:
        LOGGER.warning(
            "%d shift summary row(s) have no standardized estimate; read value_raw "
            "and n_undefined for those quantities",
            undefined,
        )
    shard = run_paths.metric(f"shift_summary_{problem_slug(problem_class)}_{dim}d")
    written = write_table(summary, shard.data, SHIFT_SUMMARY_SCHEMA)
    LOGGER.info("shift_summary: wrote %s (%d rows)", shard.data, len(written))
    return written


def run_shift_summary_stage(
    shift: pd.DataFrame,
    problem_class: str,
    dim: int,
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
) -> pd.DataFrame:
    """Aggregate the per-instance shifts into the reported summary table."""
    summary = summarize_shift(shift).assign(scale=scale, config_hash=config_hash)
    return _write_summary(summary, problem_class, dim, run_paths)
