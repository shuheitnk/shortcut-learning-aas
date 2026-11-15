"""The appendix's levels: each condition's LOFO and RRT values over the 11 seeds.

A seed's LOFO value is the median over the deployments of its pooled R²; its RRT
value is Experiment 4's unseen-family mean. A condition's level is the median over
the seeds, with the quartiles beside it.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aas_shortcut.compare_training_runs import CELL_COLUMNS, SEED_COLUMN, per_seed
from aas_shortcut.define_reporting_scopes import SCOPE_UNSEEN
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_lofo_regression import METRIC

KEYS: list[str] = list(CELL_COLUMNS)


def lofo_runs(scores: pd.DataFrame) -> pd.DataFrame:
    """One LOFO value per cell and seed: the median over the deployments."""
    rows = scores[scores["metric"] == METRIC]
    return (
        rows.groupby([*KEYS, SEED_COLUMN], observed=True)["value"]
        .median()
        .reset_index()
    )


def rrt_runs(deployments: pd.DataFrame) -> pd.DataFrame:
    """One RRT value per cell and seed, as Experiment 4's unseen-family range."""
    return per_seed(deployments, SCOPE_UNSEEN, TEST_METRICS["post"])


def seed_spread(runs: pd.DataFrame, name: str) -> pd.DataFrame:
    """Median and quartiles over the seeds, one row per cell."""
    grouped = runs.groupby(KEYS, observed=True)["value"]
    return pd.DataFrame(
        {
            f"{name}_median": grouped.median(),
            f"{name}_q1": grouped.quantile(0.25),
            f"{name}_q3": grouped.quantile(0.75),
            f"{name}_n_seeds": grouped.size(),
        }
    ).reset_index()


def level_table(lofo: pd.DataFrame, rrt: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each condition's LOFO level, with its RRT level beside it when given."""
    table = seed_spread(lofo_runs(lofo), "lofo")
    if rrt is not None:
        table = table.merge(seed_spread(rrt_runs(rrt), "rrt"), on=KEYS, how="left")
    return table.sort_values(KEYS, kind="stable").reset_index(drop=True)


def prediction_runs(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per cell and seed, the median over the deployments of three values.

    ``prediction_sd`` and ``observed_sd`` are the population standard deviations
    of the 240 predictions and targets; ``function_rho`` is Spearman's rho over the
    24 functions between the mean target and the mean prediction of each.
    """
    rows = []
    keys = [*KEYS, SEED_COLUMN, "rep"]
    for key, group in predictions.groupby(keys, observed=True, sort=True):
        means = group.groupby("function_id")[["observed", "predicted"]].mean()
        rows.append(
            {
                **dict(zip(keys, cast(tuple, key), strict=True)),
                "prediction_sd": float(np.std(group["predicted"])),
                "observed_sd": float(np.std(group["observed"])),
                "function_rho": float(
                    spearmanr(means["observed"], means["predicted"]).statistic
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .groupby([*KEYS, SEED_COLUMN], observed=True)
        .median(numeric_only=True)
        .drop(columns="rep")
        .reset_index()
    )


def summary_table(scores: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Each condition's medians over the seeds: LOFO R², spread and function rho."""
    levels = lofo_runs(scores).rename(columns={"value": "lofo_r2"})
    table = levels.merge(prediction_runs(predictions), on=[*KEYS, SEED_COLUMN])
    medians = table.groupby(KEYS, observed=True).median(numeric_only=True)
    medians = medians.drop(columns=SEED_COLUMN).reset_index()
    medians["sd_ratio"] = medians["prediction_sd"] / medians["observed_sd"]
    return medians.sort_values(KEYS, kind="stable").reset_index(drop=True)


def spread_correlations(summary: pd.DataFrame) -> pd.DataFrame:
    """Spearman's rho between the spread of the predictions and the LOFO R².

    Per group over its six conditions, on the predictions' standard deviation; and
    over every group's conditions together, on that deviation over the targets',
    since the targets' spread differs between the groups.
    """
    rows = []
    for group, rows_of in summary.groupby("group", observed=True, sort=True):
        rows.append(_rho(rows_of, "prediction_sd", str(group)))
    rows.append(_rho(summary, "sd_ratio", "all groups"))
    return pd.DataFrame(rows)


def _rho(rows: pd.DataFrame, spread: str, scope: str) -> dict:
    return {
        "scope": scope,
        "spread": spread,
        "n_points": len(rows),
        "rho": float(spearmanr(rows[spread], rows["lofo_r2"]).statistic),
    }


def rrt_correlations(levels: pd.DataFrame) -> pd.DataFrame:
    """Spearman's rho between the RRT and LOFO medians, per group."""
    rows = []
    for group, rows_of in levels.groupby("group", observed=True, sort=True):
        rows.append(
            {
                "group": str(group),
                "n_points": len(rows_of),
                "rho": float(
                    spearmanr(rows_of["rrt_median"], rows_of["lofo_median"]).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def function_rho_by_group(summary: pd.DataFrame) -> pd.DataFrame:
    """The function-level rho per group: median, min and max over its conditions."""
    grouped = summary.groupby("group", observed=True, sort=True)["function_rho"]
    return pd.DataFrame(
        {
            "n_conditions": grouped.size(),
            "function_rho_median": grouped.median(),
            "function_rho_min": grouped.min(),
            "function_rho_max": grouped.max(),
        }
    ).reset_index()
