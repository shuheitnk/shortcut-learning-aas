"""Pairwise superiorities via the Vargha-Delaney statistic."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.benchmark_algorithms import select_budget
from aas_shortcut.define_algorithm_portfolio import PAIRS, AlgorithmPair
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_table_schema import VIEW_SUPERIORITY_SCHEMA
from aas_shortcut.read_write_tables import write_table

LOGGER = logging.getLogger(__name__)

# Columns identifying one instance.
GROUP_COLUMNS = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
)

# Columns identifying one instance under one view of the objective; grouping by
# GROUP_COLUMNS alone would pool the two views' trials into one rate.
VIEW_GROUP_COLUMNS = (*GROUP_COLUMNS, "transform")


def vargha_delaney(a: np.ndarray, b: np.ndarray) -> float:
    """Return the probability that ``a`` beats ``b`` under minimisation."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError(f"empty sample: len(a)={a.size}, len(b)={b.size}")

    comparison = a[:, None] - b[None, :]
    wins = float(np.count_nonzero(comparison < 0))
    ties = float(np.count_nonzero(comparison == 0))
    return (wins + 0.5 * ties) / float(a.size * b.size)


def _pair_superiority(
    values: dict[str, np.ndarray], pair: AlgorithmPair
) -> tuple[float, int] | None:
    """Return ``(superiority, n_trials)`` for one pair, or ``None`` if incomplete."""
    a = values.get(pair.first)
    b = values.get(pair.second)
    if a is None or b is None or a.size == 0 or b.size == 0:
        return None
    return vargha_delaney(a, b), int(min(a.size, b.size))


def compute_superiorities(
    performance: pd.DataFrame,
    pairs: tuple[AlgorithmPair, ...] = PAIRS,
    group_columns: Sequence[str] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Reduce a long performance table to one superiority per instance and pair."""
    rows: list[dict] = []
    group_columns = list(group_columns)

    for key, group in performance.groupby(group_columns, observed=True, sort=True):
        identity = dict(zip(group_columns, cast(tuple, key), strict=True))
        values = cast(
            dict[str, np.ndarray],
            {
                algorithm: subset["best_y"].to_numpy(dtype=np.float64)
                for algorithm, subset in group.groupby("algorithm", observed=True)
            },
        )
        for pair in pairs:
            result = _pair_superiority(values, pair)
            if result is None:
                LOGGER.warning(
                    "skipping pair %s for %s: missing algorithm results",
                    pair.slug,
                    identity,
                )
                continue
            superiority, n_trials = result
            rows.append(
                {
                    **identity,
                    "pair": pair.slug,
                    "algorithm_a": pair.first,
                    "algorithm_b": pair.second,
                    "pair_type": pair.pair_type,
                    "n_trials": n_trials,
                    "superiority": superiority,
                }
            )

    return pd.DataFrame(rows)


def run_view_superiority_stage(
    performance: pd.DataFrame,
    problem_class: str,
    dim: int,
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
    pairs: tuple[AlgorithmPair, ...] = PAIRS,
    *,
    budget: int | None = None,
) -> pd.DataFrame:
    """Compute the superiorities of a shard that holds both views, and persist them."""
    superiorities = compute_superiorities(
        select_budget(performance, budget), pairs, VIEW_GROUP_COLUMNS
    )
    if superiorities.empty:
        raise ValueError(
            f"no superiorities computed for {problem_class} {dim}D; the performance "
            "table is empty or missing algorithms"
        )
    superiorities = superiorities.assign(scale=scale, config_hash=config_hash)

    shard = run_paths.view_superiority(problem_class, dim)
    written = write_table(superiorities, shard.data, VIEW_SUPERIORITY_SCHEMA)
    LOGGER.info("view_superiority: wrote %s (%d rows)", shard.data, len(written))
    return written


def views_of(superiorities: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Return one pair's superiority under each view, one row per instance."""
    selected = superiorities[superiorities["pair"] == pair]
    return (
        selected.pivot_table(
            index=list(GROUP_COLUMNS), columns="transform", values="superiority"
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
