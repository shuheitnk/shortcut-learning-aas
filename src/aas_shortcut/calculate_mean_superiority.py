"""Instance-wise mean superiorities: the performance the selector regresses.

The mean superiority of member ``a`` on problem ``p`` is

    J(p, a) = E[(k - R(p, a)) / (k - 1)],

where ``R(p, a)`` is ``a``'s rank among one run of each of the ``k`` members at the
search budget, smallest ``best_y`` first and ties given their average rank. The best
member scores 1 and the worst 0. Because ``k - R`` counts the members ``a`` beats (a
tie counting one half), ``J`` is the mean over the other members of
``P(a beats b)``: it is estimated as the mean Vargha-Delaney A over every pair of
trials, which for ``k = 2`` is the Vargha-Delaney A itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.benchmark_algorithms import select_budget
from aas_shortcut.define_algorithm_portfolio import ALGORITHM_GROUP, ALGORITHMS
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_table_schema import VIEW_MEAN_SUPERIORITY_SCHEMA
from aas_shortcut.read_write_tables import write_table

LOGGER = logging.getLogger(__name__)

# Columns identifying one instance.
GROUP_COLUMNS = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
)

# Columns identifying one ranking in a shard holding several views of the objective:
# members are ranked only among runs of the same view.
VIEW_GROUP_COLUMNS = (*GROUP_COLUMNS, "transform")


def mean_superiorities(samples: Sequence[np.ndarray]) -> np.ndarray:
    """Return each member's mean superiority from its trials' final values.

    Every trial is compared with every trial of every other member. Comparisons are
    counted from one sort of the pooled values, so a whole configuration grid stays
    affordable, and in integers, so exact ties survive the division.
    """
    k = len(samples)
    if k < 2:
        raise ValueError(f"a mean superiority needs at least two members, got {k}")
    arrays = [np.asarray(sample, dtype=np.float64).ravel() for sample in samples]
    sizes = np.array([array.size for array in arrays], dtype=np.int64)
    if (sizes == 0).any():
        raise ValueError(f"empty sample among {k} member(s): sizes {sizes.tolist()}")

    # One run of member b beating a run of a is worth 1/T_b of P(b beats a). In units
    # of 1/(2L), with L a common multiple of the trial counts, a win is 2L/T_b and a
    # tie half of it, so every count below is an integer.
    common = int(np.lcm.reduce(sizes))
    values = np.concatenate(arrays)
    member = np.repeat(np.arange(k), sizes)
    weight = (2 * common // sizes)[member]

    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    counts = np.diff(np.r_[starts, ordered.size])
    cumulative = np.r_[0, np.cumsum(weight[order])]
    below = cumulative[starts]
    tied = cumulative[starts + counts] - below
    beaten_pooled = np.empty(values.size, dtype=np.int64)
    beaten_pooled[order] = np.repeat(below + tied // 2, counts)

    # The same count among the member's own runs, run included, which the pooled
    # count holds and the score must not: 2 * (average rank) - 1 own ties-and-wins.
    own = np.lexsort((values, member))
    own_values, own_member = values[own], member[own]
    own_starts = np.flatnonzero(
        np.r_[
            True,
            (own_values[1:] != own_values[:-1]) | (own_member[1:] != own_member[:-1]),
        ]
    )
    own_counts = np.diff(np.r_[own_starts, own.size])
    member_starts = np.r_[0, np.cumsum(sizes)[:-1]]
    position = own_starts - member_starts[own_member[own_starts]]
    twice_rank_minus_one = np.empty(values.size, dtype=np.int64)
    twice_rank_minus_one[own] = np.repeat(2 * position + own_counts, own_counts)
    beaten = beaten_pooled - twice_rank_minus_one * (common // sizes)[member]

    totals = np.zeros(k, dtype=np.int64)
    np.add.at(totals, member, beaten)
    return 1.0 - totals / (2.0 * common * sizes * (k - 1))


def compute_mean_superiorities(
    performance: pd.DataFrame,
    algorithms: tuple[str, ...] = ALGORITHMS,
    group_columns: tuple[str, ...] = GROUP_COLUMNS,
) -> pd.DataFrame:
    """Reduce a long performance table to a mean superiority per instance and member."""
    rows: list[dict] = []
    keys = list(group_columns)
    wanted = list(algorithms)

    for key, group in performance.groupby(keys, observed=True, sort=True):
        identity = dict(zip(keys, cast(tuple, key), strict=True))
        values = {
            algorithm: subset["best_y"].dropna().to_numpy(dtype=np.float64)
            for algorithm, subset in group.groupby("algorithm", observed=True)
        }
        missing = [name for name in wanted if values.get(name, np.empty(0)).size == 0]
        if missing:
            LOGGER.warning(
                "skipping %s: no results for %s", identity, ", ".join(missing)
            )
            continue

        samples = [values[name] for name in wanted]
        n_trials = min(sample.size for sample in samples)
        for algorithm, score in zip(wanted, mean_superiorities(samples), strict=True):
            rows.append(
                {
                    **identity,
                    "algorithm": algorithm,
                    "group": ALGORITHM_GROUP[algorithm],
                    "n_trials": n_trials,
                    "mean_superiority": float(score),
                }
            )

    return pd.DataFrame(rows)


def run_view_mean_superiority_stage(
    performance: pd.DataFrame,
    problem_class: str,
    dim: int,
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
    *,
    budget: int | None = None,
    condition: str | None = None,
) -> pd.DataFrame:
    """Compute and persist the mean superiorities of a shard holding both views."""
    scores = compute_mean_superiorities(
        select_budget(performance, budget), ALGORITHMS, VIEW_GROUP_COLUMNS
    )
    if scores.empty:
        raise ValueError(
            f"no mean superiorities computed for {problem_class} {dim}D; the "
            "performance table is empty or missing algorithms"
        )
    scores = scores.assign(scale=scale, config_hash=config_hash)

    shard = run_paths.view_mean_superiority(problem_class, dim, condition)
    written = write_table(scores, shard.data, VIEW_MEAN_SUPERIORITY_SCHEMA)
    LOGGER.info("view_mean_superiority: wrote %s (%d rows)", shard.data, len(written))
    return written
