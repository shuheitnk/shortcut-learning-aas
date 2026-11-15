"""Reducing the measured grid to one mean superiority per configuration and function."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.calculate_mean_superiority import mean_superiorities
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_table_schema import TUNING_ROLE_SCHEMA
from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm, config_id
from aas_shortcut.read_write_tables import write_table

LOGGER = logging.getLogger(__name__)

STAGE = "tuning_roles"

# Completeness check only: every trial of a problem must hold the whole grid, or the
# rest would be scored against a smaller pool. Scores pool all trials (PROBLEM_COLUMNS).
COMPLETENESS_COLUMNS = ("budget", "function_id", "instance_id", "trial")

# One mean superiority is taken per problem, across every trial of every configuration.
PROBLEM_COLUMNS = ("budget", "function_id", "instance_id")

MEMBER_COLUMNS = ("algorithm", "config_id")

# Trial halves for the split-half check. Alternating rather than first/second half,
# so it is robust to a run whose later trials came from a resume.
HALVES: dict[str, int] = {"odd": 1, "even": 0}


def _pool_scores(errors: pd.DataFrame) -> pd.Series:
    """Score every configuration against the whole grid, problem by problem."""
    scores: list[pd.Series] = []
    for key, problem in errors.groupby(list(PROBLEM_COLUMNS), observed=True):
        members = problem.groupby(list(MEMBER_COLUMNS), observed=True, sort=False)
        labels = list(members.groups)
        samples = [
            members.get_group(label)["error"].to_numpy(dtype=np.float64)
            for label in labels
        ]
        index = pd.MultiIndex.from_tuples(
            [(*key, *label) for label in labels],
            names=[*PROBLEM_COLUMNS, *MEMBER_COLUMNS],
        )
        scores.append(pd.Series(mean_superiorities(samples), index=index))
    if not scores:
        # A half with no trial: leave the column undefined.
        return pd.Series(
            dtype=np.float64,
            index=pd.MultiIndex.from_tuples(
                [], names=[*PROBLEM_COLUMNS, *MEMBER_COLUMNS]
            ),
        )
    return pd.concat(scores)


def pooled_mean_superiorities(errors: pd.DataFrame) -> pd.DataFrame:
    """Score every configuration's mean superiority against the whole grid.

    The pool is every configuration of every member, so no score depends on which
    binding the tuning picks. Half-trial scores use that half's runs only.
    """
    if errors.empty:
        raise ValueError("the error table is empty; there is nothing to rank")

    sizes = errors.groupby(list(COMPLETENESS_COLUMNS), observed=True).size()
    expected = int(cast(int, sizes.max()))
    short = sizes[sizes != expected]
    if not short.empty:
        raise ValueError(
            f"{len(short)} race(s) hold fewer than {expected} configuration(s), "
            f"e.g. {short.index[0]} with {int(short.iloc[0])}; a score is only "
            "comparable across functions when the pool is the same size"
        )

    keys = ["budget", "algorithm", "config_id", "function_id", "instance_id"]
    reduced = errors.groupby(keys, observed=True).agg(n_trials=("trial", "size"))
    reduced["mean_superiority"] = _pool_scores(errors).reorder_levels(keys)
    for name, remainder in HALVES.items():
        half = cast(pd.DataFrame, errors[errors["trial"] % 2 == remainder])
        reduced[f"mean_superiority_{name}"] = _pool_scores(half).reorder_levels(keys)
    return reduced.reset_index()


def _describe(params: Mapping[str, object]) -> str:
    """Render one assignment for the table."""
    return json.dumps(dict(params), sort_keys=True, default=str)


def annotate(
    scores: pd.DataFrame,
    entries: Mapping[str, TunableAlgorithm],
    selected: Mapping[int, Mapping[str, str]],
    dimension: int,
) -> pd.DataFrame:
    """Attach the parameters and the per-budget choice to the score table."""
    by_id = {
        (name, config_id(params)): params
        for name, entry in entries.items()
        for params in entry.candidates()
    }
    defaults = {name: entry.default_config_id for name, entry in entries.items()}
    chosen = {
        (int(budget), str(algorithm), str(identifier))
        for budget, per_member in selected.items()
        for algorithm, identifier in per_member.items()
    }
    keys = list(zip(scores["algorithm"], scores["config_id"], strict=True))
    return scores.assign(
        params=[_describe(by_id[(str(a), str(c))]) for a, c in keys],
        is_default=[defaults[str(a)] == str(c) for a, c in keys],
        is_selected=[
            (int(budget), str(a), str(c)) in chosen
            for budget, (a, c) in zip(scores["budget"], keys, strict=True)
        ],
        dimension=dimension,
    )


def run_grid_rank_stage(
    scores: pd.DataFrame,
    entries: Mapping[str, TunableAlgorithm],
    selected: Mapping[int, Mapping[str, str]],
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
    dimension: int,
) -> pd.DataFrame:
    """Record the two roles' choices beside the scores and persist."""
    table = annotate(scores, entries, selected, dimension)
    table = table.assign(scale=scale, config_hash=config_hash)
    shard = run_paths.metric(STAGE)
    written = write_table(table, shard.data, TUNING_ROLE_SCHEMA)
    LOGGER.info("%s: wrote %s (%d rows)", STAGE, shard.data, len(written))
    return written
