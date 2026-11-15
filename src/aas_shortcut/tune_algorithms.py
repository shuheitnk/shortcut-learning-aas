"""Candidate algorithm tuning."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aas_shortcut.benchmark_algorithms import trial_seed
from aas_shortcut.calculate_superiority import vargha_delaney
from aas_shortcut.create_problem import create_problem
from aas_shortcut.define_algorithm_portfolio import AlgorithmPair
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_table_schema import (
    TUNING_PAIR_SCHEMA,
)
from aas_shortcut.execute_parallel_jobs import run_jobs
from aas_shortcut.optimizers.define_optimizer_base import BudgetedObjective, Runner
from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm, config_id
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS

LOGGER = logging.getLogger(__name__)

# Worker processes for the grid search.
TUNING_JOBS = int(os.environ.get("AAS_SHORTCUT_TUNING_JOBS", "-1"))


# ---------------------------------------------------------------------------
# Uniformity metrics
# ---------------------------------------------------------------------------


# The estimator's token enters the tuning's fingerprint, so a table measured under
# another estimator is never reused.
ESTIMATOR = "vargha-delaney"


def ks_distance(rates: np.ndarray) -> float:
    """Kolmogorov-Smirnov distance to ``Uniform(0, 1)``."""
    x = np.sort(np.asarray(rates, dtype=np.float64))
    n = len(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))


def comparisons_per_rate(n_trials: int) -> int:
    """Pairwise comparisons one superiority is estimated from.

    Every run is compared with every run of the other member, so a rate is a
    multiple of ``1 / (2 * n_trials ** 2)``.
    """
    return n_trials * n_trials


def floor_for(n_instances: int, n_comparisons: int) -> float:
    """Best score a sample of this granularity can reach."""
    step = 1.0 / (2 * n_comparisons)
    quantiles = (np.arange(1, n_instances + 1) - 0.5) / n_instances
    snapped = np.round(quantiles / step) * step
    return ks_distance(np.clip(snapped, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Running a pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairConditions:
    """Benchmark conditions a tuning is measured under."""

    dim: int = 10
    budget_factor: int = 1000
    function_ids: tuple[int, ...] = tuple(range(1, 25))
    # One instance per function.
    instance_ids: tuple[int, ...] = (1,)
    n_trials: int = 31
    base_seed: int = 7000

    @property
    def budget(self) -> int:
        """Evaluations per run."""
        return self.budget_factor * self.dim

    @property
    def instances(self) -> tuple[tuple[int, int], ...]:
        """The ``(function_id, instance_id)`` pairs, in a fixed order."""
        return tuple(
            (function_id, instance_id)
            for function_id in self.function_ids
            for instance_id in self.instance_ids
        )

    @property
    def trials(self) -> range:
        """The trials, numbered as the benchmark numbers its own (from 1)."""
        return range(1, self.n_trials + 1)

    def seed(
        self, algorithm: str, function_id: int, instance_id: int, trial: int
    ) -> int:
        """Seed of one run: the benchmark's for the same member, problem and trial.

        Every candidate of one member so meets the same seeds.
        """
        return trial_seed(
            self.base_seed,
            algorithm,
            PROBLEM_CLASS,
            function_id,
            instance_id,
            self.dim,
            trial,
        )

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description."""
        return {
            "dim": self.dim,
            "budget": self.budget,
            "function_ids": list(self.function_ids),
            "instance_ids": list(self.instance_ids),
            "n_trials": self.n_trials,
            "base_seed": self.base_seed,
            # How a run's seed is derived, so a table measured under another seed
            # rule is never reused.
            "seeds": "trial",
        }


def _final_error(
    runner: Runner, function_id: int, instance_id: int, seed: int, c: PairConditions
) -> float:
    """Run once and return the final distance to the optimum."""
    problem = create_problem(function_id, instance_id, c.dim)
    lb, ub = problem.bounds.lb, problem.bounds.ub
    # The benchmark's bound rule, so a candidate is tuned on the problem it is
    # later evaluated on.
    objective = BudgetedObjective(problem, c.budget, bounds=(lb, ub))
    runner(objective, c.dim, seed, c.budget, lb, ub)
    return float(problem.state.current_best.y) - float(problem.optimum.y)


# Errors of one configuration, keyed by ``(algorithm, config_id)``.
ErrorCache = dict[tuple[str, str], "Errors"]


def _run_one(
    entry: TunableAlgorithm,
    params: Mapping[str, Any],
    function_id: int,
    instance_id: int,
    seed: int,
    c: PairConditions,
) -> float:
    """One optimizer run of the tuning: the unit of work that is dispatched."""
    return _final_error(entry.runner(dict(params)), function_id, instance_id, seed, c)


def run_configurations(
    entries: Mapping[str, TunableAlgorithm],
    requests: Sequence[tuple[str, Mapping[str, Any]]],
    c: PairConditions,
    label: str,
    n_jobs: int = TUNING_JOBS,
    cache: ErrorCache | None = None,
) -> ErrorCache:
    """Measure every requested configuration over the tuning set, in one pass."""
    cache = {} if cache is None else cache
    wanted: list[tuple[tuple[str, str], str, Mapping[str, Any]]] = []
    seen = set(cache)
    for name, params in requests:
        key = (name, config_id(params))
        if key in seen:
            continue
        seen.add(key)
        wanted.append((key, name, params))
    if not wanted:
        return cache

    instances = c.instances
    n_trials = c.n_trials
    tasks = [
        (
            entries[name],
            params,
            function_id,
            instance_id,
            c.seed(name, function_id, instance_id, trial),
            c,
        )
        for _, name, params in wanted
        for function_id, instance_id in instances
        for trial in c.trials
    ]
    LOGGER.info(
        "%s: %d configuration(s) x %d instance(s) x %d trial(s) = %d run(s) on "
        "%d worker(s)",
        label,
        len(wanted),
        len(instances),
        n_trials,
        len(tasks),
        n_jobs,
    )

    values = [0.0] * len(tasks)
    for index, value in run_jobs(label, tasks, _run_one, n_jobs=n_jobs):
        values[index] = value

    per_configuration = len(instances) * n_trials
    for position, (key, _, _) in enumerate(wanted):
        block = values[
            position * per_configuration : (position + 1) * per_configuration
        ]
        cache[key] = {
            instance: np.array(block[offset * n_trials : (offset + 1) * n_trials])
            for offset, instance in enumerate(instances)
        }
    return cache


def superiorities(
    first: dict[tuple[int, int], np.ndarray],
    second: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    """Per-instance superiority of ``first`` over ``second``.

    The same estimator as the benchmark's, so the tuning flattens the quantity the
    study later regresses.
    """
    return np.array([vargha_delaney(first[i], second[i]) for i in sorted(first)])


def summarise(rates: np.ndarray, c: PairConditions) -> dict[str, Any]:
    """Score one superiority sample against ``U(0, 1)``, with context."""
    return {
        "score": ks_distance(rates),
        "metric": "ks",
        "floor": floor_for(len(rates), comparisons_per_rate(c.n_trials)),
        "mean": float(rates.mean()),
        "sd": float(rates.std()),
        "n_informative": int(((rates > 0.2) & (rates < 0.8)).sum()),
        "n_saturated": int(((rates == 0.0) | (rates == 1.0)).sum()),
        "rates": rates.tolist(),
    }


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def weaker_side(
    first: TunableAlgorithm,
    second: TunableAlgorithm,
    c: PairConditions,
    first_errors: Errors,
    second_errors: Errors,
) -> tuple[TunableAlgorithm, TunableAlgorithm, dict[str, Any]]:
    """Return ``(weaker, stronger, baseline)`` under the two defaults."""
    rates = superiorities(first_errors, second_errors)
    baseline = summarise(rates, c)
    baseline["direction"] = f"{first.name} over {second.name}"
    if rates.mean() < 0.5:
        return first, second, baseline
    return second, first, baseline


def tune_pair(
    first: TunableAlgorithm,
    second: TunableAlgorithm,
    conditions: PairConditions | None = None,
    n_jobs: int = TUNING_JOBS,
) -> dict[str, Any]:
    """Tune the weaker member so the pair's superiorities approach uniformity."""
    c = conditions or PairConditions()
    entries = {first.name: first, second.name: second}
    pair_label = f"tune {first.name} vs {second.name}"
    # Both members at their published configuration; later phases reuse these runs.
    cache: ErrorCache = run_configurations(
        entries,
        [(first.name, first.resolve()), (second.name, second.resolve())],
        c,
        f"{pair_label} (defaults)",
        n_jobs,
    )
    tuned, fixed, baseline = weaker_side(
        first,
        second,
        c,
        cache[(first.name, first.default_config_id)],
        cache[(second.name, second.default_config_id)],
    )
    LOGGER.info(
        "tune: %s is weaker than %s (mean %.3f); tuning %s over %d candidate(s)",
        tuned.name,
        fixed.name,
        baseline["mean"],
        tuned.name,
        len(tuned.candidates()),
    )

    if not tuned.grid:
        raise ValueError(
            f"{tuned.name} is the weaker member but declares no grid to tune over"
        )

    candidates = tuned.candidates()
    # Every candidate is compared against the same runs of the fixed member.
    requests = [(tuned.name, params) for params in candidates]
    cache = run_configurations(
        entries, requests, c, f"{pair_label} (grid)", n_jobs, cache
    )
    fixed_errors = cache[(fixed.name, fixed.default_config_id)]

    def rates_of(params: Mapping[str, Any]) -> np.ndarray:
        """The superiority sample of one candidate."""
        return superiorities(cache[(tuned.name, config_id(params))], fixed_errors)

    results = [
        {
            "params": params,
            "config_id": config_id(params),
            "is_default": config_id(params) == tuned.default_config_id,
            **summarise(rates_of(params), c),
        }
        for params in candidates
    ]
    # A stable sort, so a tie resolves to the configuration the grid lists first.
    results.sort(key=lambda row: row["score"])
    best = results[0]

    LOGGER.info(
        "tune: D_KS %.4f -> %.4f (floor %.4f), saturated %d -> %d",
        baseline["score"],
        best["score"],
        best["floor"],
        baseline["n_saturated"],
        best["n_saturated"],
    )
    return {
        "conditions": c.as_dict(),
        "metric": "ks",
        "tuned": tuned.name,
        "fixed": fixed.name,
        "direction": f"{tuned.name} over {fixed.name}",
        "baseline": baseline,
        "best": best,
        "results": results,
    }


# ---------------------------------------------------------------------------
# The measured grid
# ---------------------------------------------------------------------------


# Errors of one configuration, keyed by ``(function_id, instance_id)``.
Errors = dict[tuple[int, int], np.ndarray]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _pair_rows(
    pair: AlgorithmPair, record: Mapping[str, Any], c: PairConditions
) -> list[dict[str, Any]]:
    """Explode one pair's tuning into one row per configuration and instance."""
    best = record["best"]["config_id"]
    instances = sorted(c.instances)
    rows: list[dict[str, Any]] = []
    for result in record["results"]:
        shared = {
            "group": pair.pair_type.split("_")[0],
            "pair": pair.slug,
            "tuned": record["tuned"],
            "fixed": record["fixed"],
            "config_id": result["config_id"],
            "params": json.dumps(result["params"], sort_keys=True, default=str),
            "is_default": bool(result["is_default"]),
            "is_selected": result["config_id"] == best,
            "dimension": c.dim,
            "budget": c.budget,
            "score": float(result["score"]),
            "floor": float(result["floor"]),
            "n_saturated": int(result["n_saturated"]),
        }
        rows.extend(
            {
                **shared,
                "function_id": function_id,
                "instance_id": instance_id,
                "superiority": float(rate),
            }
            for (function_id, instance_id), rate in zip(
                instances, result["rates"], strict=True
            )
        )
    return rows


# The two configurations of each pair a report names: the tuned member's published
# setting and the one its tuning selected.
PAIR_TUNING_CONFIGURATIONS = ("default", "selected")


def summarize_pair_tuning(table: pd.DataFrame) -> pd.DataFrame:
    """Return each pair's default and selected configuration, from its tuning table.

    ``n_at_minimum`` counts the candidates tied at the selected score; the grid's
    order took the first.
    """
    keys = ["group", "pair", "tuned", "fixed", "config_id", "params"]
    flags = ["is_default", "is_selected"]
    configurations = (
        table.groupby([*keys, *flags], sort=False, observed=True)
        .agg(
            score=("score", "first"),
            floor=("floor", "first"),
            n_saturated=("n_saturated", "first"),
            n_problems=("function_id", "size"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for pair, candidates in configurations.groupby("pair", sort=False):
        records = candidates.to_dict("records")
        chosen = {
            name: [record for record in records if record[flag]]
            for name, flag in zip(PAIR_TUNING_CONFIGURATIONS, flags, strict=True)
        }
        for name, found in chosen.items():
            if len(found) != 1:
                raise ValueError(f"{pair}: {len(found)} {name} configuration(s)")
        minimum = float(chosen["selected"][0]["score"])
        at_minimum = sum(float(record["score"]) == minimum for record in records)
        for name, (row,) in chosen.items():
            rows.append(
                {
                    **{key: row[key] for key in keys},
                    "configuration": name,
                    "score": float(row["score"]),
                    "floor": float(row["floor"]),
                    "n_saturated": int(row["n_saturated"]),
                    "n_problems": int(row["n_problems"]),
                    "n_candidates": len(records),
                    "n_at_minimum": at_minimum,
                }
            )
    return pd.DataFrame(rows)


def run_tune_pair_stage(
    entries: Mapping[str, TunableAlgorithm],
    pairs: Sequence[AlgorithmPair],
    conditions: PairConditions,
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
    n_jobs: int = TUNING_JOBS,
) -> pd.DataFrame:
    """Run the pairwise tuning for every candidate pair."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        LOGGER.info("tune pair: %s vs %s", pair.first, pair.second)
        record = tune_pair(
            entries[pair.first], entries[pair.second], conditions, n_jobs
        )
        rows.extend(_pair_rows(pair, record, conditions))

    table = pd.DataFrame(rows).assign(scale=scale, config_hash=config_hash)
    shard = run_paths.metric("tuning_pairs")
    written = write_table(table, shard.data, TUNING_PAIR_SCHEMA)
    LOGGER.info("tune pair: wrote %s (%d rows)", shard.data, len(written))
    return written
