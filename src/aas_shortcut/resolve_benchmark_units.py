"""Benchmark units, and the superiorities each one yields.

A benchmark unit is one binding, condition, view, block and transformation. A run's
result depends only on the binding, seed and problem, so splitting a shard into
units changes no value; tables assembled from units are sorted by the schema's key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from aas_shortcut.benchmark_focus_pairs import MEMBERS
from aas_shortcut.benchmark_objective_views import (
    ViewBenchmarkConfig,
    run_view_benchmark_stage,
)
from aas_shortcut.bind_portfolio_parameters import PortfolioBinding
from aas_shortcut.calculate_superiority import run_view_superiority_stage
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS, AlgorithmPair
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import TABLE_SUFFIX, TRANSFORMS, shard_stem
from aas_shortcut.define_table_schema import SCHEMAS, TableSchema
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit
from aas_shortcut.resolve_tuning_stages import binding_provenance
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP, WarpConfig
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS

PRE, POST = TRANSFORMS


def benchmark_unit(
    context: ExperimentContext,
    *,
    algorithms: Sequence[str],
    binding: PortfolioBinding,
    view: str,
    block: Iterable[int],
    budget: int,
    condition: str | None = None,
    warp: WarpConfig = DEFAULT_WARP,
) -> ViewBenchmarkConfig:
    """Build the configuration of one benchmark unit.

    ``pre`` units always use the default warp, so experiments that differ only in
    their ``post`` family share them.
    """
    instances = tuple(block)
    last = max(instances)
    whole = instances == tuple(range(1, last + 1))
    return ViewBenchmarkConfig(
        problem_class=PROBLEM_CLASS,
        dim=context.dim,
        max_functions=context.max_functions,
        max_instances=last,
        num_trials=context.num_trials,
        budget=budget,
        base_seed=context.seed,
        scale=context.scale,
        algorithms=tuple(algorithms),
        transforms=(view,),
        instances_by_transform=None if whole else {view: instances},
        tuning=binding.as_dict(),
        condition=condition,
        warp=DEFAULT_WARP if view == PRE else warp,
        n_jobs=context.n_jobs,
        save_every=context.save_every,
    )


def benchmark_table(config: ViewBenchmarkConfig) -> str:
    """Where the benchmark stage writes a unit's shard inside the unit."""
    name = (
        "performance_views"
        if config.condition is None
        else f"performance_views_{config.condition}"
    )
    stem = shard_stem(name, config.problem_class, config.dim)
    return f"performance/{stem}{TABLE_SUFFIX}"


def resolve_benchmark(
    store: StageStore,
    config: ViewBenchmarkConfig,
    binding: PortfolioBinding,
    provenance: Mapping[str, Any],
) -> tuple[StageArtifact, pd.DataFrame]:
    """Return one benchmark unit and its rows, running it if the store lacks it."""
    unit = StageUnit("view_performance", config.as_dict(), provenance=provenance)

    def compute(paths: Any, config_hash: str) -> None:
        run_view_benchmark_stage(config, paths, config_hash, binding)

    artifact = store.resolve(unit, compute)
    return artifact, artifact.read_table(benchmark_table(config))


def superiority_settings(
    context: ExperimentContext, pairs: Sequence[AlgorithmPair], budget: int
) -> dict[str, Any]:
    """What a superiority unit is computed under, besides the benchmark it reads."""
    return {
        "stage": "view_superiority",
        "dim": context.dim,
        "pairs": [pair.slug for pair in pairs],
        "budget": budget,
    }


def superiority_table(dim: int) -> str:
    """Where the superiority stage writes a unit's table inside the unit."""
    stem = shard_stem("view_superiority", PROBLEM_CLASS, dim)
    return f"superiority/{stem}{TABLE_SUFFIX}"


def resolve_view_superiorities(
    store: StageStore,
    context: ExperimentContext,
    performance: tuple[StageArtifact, pd.DataFrame],
    budget: int,
    pairs: Sequence[AlgorithmPair] = FOCUS_PAIRS,
) -> tuple[StageArtifact, pd.DataFrame]:
    """Return the superiorities of one benchmark unit, computing them if absent."""
    source, rows = performance
    unit = StageUnit(
        "view_superiority",
        superiority_settings(context, pairs, budget),
        inputs={"performance": source.fingerprint},
    )

    def compute(paths: Any, config_hash: str) -> None:
        run_view_superiority_stage(
            rows,
            PROBLEM_CLASS,
            context.dim,
            paths,
            context.scale,
            config_hash,
            tuple(pairs),
            budget=budget,
        )

    artifact = store.resolve(unit, compute)
    return artifact, artifact.read_table(superiority_table(context.dim))


def assemble(tables: Iterable[pd.DataFrame], schema: TableSchema | str) -> pd.DataFrame:
    """Join the tables of several units in the order one consolidated shard has."""
    resolved = SCHEMAS[schema] if isinstance(schema, str) else schema
    joined = pd.concat(list(tables), ignore_index=True)
    key = [column for column in resolved.key_columns if column in joined.columns]
    return joined.sort_values(key, kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class PairUnit:
    """One resolved unit of the pair benchmark, with its superiorities."""

    benchmark: StageArtifact
    performance: pd.DataFrame
    superiority: StageArtifact
    superiorities: pd.DataFrame


def resolve_pair_unit(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    *,
    view: str,
    block: Iterable[int],
    step: str,
    warp: WarpConfig = DEFAULT_WARP,
) -> PairUnit:
    """Resolve one unit of the focus pairs' benchmark, and its superiorities."""
    config = benchmark_unit(
        context,
        algorithms=MEMBERS,
        binding=binding,
        view=view,
        block=block,
        budget=context.search_budget,
        warp=warp,
    )
    benchmark, performance = resolve_benchmark(
        store, config, binding, binding_provenance(tuning, step)
    )
    superiority, superiorities = resolve_view_superiorities(
        store, context, (benchmark, performance), context.search_budget
    )
    return PairUnit(benchmark, performance, superiority, superiorities)
