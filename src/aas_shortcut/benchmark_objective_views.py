"""Benchmarking the portfolio on both views of the same objective."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aas_shortcut.benchmark_algorithms import run_job
from aas_shortcut.bind_portfolio_parameters import PortfolioBinding
from aas_shortcut.create_problem import resolve_problem_grid
from aas_shortcut.define_algorithm_portfolio import (
    ALGORITHMS,
    PORTFOLIO_REVISION,
    minimum_fe_budget,
)
from aas_shortcut.define_paths import TRANSFORMS, RunPaths
from aas_shortcut.define_table_schema import VIEW_PERFORMANCE_SCHEMA
from aas_shortcut.execute_parallel_jobs import run_jobs
from aas_shortcut.read_write_tables import PartWriter, plan_flush_size
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP, WarpConfig

LOGGER = logging.getLogger(__name__)

STAGE = "view_performance"

# Columns that name one run of this shard.
RUN_IDENTITY = ("function_id", "instance_id", "trial", "algorithm", "transform")


def solve_view(
    binding: PortfolioBinding,
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    trial: int,
    budget: int,
    algorithm: str,
    transform: str,
    warp: WarpConfig = DEFAULT_WARP,
) -> dict:
    """Run one member once under one view and return its row, tagged with the view."""
    (row,) = run_job(
        binding,
        base_seed,
        problem_class,
        function_id,
        instance_id,
        dim,
        trial,
        budget,
        (algorithm,),
        transform,
        warp,
    )
    return {**row, "transform": transform}


@dataclass(frozen=True)
class ViewBenchmarkConfig:
    """Grid, budget and binding of a two-view benchmark."""

    problem_class: str = "bbob"
    dim: int = 10
    max_functions: int = 24
    max_instances: int = 1
    num_trials: int = 31
    budget: int = 10_000
    base_seed: int = 42
    scale: str = "full"
    algorithms: tuple[str, ...] = field(default_factory=lambda: ALGORITHMS)
    transforms: tuple[str, ...] = field(default_factory=lambda: TRANSFORMS)
    instances_by_transform: Mapping[str, Sequence[int]] | None = None
    tuning: Mapping[str, Any] | None = None
    # Portfolio condition, which names the shard. ``None`` is the unnamed shard of a
    # run with one condition; Experiments 3 and 5 need ``selection`` and ``sbs``.
    condition: str | None = None
    # The transformation of the ``post`` view.
    warp: WarpConfig = DEFAULT_WARP
    n_jobs: int = 1
    save_every: int = 25

    def grid(self) -> tuple[range, range]:
        """Return the ``(functions, instances)`` ranges to sweep."""
        return resolve_problem_grid(
            self.problem_class, self.max_functions, self.max_instances
        )

    def instances_of(self, transform: str) -> tuple[int, ...]:
        """Return the instances one view is run over."""
        _, instances = self.grid()
        if self.instances_by_transform is None:
            return tuple(instances)
        return tuple(self.instances_by_transform.get(transform, tuple(instances)))

    def jobs(self) -> list[tuple[int, int, int, str, str]]:
        """Return every ``(function, instance, trial, algorithm, view)``, in order."""
        functions, instances = self.grid()
        covered = {
            transform: set(self.instances_of(transform))
            for transform in self.transforms
        }
        return [
            (function_id, instance_id, trial, algorithm, transform)
            for function_id in functions
            for instance_id in instances
            for trial in range(1, self.num_trials + 1)
            for algorithm in self.algorithms
            for transform in self.transforms
            if instance_id in covered[transform]
        ]

    def expected_rows(self) -> int:
        """Rows a complete shard must contain."""
        return len(self.jobs())

    def as_dict(self) -> dict:
        """Fingerprintable description of the configuration."""
        config: dict = {
            "stage": STAGE,
            "problem_class": self.problem_class,
            "dim": self.dim,
            "budget": self.budget,
            "num_trials": self.num_trials,
            "max_functions": self.max_functions,
            "max_instances": self.max_instances,
            "base_seed": self.base_seed,
            "algorithms": list(self.algorithms),
            "transforms": list(self.transforms),
            "portfolio_revision": PORTFOLIO_REVISION,
            # A different warp is a different `post` view, so it must not reuse
            # a shard.
            "transform": self.warp.identifier,
        }
        if self.instances_by_transform is not None:
            config["instances_by_transform"] = {
                transform: list(self.instances_of(transform))
                for transform in self.transforms
            }
        if self.tuning is not None:
            config["tuning"] = dict(self.tuning)
        if self.condition is not None:
            config["condition"] = self.condition
        return config

    def validate(self) -> None:
        """Reject a grid or budget the stage cannot run with."""
        floor = minimum_fe_budget(self.dim)
        if self.budget < floor:
            raise ValueError(
                f"a budget of {self.budget} is below the {floor} the portfolio "
                f"needs at {self.dim}D"
            )
        unknown = [name for name in self.algorithms if name not in ALGORITHMS]
        if unknown:
            raise ValueError(f"unknown portfolio member(s) {unknown}")
        views = [name for name in self.transforms if name not in TRANSFORMS]
        if views:
            raise ValueError(f"unknown objective view(s) {views}")
        if self.instances_by_transform is not None:
            _, instances = self.grid()
            for transform, ids in self.instances_by_transform.items():
                if transform not in self.transforms:
                    raise ValueError(
                        f"instances_by_transform names view {transform!r}, which is "
                        f"not one of {list(self.transforms)}"
                    )
                outside = sorted(set(ids) - set(instances))
                if outside:
                    raise ValueError(
                        f"view {transform!r} is restricted to instance(s) {outside}, "
                        f"which the grid {instances} does not cover"
                    )
        if self.num_trials < 1:
            raise ValueError(f"num_trials must be >= 1, got {self.num_trials}")


def _completed(existing: pd.DataFrame) -> set[tuple]:
    """Return the runs a previous invocation already finished."""
    if existing.empty or any(column not in existing.columns for column in RUN_IDENTITY):
        return set()
    return set(
        existing[list(RUN_IDENTITY)].itertuples(index=False, name=None)  # type: ignore[arg-type]
    )


def run_view_benchmark_stage(
    config: ViewBenchmarkConfig,
    run_paths: RunPaths,
    config_hash: str,
    binding: PortfolioBinding,
    *,
    batch_size: int | None = None,
) -> pd.DataFrame:
    """Benchmark every member under both views and persist the shard."""
    config.validate()
    shard = run_paths.view_performance(
        config.problem_class, config.dim, config.condition
    )
    writer = PartWriter(shard, VIEW_PERFORMANCE_SCHEMA, fingerprint=config_hash)

    cached = writer.reuse_or_reset()
    if cached is not None and len(cached) == config.expected_rows():
        LOGGER.info("%s: %s already complete", STAGE, shard.data)
        return cached

    done = _completed(writer.existing(columns=RUN_IDENTITY))
    pending = [job for job in config.jobs() if job not in done]
    LOGGER.info(
        "%s: %s %dD | %d run(s) pending, %d done | budget=%d | views=%s",
        STAGE,
        config.problem_class,
        config.dim,
        len(pending),
        len(done),
        config.budget,
        ", ".join(config.transforms),
    )
    LOGGER.info("%s", binding.describe())

    tasks = [
        (
            binding,
            config.base_seed,
            config.problem_class,
            function_id,
            instance_id,
            config.dim,
            trial,
            config.budget,
            algorithm,
            transform,
            config.warp,
        )
        for function_id, instance_id, trial, algorithm, transform in pending
    ]
    flush = plan_flush_size(len(pending), batch_size or config.save_every)

    buffer: list[dict] = []
    for finished, (_, row) in enumerate(
        run_jobs(STAGE, tasks, solve_view, n_jobs=config.n_jobs),
        start=1,
    ):
        buffer.append({**row, "scale": config.scale, "config_hash": config_hash})
        if finished % flush == 0:
            writer.append(buffer)
            buffer.clear()
    if buffer:
        writer.append(buffer)

    table = writer.consolidate()
    if len(table) != config.expected_rows():
        raise ValueError(
            f"{shard.data} holds {len(table)} row(s), expected {config.expected_rows()}"
        )
    return table
