"""Running every configuration of every member over the tuning problems."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from aas_shortcut.define_algorithm_portfolio import PORTFOLIO_REVISION
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_table_schema import GRID_ERROR_SCHEMA
from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm, config_id
from aas_shortcut.read_write_tables import PartWriter
from aas_shortcut.tune_algorithms import PairConditions, run_configurations

LOGGER = logging.getLogger(__name__)

STAGE = "grid_errors"


@dataclass(frozen=True)
class GridBenchmarkConfig:
    """Which configurations to run, on what, and at which budgets."""

    by_budget: Mapping[int, PairConditions]
    n_jobs: int = 1
    save_every: int = 25
    # Search space description, fingerprinted so that a widened grid invalidates
    # the shard.
    theta: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description of the configuration."""
        return {
            "stage": STAGE,
            "conditions": [
                conditions.as_dict() for _, conditions in sorted(self.by_budget.items())
            ],
            "theta": dict(self.theta),
            # Names alone do not identify the portfolio: a member can keep its name
            # and still produce different trajectories.
            "portfolio_revision": PORTFOLIO_REVISION,
        }

    def expected_rows(self, entries: Mapping[str, TunableAlgorithm]) -> int:
        """Rows a complete shard holds."""
        per_member = sum(len(entry.candidates()) for entry in entries.values())
        return sum(
            per_member * len(conditions.instances) * conditions.n_trials
            for conditions in self.by_budget.values()
        )


def _rows(
    algorithm: str,
    params: Mapping[str, Any],
    errors: Mapping[tuple[int, int], Any],
    conditions: PairConditions,
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Explode one configuration's runs into long-format rows."""
    identifier = config_id(params)
    return [
        {
            "budget": conditions.budget,
            "algorithm": algorithm,
            "config_id": identifier,
            "function_id": function_id,
            "instance_id": instance_id,
            "trial": trial,
            "error": float(value),
            **provenance,
        }
        for (function_id, instance_id), values in errors.items()
        for trial, value in enumerate(values, start=1)
    ]


def _batches(
    requests: Sequence[tuple[str, Mapping[str, Any]]], size: int
) -> list[Sequence[tuple[str, Mapping[str, Any]]]]:
    """Split the pending configurations into checkpointing batches."""
    return [requests[start : start + size] for start in range(0, len(requests), size)]


def run_grid_benchmark_stage(
    entries: Mapping[str, TunableAlgorithm],
    config: GridBenchmarkConfig,
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
) -> pd.DataFrame:
    """Run the whole grid at every budget and persist the errors."""
    provenance = {"scale": scale, "config_hash": config_hash}
    shard = run_paths.metric(STAGE)
    writer = PartWriter(shard, GRID_ERROR_SCHEMA, fingerprint=config_hash)
    cached = writer.reuse_or_reset()
    expected = config.expected_rows(entries)
    if cached is not None and len(cached) == expected:
        LOGGER.info("%s: %s already complete", STAGE, shard.data)
        return cached

    done = writer.existing(columns=["budget", "algorithm", "config_id"])
    measured = (
        {
            (int(budget), str(algorithm), str(identifier))
            for budget, algorithm, identifier in done.itertuples(index=False)
        }
        if not done.empty
        else set()
    )

    for budget, conditions in sorted(config.by_budget.items()):
        requests = [
            (name, params)
            for name, entry in sorted(entries.items())
            for params in entry.candidates()
            if (budget, name, config_id(params)) not in measured
        ]
        total = sum(len(entry.candidates()) for entry in entries.values())
        if not requests:
            LOGGER.info("%s: budget %d already measured", STAGE, budget)
            continue
        LOGGER.info(
            "%s: budget %d, %d of %d configuration(s) to measure, %d run(s) each",
            STAGE,
            budget,
            len(requests),
            total,
            len(conditions.instances) * conditions.n_trials,
        )
        for index, batch in enumerate(_batches(requests, config.save_every), start=1):
            # A fresh cache per batch: the whole grid's errors at once would be the
            # shard held twice in memory, and the batch is what gets checkpointed.
            cache = run_configurations(
                entries,
                batch,
                conditions,
                f"{STAGE} (budget {budget}, batch {index})",
                config.n_jobs,
            )
            rows = [
                row
                for name, params in batch
                for row in _rows(
                    name,
                    params,
                    cache[(name, config_id(params))],
                    conditions,
                    provenance,
                )
            ]
            writer.append(rows)
            LOGGER.info(
                "%s: budget %d, checkpoint %d/%d",
                STAGE,
                budget,
                index,
                len(_batches(requests, config.save_every)),
            )

    written = writer.consolidate()
    LOGGER.info("%s: wrote %s (%d rows)", STAGE, shard.data, len(written))
    return written
