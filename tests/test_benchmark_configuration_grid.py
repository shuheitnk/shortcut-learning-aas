"""Tests for the grid measurement's bookkeeping, with the optimizer runs stubbed.

At 1,830,240 runs, the key property is that an interrupted run resumes where it
stopped.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.benchmark_configuration_grid import (
    GridBenchmarkConfig,
    _batches,
    _rows,
    run_grid_benchmark_stage,
)
from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm, config_id
from aas_shortcut.tune_algorithms import PairConditions

CONDITIONS = {
    200: PairConditions(
        dim=2, budget_factor=100, function_ids=(1, 2), instance_ids=(11,), n_trials=3
    ),
    300: PairConditions(
        dim=2, budget_factor=150, function_ids=(1, 2), instance_ids=(11,), n_trials=3
    ),
}


def make_entries(sizes: dict[str, int]) -> dict[str, TunableAlgorithm]:
    """A search space of the requested grid sizes."""
    return {
        name: TunableAlgorithm(
            name=name,
            slug=name.lower(),
            factory=cast(Any, lambda **params: lambda *a, **k: 0),
            defaults={"x": 0.0},
            grid={"x": [float(value) for value in range(size)]},
        )
        for name, size in sizes.items()
    }


@pytest.fixture
def stub(monkeypatch):
    """Replace the optimizer runs with a deterministic stand-in, and count them."""
    calls: list[int] = []

    def fake(entries, requests, c, label, n_jobs=1, cache=None):
        calls.append(len(requests))
        cache = {} if cache is None else cache
        for name, params in requests:
            cache[(name, config_id(params))] = {
                instance: np.arange(c.n_trials, dtype=np.float64) + float(params["x"])
                for instance in c.instances
            }
        return cache

    monkeypatch.setattr(
        "aas_shortcut.benchmark_configuration_grid.run_configurations", fake
    )
    return calls


class TestConfig:
    def test_the_expected_row_count_covers_every_budget(self):
        entries = make_entries({"A": 3, "B": 2})
        config = GridBenchmarkConfig(by_budget=CONDITIONS)
        # (3 + 2) configurations x 2 functions x 3 trials, at each of two budgets.
        assert config.expected_rows(entries) == 5 * 2 * 3 * 2

    def test_the_search_space_is_part_of_the_fingerprint(self):
        """A widened grid is a different measurement and must not be reused."""
        narrow = GridBenchmarkConfig(by_budget=CONDITIONS, theta={"A": {"grid": [1]}})
        wide = GridBenchmarkConfig(by_budget=CONDITIONS, theta={"A": {"grid": [1, 2]}})
        assert narrow.as_dict() != wide.as_dict()

    def test_parallelism_is_not_part_of_the_fingerprint(self):
        """The same runs on more cores are the same runs."""
        assert (
            GridBenchmarkConfig(by_budget=CONDITIONS, n_jobs=1).as_dict()
            == GridBenchmarkConfig(by_budget=CONDITIONS, n_jobs=26).as_dict()
        )

    def test_both_budgets_are_fingerprinted(self):
        one = GridBenchmarkConfig(by_budget={200: CONDITIONS[200]})
        assert one.as_dict() != GridBenchmarkConfig(by_budget=CONDITIONS).as_dict()


class TestBatches:
    def test_it_splits_into_checkpointing_units(self):
        assert [len(batch) for batch in _batches(cast(Any, list(range(10))), 4)] == [
            4,
            4,
            2,
        ]

    def test_an_exact_multiple_leaves_no_remainder(self):
        assert [len(batch) for batch in _batches(cast(Any, list(range(8))), 4)] == [
            4,
            4,
        ]

    def test_nothing_pending_is_no_batches(self):
        assert _batches([], 4) == []


class TestRows:
    def test_one_row_per_trial(self):
        rows = _rows(
            "A",
            {"x": 1.0},
            {(1, 11): np.array([0.5, 0.25])},
            CONDITIONS[200],
            {"scale": "debug", "config_hash": "h"},
        )
        assert [row["trial"] for row in rows] == [1, 2]
        assert [row["error"] for row in rows] == [0.5, 0.25]

    def test_the_budget_comes_from_the_conditions(self):
        rows = _rows("A", {"x": 1.0}, {(1, 11): np.array([0.0])}, CONDITIONS[300], {})
        assert rows[0]["budget"] == 300

    def test_provenance_is_stamped(self):
        rows = _rows(
            "A",
            {"x": 1.0},
            {(1, 11): np.array([0.0])},
            CONDITIONS[200],
            {"scale": "debug", "config_hash": "abc"},
        )
        assert rows[0]["scale"] == "debug"
        assert rows[0]["config_hash"] == "abc"


class TestStage:
    def test_it_writes_the_whole_grid(self, run_paths, stub):
        entries = make_entries({"A": 3, "B": 2})
        config = GridBenchmarkConfig(by_budget=CONDITIONS, save_every=2)
        table = run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")
        assert len(table) == config.expected_rows(entries)
        assert sorted(table["budget"].unique()) == [200, 300]

    def test_a_second_call_measures_nothing(self, run_paths, stub):
        """The completed shard is recognised as its own, not remeasured."""
        entries = make_entries({"A": 3, "B": 2})
        config = GridBenchmarkConfig(by_budget=CONDITIONS, save_every=2)
        run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")
        measured = sum(stub)
        run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")
        assert sum(stub) == measured

    def test_an_interrupted_run_resumes_rather_than_restarts(
        self, run_paths, stub, monkeypatch
    ):
        """An interrupted grid resumes without remeasuring finished runs."""
        entries = make_entries({"A": 3, "B": 2})
        config = GridBenchmarkConfig(by_budget=CONDITIONS, save_every=2)

        real = run_grid_benchmark_stage.__globals__["run_configurations"]
        state = {"calls": 0}

        def crash_after_two(*args, **kwargs):
            state["calls"] += 1
            if state["calls"] > 2:
                raise KeyboardInterrupt("interrupted")
            return real(*args, **kwargs)

        monkeypatch.setattr(
            "aas_shortcut.benchmark_configuration_grid.run_configurations",
            crash_after_two,
        )
        with pytest.raises(KeyboardInterrupt):
            run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")

        monkeypatch.setattr(
            "aas_shortcut.benchmark_configuration_grid.run_configurations", real
        )
        before = sum(stub)
        table = run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")
        assert len(table) == config.expected_rows(entries)
        # Four of the five configurations at the first budget were already on disk.
        assert sum(stub) - before < len(entries) * 3 * 2

    def test_every_configuration_and_function_is_present(self, run_paths, stub):
        entries = make_entries({"A": 3, "B": 2})
        config = GridBenchmarkConfig(by_budget=CONDITIONS, save_every=2)
        table = run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")
        counts = table.groupby(
            ["budget", "algorithm", "config_id"], observed=True
        ).size()
        assert set(counts) == {2 * 3}

    def test_the_table_can_be_scored(self, run_paths, stub):
        """The shard's only consumer is the reduction; it must accept it."""
        from aas_shortcut.rank_configuration_grid import pooled_mean_superiorities

        entries = make_entries({"A": 3, "B": 2})
        config = GridBenchmarkConfig(by_budget=CONDITIONS, save_every=2)
        table = run_grid_benchmark_stage(entries, config, run_paths, "debug", "h")
        scores = pooled_mean_superiorities(table)
        assert len(scores) == 5 * 2 * 2
        assert scores["mean_superiority"].between(0, 1).all()


class TestSchema:
    def test_the_written_table_conforms(self, run_paths, stub):
        from aas_shortcut.define_table_schema import GRID_ERROR_SCHEMA

        entries = make_entries({"A": 2, "B": 2})
        table = run_grid_benchmark_stage(
            entries, GridBenchmarkConfig(by_budget=CONDITIONS), run_paths, "debug", "h"
        )
        assert list(table.columns) == list(GRID_ERROR_SCHEMA.columns)
        assert not pd.isna(table["error"]).any()
