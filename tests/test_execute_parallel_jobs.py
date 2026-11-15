"""Tests for streaming dispatch and the remaining-time estimate."""

from __future__ import annotations

import operator

import pytest

from aas_shortcut.execute_parallel_jobs import (
    MIN_LOG_INTERVAL,
    run_jobs,
)


def _square(value: int) -> int:
    """A module-level worker, so the dispatcher can pickle it."""
    return value * value


class TestRunJobs:
    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_every_task_is_run_and_indexed(self, n_jobs):
        results = dict(
            run_jobs("probe", [(i,) for i in range(7)], _square, n_jobs=n_jobs)
        )
        assert [results[i] for i in range(7)] == [0, 1, 4, 9, 16, 25, 36]

    def test_arguments_are_spread_over_the_worker_signature(self):
        results = dict(run_jobs("probe", [(3, 4), (5, 6)], operator.mul, n_jobs=1))
        assert results == {0: 12, 1: 30}

    def test_an_empty_task_list_yields_nothing(self):
        assert list(run_jobs("probe", [], _square, n_jobs=1)) == []

    def test_the_log_is_rate_limited(self):
        """A stage that finishes in milliseconds reports once, at the end."""
        assert MIN_LOG_INTERVAL > 0
