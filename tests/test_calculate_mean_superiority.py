"""Instance-wise mean superiorities.

The target is the normalised expected rank: the Vargha-Delaney A for two members,
blind to the objective's scale, and undefined where a member did not run.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from aas_shortcut.calculate_mean_superiority import (
    compute_mean_superiorities,
    mean_superiorities,
    run_view_mean_superiority_stage,
)
from aas_shortcut.calculate_superiority import vargha_delaney
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.define_table_schema import (
    VIEW_MEAN_SUPERIORITY_SCHEMA,
    validate_table,
)
from tests.check_stage_artifacts import check_mean_superiority
from tests.conftest import MEAN_SUPERIORITY_SCHEMA


class TestMeanSuperiorities:
    def test_the_best_member_scores_one_and_the_worst_zero(self):
        scores = mean_superiorities([np.array([3.0]), np.array([1.0]), np.array([2.0])])
        assert list(scores) == [0.0, 1.0, 0.5]

    def test_ties_take_the_average_rank(self):
        scores = mean_superiorities([np.array([1.0]), np.array([1.0]), np.array([2.0])])
        assert list(scores) == [0.75, 0.75, 0.0]

    def test_two_members_reduce_to_the_vargha_delaney_a(self):
        rng = np.random.default_rng(0)
        first, second = rng.normal(size=31), rng.normal(size=29)
        scores = mean_superiorities([first, second])
        assert scores[0] == pytest.approx(vargha_delaney(first, second))
        assert scores[1] == pytest.approx(1.0 - vargha_delaney(first, second))

    def test_it_is_the_normalised_expected_rank_over_every_combination(self):
        """J = E[(k - R) / (k - 1)], one run of each member drawn uniformly."""
        rng = np.random.default_rng(1)
        samples = [rng.integers(0, 4, size=size).astype(float) for size in (3, 4, 2)]
        k = len(samples)
        normalised = [
            (k - rankdata(runs, method="average")) / (k - 1)
            for runs in itertools.product(*samples)
        ]
        expected = np.mean(normalised, axis=0)
        np.testing.assert_allclose(mean_superiorities(samples), expected)

    def test_one_member_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            mean_superiorities([np.array([1.0])])


class TestComputeMeanSuperiorities:
    def test_every_instance_and_algorithm_is_present(self, performance):
        table = compute_mean_superiorities(performance)
        instances = performance["instance_id"].nunique()
        assert set(table["algorithm"]) == set(ALGORITHMS)
        assert len(table) == instances * len(ALGORITHMS)

    def test_scores_sum_to_half_the_portfolio(self, performance):
        """Every pair hands out one win between them, so the k scores sum to k/2."""
        table = compute_mean_superiorities(performance)
        totals = table.groupby("instance_id", observed=True, sort=True)[
            "mean_superiority"
        ].sum()
        assert np.allclose(totals.to_numpy(), len(ALGORITHMS) / 2)

    def test_scores_stay_within_the_unit_interval(self, performance):
        table = compute_mean_superiorities(performance)
        assert table["mean_superiority"].min() >= 0.0
        assert table["mean_superiority"].max() <= 1.0

    def test_invariant_under_a_monotonic_rescaling(self, performance):
        """The target is a rank statistic: scaling ``best_y`` cannot move it."""
        rescaled = performance.assign(
            best_y=np.log1p(performance["best_y"] - performance["best_y"].min() + 1.0)
        )
        pd.testing.assert_frame_equal(
            compute_mean_superiorities(performance),
            compute_mean_superiorities(rescaled),
        )

    def test_an_instance_missing_an_algorithm_is_dropped(self, performance):
        partial = performance[
            ~(
                (performance["instance_id"] == 1)
                & (performance["algorithm"] == ALGORITHMS[0])
            )
        ]
        table = compute_mean_superiorities(partial)
        assert 1 not in set(table["instance_id"])
        assert set(table["algorithm"]) == set(ALGORITHMS)

    def test_the_written_table_satisfies_its_schema(self, mean_superiorities):
        assert validate_table(mean_superiorities, MEAN_SUPERIORITY_SCHEMA) == []
        assert check_mean_superiority(mean_superiorities) == []

    def test_an_incomplete_ranking_is_caught_by_validation(self, mean_superiorities):
        """A rescaled score passes the schema but fails the sum identity."""
        broken = mean_superiorities.copy()
        broken.loc[broken.index[0], "mean_superiority"] = 1.0
        broken.loc[broken.index[1], "mean_superiority"] = 1.0
        violations = check_mean_superiority(broken)
        assert any(violation.kind == "incomplete" for violation in violations)


class TestRunViewMeanSuperiorityStage:
    """The stage the two-view experiments score their shards with."""

    @staticmethod
    def views(performance: pd.DataFrame) -> pd.DataFrame:
        """The one-view fixture doubled into a shard that carries both views."""
        return pd.concat(
            [performance.assign(transform=view) for view in TRANSFORMS],
            ignore_index=True,
        )

    def test_each_view_is_ranked_among_its_own_runs(self, performance, run_paths):
        table = run_view_mean_superiority_stage(
            self.views(performance), "bbob", 2, run_paths, "debug", "hash"
        )
        assert set(table["transform"]) == set(TRANSFORMS)
        assert validate_table(table, VIEW_MEAN_SUPERIORITY_SCHEMA) == []

    def test_the_whole_portfolio_is_ranked_by_default(self, performance, run_paths):
        table = run_view_mean_superiority_stage(
            self.views(performance), "bbob", 2, run_paths, "debug", "hash"
        )
        assert set(table["algorithm"]) == set(ALGORITHMS)
