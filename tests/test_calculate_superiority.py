"""Pairwise superiorities via the Vargha-Delaney statistic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.calculate_superiority import (
    VIEW_GROUP_COLUMNS,
    compute_superiorities,
    run_view_superiority_stage,
    vargha_delaney,
    views_of,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS, PAIRS
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.define_table_schema import VIEW_SUPERIORITY_SCHEMA, validate_table


class TestVarghaDelaney:
    def test_a_strictly_better_sample_wins_outright(self):
        """Smaller is better: the problems are minimised."""
        assert vargha_delaney(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == 1.0

    def test_a_strictly_worse_sample_loses_outright(self):
        assert vargha_delaney(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 0.0

    def test_identical_samples_tie_at_one_half(self):
        assert vargha_delaney(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5

    def test_ties_count_as_half(self):
        assert vargha_delaney(np.array([1.0, 3.0]), np.array([1.0, 1.0])) == 0.25

    def test_result_is_always_a_probability(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.normal(size=31)
            b = rng.normal(size=31)
            assert 0.0 <= vargha_delaney(a, b) <= 1.0

    def test_is_antisymmetric(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=31), rng.normal(size=31)
        assert vargha_delaney(a, b) + vargha_delaney(b, a) == pytest.approx(1.0)

    def test_is_invariant_under_a_monotone_rescaling(self):
        """A12 depends only on ranks, so a monotone rescaling leaves it unchanged."""
        rng = np.random.default_rng(2)
        a, b = np.abs(rng.normal(size=31)), np.abs(rng.normal(size=31))
        assert vargha_delaney(a, b) == vargha_delaney(np.log1p(a), np.log1p(b))

    def test_empty_sample_is_rejected(self):
        with pytest.raises(ValueError, match="empty sample"):
            vargha_delaney(np.array([]), np.array([1.0]))


class TestComputeSuperiorities:
    def test_one_row_per_instance_and_pair(self, performance):
        table = compute_superiorities(performance)
        n_instances = performance["instance_id"].nunique()
        assert len(table) == n_instances * len(PAIRS)

    def test_pair_types_are_carried(self, performance):
        table = compute_superiorities(performance)
        assert set(table["pair_type"]) == {pair.pair_type for pair in PAIRS}


def _both_views(performance):
    """Duplicate a performance table under both views of the objective.

    Both views hold the same ``best_y``, as a comparison-based pair produces under a
    strictly increasing warp, so their superiorities must agree exactly.
    """
    return pd.concat(
        [performance.assign(transform=transform) for transform in TRANSFORMS],
        ignore_index=True,
    )


class TestPerView:
    def test_each_view_gets_its_own_superiority(self, performance):
        table = compute_superiorities(
            _both_views(performance), PAIRS, VIEW_GROUP_COLUMNS
        )
        n_instances = performance["instance_id"].nunique()
        assert len(table) == n_instances * len(PAIRS) * len(TRANSFORMS)
        assert set(table["transform"]) == set(TRANSFORMS)

    def test_grouping_by_the_view_does_not_pool_the_two(self, performance):
        """Without the view in the grouping the two views' trials become one sample."""
        pooled = compute_superiorities(_both_views(performance), PAIRS)
        assert len(pooled) == performance["instance_id"].nunique() * len(PAIRS)

    def test_identical_trajectories_give_identical_rates(self, performance):
        table = compute_superiorities(
            _both_views(performance), PAIRS, VIEW_GROUP_COLUMNS
        )
        views = views_of(table, PAIRS[0].slug)
        assert (views["pre"] == views["post"]).all()

    def test_the_stage_writes_a_conforming_shard(self, performance, run_paths):
        table = run_view_superiority_stage(
            _both_views(performance),
            "bbob",
            2,
            run_paths,
            "debug",
            "0123456789abcdef",
            FOCUS_PAIRS,
        )
        assert run_paths.view_superiority("bbob", 2).exists()
        assert validate_table(table, VIEW_SUPERIORITY_SCHEMA) == []
