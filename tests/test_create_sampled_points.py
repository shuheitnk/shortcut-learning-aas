"""Tests for the shared sample."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from aas_shortcut.create_sampled_points import (
    create_sampled_points,
    ela_inputs,
    latin_hypercube,
    normalize,
    sample_seed,
)
from aas_shortcut.derive_seeds import SEED_MODULUS, derive_seed
from aas_shortcut.sample_monotone_transform import transform_for_instance


class TestLatinHypercube:
    def test_shape_and_range(self):
        sample = latin_hypercube(50, 3, np.random.default_rng(0))
        assert sample.shape == (50, 3)
        assert np.all((sample > 0.0) & (sample < 1.0))

    def test_every_stratum_is_occupied_once(self):
        """The defining property of a Latin hypercube."""
        n = 40
        sample = latin_hypercube(n, 4, np.random.default_rng(1))
        for axis in range(4):
            strata = np.floor(sample[:, axis] * n).astype(int)
            assert sorted(strata) == list(range(n))

    def test_is_a_pure_function_of_the_generator(self):
        first = latin_hypercube(20, 2, np.random.default_rng(7))
        second = latin_hypercube(20, 2, np.random.default_rng(7))
        assert np.array_equal(first, second)

    def test_different_seeds_differ(self):
        first = latin_hypercube(20, 2, np.random.default_rng(7))
        second = latin_hypercube(20, 2, np.random.default_rng(8))
        assert not np.array_equal(first, second)

    def test_does_not_touch_the_global_rng(self):
        np.random.seed(3)
        before = np.random.random()
        np.random.seed(3)
        latin_hypercube(30, 2, np.random.default_rng(0))
        assert np.random.random() == before

    @pytest.mark.parametrize(("n", "dim"), [(0, 2), (5, 0), (-1, 2)])
    def test_rejects_degenerate_shapes(self, n, dim):
        with pytest.raises(ValueError):
            latin_hypercube(n, dim, np.random.default_rng(0))


class TestSeedDerivation:
    def test_seeds_stay_in_the_supported_range(self):
        for rep in range(50):
            seed = sample_seed(42, "bbob", 1, rep, 10, 1)
            assert 0 <= seed < SEED_MODULUS

    def test_distinct_coordinates_give_distinct_seeds(self):
        seeds = {
            sample_seed(42, "bbob", f, i, 2, r)
            for f in range(1, 6)
            for i in range(1, 6)
            for r in range(1, 6)
        }
        assert len(seeds) == 125

    def test_separator_cannot_be_forged(self):
        """Two different coordinate tuples must not collide."""
        assert derive_seed("a|b", "c") != derive_seed("a", "b|c")

    def test_unknown_problem_class_is_rejected(self):
        with pytest.raises(ValueError):
            sample_seed(42, "not-a-suite", 1, 1, 2, 1)


@pytest.fixture(scope="module")
def sample():
    """One real sample, reused across the assertions about it."""
    return create_sampled_points(42, "bbob", 1, 1, 2, 1, 100)


class TestSampledPoints:
    def test_shapes(self, sample):
        assert sample.X_unit.shape == (100, 2)
        assert sample.y_pre.shape == (100,)
        assert sample.y_post.shape == (100,)
        assert sample.sample_size == 100

    def test_points_are_in_the_unit_cube(self, sample):
        assert np.all((sample.X_unit >= 0.0) & (sample.X_unit <= 1.0))

    def test_post_is_the_instances_warp_applied_to_pre(self, sample):
        """The post view is the instance's own strictly increasing map."""
        warp = transform_for_instance(42, "bbob", 1, 1, 2)
        assert np.array_equal(sample.y_post, warp(sample.y_pre))

    def test_post_is_the_same_warp_in_every_repetition(self):
        """The warp is a property of the instance, not of the repetition."""
        first = create_sampled_points(42, "bbob", 1, 1, 2, 1, 100)
        second = create_sampled_points(42, "bbob", 1, 1, 2, 2, 100)
        warp = transform_for_instance(42, "bbob", 1, 1, 2)
        assert np.array_equal(first.y_post, warp(first.y_pre))
        assert np.array_equal(second.y_post, warp(second.y_pre))

    def test_objective_selects_the_transform(self, sample):
        assert np.array_equal(sample.objective("pre"), sample.y_pre)
        assert np.array_equal(sample.objective("post"), sample.y_post)
        with pytest.raises(ValueError):
            sample.objective("middle")

    def test_ranks_agree(self, sample):
        assert sample.ranks_agree()

    def test_the_same_sample_serves_all_four_conditions(self, sample):
        """One sample serves pre == post and value == rank alike."""
        matrices = [
            ela_inputs(sample, transform, encoding)[0]
            for transform in ("pre", "post")
            for encoding in ("value", "rank")
        ]
        for other in matrices[1:]:
            assert matrices[0].equals(other)

    def test_rank_encoding_is_identical_across_transforms(self, sample):
        _, pre = ela_inputs(sample, "pre", "rank")
        _, post = ela_inputs(sample, "post", "rank")
        assert np.array_equal(pre.to_numpy(), post.to_numpy())

    def test_value_encoding_differs_across_transforms(self, sample):
        _, pre = ela_inputs(sample, "pre", "value")
        _, post = ela_inputs(sample, "post", "value")
        assert not np.array_equal(pre.to_numpy(), post.to_numpy())

    def test_rank_encoding_is_the_ranking(self, sample):
        _, y = ela_inputs(sample, "pre", "rank")
        expected = normalize(rankdata(sample.y_pre, method="average"))
        assert np.allclose(y.to_numpy(), expected)

    def test_unknown_encoding_is_rejected(self, sample):
        with pytest.raises(ValueError):
            ela_inputs(sample, "pre", "quantile")


class TestNormalize:
    def test_maps_to_the_unit_interval(self):
        scaled = normalize(np.array([3.0, 5.0, 11.0]))
        assert scaled.min() == 0.0
        assert scaled.max() == 1.0

    def test_constant_input_becomes_zeros(self):
        """Otherwise a flat objective would divide by zero."""
        assert normalize(np.full(4, 2.5)).tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_is_affine_invariant(self):
        """The encodings can differ only through the non-linear part."""
        y = np.array([1.0, 4.0, 9.0, 16.0])
        assert np.allclose(normalize(y), normalize(3.0 * y + 7.0))
