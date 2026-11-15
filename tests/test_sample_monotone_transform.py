"""The per-instance random strictly increasing objective transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from aas_shortcut.create_problem import create_problem
from aas_shortcut.derive_seeds import SEED_MODULUS
from aas_shortcut.sample_monotone_transform import (
    AUGMENTATION_DOMAIN,
    DEFAULT_WARP,
    HELD_OUT_DOMAIN,
    LINEAR_INTERPOLATION,
    MAX_ESTIMATE_SAMPLE_SIZE,
    MINIMUM_KNOTS,
    SPLINE_KNOTS,
    WARP_FAMILIES,
    WARP_ID,
    MonotoneTransform,
    WarpConfig,
    _sample_warp,
    estimate_maximum,
    maximum_seed,
    sample_monotone_transform,
    warp_seed,
    warp_strength,
    warp_strengths,
)

COORDINATES = (0, "bbob", 8, 1, 2)


def draw(seed: int, **settings):
    """Draw one warp on ``[0, 1]``, defaulting to the study's own setting."""
    return _sample_warp(np.random.default_rng(seed), WarpConfig(**settings))


@pytest.fixture(scope="module")
def problem():
    return create_problem(function_id=8, instance_id=1, dim=2)


@pytest.fixture(scope="module")
def transform(problem):
    return sample_monotone_transform(problem, *COORDINATES)


class TestWarp:
    def test_is_strictly_increasing(self):
        grid, values, _ = draw(0)
        assert np.all(np.diff(values) > 0.0)
        assert grid[0] == 0.0 and grid[-1] == 1.0

    def test_pins_both_endpoints(self):
        for seed in range(20):
            _, values, _ = draw(seed)
            assert values[0] == 0.0
            assert values[-1] == 1.0

    def test_shape_varies_between_draws(self):
        _, first, _ = draw(1)
        _, second, _ = draw(2)
        assert not np.allclose(first, second)


class TestFamilies:
    """Every family has to be a warp, and each has to be its own kind of one."""

    @pytest.mark.parametrize("family", WARP_FAMILIES)
    def test_is_strictly_increasing_with_both_endpoints_pinned(self, family):
        for seed in range(10):
            grid, values, _ = draw(seed, family=family)
            assert np.all(np.diff(grid) > 0.0)
            assert np.all(np.diff(values) > 0.0)
            assert (grid[0], grid[-1]) == (0.0, 1.0)
            assert (values[0], values[-1]) == (0.0, 1.0)

    def test_the_piecewise_family_is_read_without_smoothing(self):
        """Its corners are the property it is drawn for."""
        _, _, interpolation = draw(0, family="piecewise-linear")
        assert interpolation == LINEAR_INTERPOLATION

    def test_the_power_family_reaches_slopes_the_spline_family_cannot(self):
        """Unbounded derivative at the optimum: what makes it a held-out family."""
        steepest = {
            family: max(
                np.max(np.diff(values) / np.diff(grid))
                for grid, values, _ in (
                    draw(seed, family=family, alpha=2.0) for seed in range(20)
                )
            )
            for family in ("spline", "power")
        }
        assert steepest["power"] > 10 * steepest["spline"]

    def test_families_draw_different_shapes(self):
        """Otherwise a held-out family would not be holding anything out.

        Compared over draws rather than on one seed: any family can happen to
        draw something near the identity, so two of them agreeing once says
        nothing about whether they are the same family.
        """
        common = np.linspace(0.0, 1.0, 101)
        shapes = {
            family: np.array(
                [
                    np.interp(common, grid, values)
                    for grid, values, _ in (
                        draw(seed, family=family) for seed in range(20)
                    )
                ]
            )
            for family in WARP_FAMILIES
        }
        for i, first in enumerate(WARP_FAMILIES):
            for second in WARP_FAMILIES[i + 1 :]:
                separation = np.mean(
                    np.max(np.abs(shapes[first] - shapes[second]), axis=1)
                )
                assert separation > 0.01, f"{first} and {second} draw alike"


class TestWarpConfig:
    def test_the_default_keeps_the_identifier_artifacts_were_written_under(self):
        """Changing it would invalidate every result the study has measured."""
        assert DEFAULT_WARP.identifier == WARP_ID

    @pytest.mark.parametrize(
        "settings",
        [
            {"alpha": 0.5},
            {"family": "power"},
            {"domain": HELD_OUT_DOMAIN},
            {"knots": (3, 5)},
        ],
    )
    def test_any_departure_from_the_default_changes_the_identifier(self, settings):
        """This is what stops a sweep being served from the previous setting."""
        assert WarpConfig(**settings).identifier != WARP_ID

    @pytest.mark.parametrize(
        "settings",
        [
            {"family": "quadratic"},
            {"domain": "warptest"},
            {"alpha": -1.0},
            # Fewer control points than a cubic spline is defined for.
            {"knots": (MINIMUM_KNOTS - 1, 5)},
            # A range whose ends are not apart draws from nothing.
            {"knots": (5, 5)},
            {"knots": (5,)},
            # Only the spline family reads the count, so this would be dropped.
            {"family": "power", "knots": (3, 5)},
        ],
    )
    def test_a_setting_no_draw_is_defined_for_is_refused(self, settings):
        with pytest.raises(ValueError):
            WarpConfig(**settings)


class TestControlPoints:
    """The axis Experiment 5 adds: the same construction, another knot count."""

    def test_the_default_range_is_the_one_the_study_trains_on(self):
        assert DEFAULT_WARP.knots == SPLINE_KNOTS

    def test_the_identifier_only_names_the_count_once_it_moves(self):
        """The default identifier omits the knot count, keeping fingerprints valid."""
        held_out = WarpConfig(domain=HELD_OUT_DOMAIN)
        assert "knots" not in held_out.identifier
        assert (
            "knots=3-5" in WarpConfig(domain=HELD_OUT_DOMAIN, knots=(3, 5)).identifier
        )

    def test_two_ranges_are_two_settings(self):
        assert (
            WarpConfig(knots=(3, 5)).identifier != WarpConfig(knots=(15, 21)).identifier
        )

    @pytest.mark.parametrize("knots", [(3, 5), (15, 21)])
    def test_a_moved_range_still_draws_a_strictly_increasing_map(self, knots):
        grid, values, _ = draw(0, knots=knots)
        assert np.all(np.diff(values) > 0.0)
        assert (values[0], values[-1]) == (0.0, 1.0)
        assert grid.size == values.size

    def test_a_sparse_draw_bends_in_fewer_places_than_a_dense_one(self):
        """What the two audit points are for: the same family, another shape."""

        def turning_points(knots):
            counts = []
            for seed in range(24):
                _, values, _ = draw(seed, knots=knots)
                slope = np.diff(values)
                counts.append(int(np.count_nonzero(np.diff(np.sign(np.diff(slope))))))
            return float(np.mean(counts))

        assert turning_points((3, 5)) < turning_points((15, 21))

    def test_the_count_changes_the_map_a_given_seed_draws(self):
        _, sparse, _ = draw(7, knots=(3, 5))
        _, dense, _ = draw(7, knots=(15, 21))
        assert not np.allclose(sparse, dense)


class TestSeedDomains:
    def test_the_default_domain_is_the_one_every_artifact_was_drawn_under(self):
        assert DEFAULT_WARP.domain == AUGMENTATION_DOMAIN
        assert warp_seed(*COORDINATES) == warp_seed(*COORDINATES, AUGMENTATION_DOMAIN)

    def test_the_held_out_domain_gives_the_same_instance_another_map(self):
        """An audit drawing the training map would report an optimistic score."""
        assert warp_seed(*COORDINATES, HELD_OUT_DOMAIN) != warp_seed(*COORDINATES)

    @pytest.mark.parametrize(
        "settings", [{"domain": "warpdiag"}, {"family": "log-tail"}]
    )
    def test_a_removed_domain_or_family_is_refused(self, settings):
        """Domains and families no paper experiment uses are refused."""
        with pytest.raises(ValueError):
            WarpConfig(**settings)

    def test_an_unknown_domain_is_refused(self):
        with pytest.raises(ValueError):
            warp_seed(*COORDINATES, "warpsomething")


class TestRepetitions:
    """Every experiment of the paper uses draw 0."""

    def test_the_first_repetition_is_what_every_artifact_was_drawn_under(self):
        """Draw 0 must leave the seed and the identifier untouched."""
        assert DEFAULT_WARP.draw == 0
        assert warp_seed(*COORDINATES, AUGMENTATION_DOMAIN, 0) == warp_seed(
            *COORDINATES
        )
        assert WarpConfig(draw=0).identifier == WARP_ID


class TestMaximumEstimate:
    def test_is_a_pure_function_of_the_seed(self, problem):
        assert estimate_maximum(problem, 3) == estimate_maximum(problem, 3)

    def test_different_seeds_give_different_estimates(self, problem):
        assert estimate_maximum(problem, 3) != estimate_maximum(problem, 4)

    def test_lies_above_the_optimum(self, problem):
        assert estimate_maximum(problem, 3) > float(problem.optimum.y)

    def test_reads_the_documented_number_of_points(self, problem):
        """The estimate is the sample maximum of that many uniform draws."""
        lb = np.asarray(problem.bounds.lb, dtype=np.float64)
        ub = np.asarray(problem.bounds.ub, dtype=np.float64)
        rng = np.random.default_rng(11)
        points = rng.uniform(lb, ub, size=(MAX_ESTIMATE_SAMPLE_SIZE, lb.size))
        expected = max(problem(point) for point in points)
        assert estimate_maximum(problem, 11) == expected


class TestSeeds:
    def test_lie_in_range(self):
        assert 0 <= warp_seed(*COORDINATES) < SEED_MODULUS
        assert 0 <= maximum_seed(*COORDINATES) < SEED_MODULUS

    def test_the_two_domains_are_separate(self):
        assert warp_seed(*COORDINATES) != maximum_seed(*COORDINATES)

    def test_every_coordinate_moves_the_warp(self):
        seeds = {
            warp_seed(0, "bbob", 8, 1, 2),
            warp_seed(1, "bbob", 8, 1, 2),
            warp_seed(0, "mabbob", 8, 1, 2),
            warp_seed(0, "bbob", 9, 1, 2),
            warp_seed(0, "bbob", 8, 2, 2),
            warp_seed(0, "bbob", 8, 1, 5),
        }
        assert len(seeds) == 6


class TestTransform:
    def test_maps_the_two_anchors_to_zero_and_one(self, transform):
        assert transform(transform.optimum_y) == pytest.approx(0.0, abs=1e-12)
        assert transform(transform.maximum_y) == pytest.approx(1.0, abs=1e-12)

    def test_is_strictly_increasing_across_the_join(self, transform):
        span = transform.maximum_y - transform.optimum_y
        y = transform.optimum_y + np.linspace(0.0, 3.0, 4001) * span
        assert np.all(np.diff(transform(y)) > 0.0)

    def test_is_the_identity_above_the_estimated_maximum(self, transform):
        span = transform.maximum_y - transform.optimum_y
        z = np.array([1.5, 2.0, 10.0, 1e6])
        assert transform(transform.optimum_y + z * span) == pytest.approx(z)

    def test_is_continuous_at_the_join(self, transform):
        span = transform.maximum_y - transform.optimum_y
        below = transform(transform.optimum_y + (1.0 - 1e-9) * span)
        above = transform(transform.optimum_y + (1.0 + 1e-9) * span)
        assert float(above) - float(below) == pytest.approx(0.0, abs=1e-7)

    def test_clamps_values_below_the_tabulated_optimum(self, transform):
        """Float noise can put a near-optimal evaluation under the optimum."""
        assert transform(transform.optimum_y - 1e-9) == pytest.approx(0.0, abs=1e-12)

    def test_preserves_the_ranking_of_any_sample(self, transform):
        span = transform.maximum_y - transform.optimum_y
        rng = np.random.default_rng(5)
        # Deliberately straddling the join: a third of these land above it.
        y = transform.optimum_y + rng.uniform(0.0, 1.5, 500) * span
        assert np.array_equal(
            rankdata(y, method="average"),
            rankdata(transform(y), method="average"),
        )

    def test_departs_from_the_identity(self, transform):
        assert transform.strength > 0.0

    def test_rejects_a_degenerate_range(self):
        grid = np.linspace(0.0, 1.0, 5)
        with pytest.raises(ValueError, match="must exceed"):
            MonotoneTransform(1.0, 1.0, grid, grid, seed=0)


class TestSampling:
    def test_is_a_pure_function_of_the_coordinates(self, problem, transform):
        again = sample_monotone_transform(problem, *COORDINATES)
        assert again.seed == transform.seed
        assert again.maximum_y == transform.maximum_y
        assert np.array_equal(again.values, transform.values)

    def test_different_instances_are_warped_differently(self, problem):
        first = sample_monotone_transform(problem, 0, "bbob", 8, 1, 2)
        second = sample_monotone_transform(problem, 0, "bbob", 8, 2, 2)
        assert first.seed != second.seed
        assert not np.allclose(first.values, second.values)

    def test_anchors_on_the_instance_optimum(self, problem, transform):
        assert transform.optimum_y == float(problem.optimum.y)


class TestStrength:
    """How far a drawn map departs from the identity, without the problem."""

    @pytest.mark.parametrize(
        "config",
        [DEFAULT_WARP, WarpConfig(domain=HELD_OUT_DOMAIN), WarpConfig(family="power")],
    )
    def test_is_the_drawn_transform_s_own(self, problem, config):
        drawn = sample_monotone_transform(problem, *COORDINATES, config)
        assert warp_strength(*COORDINATES, config) == drawn.strength

    def test_one_row_per_problem_in_problem_order(self):
        base_seed, problem_class, _, _, dim = COORDINATES
        problems = pd.DataFrame(
            {
                "problem_class": [problem_class] * 4,
                "function_id": [8, 1, 8, 1],
                "instance_id": [2, 1, 2, 3],
                "pair": ["a", "a", "b", "a"],
            }
        )
        table = warp_strengths(problems, base_seed, dim, DEFAULT_WARP)
        assert list(zip(table["function_id"], table["instance_id"], strict=True)) == [
            (1, 1),
            (1, 3),
            (8, 2),
        ]
        assert set(table["domain"]) == {AUGMENTATION_DOMAIN}
        assert table["strength"].iloc[2] == warp_strength(
            base_seed, problem_class, 8, 2, dim
        )
