"""The appendix's bootstrap: four units resampled, levels recomputed as reported."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.bootstrap_reported_levels import (
    FUNCTIONS_AND_INSTANCES,
    INSTANCES_WITHIN_FUNCTIONS,
    draw_samples,
    identity_draws,
    level,
    percentile_interval,
    problem_order,
    regression_cube,
    seed_values,
    selection_cube,
    weighted_mean,
    weighted_r2,
)
from aas_shortcut.predict_superiority import r2_score

FUNCTIONS, INSTANCES, SEEDS, REPS = 4, 3, 5, 3


def _problems() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"function_id": f, "instance_id": i}
            for f in range(1, FUNCTIONS + 1)
            for i in range(1, INSTANCES + 1)
        ]
    )


@pytest.fixture
def draws():
    return draw_samples(_problems(), SEEDS, REPS, n_samples=50, base_seed=42)


class TestDraws:
    def test_every_sample_draws_as_many_problems_seeds_and_deployments(self, draws):
        assert (draws.weights.sum(axis=1) == FUNCTIONS * INSTANCES).all()
        assert draws.seeds.shape == draws.other_seeds.shape == (50, SEEDS)
        assert draws.reps.shape == (50, REPS)

    def test_the_draws_are_fixed_by_the_base_seed(self, draws):
        again = draw_samples(_problems(), SEEDS, REPS, n_samples=50, base_seed=42)
        other = draw_samples(_problems(), SEEDS, REPS, n_samples=50, base_seed=43)
        assert np.array_equal(draws.weights, again.weights)
        assert not np.array_equal(draws.weights, other.weights)

    def test_instances_are_drawn_within_the_drawn_functions(self):
        drawn = draw_samples(_problems(), SEEDS, REPS, n_samples=500, base_seed=1)
        per_function = drawn.weights.reshape(500, FUNCTIONS, INSTANCES).sum(axis=2)
        # A function is drawn whole-number times, each time with INSTANCES draws.
        assert (per_function % INSTANCES == 0).all()
        assert (per_function == 0).any()

    def test_within_functions_every_function_is_kept_once(self):
        drawn = draw_samples(
            _problems(),
            SEEDS,
            REPS,
            n_samples=200,
            base_seed=1,
            resampling=INSTANCES_WITHIN_FUNCTIONS,
        )
        per_function = drawn.weights.reshape(200, FUNCTIONS, INSTANCES).sum(axis=2)
        assert (per_function == INSTANCES).all()
        assert (drawn.weights != 1.0).any()

    def test_the_main_version_s_draws_do_not_depend_on_the_other(self):
        main = draw_samples(_problems(), SEEDS, REPS, n_samples=20, base_seed=42)
        named = draw_samples(
            _problems(),
            SEEDS,
            REPS,
            n_samples=20,
            base_seed=42,
            resampling=FUNCTIONS_AND_INSTANCES,
        )
        assert np.array_equal(main.weights, named.weights)

    def test_functions_of_unequal_size_are_refused(self):
        problems = _problems().iloc[1:]
        with pytest.raises(ValueError, match="as many instances"):
            draw_samples(problems, SEEDS, REPS, n_samples=2)

    def test_the_problems_are_laid_out_by_function_then_instance(self):
        order = problem_order(_problems().sample(frac=1.0, random_state=0))
        assert list(order.itertuples(index=False, name=None))[:2] == [(1, 1), (1, 2)]


class TestWeighted:
    def test_r2_is_r2_score_on_the_problems_repeated_as_drawn(self, draws):
        rng = np.random.default_rng(0)
        observed = rng.random(FUNCTIONS * INSTANCES)
        predicted = rng.random((SEEDS, REPS, FUNCTIONS * INSTANCES))
        got = weighted_r2(observed, predicted, draws.weights)
        for sample in (0, 7):
            repeated = np.repeat(
                np.arange(observed.size), draws.weights[sample].astype(int)
            )
            for seed, rep in ((0, 0), (3, 2)):
                expected = r2_score(observed[repeated], predicted[seed, rep][repeated])
                assert got[sample, seed, rep] == pytest.approx(expected, abs=1e-12)

    def test_a_target_with_no_spread_scores_zero(self):
        observed = np.array([0.5, 0.5, 0.9])
        weights = np.array([[2.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        got = weighted_r2(observed, np.zeros((1, 1, 3)), weights)
        assert got[0, 0, 0] == 0.0
        assert got[1, 0, 0] == pytest.approx(r2_score(observed, np.zeros(3)))

    def test_the_mean_is_weighted_by_the_draws(self):
        values = np.array([[[0.0, 1.0, 2.0]]])
        got = weighted_mean(values, np.array([[3.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        assert got[:, 0, 0].tolist() == [0.0, 1.0]


class TestReduction:
    def test_the_data_itself_gives_the_reported_reduction(self):
        rng = np.random.default_rng(1)
        families = [rng.random((1, SEEDS, REPS)) for _ in range(2)]
        point = identity_draws(SEEDS, REPS, 1)
        expected = np.median(
            np.mean([np.median(f[0], axis=1) for f in families], axis=0)
        )
        assert level(seed_values(families, point), point.seeds)[0] == expected

    def test_a_drawn_deployment_counts_as_often_as_drawn(self):
        values = np.array([[[0.0, 1.0, 5.0]]])
        draws = identity_draws(1, 3, 1)
        draws = type(draws)(
            draws.seeds, draws.seeds, np.array([[2, 2, 0]]), draws.weights
        )
        assert seed_values([values], draws)[0, 0] == 5.0

    def test_the_interval_is_the_central_percentiles(self):
        low, high = percentile_interval(np.arange(101, dtype=float), 0.9)
        assert (low, high) == pytest.approx((5.0, 95.0))


class TestCubes:
    def _rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "seed": s,
                    "rep": r,
                    "function_id": f,
                    "instance_id": 1,
                    "observed": float(f),
                    "predicted": float(10 * s + r + f),
                    "superiority_vs_sbs": float(s + r),
                }
                for s in (42, 43)
                for r in (1, 2)
                for f in (2, 1)
            ]
        )

    def test_rows_are_laid_out_by_seed_deployment_and_problem(self):
        order = problem_order(self._rows())
        observed, predicted = regression_cube(self._rows(), order, (42, 43), (1, 2))
        assert observed.tolist() == [1.0, 2.0]
        assert predicted[1, 0].tolist() == [432.0, 433.0]
        assert selection_cube(self._rows(), order, (42, 43), (1, 2))[1, 1, 0] == 45.0

    def test_a_short_cell_is_refused(self):
        rows = self._rows().iloc[1:]
        with pytest.raises(ValueError, match="expected"):
            regression_cube(rows, problem_order(self._rows()), (42, 43), (1, 2))

    def test_a_target_that_moves_between_deployments_is_refused(self):
        rows = self._rows()
        rows.loc[0, "observed"] = 9.0
        with pytest.raises(ValueError, match="target differs"):
            regression_cube(rows, problem_order(rows), (42, 43), (1, 2))
