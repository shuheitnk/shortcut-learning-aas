"""Tests for candidate tuning (``aas_shortcut.tune_algorithms``)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut import tune_algorithms
from aas_shortcut.benchmark_algorithms import trial_seed
from aas_shortcut.calculate_superiority import compute_superiorities
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_tunable_portfolio import TUNABLE
from aas_shortcut.optimizers.define_tunable_algorithm import config_id as config_id_of
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS
from aas_shortcut.tune_algorithms import (
    ESTIMATOR,
    PairConditions,
    comparisons_per_rate,
    floor_for,
    ks_distance,
    run_configurations,
    summarise,
    summarize_pair_tuning,
    superiorities,
)

# The sample a perfectly tuned pair would produce: the 24 uniform quantiles, which
# is the best any continuous sample of this size can do.
IDEAL = (np.arange(1, 25) - 0.5) / 24

# The pathology the tuning exists to remove: every instance decided outright, half
# each way.
SATURATED = np.array([0.0] * 12 + [1.0] * 12)


class TestKsDistance:
    def test_the_ideal_sample_scores_far_below_the_saturated_one(self):
        assert ks_distance(IDEAL) < ks_distance(SATURATED)

    def test_the_saturated_sample_has_a_mean_of_exactly_one_half(self):
        """The saturated sample has mean one half, so a mean objective accepts it."""
        assert SATURATED.mean() == 0.5

    def test_a_constant_sample_is_worse_than_a_spread_one(self):
        assert ks_distance(np.full(24, 0.5)) > ks_distance(IDEAL)

    def test_the_distance_ignores_the_order_of_the_sample(self):
        shuffled = np.random.default_rng(0).permutation(IDEAL)
        assert ks_distance(shuffled) == pytest.approx(ks_distance(IDEAL))

    def test_the_saturated_sample_has_the_expected_largest_gap(self):
        """The saturated sample's largest CDF gap is one half."""
        assert ks_distance(SATURATED) == pytest.approx(0.5)


class TestDiscretenessFloor:
    def test_the_floor_is_above_the_continuous_ideal(self):
        """The discreteness floor is at or above the continuous ideal."""
        assert floor_for(24, 9) >= ks_distance(IDEAL)

    def test_more_comparisons_lower_the_floor(self):
        """More comparisons per rate lower the floor."""
        assert floor_for(24, 31) < floor_for(24, 9)

    def test_the_floor_is_reported_alongside_every_score(self):
        summary = summarise(IDEAL, PairConditions(n_trials=9))
        assert summary["floor"] == floor_for(24, comparisons_per_rate(9))
        assert summary["score"] == ks_distance(IDEAL)

    def test_the_grid_is_the_estimator_s_own(self):
        """A rate takes one of ``2 T^2 + 1`` values, not ``2 T + 1``."""
        assert comparisons_per_rate(31) == 31 * 31
        assert summarise(IDEAL, PairConditions())["floor"] == floor_for(24, 961)

    def test_the_finer_grid_stops_binding_at_this_sample_size(self):
        """At 961 comparisons the floor is set by the 24 problems, not the grid."""
        assert floor_for(24, 961) == pytest.approx(0.5 / 24, abs=5e-4)
        assert floor_for(24, 961) < floor_for(24, 31)


class TestSuperiorities:
    def test_the_estimator_is_the_one_the_target_is_measured_with(self):
        """The tuning's superiorities equal `compute_superiorities` on the same runs."""
        rng = np.random.default_rng(0)
        first = {(f, 1): rng.normal(size=7) for f in (1, 2, 3)}
        second = {(f, 1): rng.normal(size=7) for f in (1, 2, 3)}
        pair = FOCUS_PAIRS[0]
        frame = pd.DataFrame(
            [
                {
                    "problem_class": "bbob",
                    "function_id": function_id,
                    "instance_id": instance_id,
                    "dimension": 10,
                    "algorithm": algorithm,
                    "best_y": value,
                }
                for algorithm, source in (
                    (pair.first, first),
                    (pair.second, second),
                )
                for (function_id, instance_id), values in source.items()
                for value in values
            ]
        )
        through_the_target = compute_superiorities(frame, pairs=(pair,)).sort_values(
            ["function_id", "instance_id"]
        )["superiority"]
        assert superiorities(first, second) == pytest.approx(
            through_the_target.to_numpy()
        )

    def test_every_run_is_compared_against_every_run(self):
        """Equal samples in different orders score one half."""
        first = {(1, 1): np.array([1.0, 2.0, 3.0])}
        second = {(1, 1): np.array([3.0, 1.0, 2.0])}
        assert superiorities(first, second)[0] == pytest.approx(0.5)
        assert sorted(first[1, 1]) == sorted(second[1, 1])

    def test_the_run_conditions_do_not_name_the_estimator(self):
        """The run conditions omit the estimator, so `grid_errors` is not redone."""
        assert "estimator" not in PairConditions().as_dict()

    def test_an_exact_tie_scores_half(self):
        first = {(1, 1): np.array([1.0, 1.0])}
        second = {(1, 1): np.array([1.0, 1.0])}
        assert superiorities(first, second)[0] == 0.5

    def test_one_rate_per_instance_in_sorted_order(self):
        first = {(1, 2): np.array([0.0]), (1, 1): np.array([1.0])}
        second = {(1, 2): np.array([1.0]), (1, 1): np.array([0.0])}
        # f1 i1 loses, f1 i2 wins, and the result is ordered by instance key.
        assert list(superiorities(first, second)) == [0.0, 1.0]


class TestSummary:
    def test_saturation_counts_only_the_exact_extremes(self):
        rates = np.array([0.0, 0.01, 0.5, 0.99, 1.0])
        assert summarise(rates, PairConditions())["n_saturated"] == 2

    def test_the_informative_band_excludes_the_edges(self):
        rates = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        assert summarise(rates, PairConditions())["n_informative"] == 1


class TestThePairTuningSummary:
    """Each pair's default and selected configuration, one row each."""

    @staticmethod
    def _table() -> pd.DataFrame:
        # Three candidates on two instances: the default scores worst, and the
        # selected one ties the third, which the grid lists after it.
        candidates = [
            ("default", True, False, 0.4, 5),
            ("chosen", False, True, 0.2, 1),
            ("tied", False, False, 0.2, 2),
        ]
        return pd.DataFrame(
            [
                {
                    "group": "G1",
                    "pair": "a__b",
                    "tuned": "B",
                    "fixed": "A",
                    "config_id": config,
                    "params": "{}",
                    "is_default": is_default,
                    "is_selected": is_selected,
                    "function_id": function_id,
                    "instance_id": 11,
                    "score": score,
                    "floor": 0.02,
                    "n_saturated": saturated,
                }
                for config, is_default, is_selected, score, saturated in candidates
                for function_id in (1, 2)
            ]
        )

    def test_the_default_and_the_selected_configuration(self):
        summary = summarize_pair_tuning(self._table()).set_index("configuration")
        assert list(summary.index) == ["default", "selected"]
        assert list(summary["config_id"]) == ["default", "chosen"]
        assert list(summary["score"]) == [0.4, 0.2]
        assert list(summary["n_saturated"]) == [5, 1]
        assert (summary["n_problems"] == 2).all()
        assert (summary["n_candidates"] == 3).all()
        assert (summary["n_at_minimum"] == 2).all()

    def test_a_pair_without_a_selection_is_rejected(self):
        table = self._table()
        table["is_selected"] = False
        with pytest.raises(ValueError, match="0 selected"):
            summarize_pair_tuning(table)


class TestPairConditions:
    def test_the_budget_follows_the_project_convention(self):
        assert PairConditions(dim=10, budget_factor=1000).budget == 10_000

    def test_the_defaults_are_the_tuning_set(self):
        """The defaults are 24 functions x 1 instance, 31 trials."""
        conditions = PairConditions()
        assert len(conditions.instances) == 24
        assert conditions.n_trials == 31

    def test_the_trials_are_numbered_as_the_benchmark_numbers_them(self):
        assert list(PairConditions(n_trials=5).trials) == [1, 2, 3, 4, 5]

    def test_a_run_takes_the_benchmark_s_seed(self):
        """A run takes the seed the benchmark gives the same member, problem, trial."""
        conditions = PairConditions(dim=10, instance_ids=(11,), base_seed=42)
        expected = trial_seed(42, "SHADE", PROBLEM_CLASS, 3, 11, 10, 1)
        assert conditions.seed("SHADE", 3, 11, 1) == expected

    def test_the_seeding_is_part_of_the_fingerprint(self):
        """The seeding scheme is part of the fingerprint."""
        assert PairConditions().as_dict()["seeds"] == "trial"


class TestRunSeeds:
    """The seed each tuning run is given."""

    CONDITIONS = PairConditions(
        dim=2, function_ids=(1, 2), instance_ids=(11,), n_trials=3, base_seed=42
    )

    def _seeds_run(self, monkeypatch, requests):
        """Run ``requests`` with a stub optimizer that returns its seed."""

        def seed_as_error(entry, params, function_id, instance_id, seed, c):
            return float(seed)

        monkeypatch.setattr(tune_algorithms, "_run_one", seed_as_error)
        entries = {name: TUNABLE[name] for name, _ in requests}
        return run_configurations(entries, requests, self.CONDITIONS, "test", 1)

    def test_every_run_takes_its_own_member_s_benchmark_seed(self, monkeypatch):
        requests = [("SHADE", TUNABLE["SHADE"].resolve())]
        requests.append(("L-SHADE", TUNABLE["L-SHADE"].resolve()))
        cache = self._seeds_run(monkeypatch, requests)
        c = self.CONDITIONS
        for name, params in requests:
            for (function_id, instance_id), seeds in cache[
                (name, config_id_of(params))
            ].items():
                expected = [c.seed(name, function_id, instance_id, t) for t in c.trials]
                assert list(seeds) == expected

    def test_every_candidate_of_a_member_meets_the_same_seeds(self, monkeypatch):
        """All candidates of a member share seeds (common random numbers)."""
        first, second = TUNABLE["SHADE"].candidates()[:2]
        cache = self._seeds_run(monkeypatch, [("SHADE", first), ("SHADE", second)])
        one = cache[("SHADE", config_id_of(first))]
        other = cache[("SHADE", config_id_of(second))]
        assert all((one[instance] == other[instance]).all() for instance in one)

    def test_the_two_members_meet_different_seeds(self, monkeypatch):
        requests = [("SHADE", TUNABLE["SHADE"].resolve())]
        requests.append(("L-SHADE", TUNABLE["L-SHADE"].resolve()))
        cache = self._seeds_run(monkeypatch, requests)
        shade = cache[("SHADE", config_id_of(requests[0][1]))]
        l_shade = cache[("L-SHADE", config_id_of(requests[1][1]))]
        assert not any((shade[i] == l_shade[i]).any() for i in shade)


# ---------------------------------------------------------------------------
# What the estimator selects, against the measured grid
# ---------------------------------------------------------------------------

# The paper's `grid_errors` unit keeps every candidate's per-trial errors, so each
# pair's choice can be checked without running an optimizer. It is found by its
# fingerprint in any stage store under outputs/; outputs are not in the repository,
# so these tests skip where it has not been produced (as on CI).
GRID_UNIT = "grid_errors/f5d081b19e4b7a71/metrics/grid_errors.parquet"
_OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
GRID_SHARD = next(
    iter(
        sorted(
            (
                *_OUTPUTS.glob(f"_stages/{GRID_UNIT}"),
                *_OUTPUTS.glob(f"*/_stages/{GRID_UNIT}"),
            )
        )
    ),
    _OUTPUTS / "_stages" / GRID_UNIT,
)
GRID_BUDGET = 10_000
GRID_INSTANCE = 11

needs_the_grid = pytest.mark.skipif(
    not GRID_SHARD.exists(),
    reason=f"{GRID_SHARD} has not been produced in this checkout",
)

# What `tune_pair` selects for each focus pair, measured off `GRID_SHARD`: the
# binding the paper's results use (its `tune_pair` unit ccdef07d6e569b5b).
SELECTED_UNDER_THE_ESTIMATOR = {
    "bipop-cmaes__adaptive-nelder-mead": "0881cc94d087",
    "shade__l-shade": "0a405cc922b0",
    "bobyqa__dual-annealing": "1136c8a03c82",
}


def _measured_errors():
    """Every configuration's per-trial errors, keyed by member and config."""
    grid = pd.read_parquet(GRID_SHARD)
    grid = grid[(grid.budget == GRID_BUDGET) & (grid.instance_id == GRID_INSTANCE)]
    return {
        (str(algorithm), str(config)): {
            int(cast(int, function_id)): rows.sort_values("trial")["error"].to_numpy(
                dtype=np.float64
            )
            for function_id, rows in block.groupby("function_id", observed=True)
        }
        for (algorithm, config), block in grid.groupby(
            ["algorithm", "config_id"], observed=True
        )
    }


def _rates_of(tuned_errors, fixed_errors):
    """One candidate's superiority sample, on the tuning's own instance order."""
    functions = sorted(fixed_errors)
    return superiorities(
        {(f, GRID_INSTANCE): tuned_errors[f] for f in functions},
        {(f, GRID_INSTANCE): fixed_errors[f] for f in functions},
    )


def _sides_of(pair, errors):
    """Which member is tuned and which is held at its published configuration."""
    keys = {
        name: (name, config_id_of(TUNABLE[name].resolve()))
        for name in (pair.first, pair.second)
    }
    rates = _rates_of(errors[keys[pair.first]], errors[keys[pair.second]])
    tuned = pair.first if rates.mean() < 0.5 else pair.second
    fixed = pair.second if tuned == pair.first else pair.first
    return keys[tuned], keys[fixed]


def _selected_for(pair, errors):
    """The candidate `tune_pair` would keep; ties go to the first in grid order."""
    (tuned, _), fixed = _sides_of(pair, errors)
    scored = [
        (
            ks_distance(
                _rates_of(errors[(tuned, config_id_of(params))], errors[fixed])
            ),
            position,
            config_id_of(params),
        )
        for position, params in enumerate(TUNABLE[tuned].candidates())
    ]
    return min(scored)[2]


class TestWhatTheEstimatorSelects:
    """The binding the study runs under, pinned against the measured grid."""

    @needs_the_grid
    @pytest.mark.parametrize("pair", FOCUS_PAIRS, ids=lambda pair: pair.slug)
    def test_the_selected_candidate_is_the_pinned_one(self, pair):
        selected = _selected_for(pair, _measured_errors())
        assert selected == SELECTED_UNDER_THE_ESTIMATOR[pair.slug], (
            f"{pair.slug} selects {selected} under {ESTIMATOR}, not the pinned "
            f"{SELECTED_UNDER_THE_ESTIMATOR[pair.slug]}; the binding every "
            "downstream benchmark runs under has moved"
        )

    @needs_the_grid
    @pytest.mark.parametrize("pair", FOCUS_PAIRS, ids=lambda pair: pair.slug)
    def test_the_tuned_member_is_decided_by_the_same_estimator(self, pair):
        """The tuned member is the weaker side by the same estimator."""
        (tuned, _), (fixed, _) = _sides_of(pair, _measured_errors())
        assert {tuned, fixed} == {pair.first, pair.second}
        assert tuned != fixed
