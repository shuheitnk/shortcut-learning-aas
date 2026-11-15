"""Benchmarking the portfolio on both views of the objective (the preanalysis)."""

from __future__ import annotations

import numpy as np
import pytest

from aas_shortcut.benchmark_algorithms import trial_seed
from aas_shortcut.benchmark_objective_views import (
    ViewBenchmarkConfig,
    solve_view,
)
from aas_shortcut.define_algorithm_portfolio import (
    GROUP_RANK_SEARCH,
    PORTFOLIO,
)
from aas_shortcut.sample_monotone_transform import transform_for_instance
from tests.conftest import published_binding

DIM = 5
BUDGET = 400
SEED = 42


def config(**overrides) -> ViewBenchmarkConfig:
    """A small grid, overridable field by field."""
    settings = {
        "dim": DIM,
        "max_functions": 3,
        "max_instances": 1,
        "num_trials": 2,
        "budget": BUDGET,
    }
    return ViewBenchmarkConfig(**{**settings, **overrides})


class TestViewBenchmarkConfig:
    def test_the_grid_covers_every_member_under_every_view(self):
        settings = config()
        assert len(settings.jobs()) == 3 * 1 * 2 * 6 * 2
        assert settings.expected_rows() == len(settings.jobs())

    def test_both_views_are_fingerprinted(self):
        """A shard computed under one view set must not be reused under another."""
        assert config().as_dict()["transforms"] == ["pre", "post"]

    def test_the_warp_identity_is_fingerprinted(self):
        """A different warp is a different `post`, so the shard is not reusable."""
        assert "spline-warp" in config().as_dict()["transform"]

    def test_a_budget_below_the_portfolio_floor_is_rejected(self):
        with pytest.raises(ValueError, match="below the"):
            config(budget=10).validate()

    def test_an_unknown_member_is_rejected(self):
        with pytest.raises(ValueError, match="unknown portfolio member"):
            config(algorithms=("Nonesuch",)).validate()

    def test_an_unknown_view_is_rejected(self):
        with pytest.raises(ValueError, match="unknown objective view"):
            config(transforms=("pre", "sideways")).validate()

    def test_a_non_positive_trial_count_is_rejected(self):
        with pytest.raises(ValueError, match="num_trials"):
            config(num_trials=0).validate()


class TestSeedDomain:
    def test_the_view_is_not_part_of_a_run_seed(self):
        """Both views of one trial share a seed, so the transform stays out of it."""
        seeds = {
            trial_seed(SEED, spec.name, "bbob", 1, 1, DIM, trial)
            for spec in PORTFOLIO
            for trial in (1, 2)
        }
        # One seed per (member, trial) and not one per (member, trial, view).
        assert len(seeds) == len(PORTFOLIO) * 2


class TestPerViewGrid:
    """Experiment 1 runs the two views over different instance blocks."""

    def test_by_default_every_view_covers_the_whole_grid(self):
        settings = config(max_instances=4)
        assert settings.instances_of("pre") == settings.instances_of("post")
        assert settings.instances_of("post") == (1, 2, 3, 4)

    def test_a_restricted_view_runs_fewer_jobs(self):
        whole = config(max_instances=4)
        restricted = config(max_instances=4, instances_by_transform={"post": (1, 2)})
        assert restricted.instances_of("pre") == (1, 2, 3, 4)
        assert restricted.instances_of("post") == (1, 2)
        assert restricted.expected_rows() < whole.expected_rows()
        views = {transform for *_, transform in restricted.jobs()}
        assert views == {"pre", "post"}
        assert {
            instance for _, instance, _, _, view in restricted.jobs() if view == "post"
        } == {1, 2}

    def test_an_instance_outside_the_grid_is_rejected(self):
        with pytest.raises(ValueError, match="does not cover"):
            config(max_instances=2, instances_by_transform={"post": (9,)}).validate()

    def test_an_unknown_view_is_rejected(self):
        with pytest.raises(ValueError, match="not one of"):
            config(instances_by_transform={"sideways": (1,)}).validate()

    def test_an_unrestricted_configuration_hashes_as_it_always_did(self):
        """Leaving both fields unset keeps them out of the fingerprint."""
        assert "instances_by_transform" not in config().as_dict()
        assert "tuning" not in config().as_dict()

    def test_a_restriction_and_a_binding_both_reach_the_fingerprint(self):
        restricted = config(instances_by_transform={"post": (1,)}).as_dict()
        assert restricted["instances_by_transform"]["post"] == [1]
        tuned = config(tuning={"params": {"SHADE": {"f0": 0.5}}}).as_dict()
        assert tuned["tuning"] == {"params": {"SHADE": {"f0": 0.5}}}
        assert config().as_dict() != tuned


@pytest.mark.slow
class TestSolveView:
    """What the two views actually do to a run."""

    def run(self, algorithm: str, transform: str, function_id: int = 8) -> dict:
        return solve_view(
            published_binding("views", BUDGET),
            SEED,
            "bbob",
            function_id,
            1,
            DIM,
            1,
            BUDGET,
            algorithm,
            transform,
        )

    def test_best_y_is_measured_on_the_raw_objective(self):
        """`best_y` is read off the raw objective, not the warped one.

        The warp maps the sampled range into [0, 1], so an optimum above one tells
        the two apart.
        """
        row = self.run("BIPOP-CMA-ES", "post")
        warp = transform_for_instance(SEED, "bbob", 8, 1, DIM)
        assert float(warp(row["optimum_y"])) == 0.0
        assert row["optimum_y"] > 1.0, "the check needs an optimum above the warp"
        assert row["best_y"] >= row["optimum_y"]

    def test_a_rank_reading_member_reproduces_its_run_under_the_warp(self):
        """Every G1 member gives the same run under the warp as without it."""
        for spec in PORTFOLIO:
            if spec.group != GROUP_RANK_SEARCH:
                continue
            pre = self.run(spec.name, "pre")
            post = self.run(spec.name, "post")
            assert post["best_y"] == pre["best_y"], spec.name
            assert post["evaluations_used"] == pre["evaluations_used"], spec.name
            assert post["restarts"] == pre["restarts"], spec.name

    def test_the_view_is_recorded_on_the_row(self):
        assert self.run("SHADE", "post")["transform"] == "post"

    def test_the_budget_is_spent_exactly_under_both_views(self):
        for transform in ("pre", "post"):
            assert self.run("SHADE", transform)["evaluations_used"] == BUDGET


@pytest.mark.slow
class TestValueReadingMembersMove:
    """The other half of the control: G3 is not invariant, by design."""

    def test_a_value_reading_member_can_land_somewhere_else(self):
        binding = published_binding("views", BUDGET)
        moved = []
        for function_id in (1, 8, 15, 21):
            rows = {
                transform: solve_view(
                    binding,
                    SEED,
                    "bbob",
                    function_id,
                    1,
                    DIM,
                    1,
                    BUDGET,
                    "Dual Annealing",
                    transform,
                )
                for transform in ("pre", "post")
            }
            moved.append(not np.isclose(rows["pre"]["best_y"], rows["post"]["best_y"]))
        assert any(moved), "no instance moved; the warp reached nothing"
