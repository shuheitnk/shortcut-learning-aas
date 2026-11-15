"""The tuning search space and the two objectives it is searched with.

One space, shared by the pairwise and the portfolio tuning.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, cast

from aas_shortcut.define_algorithm_portfolio import ALGORITHMS, FOCUS_PAIRS
from aas_shortcut.define_tunable_portfolio import TUNABLE
from aas_shortcut.optimizers.define_optimizer_base import EPS_X
from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm
from aas_shortcut.tune_algorithms import PairConditions

# Parameters that define an algorithm rather than configure it.
FORBIDDEN_AXES = (
    # Fixed rather than searched.
    "xtol_rel",
    "p",
    # Algorithm identity.
    "lpsr",
    "adaptive",
    "active",
    "restarts",
    "ftol_rel",
    "no_local_search",
    # Dual Annealing's L-BFGS-B step tolerance: fixed at 1e-4, because L-BFGS-B's
    # own ftol and gtol both read the objective.
    "local_step_tolerance",
)

# The axes the design fixes, per member.
EXPECTED_AXES = {
    "BIPOP-CMA-ES": {"sigma0_factor", "lambda0_factor", "eps_x"},
    "Adaptive Nelder-Mead": {"simplex_factor", "eps_x"},
    "SHADE": {"population_factor", "f0", "cr0", "eps_x"},
    "L-SHADE": {"population_factor", "f0", "cr0", "eps_x"},
    "BOBYQA": {"initial_step_factor", "eps_x"},
    # Here `eps_x` is the re-annealing ratio: a displacement criterion does not fit
    # its visiting distribution. Its L-BFGS-B tolerance is fixed, not an axis.
    "Dual Annealing": {"initial_temp", "visit", "accept", "eps_x"},
}


class TestSearchSpace:
    def test_every_portfolio_member_has_an_entry(self):
        assert tuple(TUNABLE) == ALGORITHMS

    def test_the_axes_are_exactly_the_designs(self):
        """These are the listed axes; nothing else may be searched."""
        assert {name: set(e.grid) for name, e in TUNABLE.items()} == EXPECTED_AXES

    def test_eps_x_is_searched_from_a_shared_three_value_grid(self):
        """Every member searches the same three log-spaced eps_x values."""
        from aas_shortcut.define_tunable_portfolio import EPS_X_GRID

        assert EPS_X_GRID == (1e-8, 1e-6, EPS_X)
        for name, entry in TUNABLE.items():
            assert entry.grid["eps_x"] == EPS_X_GRID, name
            assert entry.defaults["eps_x"] == EPS_X, name

    def test_shade_keeps_p_at_its_published_behaviour(self):
        """`p` is fixed at its published default to bound the search space."""
        for name in ("SHADE", "L-SHADE"):
            assert "p" not in TUNABLE[name].grid, name
            assert TUNABLE[name].defaults["p"] is None, name

    def test_f_and_cr_remain_in_the_de_members_space(self):
        for name in ("SHADE", "L-SHADE"):
            assert {"f0", "cr0"} <= set(TUNABLE[name].grid), name

    def test_no_axis_defines_the_algorithm(self):
        for name, entry in TUNABLE.items():
            forbidden = [axis for axis in FORBIDDEN_AXES if axis in entry.grid]
            assert not forbidden, f"{name}: {forbidden}"

    def test_shade_and_lshade_share_a_parameter_set(self):
        """The population reduction is a mechanism, not a setting."""
        shade, lshade = TUNABLE["SHADE"], TUNABLE["L-SHADE"]
        assert dict(shade.defaults) == dict(lshade.defaults)
        assert {k: list(v) for k, v in shade.grid.items()} == {
            k: list(v) for k, v in lshade.grid.items()
        }

    def test_bobyqa_carries_eps_x_rather_than_xtol_rel(self):
        """NLopt's xtol_rel is this member's displacement threshold."""
        entry = TUNABLE["BOBYQA"]
        assert "xtol_rel" not in entry.defaults
        assert entry.runner({"eps_x": 1e-3}) is not None

    def test_every_default_configuration_builds_a_runner(self):
        for name, entry in TUNABLE.items():
            assert entry.runner() is not None, name

    def test_every_grid_value_builds_a_runner(self):
        for name, entry in TUNABLE.items():
            for params in entry.candidates():
                assert entry.runner(params) is not None, (name, params)

    def test_the_search_is_full_factorial(self):
        """The product over the axes, interactions included."""
        for name, entry in TUNABLE.items():
            expected = math.prod(len(values) for values in entry.grid.values())
            assert len(entry.candidates()) == expected, name

    def test_every_combination_of_every_axis_is_present(self):
        """The factorial spelled out: no pair of values goes unsearched."""
        for name, entry in TUNABLE.items():
            axes = sorted(entry.grid)
            searched = {
                tuple(params[axis] for axis in axes) for params in entry.candidates()
            }
            expected = set(itertools.product(*(entry.grid[axis] for axis in axes)))
            assert searched == expected, name

    def test_a_two_axis_grid_searches_the_product_not_the_cross(self):
        """A regression guard: coordinate-wise would give 1 + 2 + 2 = 5, not 9."""
        entry = TunableAlgorithm(
            name="probe",
            slug="probe",
            factory=cast(Any, lambda a, b: (a, b)),
            defaults={"a": 0, "b": 0},
            grid={"a": (0, 1, 2), "b": (0, 1, 2)},
        )
        assert len(entry.candidates()) == 9

    def test_the_candidate_counts_are_the_documented_ones(self):
        """Each member's grid has the documented candidate count."""
        counts = {name: len(entry.candidates()) for name, entry in TUNABLE.items()}
        assert counts == {
            "BIPOP-CMA-ES": 75,
            "Adaptive Nelder-Mead": 15,
            "SHADE": 375,
            "L-SHADE": 375,
            "BOBYQA": 15,
            "Dual Annealing": 375,
        }
        assert sum(counts.values()) == 1230

    def test_the_default_leads_the_candidate_list(self):
        for name, entry in TUNABLE.items():
            first = entry.candidates()[0]
            assert first == entry.resolve(), name


class TestConditions:
    def test_tuning_uses_one_instance_per_function(self):
        """24 functions x 1 instance, not the evaluation's 240."""
        conditions = PairConditions()
        assert len(conditions.function_ids) == 24
        assert conditions.instance_ids == (1,)
        assert len(conditions.instances) == 24
        assert conditions.n_trials == 31
        assert conditions.dim == 10

    def test_the_composed_run_tunes_on_24_and_evaluates_on_240(self, tmp_path):
        """The composed run tunes on 24 problems and evaluates on 240."""
        from aas_shortcut.run_experiment_spec import build_context, resolve_run_paths
        from tests.conftest import compose_config

        cfg = compose_config("preset=full", f"paths.output_root={tmp_path}")
        context = build_context(cfg, resolve_run_paths(cfg))
        assert len(context.tuning_conditions().instances) == 24
        assert context.max_functions * context.max_instances == 240

    def test_there_is_one_focus_pair_per_group(self):
        assert len(FOCUS_PAIRS) == 3
        assert {pair.pair_type for pair in FOCUS_PAIRS} == {"G1_G1", "G2_G2", "G3_G3"}


class TestWiring:
    """The two tunings are two objectives over one search space, not two spaces."""

    def _units(self, tmp_path):
        """The three tuning units the experiments resolve, keyed by their stage."""
        from aas_shortcut.resolve_tuning_stages import (
            grid_unit,
            pair_tuning_unit,
            role_unit,
        )
        from aas_shortcut.run_experiment_spec import build_context, resolve_run_paths
        from tests.conftest import compose_config

        cfg = compose_config("preset=debug", f"paths.output_root={tmp_path}")
        context = build_context(cfg, resolve_run_paths(cfg))
        units = (pair_tuning_unit(context), grid_unit(context), role_unit(context))
        return {unit.stage: unit for unit in units}

    def test_both_tunings_reference_the_same_theta(self, tmp_path):
        units = self._units(tmp_path)
        pair = units["tune_pair"].settings["theta"]
        grid = units["grid_errors"].settings["theta"]
        assert pair == grid
        assert set(pair) == set(ALGORITHMS)

    def test_the_two_objectives_differ(self, tmp_path):
        units = self._units(tmp_path)
        assert units["tune_pair"].settings["objective"] == "ks"
        assert set(units["tuning_roles"].settings["objectives"].values()) == {
            "generalist",
            "oracle",
        }

    def test_the_role_tuning_covers_both_budgets(self, tmp_path):
        """1000d and 1050d are measured on one grid and bound separately."""
        units = self._units(tmp_path)
        budgets = {
            conditions["budget"]
            for conditions in units["grid_errors"].settings["conditions"]
        }
        assert len(budgets) == 2
        assert set(units["tuning_roles"].settings["objectives"]) == {
            str(budget) for budget in budgets
        }

    def test_neither_tuning_reads_a_feature(self, tmp_path):
        """The tuning is independent of every representation."""
        units = self._units(tmp_path)
        assert all(not unit.inputs for unit in units.values())
        # The role selection is chosen from the grid, which its settings carry.
        assert units["tuning_roles"].settings["grid"] == units["grid_errors"].settings
