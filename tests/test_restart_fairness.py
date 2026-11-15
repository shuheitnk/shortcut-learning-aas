"""Tests for the comparison's fairness, given that the restart mechanisms differ.

Fairness rests on everything around the restart rule being shared, and on no
member's own mechanism being altered to suit the study.
"""

from __future__ import annotations

import numpy as np
import pytest

from aas_shortcut.create_problem import create_problem
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS, PORTFOLIO
from aas_shortcut.optimizers.define_optimizer_base import BudgetedObjective
from tests.define_restart_mechanism import MECHANISMS

DIM = 5
BUDGET = 1200
LB, UB = np.full(DIM, -5.0), np.full(DIM, 5.0)


class TestSharedConditions:
    """What every member gets, whatever its restart rule."""

    @pytest.mark.parametrize("name", ALGORITHMS)
    def test_the_whole_budget_is_spent_and_never_exceeded(self, name: str):
        """Restarting more often must not buy or cost evaluations."""
        spec = next(s for s in PORTFOLIO if s.name == name)
        problem = create_problem(8, 1, DIM)
        objective = BudgetedObjective(
            lambda x: float(problem(x)), BUDGET, bounds=(LB, UB)
        )
        spec.runner(objective, DIM, 3, BUDGET, LB, UB)
        assert objective.evaluations == BUDGET

    @pytest.mark.parametrize("name", ALGORITHMS)
    def test_no_member_evaluates_outside_the_box(self, name: str):
        """One domain and one bound rule; a member searching a larger box would win."""
        spec = next(s for s in PORTFOLIO if s.name == name)
        problem = create_problem(8, 1, DIM)
        seen: list[np.ndarray] = []

        def objective(x):
            seen.append(np.asarray(x, dtype=np.float64).copy())
            return float(problem(x))

        wrapped = BudgetedObjective(objective, BUDGET, bounds=(LB, UB))
        spec.runner(wrapped, DIM, 3, BUDGET, LB, UB)
        points = np.asarray(seen)
        assert np.all(points >= LB) and np.all(points <= UB)


class TestNoMemberWasTunedToRestart:
    """Every member's restart threshold is tuned from one declared grid.

    No member's schedule is set by hand: each draws from the same grid, anchored on
    the value it would otherwise have had.
    """

    def test_the_native_threshold_no_longer_follows_the_dependency_default(self):
        """Dual Annealing's re-annealing ratio is eps_x, not SciPy's 2e-5.

        The mechanism stays SciPy's; pinning the threshold to SciPy's default would be
        the asymmetry this class rules out.
        """
        import inspect

        from scipy.optimize import dual_annealing

        from aas_shortcut.define_tunable_portfolio import EPS_X_GRID, TUNABLE

        published = (
            inspect.signature(dual_annealing).parameters["restart_temp_ratio"].default
        )
        assert published not in EPS_X_GRID
        assert TUNABLE["Dual Annealing"].defaults["eps_x"] in EPS_X_GRID

    def test_the_restart_threshold_stays_inside_the_dependency_domain(self):
        """SciPy rejects a re-annealing ratio outside (0, 1)."""
        from aas_shortcut.define_tunable_portfolio import EPS_X_GRID

        assert all(0.0 < e < 1.0 for e in EPS_X_GRID)

    def test_every_member_has_a_tuned_restart_threshold(self):
        """Every member tunes ``eps_x``; no member is left on a fixed threshold.

        For Dual Annealing it bounds a temperature ratio rather than a displacement,
        which does not apply to its visiting distribution, but from the same values.
        """
        from aas_shortcut.define_tunable_portfolio import TUNABLE

        for name, mechanism in MECHANISMS.items():
            assert mechanism.threshold_is_tuned, f"{name} restarts on a fixed threshold"
            assert mechanism.uses_eps_x_for_restart, (
                f"{name} restarts on something else"
            )
            assert "eps_x" in TUNABLE[name].grid, f"{name} does not search eps_x"

    def test_research_side_thresholds_come_from_one_declared_grid(self):
        """No member gets a threshold of its own choosing."""
        from aas_shortcut.define_tunable_portfolio import EPS_X_GRID, TUNABLE

        for name, mechanism in MECHANISMS.items():
            grid = TUNABLE[name].as_dict().get("grid", {})
            if not mechanism.uses_eps_x_for_restart:
                continue
            assert tuple(grid["eps_x"]) == tuple(EPS_X_GRID), (
                f"{name} draws its restart threshold from a different set"
            )

    def test_the_restart_threshold_axis_has_one_resolution(self):
        """One axis, three values, for every member: no member searches finer."""
        from aas_shortcut.define_tunable_portfolio import EPS_X_GRID, TUNABLE

        assert len(EPS_X_GRID) == 3
        for name in MECHANISMS:
            assert tuple(TUNABLE[name].grid["eps_x"]) == tuple(EPS_X_GRID), name

    def test_every_member_is_scored_on_the_same_quantity(self):
        """best-so-far raw objective, whatever the member saw during the search."""
        from aas_shortcut.define_table_schema import VIEW_PERFORMANCE_SCHEMA

        assert "best_y" in VIEW_PERFORMANCE_SCHEMA.columns
        # Diagnostics cannot enter a comparison: they are optional columns and no
        # analysis stage reads them.
        assert set(VIEW_PERFORMANCE_SCHEMA.optional_columns) == {
            "multistarts",
            "native_reanneals",
        }
