"""Tests for the shared optimizers in ``aas_shortcut.optimizers``.

Each portfolio member binds parameters to a factory here; the code is shared.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from aas_shortcut.optimizers.build_cmaes import make_bipop_cmaes
from aas_shortcut.optimizers.build_differential_evolution import make_shade
from aas_shortcut.optimizers.build_dual_annealing import make_dual_annealing
from aas_shortcut.optimizers.build_nelder_mead import make_nelder_mead
from aas_shortcut.optimizers.build_nlopt_local import make_bobyqa
from aas_shortcut.optimizers.define_optimizer_base import EPS_X, BudgetedObjective

pytest.importorskip("ioh")

DIM = 10
BUDGET = 1000 * DIM
INSTANCE = 1


def run(runner, function_id: int, seed: int) -> tuple[float, int, int]:
    """Run once; return the final error, the restart count and the evaluations."""
    import ioh

    problem = ioh.get_problem(function_id, INSTANCE, DIM, ioh.ProblemClass.BBOB)
    objective = BudgetedObjective(cast(Any, problem), BUDGET)
    restarts = runner(
        objective, DIM, seed, BUDGET, np.full(DIM, -5.0), np.full(DIM, 5.0)
    )
    error = float(problem.state.current_best.y) - float(problem.optimum.y)
    return error, int(restarts), int(objective.evaluations)


# ---------------------------------------------------------------------------
# Structure: what belongs in this package and what does not
# ---------------------------------------------------------------------------


def test_optimizers_do_not_import_any_portfolio() -> None:
    """No implementation may depend on a portfolio that configures it."""
    import aas_shortcut.optimizers as package

    forbidden = ("define_algorithm_portfolio", "tune_algorithms")
    for module_path in Path(package.__file__).parent.glob("*.py"):
        source = module_path.read_text(encoding="utf-8")
        for line in source.splitlines():
            if not line.startswith(("import ", "from ")):
                continue
            assert not any(f"aas_shortcut.{name}" in line for name in forbidden), (
                f"{module_path.name} imports a portfolio: {line!r}"
            )


# ---------------------------------------------------------------------------
# The bindings the portfolio makes
# ---------------------------------------------------------------------------


def test_bobyqa_runs_with_its_capability_criterion_on() -> None:
    """``run_bobyqa`` binds ``xtol_rel=EPS_X``, not NLopt's own ``0``."""
    from aas_shortcut.define_algorithm_portfolio import run_bobyqa

    bound = run(run_bobyqa, 1, 7)
    assert bound == run(make_bobyqa(xtol_rel=EPS_X), 1, 7)
    assert bound != run(make_bobyqa(xtol_rel=0.0), 1, 7)
    # The criterion is what buys the extra starts.
    assert bound[1] > run(make_bobyqa(xtol_rel=0.0), 1, 7)[1]


def test_lshade_is_shade_with_lpsr() -> None:
    """L-SHADE is SHADE with LPSR, and each part changes the run."""
    from aas_shortcut.define_algorithm_portfolio import run_lshade, run_shade

    assert run(run_shade, 8, 7) != run(run_lshade, 8, 7)
    settings: dict[str, Any] = {"population_factor": 18.0, "f0": 0.5, "cr0": 0.5}
    assert run(run_shade, 8, 7) == run(make_shade(**settings, lpsr=False), 8, 7)
    assert run(run_lshade, 8, 7) == run(make_shade(**settings, lpsr=True), 8, 7)


def test_adaptive_nelder_mead_binds_the_gao_han_coefficients() -> None:
    """Adaptive Nelder-Mead is the shared implementation with ``adaptive=True``."""
    from aas_shortcut.define_algorithm_portfolio import run_nelder_mead

    assert run(run_nelder_mead, 8, 7) == run(make_nelder_mead(adaptive=True), 8, 7)
    assert run(run_nelder_mead, 8, 7) != run(make_nelder_mead(adaptive=False), 8, 7)


# ---------------------------------------------------------------------------
# The restart specification
# ---------------------------------------------------------------------------


MEMBERS = {
    "BIPOP-CMA-ES": make_bipop_cmaes(),
    "Adaptive Nelder-Mead": make_nelder_mead(adaptive=True),
    "SHADE": make_shade(),
    "L-SHADE": make_shade(lpsr=True),
    "BOBYQA": make_bobyqa(xtol_rel=EPS_X),
    "Dual Annealing": make_dual_annealing(),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_every_member_spends_the_whole_budget(name: str) -> None:
    """Every member spends the whole budget."""
    _, _, evaluations = run(MEMBERS[name], 21, 7)
    assert evaluations == BUDGET


def test_each_restart_criterion_is_the_geometric_quantity_the_spec_names() -> None:
    """Each restart criterion is geometric and never reads an objective value."""
    from types import SimpleNamespace

    from aas_shortcut.optimizers import build_cmaes as cmaes
    from aas_shortcut.optimizers import build_differential_evolution as de
    from aas_shortcut.optimizers import build_nelder_mead as nelder_mead

    lb, ub = np.full(3, -5.0), np.full(3, 5.0)

    # ANM: the simplex diameter in normalised coordinates.
    simplex = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    assert nelder_mead.capability_lost(simplex, lb, ub, 0.11)
    assert not nelder_mead.capability_lost(simplex, lb, ub, 0.09)

    # SHADE/L-SHADE: twice the diameter of population and archive. The archive counts
    # because x_r2 is drawn from it.
    population = SimpleNamespace(x=np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    tight = cast(Any, SimpleNamespace(population=population, archive=None))
    assert de.capability_lost(tight, lb, ub, 0.21)
    assert not de.capability_lost(tight, lb, ub, 0.19)
    spread = cast(
        Any,
        SimpleNamespace(
            population=population,
            archive=SimpleNamespace(x=np.array([[5.0], [0.0], [0.0]])),
        ),
    )
    assert not de.capability_lost(spread, lb, ub, 0.21)

    # BIPOP-CMA-ES: sigma times the largest principal standard deviation, normalised by
    # the box width. `adaptation.d` holds the square roots of the eigenvalues of C.
    optimizer = cast(
        Any,
        SimpleNamespace(
            p=SimpleNamespace(
                mutation=SimpleNamespace(sigma=0.5),
                adaptation=SimpleNamespace(d=np.array([0.25, 2.0, 1.0])),
            )
        ),
    )
    assert cmaes.search_capability(optimizer, 10.0) == pytest.approx(0.1)

    # None of the three takes an objective, a value array or a callable.
    for criterion in (
        nelder_mead.capability_lost,
        de.capability_lost,
        cmaes.search_capability,
    ):
        taken = set(inspect.signature(criterion).parameters)
        assert not taken & {"func", "values", "f", "objective", "fitness"}, criterion


@pytest.mark.slow
def test_restarts_fire_on_a_collapsing_search_and_keep_the_global_best() -> None:
    """A small case built to force restarts, checking the budget rules together."""
    dim, budget = 2, 120
    lb, ub = np.full(dim, -5.0), np.full(dim, 5.0)

    for name in ("Adaptive Nelder-Mead", "BOBYQA"):
        best = [np.inf]

        def sphere(x, best=best) -> float:
            value = float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
            best[0] = min(best[0], value)
            return value

        objective = BudgetedObjective(sphere, budget)
        restarts = int(MEMBERS[name](objective, dim, 3, budget, lb, ub))

        assert restarts >= 1, f"{name} never restarted on a 2-D sphere"
        assert objective.evaluations == budget, name
        # The incumbent is monotone: no restart discarded it.
        assert best[0] < 1e-6, name


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(MEMBERS))
def test_every_member_keeps_itself_inside_the_domain(name: str) -> None:
    """Every member keeps its evaluated points inside the domain."""
    import ioh

    # Above float64 rounding on a bound of 5, below anything a search does.
    tolerance = 1e-9

    lb, ub = np.full(DIM, -5.0), np.full(DIM, 5.0)
    for function_id in (1, 21):
        problem = ioh.get_problem(function_id, INSTANCE, DIM, ioh.ProblemClass.BBOB)
        objective = BudgetedObjective(cast(Any, problem), BUDGET, bounds=(lb, ub))
        MEMBERS[name](objective, DIM, 7, BUDGET, lb, ub)
        assert objective.max_violation < tolerance, (
            f"{name} searched {objective.max_violation:g} outside the domain on "
            f"f{function_id} ({objective.projections} point(s))"
        )


def test_the_wrapper_projects_rather_than_trusting_the_solver() -> None:
    """The wrapper evaluates an infeasible proposal at its projection."""
    seen: list[np.ndarray] = []
    objective = BudgetedObjective(
        lambda x: seen.append(np.array(x)) or 0.0,  # type: ignore[func-returns-value]
        budget=4,
        bounds=(np.full(2, -1.0), np.full(2, 1.0)),
    )
    objective(np.array([0.5, -0.5]))
    objective(np.array([9.0, -9.0]))
    assert objective.projections == 1
    assert np.array_equal(seen[0], [0.5, -0.5])
    assert np.array_equal(seen[1], [1.0, -1.0])


@pytest.mark.slow
@pytest.mark.parametrize("eps_x", [1e-8, 1e-6, 1e-4, 1e-2])
def test_the_budget_is_exhausted_at_every_threshold(eps_x: float) -> None:
    """The budget rule holds whatever ``eps_x`` is, not only at the shipped value."""
    from aas_shortcut.optimizers.build_dual_annealing import make_dual_annealing

    lb, ub = np.full(DIM, -5.0), np.full(DIM, 5.0)
    runners = {
        "BIPOP-CMA-ES": make_bipop_cmaes(eps_x=eps_x),
        "Adaptive Nelder-Mead": make_nelder_mead(adaptive=True, eps_x=eps_x),
        "SHADE": make_shade(eps_x=eps_x),
        "L-SHADE": make_shade(eps_x=eps_x, lpsr=True),
        # eps_x reaches BOBYQA as the trust-region ratio NLopt exposes.
        "BOBYQA": make_bobyqa(xtol_rel=eps_x),
        "Dual Annealing": make_dual_annealing(eps_x=eps_x),
    }
    import ioh

    for name, runner in runners.items():
        for function_id in (3, 21):
            problem = ioh.get_problem(function_id, INSTANCE, DIM, ioh.ProblemClass.BBOB)
            objective = BudgetedObjective(cast(Any, problem), BUDGET)
            runner(objective, DIM, 7, BUDGET, lb, ub)
            assert objective.evaluations == BUDGET, (
                f"{name} left {BUDGET - objective.evaluations} evaluation(s) "
                f"unspent on f{function_id} at eps_x={eps_x:g}"
            )


def test_dual_annealing_restarts_are_its_native_reannealings() -> None:
    """Dual Annealing's restarts are exactly SciPy's re-annealings."""
    from aas_shortcut.optimizers.build_dual_annealing import make_dual_annealing

    dim, budget = 2, 4000
    lb, ub = np.full(dim, -5.0), np.full(dim, 5.0)

    def sphere(x) -> float:
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))

    # A ratio high enough that re-annealing fires inside this budget.
    objective = BudgetedObjective(sphere, budget, bounds=(lb, ub))
    report = make_dual_annealing(eps_x=0.1)(objective, dim, 3, budget, lb, ub)
    assert report.native_reanneals >= 1
    assert report.restarts == report.native_reanneals
    assert report.multistarts == 0
    assert objective.evaluations == budget


@pytest.mark.parametrize("eps_x", [1e-8, 1e-6, 1e-4])
def test_eps_x_is_dual_annealings_re_annealing_ratio(eps_x: float) -> None:
    """``eps_x`` sets the re-annealing ratio, and every restart is a re-annealing."""
    from aas_shortcut.optimizers.build_dual_annealing import make_dual_annealing
    from tests.define_restart_mechanism import mechanism_for

    assert mechanism_for("Dual Annealing").uses_eps_x_for_restart

    dim, budget = 2, 4000
    lb, ub = np.full(dim, -5.0), np.full(dim, 5.0)

    def sphere(x) -> float:
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))

    objective = BudgetedObjective(sphere, budget, bounds=(lb, ub))
    report = make_dual_annealing(eps_x=eps_x)(objective, dim, 3, budget, lb, ub)
    assert report.restarts == report.native_reanneals
    assert report.multistarts == 0
    assert objective.evaluations == budget


def test_a_larger_eps_x_restarts_dual_annealing_sooner() -> None:
    """A larger ``eps_x`` restarts Dual Annealing sooner, as for the other members.

    Here ``eps_x`` bounds a temperature ratio, so a shared grid needs the same
    direction.
    """
    from aas_shortcut.optimizers.build_dual_annealing import make_dual_annealing

    dim, budget = 2, 6000
    lb, ub = np.full(dim, -5.0), np.full(dim, 5.0)

    def sphere(x) -> float:
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))

    counts = []
    for eps_x in (1e-8, 1e-2):
        objective = BudgetedObjective(sphere, budget, bounds=(lb, ub))
        counts.append(
            make_dual_annealing(eps_x=eps_x)(
                objective, dim, 3, budget, lb, ub
            ).native_reanneals
        )
    assert counts[1] > counts[0], f"{counts[1]} re-annealings at 1e-2 vs {counts[0]}"


def test_the_local_search_does_not_stop_on_an_objective_value() -> None:
    """The local search switches off L-BFGS-B's ftol and gtol, which read f.

    A strictly increasing g changes both, so the phase stops on a fixed displacement.
    """
    import inspect

    from scipy.optimize._lbfgsb_py import _minimize_lbfgsb

    from aas_shortcut.optimizers.build_dual_annealing import (
        LOCAL_STEP_TOLERANCE,
        StepLimitedLocalSearch,
    )

    defaults = inspect.signature(_minimize_lbfgsb).parameters
    assert defaults["ftol"].default > 0.0
    assert defaults["gtol"].default > 0.0

    search = StepLimitedLocalSearch.__init__
    source = inspect.getsource(search)
    assert '"ftol": 0.0' in source and '"gtol": 0.0' in source, (
        "L-BFGS-B's objective-reading criteria must be switched off"
    )
    assert LOCAL_STEP_TOLERANCE == 1e-4


def test_dual_annealing_projects_its_visiting_proposals() -> None:
    """The toroidal wrap is replaced by projection."""
    from scipy.optimize._dual_annealing import VisitingDistribution

    from aas_shortcut.optimizers.build_dual_annealing import (
        ProjectingVisitingDistribution,
    )

    lb, ub = np.array([-5.0]), np.array([5.0])

    def on_face(cls) -> tuple[float, np.ndarray]:
        distribution = cls(lb, ub, 2.62, np.random.default_rng(0))
        proposals = np.array(
            [distribution.visiting(np.array([4.9]), 0, 5.0)[0] for _ in range(2000)]
        )
        touching = np.isclose(proposals, lb[0], atol=1e-8) | np.isclose(
            proposals, ub[0], atol=1e-8
        )
        return float(touching.mean()), proposals

    projected, proposals = on_face(ProjectingVisitingDistribution)
    wrapped, _ = on_face(VisitingDistribution)

    assert np.all(proposals >= lb[0]) and np.all(proposals <= ub[0])
    assert projected > 0.9, f"only {projected:.1%} of proposals reached a face"
    assert wrapped < 0.01, "SciPy's own class is expected to wrap, not project"


def test_a_restart_never_resets_the_evaluation_counter() -> None:
    """The budget is shared across episodes, not granted per episode."""
    dim, budget = 2, 60
    objective = BudgetedObjective(
        lambda x: float(np.sum(np.asarray(x, dtype=np.float64) ** 2)), budget
    )
    restarts = int(
        make_bobyqa(xtol_rel=EPS_X)(
            objective, dim, 3, budget, np.full(dim, -5.0), np.full(dim, 5.0)
        )
    )
    assert restarts >= 1
    assert objective.evaluations == budget


def test_the_wrapper_is_the_only_authority_on_the_count() -> None:
    """``objective.state``, which ``modde`` reads, is the wrapper's own count."""
    objective = BudgetedObjective(lambda x: 0.0, budget=3)
    assert objective.state.evaluations == 0
    objective(np.zeros(2))
    assert objective.state.evaluations == 1
    # Stopping on a target precision would end a run with budget left and is an
    # objective-value condition.
    assert objective.state.optimum_found is False


def test_pbest_is_reachable_rather_than_left_to_the_library() -> None:
    """``p=None`` differs from a fixed ``p``, since ``modde`` then draws it."""
    fixed = run(make_shade(p=0.05), 8, 7)
    drawn = run(make_shade(p=None), 8, 7)
    assert fixed != drawn
