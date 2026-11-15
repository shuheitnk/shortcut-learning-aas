"""Dual Annealing, built from SciPy's components with projected bounds."""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt
from scipy.optimize._dual_annealing import (
    EnergyState,
    LocalSearchWrapper,
    ObjectiveFunWrapper,
    StrategyChain,
    VisitingDistribution,
)

from aas_shortcut.optimizers.define_optimizer_base import (
    EPS_X,
    BudgetedObjective,
    RestartReport,
    Runner,
    normalise,
    run_within_budget,
)

LOGGER = logging.getLogger(__name__)

# SciPy's defaults, named so the portfolio can bind them explicitly.
INITIAL_TEMP = 5230.0
VISIT_PARAM = 2.62
ACCEPT_PARAM = -5.0
# Displacement below which the L-BFGS-B phase stops, in normalised coordinates.
# Fixed rather than tuned, and not ``eps_x``, which here is the restart threshold.
# L-BFGS-B's own criteria are switched off because both read the objective, so a
# strictly increasing ``g`` would move the local phase's stopping point.
LOCAL_STEP_TOLERANCE = 1e-4


class ProjectingVisitingDistribution(VisitingDistribution):
    """SciPy's visiting distribution with the toroidal wrap replaced."""

    def visiting(self, x, step, temperature):
        """Return a proposal inside the box, projecting rather than wrapping."""
        dim = x.size
        if step < dim:
            visits = self.visit_fn(temperature, dim)
            upper_sample, lower_sample = self.rand_gen.uniform(size=2)
            visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
            visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
            x_visit = np.clip(visits + x, self.lower, self.upper)
            x_visit[np.abs(x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1e-10
            return x_visit

        x_visit = np.copy(x)
        visit = self.visit_fn(temperature, 1)[0]
        if visit > self.TAIL_LIMIT:
            visit = self.TAIL_LIMIT * self.rand_gen.uniform()
        elif visit < -self.TAIL_LIMIT:
            visit = -self.TAIL_LIMIT * self.rand_gen.uniform()
        index = step - dim
        x_visit[index] = min(
            max(visit + x[index], self.lower[index]), self.upper[index]
        )
        if np.abs(x_visit[index] - self.lower[index]) < self.MIN_VISIT_BOUND:
            x_visit[index] += self.MIN_VISIT_BOUND
        return x_visit


class _StepConverged(Exception):
    """Raised inside the local search once its step falls below ``eps_x``."""


class StepLimitedLocalSearch(LocalSearchWrapper):
    """L-BFGS-B stopped on step displacement, as the restart rule asks."""

    def __init__(self, search_bounds, func_wrapper, lower, upper, eps_x, **kwargs):
        super().__init__(search_bounds, func_wrapper, **kwargs)
        self._box = (np.asarray(lower, float), np.asarray(upper, float))
        self._eps_x = float(eps_x)
        # `ftol=0` does not disable the reduction test but does stop it firing early on
        # a merely small improvement, which is the part that would otherwise pre-empt
        # the step criterion.
        self.kwargs.setdefault("options", {})
        self.kwargs["options"] = {**self.kwargs["options"], "ftol": 0.0, "gtol": 0.0}

    def local_search(self, x, e):
        """Run the local search, ending it on the step criterion."""
        iterates: list[np.ndarray] = [np.asarray(x, dtype=np.float64)]
        lower, upper = self._box

        def callback(xk) -> None:
            iterates.append(np.array(xk, dtype=np.float64))
            step = normalise(iterates[-1], lower, upper) - normalise(
                iterates[-2], lower, upper
            )
            if float(np.linalg.norm(step)) <= self._eps_x:
                raise _StepConverged

        previous = self.kwargs.pop("callback", None)
        self.kwargs["callback"] = callback
        try:
            return super().local_search(x, e)
        except _StepConverged:
            # The last iterate is the result; its value costs one evaluation, which the
            # budget wrapper accounts for like any other.
            best = iterates[-1]
            value = float(self.func_wrapper.fun(best))
            return (value, best) if value < e else (e, np.copy(x))
        finally:
            if previous is None:
                self.kwargs.pop("callback", None)
            else:
                self.kwargs["callback"] = previous


def _drive(
    objective: BudgetedObjective,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    seed: int,
    *,
    initial_temp: float,
    visit: float,
    accept: float,
    eps_x: float,
    no_local_search: bool,
    # Single-element counter, incremented once per native re-annealing.
    reannealings: list[int],
) -> None:
    """Run Dual Annealing until the budget ends, re-annealing as SciPy defines it."""
    rng = np.random.default_rng(seed)
    func_wrapper = ObjectiveFunWrapper(objective, maxfun=float("inf"))
    bounds = list(zip(lb, ub, strict=True))
    minimizer = StepLimitedLocalSearch(
        bounds, func_wrapper, lb, ub, LOCAL_STEP_TOLERANCE
    )
    energy_state = EnergyState(lb, ub)
    energy_state.reset(func_wrapper, rng, np.asarray(x0, dtype=np.float64))
    distribution = ProjectingVisitingDistribution(lb, ub, visit, rng)
    chain = StrategyChain(
        accept, distribution, func_wrapper, minimizer, rng, energy_state
    )

    t1 = np.exp((visit - 1.0) * np.log(2.0)) - 1.0
    # `eps_x` is this member's restart threshold, as it is for the other five; the
    # quantity it bounds is the temperature ratio rather than a displacement.
    temperature_restart = initial_temp * eps_x
    iteration = 0
    while True:
        temperature = (
            initial_temp
            * t1
            / (np.exp((visit - 1.0) * np.log(float(iteration) + 2.0)) - 1.0)
        )
        if temperature < temperature_restart:
            # This member's restart: SciPy's own re-annealing, a fresh energy state and
            # the temperature schedule started over, at the tuned `eps_x`.
            energy_state.reset(func_wrapper, rng)
            reannealings[0] += 1
            iteration = 0
            continue
        if chain.run(iteration, temperature) is not None:
            return
        if not no_local_search and chain.local_search() is not None:
            return
        # No displacement-based termination: the episode ends with the budget, and
        # re-annealing above is this member's restart.
        iteration += 1


def make_dual_annealing(
    *,
    initial_temp: float = INITIAL_TEMP,
    visit: float = VISIT_PARAM,
    accept: float = ACCEPT_PARAM,
    no_local_search: bool = False,
    eps_x: float = EPS_X,
) -> Runner:
    """Build a Dual Annealing runner that restarts on loss of search capability.

    Its restarts are the native re-annealings, reported both as ``restarts`` and as
    ``native_reanneals`` so they are not mistaken for a displacement criterion.
    """

    def runner(
        objective: BudgetedObjective,
        dim: int,
        seed: int,
        budget: int,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
    ) -> RestartReport:
        del budget  # The cap lives on the objective, not on the loop.
        rng = np.random.default_rng(seed)
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        episodes = 0
        reannealings = [0]
        while objective.remaining > 0:
            before = objective.evaluations
            run_within_budget(
                lambda: _drive(
                    objective,
                    rng.uniform(lb, ub, size=dim),
                    lb,
                    ub,
                    int(rng.integers(0, 2**31 - 1)),
                    initial_temp=initial_temp,
                    visit=visit,
                    accept=accept,
                    eps_x=eps_x,
                    no_local_search=no_local_search,
                    reannealings=reannealings,
                ),
                objective,
            )
            episodes += 1
            if objective.evaluations == before:
                break

        # `episodes` counts outer loop passes, which end only when the budget does;
        # this member's re-initialisations are the re-annealings.
        LOGGER.debug(
            "dual annealing: %d native re-annealing(s) over %d episode(s), %d FE",
            reannealings[0],
            episodes,
            objective.evaluations,
        )
        return RestartReport(restarts=reannealings[0], native_reanneals=reannealings[0])

    return runner
