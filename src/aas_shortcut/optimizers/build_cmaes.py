"""CMA-ES, and BIPOP restarts driven by this study's own criterion."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from modcma.c_maes import ModularCMAES, Settings, cmaescpp, options, parameters

from aas_shortcut.optimizers.define_optimizer_base import (
    EPS_X,
    BudgetedObjective,
    RestartReport,
    Runner,
    run_within_budget,
)

# Initial step size as a fraction of the mean box width.
SIGMA0_FACTOR = 0.3


def default_population(dim: int) -> int:
    """Return CMA-ES's default population size, ``4 + floor(3 ln n)``."""
    return int(4 + np.floor(3.0 * np.log(max(1, int(dim)))))


def search_capability(optimizer: ModularCMAES, box_width: float) -> float:
    """Return ``sigma * sqrt(lambda_max(C))`` in normalised coordinates."""
    sigma = float(optimizer.p.mutation.sigma)
    root_eigenvalues = np.asarray(optimizer.p.adaptation.d, dtype=np.float64)
    return sigma * float(np.max(root_eigenvalues)) / box_width


def _run_episode(
    objective: BudgetedObjective,
    dim: int,
    x0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    lambda0: int,
    sigma0: float,
    eps_x: float,
) -> None:
    """Run one CMA-ES episode until its search capability collapses."""
    modules = parameters.Modules()
    modules.restart_strategy = options.RestartStrategy.NONE
    # The plain rank-mu update, not aCMA-ES.
    modules.active = False
    # The bound rule every portfolio member shares.
    modules.bound_correction = options.CorrectionMethod.SATURATE
    optimizer = ModularCMAES(
        Settings(
            dim,
            x0=x0,
            modules=modules,
            lb=lb,
            ub=ub,
            sigma0=sigma0,
            budget=objective.remaining + lambda0,
            lambda0=lambda0,
        )
    )
    box_width = float(np.mean(ub - lb))

    def solve() -> None:
        while True:
            # Generate, evaluate, *then* decide.
            if not optimizer.step(objective):
                return
            if search_capability(optimizer, box_width) <= eps_x:
                return

    run_within_budget(solve, objective)


def make_bipop_cmaes(
    *,
    sigma0_factor: float = SIGMA0_FACTOR,
    lambda0_factor: float | None = None,
    eps_x: float = EPS_X,
) -> Runner:
    """Build a BIPOP-CMA-ES runner that reports the completed restarts."""

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
        np.random.seed(seed)
        cmaescpp.utils.set_seed(seed)

        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        box_width = float(np.mean(ub - lb))
        sigma0 = float(sigma0_factor) * box_width
        default = default_population(dim)
        lambda0 = (
            default
            if lambda0_factor is None
            else max(4, round(lambda0_factor * default))
        )

        # BIPOP's budget balancing (Hansen 2009): the two regimes are alternated by
        # whichever has consumed less, so neither can monopolise the budget.
        spent_large = spent_small = 0
        large_restarts = 0
        episodes = 0

        while objective.remaining > 0:
            if episodes == 0:
                lambda_, sigma, small = lambda0, sigma0, False
            elif spent_small < spent_large:
                # Small regime: a population between lambda0 and half the current large
                # one, and a step size damped by up to 1e-2.
                lambda_large = lambda0 * 2**large_restarts
                exponent = float(rng.random()) ** 2
                lambda_ = max(
                    4, int(lambda0 * (lambda_large / (2 * lambda0)) ** exponent)
                )
                # Damped as Hansen's BIPOP damps it, with no floor: the episode
                # evaluates a generation before the criterion is consulted, so a
                # regime starting below the threshold still searches once.
                sigma = sigma0 * 10.0 ** (-2.0 * float(rng.random()))
                small = True
            else:
                large_restarts += 1
                lambda_ = lambda0 * 2**large_restarts
                sigma, small = sigma0, False

            before = objective.evaluations
            _run_episode(
                objective,
                dim,
                rng.uniform(lb, ub, size=dim),
                lb,
                ub,
                lambda0=lambda_,
                sigma0=sigma,
                eps_x=eps_x,
            )
            spent = objective.evaluations - before
            if small:
                spent_small += spent
            else:
                spent_large += spent
            episodes += 1

            if spent == 0:
                # No budget left for even one evaluation; another episode would loop
                # forever.
                break

        # Episodes beyond the first are the restarts.
        return RestartReport(restarts=max(0, episodes - 1))

    return runner
