"""NLopt's derivative-free local methods, behind one shared start."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable

import nlopt
import numpy as np
import numpy.typing as npt

from aas_shortcut.optimizers.define_optimizer_base import (
    EPS_X,
    BudgetedObjective,
    RestartReport,
    Runner,
    multistart,
)


def nlopt_start(
    algorithm: int,
    *,
    initial_step_factor: float | None = None,
    xtol_rel: float = 0.0,
    ftol_rel: float = 0.0,
) -> Callable[..., None]:
    """Build a single local start for one NLopt derivative-free method."""

    def start(
        objective: BudgetedObjective,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        maxfun: int,
        seed: int,
    ) -> None:
        # NLopt's generator is global; seed it so no run depends on ambient state.
        nlopt.srand(int(seed))

        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)
        opt = nlopt.opt(algorithm, len(x0))
        opt.set_lower_bounds(lb)
        opt.set_upper_bounds(ub)
        opt.set_maxeval(int(maxfun))
        if initial_step_factor is not None:
            opt.set_initial_step(float(initial_step_factor) * (ub - lb))
        if xtol_rel:
            opt.set_xtol_rel(float(xtol_rel))
        if ftol_rel:
            opt.set_ftol_rel(float(ftol_rel))
        opt.set_min_objective(lambda x, _grad: objective(x))

        # NLopt raises on a collapsed trust region; for a multistarted solver that is
        # an ordinary end of a start.
        with (
            warnings.catch_warnings(),
            contextlib.suppress(nlopt.RoundoffLimited, RuntimeError),
        ):
            warnings.simplefilter("ignore")
            opt.optimize(np.asarray(x0, dtype=np.float64))

    return start


def _multistarted(start: Callable[..., None]) -> Runner:
    """Wrap a local start as a full runner."""

    def runner(
        objective: BudgetedObjective,
        dim: int,
        seed: int,
        budget: int,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
    ) -> RestartReport:
        del budget  # The cap lives on the objective, not on the loop.
        return multistart(start, objective, dim, seed, lb, ub)

    return runner


def make_bobyqa(
    *,
    initial_step_factor: float | None = None,
    xtol_rel: float = EPS_X,
    ftol_rel: float = 0.0,
) -> Runner:
    """Build a multistart BOBYQA runner."""
    return _multistarted(
        nlopt_start(
            nlopt.LN_BOBYQA,
            initial_step_factor=initial_step_factor,
            xtol_rel=xtol_rel,
            ftol_rel=ftol_rel,
        )
    )
