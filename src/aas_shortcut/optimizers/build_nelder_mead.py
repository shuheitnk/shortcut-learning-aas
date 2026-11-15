"""Nelder-Mead with its four coefficients and its termination rule exposed."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from aas_shortcut.optimizers.define_optimizer_base import (
    EPS_X,
    BudgetedObjective,
    RestartReport,
    Runner,
    collapsed,
    multistart,
    normalise,
)


def initial_simplex(
    x0: np.ndarray, lb: np.ndarray, ub: np.ndarray, factor: float
) -> np.ndarray:
    """Build a ``(dim + 1, dim)`` Nelder-Mead starting simplex inside the box."""
    dim = len(x0)
    simplex = np.repeat(np.asarray(x0, dtype=np.float64)[None, :], dim + 1, axis=0)
    step = float(factor) * (ub - lb)
    for index in range(dim):
        forward = x0[index] + step[index]
        simplex[index + 1, index] = (
            forward if forward <= ub[index] else x0[index] - step[index]
        )
    return np.clip(simplex, lb, ub)


class Coefficients(TypedDict):
    """The four simplex coefficients, as a complete assignment."""

    rho: float
    chi: float
    psi: float
    sigma: float


# Nelder and Mead's original coefficients, SciPy's ``adaptive=False``.
STANDARD_COEFFICIENTS: Coefficients = {
    "rho": 1.0,
    "chi": 2.0,
    "psi": 0.5,
    "sigma": 0.5,
}


def gao_han_coefficients(dim: int) -> Coefficients:
    """Return Gao and Han's dimension-dependent coefficients."""
    n = float(dim)
    return {
        "rho": 1.0,
        "chi": 1.0 + 2.0 / n,
        "psi": 0.75 - 1.0 / (2.0 * n),
        "sigma": 1.0 - 1.0 / n,
    }


def resolve_coefficients(
    dim: int,
    *,
    adaptive: bool = True,
    rho: float | None = None,
    chi: float | None = None,
    psi: float | None = None,
    sigma: float | None = None,
) -> Coefficients:
    """Resolve the four coefficients, each override applied over the preset."""
    preset = gao_han_coefficients(dim) if adaptive else STANDARD_COEFFICIENTS
    return {
        "rho": preset["rho"] if rho is None else float(rho),
        "chi": preset["chi"] if chi is None else float(chi),
        "psi": preset["psi"] if psi is None else float(psi),
        "sigma": preset["sigma"] if sigma is None else float(sigma),
    }


def capability_lost(
    simplex: np.ndarray, lb: np.ndarray, ub: np.ndarray, eps_x: float = EPS_X
) -> bool:
    """Whether the simplex has collapsed past the point of generating moves."""
    return collapsed(normalise(simplex, lb, ub), eps_x)


def minimize_neldermead(
    func: Callable[[np.ndarray], float],
    simplex: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    maxfun: int,
    rho: float,
    chi: float,
    psi: float,
    sigma: float,
    eps_x: float = EPS_X,
) -> tuple[np.ndarray, float]:
    """Run one Nelder-Mead start from a given simplex, returning its best vertex.

    The start ends on the simplex's displacement alone.
    """
    simplex = np.clip(np.array(simplex, dtype=np.float64, copy=True), lb, ub)
    n = simplex.shape[1]
    calls = 0

    def evaluate(x: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(func(x))

    values = np.full(n + 1, np.inf, dtype=np.float64)
    for k in range(n + 1):
        if calls >= maxfun:
            break
        values[k] = evaluate(simplex[k])

    order = np.argsort(values)
    simplex, values = np.take(simplex, order, 0), np.take(values, order, 0)

    while calls < maxfun:
        if capability_lost(simplex, lb, ub, eps_x):
            break

        centroid = np.add.reduce(simplex[:-1], 0) / n
        reflected = np.clip((1 + rho) * centroid - rho * simplex[-1], lb, ub)
        if calls >= maxfun:
            break
        f_reflected = evaluate(reflected)
        shrink = False

        if f_reflected < values[0]:
            expanded = np.clip(
                (1 + rho * chi) * centroid - rho * chi * simplex[-1], lb, ub
            )
            if calls >= maxfun:
                simplex[-1], values[-1] = reflected, f_reflected
                order = np.argsort(values)
                simplex = np.take(simplex, order, 0)
                values = np.take(values, order, 0)
                break
            f_expanded = evaluate(expanded)
            if f_expanded < f_reflected:
                simplex[-1], values[-1] = expanded, f_expanded
            else:
                simplex[-1], values[-1] = reflected, f_reflected
        elif f_reflected < values[-2]:
            simplex[-1], values[-1] = reflected, f_reflected
        elif f_reflected < values[-1]:
            # Outside contraction.
            contracted = np.clip(
                (1 + psi * rho) * centroid - psi * rho * simplex[-1], lb, ub
            )
            if calls >= maxfun:
                break
            f_contracted = evaluate(contracted)
            if f_contracted <= f_reflected:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                shrink = True
        else:
            # Inside contraction.
            contracted = np.clip((1 - psi) * centroid + psi * simplex[-1], lb, ub)
            if calls >= maxfun:
                break
            f_contracted = evaluate(contracted)
            if f_contracted < values[-1]:
                simplex[-1], values[-1] = contracted, f_contracted
            else:
                shrink = True

        if shrink:
            for j in range(1, n + 1):
                simplex[j] = np.clip(
                    simplex[0] + sigma * (simplex[j] - simplex[0]), lb, ub
                )
                if calls >= maxfun:
                    break
                values[j] = evaluate(simplex[j])

        order = np.argsort(values)
        simplex, values = np.take(simplex, order, 0), np.take(values, order, 0)

    return simplex[0], float(values[0])


def make_nelder_mead(
    *,
    simplex_factor: float = 0.05,
    adaptive: bool = True,
    rho: float | None = None,
    chi: float | None = None,
    psi: float | None = None,
    sigma: float | None = None,
    eps_x: float = EPS_X,
) -> Runner:
    """Build a Nelder-Mead runner that restarts on loss of search capability."""

    def local_start(
        objective: BudgetedObjective,
        x0: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        maxfun: int,
        seed: int,
    ) -> None:
        del seed  # Nelder-Mead is deterministic given its starting simplex.
        minimize_neldermead(
            objective,
            initial_simplex(x0, lb, ub, simplex_factor),
            lb,
            ub,
            maxfun=int(maxfun),
            eps_x=float(eps_x),
            **resolve_coefficients(
                len(x0), adaptive=adaptive, rho=rho, chi=chi, psi=psi, sigma=sigma
            ),
        )

    def runner(
        objective: BudgetedObjective,
        dim: int,
        seed: int,
        budget: int,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
    ) -> RestartReport:
        del budget  # The cap lives on the objective, not on the loop.
        return multistart(local_start, objective, dim, seed, lb, ub)

    return runner
