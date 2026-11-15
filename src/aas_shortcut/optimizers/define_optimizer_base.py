"""Primitives every optimizer in this project is built on."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt

from aas_shortcut.derive_seeds import SEED_MODULUS

LOGGER = logging.getLogger(__name__)


# Displacement below which the search operator counts as collapsed.
EPS_X = 1e-4


@dataclass(frozen=True)
class RestartReport:
    """What one run did about restarting, with the mechanisms kept apart."""

    restarts: int = 0
    multistarts: int = 0
    native_reanneals: int = 0

    def __int__(self) -> int:
        """Return :attr:`restarts`, for callers that want the headline number."""
        return int(self.restarts)

    def as_dict(self) -> dict[str, int]:
        """Render the three counters for a table row."""
        return {
            "restarts": int(self.restarts),
            "multistarts": int(self.multistarts),
            "native_reanneals": int(self.native_reanneals),
        }


class BudgetExhausted(Exception):
    """Raised by :class:`BudgetedObjective` when the budget is used up."""


class _EvaluationState:
    """The ``state`` a solver reads off the objective to check its own budget."""

    def __init__(self, objective: BudgetedObjective) -> None:
        self._objective = objective

    @property
    def evaluations(self) -> int:
        """Evaluations consumed so far, restarts included."""
        return self._objective.evaluations

    @property
    def optimum_found(self) -> bool:
        """Always ``False``, so a solver never stops early on the optimum."""
        return False


def normalise(x, lb, ub) -> np.ndarray:
    """Map raw coordinates into ``[0, 1]^d``."""
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    return (np.asarray(x, dtype=np.float64) - lb) / (ub - lb)


def collapsed(points: np.ndarray, eps_x: float = EPS_X) -> bool:
    """Whether the L2 diameter of ``points`` is at most ``eps_x``."""
    if len(points) < 2:
        return True
    if float(np.max(points.max(axis=0) - points.min(axis=0))) > eps_x:
        return False
    delta = points[:, None, :] - points[None, :, :]
    return bool(np.sqrt((delta**2).sum(-1)).max() <= eps_x)


class BudgetedObjective:
    """Callable wrapper that counts evaluations and caps them at a budget."""

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        budget: int,
        *,
        bounds: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    ) -> None:
        self.func = func
        self.budget = int(budget)
        self.evaluations = 0
        self.exhausted = False
        # Box every evaluated point is projected into, or ``None`` to evaluate whatever
        # the solver proposes.
        self.bounds = (
            None
            if bounds is None
            else (
                np.asarray(bounds[0], dtype=np.float64),
                np.asarray(bounds[1], dtype=np.float64),
            )
        )
        # Proposals that had to be projected.
        self.projections = 0
        # Largest distance by which a proposal fell outside the box.
        self.max_violation = 0.0
        # What a solver reads to check its own budget; see :class:`_EvaluationState`.
        self.state = _EvaluationState(self)

    @property
    def remaining(self) -> int:
        """Evaluations still available."""
        return max(0, self.budget - self.evaluations)

    def __call__(self, x) -> float:
        """Evaluate the wrapped objective, refusing to exceed the budget."""
        if self.evaluations >= self.budget:
            self.exhausted = True
            raise BudgetExhausted(f"budget of {self.budget} evaluation(s) exhausted")
        point = np.asarray(x, dtype=np.float64)
        bounds = self.bounds
        if bounds is not None:
            lower, upper = bounds
            if ((point < lower) | (point > upper)).any():
                projected = np.clip(point, lower, upper)
                self.projections += 1
                self.max_violation = max(
                    self.max_violation, float(np.max(np.abs(point - projected)))
                )
                point = projected
        self.evaluations += 1
        return float(self.func(point))

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped objective.

        Dunder lookups raise, so copy, pickle and ``inspect`` see a normal object.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.func, name)


# The signature every portfolio member's entry point has: ``(objective, dim, seed,
# budget, lb, ub) -> RestartReport``.
Runner = Callable[
    [BudgetedObjective, int, int, int, npt.ArrayLike, npt.ArrayLike], "RestartReport"
]


def run_within_budget(
    solve: Callable[[], object], objective: BudgetedObjective
) -> None:
    """Run ``solve``, treating budget exhaustion as a normal termination."""
    try:
        solve()
    except BudgetExhausted:
        pass
    except Exception:
        if not objective.exhausted:
            raise
        LOGGER.debug("solver reported budget exhaustion through its own exception type")


# Evaluations a local solver must be able to spend for a start to be worth making.
MIN_LOCAL_BUDGET_FACTOR = 3


def multistart(
    local_start: Callable[..., None],
    objective: BudgetedObjective,
    dim: int,
    seed: int,
    lb,
    ub,
) -> RestartReport:
    """Restart a local solver from fresh points until the budget is gone.

    ``restarts`` is one fewer than the starts: the first start initialises the search.
    """
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    min_local_budget = MIN_LOCAL_BUDGET_FACTOR * (dim + 1)

    restarts = 0
    while objective.remaining > 0:
        x0 = rng.uniform(lb, ub, size=dim)
        maxfun = objective.remaining
        restarts += 1
        start_seed = int(rng.integers(0, SEED_MODULUS))

        before = objective.evaluations
        run_within_budget(
            partial(local_start, objective, x0, lb, ub, maxfun, start_seed), objective
        )
        if objective.exhausted:
            break
        if objective.evaluations == before:
            # A start that consumed nothing would loop forever.
            if objective.remaining >= min_local_budget:
                raise RuntimeError(
                    f"local solver consumed no evaluations from x0={x0!r}; "
                    f"remaining budget {objective.remaining}"
                )
            restarts -= 1
            break

    return RestartReport(restarts=max(0, restarts - 1), multistarts=restarts)
