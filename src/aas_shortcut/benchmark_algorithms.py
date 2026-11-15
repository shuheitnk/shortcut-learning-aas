"""Stage ``performance``: run the portfolio on every instance."""

from __future__ import annotations

import logging
import time

import pandas as pd

from aas_shortcut.bind_portfolio_parameters import (
    PortfolioBinding,
)
from aas_shortcut.create_problem import create_problem
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS, algorithm_seed
from aas_shortcut.define_paths import validate_transform
from aas_shortcut.optimizers.define_optimizer_base import BudgetedObjective
from aas_shortcut.sample_monotone_transform import (
    DEFAULT_WARP,
    WarpConfig,
    transform_for_instance,
)
from aas_shortcut.transform_objective import FuncTransform

LOGGER = logging.getLogger(__name__)


class BudgetViolation(RuntimeError):
    """Raised when a run consumed more evaluations than its budget."""


def trial_seed(
    base_seed: int,
    algorithm: str,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    trial: int,
) -> int:
    """Derive the seed of one algorithm run."""
    return algorithm_seed(
        base_seed, algorithm, problem_class, function_id, instance_id, dim, trial
    )


def _solve(
    algorithm: str,
    binding: PortfolioBinding,
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    trial: int,
    budget: int,
    transform: str = "pre",
    warp: WarpConfig = DEFAULT_WARP,
) -> dict:
    """Run one algorithm once, under one binding, and return its row."""
    validate_transform(transform)
    problem = create_problem(function_id, instance_id, dim)
    optimum_y = float(problem.optimum.y)
    lb, ub = problem.bounds.lb, problem.bounds.ub

    # The warp calls straight through to the raw problem, so evaluations and
    # `best_y` are counted on the raw objective whichever view is optimized.
    view = problem
    if transform == "post":
        view = FuncTransform(
            problem,
            transform_for_instance(
                base_seed, problem_class, function_id, instance_id, dim, warp
            ),
        )

    objective = BudgetedObjective(
        view,
        budget,
        # No evaluation outside the domain, whatever the solver proposes.
        bounds=(lb, ub),
    )
    seed = trial_seed(
        base_seed, algorithm, problem_class, function_id, instance_id, dim, trial
    )

    started = time.perf_counter()
    report = binding.runner(algorithm)(objective, dim, seed, budget, lb, ub)
    elapsed = time.perf_counter() - started

    evaluations_used = int(problem.state.evaluations)
    # Using fewer evaluations than the budget is a violation too.
    if evaluations_used != budget:
        raise BudgetViolation(
            f"{algorithm} used {evaluations_used} evaluation(s) on "
            f"{problem_class} f{function_id} i{instance_id} trial {trial} "
            f"with a budget of {budget}"
        )

    identity = {
        "problem_class": problem_class,
        "function_id": function_id,
        "instance_id": instance_id,
        "dimension": dim,
        "algorithm": algorithm,
        "trial": trial,
        "seed": seed,
        "optimum_y": optimum_y,
    }
    row = {
        **identity,
        "budget": budget,
        "evaluations_used": evaluations_used,
        **report.as_dict(),
        "best_y": float(problem.state.current_best.y),
    }
    LOGGER.debug(
        "%s f%02d i%03d trial %02d | %d evals | %d restart(s) "
        "(%d multistart, %d native re-anneal) | %.2fs",
        algorithm,
        function_id,
        instance_id,
        trial,
        evaluations_used,
        report.restarts,
        report.multistarts,
        report.native_reanneals,
        elapsed,
    )

    return row


def run_job(
    binding: PortfolioBinding,
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    trial: int,
    budget: int,
    algorithms: tuple[str, ...] = ALGORITHMS,
    transform: str = "pre",
    warp: WarpConfig = DEFAULT_WARP,
) -> list[dict]:
    """Run the given algorithms on one job and return long-format rows."""
    return [
        _solve(
            algorithm,
            binding,
            base_seed,
            problem_class,
            function_id,
            instance_id,
            dim,
            trial,
            budget,
            transform,
            warp,
        )
        for algorithm in algorithms
    ]


def select_budget(performance: pd.DataFrame, budget: int | None) -> pd.DataFrame:
    """Restrict a performance table to the runs observed at one budget."""
    if budget is None:
        return performance
    selected = performance[performance["budget"] == budget]
    if selected.empty:
        available = sorted(performance["budget"].unique())
        raise ValueError(
            f"the performance table holds no run at budget {budget}; it has {available}"
        )
    return selected
