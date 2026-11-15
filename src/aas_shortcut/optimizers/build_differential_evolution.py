"""Differential evolution: SHADE and L-SHADE from one parameter block."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from modde import ModularDE
from modde.parameters import Parameters

from aas_shortcut.optimizers.define_optimizer_base import (
    EPS_X,
    BudgetedObjective,
    RestartReport,
    Runner,
    collapsed,
    normalise,
    run_within_budget,
)

# SHADE's memory depth, ``H`` in the paper.
SHADE_MEMORY_SIZE = 6

# ``modde`` options both DE members share.
DE_STRUCTURE: dict[str, object] = {
    "base_sampler": "uniform",
    "mutation_base": "target",
    "mutation_reference": "pbest",
    "crossover": "bin",
    "init_stats": True,
}

# Projection onto the box, the bound rule every portfolio member applies (not
# ``modde``'s default).
DE_BOUND_CORRECTION = "saturate"


def build_parameters(
    dim: int,
    seed: int,
    budget: int,
    lb: npt.ArrayLike,
    ub: npt.ArrayLike,
    *,
    population: int,
    f0: float,
    cr0: float,
    p: float | None,
    memory_size: int = SHADE_MEMORY_SIZE,
    lpsr: bool = False,
) -> Parameters:
    """Build the ``modde`` parameter block both DE members share."""
    return Parameters(
        dim,
        lambda_=int(population),
        pbest_value=None if p is None else float(p),
        memory_size=int(memory_size),
        lpsr=bool(lpsr),
        eigenvalue_crossover=False,
        bound_correction=DE_BOUND_CORRECTION,
        use_archive=True,
        # Scalars broadcast over the population; they only seed the SHADE memory.
        F=np.array([float(f0)]),
        CR=np.array([float(cr0)]),
        adaptation_method_F="shade",
        adaptation_method_CR="shade",
        lb=np.asarray(lb, dtype=np.float64).reshape(-1, 1),
        ub=np.asarray(ub, dtype=np.float64).reshape(-1, 1),
        seed=seed,
        budget=budget,
        **DE_STRUCTURE,
    )


def population_size(population_factor: float, dim: int) -> int:
    """Return the population size, floored at the four a DE mutation needs."""
    return max(4, round(float(population_factor) * int(dim)))


def mutation_points(parameters: Parameters, lb, ub) -> np.ndarray:
    """Return the normalised points a mutation displacement can be built from."""
    blocks = [np.asarray(parameters.population.x, dtype=np.float64).T]
    archive = getattr(parameters, "archive", None)
    archive_x = getattr(archive, "x", None) if archive is not None else None
    if archive_x is not None:
        points = np.asarray(archive_x, dtype=np.float64).T
        if points.ndim == 2 and len(points):
            blocks.append(points)
    return normalise(np.vstack(blocks), lb, ub)


def capability_lost(parameters: Parameters, lb, ub, eps_x: float = EPS_X) -> bool:
    """Whether no mutation can displace an individual by more than ``eps_x``."""
    return collapsed(mutation_points(parameters, lb, ub), eps_x / 2.0)


def scheduled_population(parameters: Parameters, used_budget: int, budget: int) -> int:
    """Return the population size L-SHADE's LPSR schedule calls for."""
    initial = int(parameters.initial_lambda_)
    minimum = int(parameters.min_lambda)
    reduced = round((minimum - initial) / budget * used_budget + initial)
    return max(minimum, min(initial, int(reduced)))


def make_shade(
    *,
    population_factor: float = 18.0,
    f0: float = 0.5,
    cr0: float = 0.5,
    p: float | None = None,
    memory_size: int = SHADE_MEMORY_SIZE,
    lpsr: bool = False,
    eps_x: float = EPS_X,
) -> Runner:
    """Build a SHADE or L-SHADE runner."""

    def runner(
        objective: BudgetedObjective,
        dim: int,
        seed: int,
        budget: int,
        lb: npt.ArrayLike,
        ub: npt.ArrayLike,
    ) -> RestartReport:
        np.random.seed(seed)
        initial = population_size(population_factor, dim)
        episodes = 0

        while objective.remaining > 0:
            before = objective.evaluations
            population = initial
            if lpsr and episodes:
                reference = build_parameters(
                    dim,
                    seed,
                    budget,
                    lb,
                    ub,
                    population=initial,
                    f0=f0,
                    cr0=cr0,
                    p=p,
                    memory_size=memory_size,
                    lpsr=True,
                )
                population = scheduled_population(reference, before, budget)

            parameters = build_parameters(
                dim,
                seed + episodes,
                budget,
                lb,
                ub,
                population=population,
                f0=f0,
                cr0=cr0,
                p=p,
                memory_size=memory_size,
                lpsr=lpsr,
            )
            if lpsr:
                # `Parameters` sets `initial_lambda_` from `population`, which would
                # restart the LPSR line from the reduced size and flatten it.
                parameters.initial_lambda_ = initial
                parameters.used_budget = before

            def episode(parameters: Parameters = parameters) -> None:
                # Construction evaluates the initial population, so it stays inside the
                # budget guard: running out there is an ordinary end of run.
                solver = ModularDE(cast(Any, objective), parameters=parameters)
                while solver.step():
                    if capability_lost(solver.parameters, lb, ub, eps_x):
                        return

            with _quiet_correction_statistic():
                run_within_budget(episode, objective)
            episodes += 1
            if objective.evaluations == before:
                break

        return RestartReport(restarts=max(0, episodes - 1))

    return runner


@contextmanager
def _quiet_correction_statistic() -> Iterator[None]:
    """Silence the one warning ``modde``'s bound-correction statistic emits."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
            module=r"scipy\.spatial\.distance",
        )
        yield
