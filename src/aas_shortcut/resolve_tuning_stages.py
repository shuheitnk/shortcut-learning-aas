"""Resolving the two tunings, the bindings they choose, and the pair calibration.

A binding is not a unit: stages run under it carry its parameters in their own
settings, and the tuning that chose them is recorded as provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from aas_shortcut.benchmark_configuration_grid import (
    GridBenchmarkConfig,
    run_grid_benchmark_stage,
)
from aas_shortcut.bind_portfolio_parameters import (
    PortfolioBinding,
    read_pair_binding,
    read_role_bindings,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_tunable_portfolio import TUNABLE
from aas_shortcut.measure_superiority_calibration import (
    CalibrationConfig,
    run_calibration_stage,
)
from aas_shortcut.rank_configuration_grid import (
    pooled_mean_superiorities,
    run_grid_rank_stage,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit
from aas_shortcut.select_portfolio_configurations import (
    GENERALIST,
    ORACLE,
    select_for_budgets,
)
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS
from aas_shortcut.tune_algorithms import ESTIMATOR, run_tune_pair_stage

LOGGER = logging.getLogger(__name__)


# Where each stage writes its table inside a unit.
PAIR_TABLE = "metrics/tuning_pairs.parquet"
GRID_TABLE = "metrics/grid_errors.parquet"
ROLE_TABLE = "metrics/tuning_roles.parquet"
CALIBRATION_TABLES = (
    "metrics/superiority_calibration.parquet",
    "metrics/superiority_calibration_deciles.parquet",
)


def tuning_theta() -> dict[str, Any]:
    """Return every tunable member's declared grid, as both tunings fingerprint it."""
    return {name: entry.as_dict() for name, entry in TUNABLE.items()}


def pair_tuning_config(
    context: ExperimentContext, theta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return the pair tuning's fingerprintable configuration."""
    return {
        "stage": "tune_pair",
        "objective": "ks",
        # Not in `conditions`, which describe the runs and are shared with
        # `grid_errors`.
        "estimator": ESTIMATOR,
        # Version of the table's column names, so renaming a column changes the
        # fingerprint.
        "names": 2,
        "conditions": context.tuning_conditions().as_dict(),
        "pairs": [pair.slug for pair in FOCUS_PAIRS],
        "theta": theta if theta is not None else tuning_theta(),
    }


@dataclass(frozen=True)
class RoleTuning:
    """Experiments 3 and 5's tuning: one grid, and the two roles chosen from it."""

    grid: GridBenchmarkConfig
    # Budget -> objective: each side is tuned for the quantity it is scored on (the
    # oracle at the search budget, the best single member at the total budget).
    objectives: dict[int, str]

    @property
    def grid_config(self) -> dict[str, Any]:
        """Return the grid benchmark's fingerprintable configuration."""
        return self.grid.as_dict()

    @property
    def role_config(self) -> dict[str, Any]:
        """Return the role selection's fingerprintable configuration."""
        return {
            "stage": "tuning_roles",
            "objectives": {
                str(budget): objective
                for budget, objective in sorted(self.objectives.items())
            },
            # Names the score both objectives aggregate, so a tuning on another score
            # gets another fingerprint.
            "score": {"kind": "mean-superiority", "pool": "grid"},
            "grid": self.grid_config,
        }


def role_tuning(
    context: ExperimentContext, theta: dict[str, Any] | None = None
) -> RoleTuning:
    """Build the role tuning's grid and objectives from the context."""
    reference_conditions = context.tuning_conditions(
        context.fe_budget_factor + context.sampling_factor
    )
    return RoleTuning(
        grid=GridBenchmarkConfig(
            by_budget={
                context.search_budget: context.tuning_conditions(),
                context.total_budget: reference_conditions,
            },
            n_jobs=context.n_jobs,
            save_every=context.save_every,
            theta=theta if theta is not None else tuning_theta(),
        ),
        objectives={
            context.search_budget: ORACLE,
            context.total_budget: GENERALIST,
        },
    )


def write_tuning_roles(
    grid_errors: pd.DataFrame,
    tuning: RoleTuning,
    paths: RunPaths,
    scale: str,
    config_hash: str,
    dim: int,
) -> pd.DataFrame:
    """Choose both roles from the grid, and write the table."""
    scores = pooled_mean_superiorities(grid_errors)
    selections = select_for_budgets(scores, tuning.objectives)
    for budget, selection in sorted(selections.items()):
        LOGGER.info("tuning_roles: budget %d, %s", budget, selection.describe())
    return run_grid_rank_stage(
        scores,
        TUNABLE,
        {budget: selection.chosen for budget, selection in selections.items()},
        paths,
        scale,
        config_hash,
        dim,
    )


def pair_tuning_unit(context: ExperimentContext) -> StageUnit:
    """Return the pair tuning the preanalysis and Experiments 1, 2 and 4 bind from."""
    return StageUnit("tune_pair", pair_tuning_config(context))


def resolve_pair_tuning(store: StageStore, context: ExperimentContext) -> StageArtifact:
    """Return the pair tuning, running it if the store does not hold it."""
    conditions = context.tuning_conditions()

    def compute(paths: Any, config_hash: str) -> None:
        run_tune_pair_stage(
            TUNABLE,
            FOCUS_PAIRS,
            conditions,
            paths,
            context.scale,
            config_hash,
            n_jobs=context.n_jobs,
        )

    return store.resolve(pair_tuning_unit(context), compute)


def binding_provenance(tuning: StageArtifact, step: str) -> dict[str, Any]:
    """Return what a stage run under a binding records about the binding's origin."""
    return {
        "binding": {
            "stage": tuning.unit.stage,
            "fingerprint": tuning.fingerprint,
            "step": step,
        }
    }


def resolve_pair_binding(
    store: StageStore, context: ExperimentContext, step: str
) -> tuple[PortfolioBinding, StageArtifact]:
    """Return the binding the pair benchmark runs under, and the tuning behind it."""
    tuning = resolve_pair_tuning(store, context)
    return read_pair_binding(tuning.run_paths, context.search_budget, step), tuning


def grid_unit(context: ExperimentContext) -> StageUnit:
    """Return the unit of the candidate grid the role tuning chooses from."""
    return StageUnit("grid_errors", role_tuning(context).grid_config)


def role_unit(context: ExperimentContext) -> StageUnit:
    """Return the role selection unit: the AAS and SBS sides' configurations."""
    return StageUnit("tuning_roles", role_tuning(context).role_config)


def resolve_role_tuning(store: StageStore, context: ExperimentContext) -> StageArtifact:
    """Return the role selection, running the grid and the selection if absent."""
    tuning = role_tuning(context)

    def compute_grid(paths: Any, config_hash: str) -> None:
        run_grid_benchmark_stage(
            TUNABLE, tuning.grid, paths, context.scale, config_hash
        )

    grid = store.resolve(grid_unit(context), compute_grid)

    def compute_roles(paths: Any, config_hash: str) -> None:
        write_tuning_roles(
            grid.read_table(GRID_TABLE),
            tuning,
            paths,
            context.scale,
            config_hash,
            context.dim,
        )

    return store.resolve(role_unit(context), compute_roles)


def resolve_role_bindings(
    store: StageStore, context: ExperimentContext
) -> tuple[dict[str, PortfolioBinding], StageArtifact]:
    """Return the selection and SBS bindings, and the role selection behind them."""
    roles = resolve_role_tuning(store, context)
    bindings = read_role_bindings(
        roles.run_paths, context.search_budget, context.total_budget
    )
    return bindings, roles


def calibration_unit(
    context: ExperimentContext, superiority: StageArtifact
) -> tuple[StageUnit, CalibrationConfig]:
    """Return the pair calibration unit, on the untransformed training block."""
    config = CalibrationConfig(problem_class=PROBLEM_CLASS, dim=context.dim)
    unit = StageUnit(
        "superiority_calibration",
        config.as_dict(),
        inputs={"superiority": superiority.fingerprint},
    )
    return unit, config


def resolve_calibration(
    store: StageStore,
    context: ExperimentContext,
    superiority: StageArtifact,
    superiorities: pd.DataFrame,
) -> StageArtifact:
    """Return the calibration of the pair tuning, measuring it if absent.

    ``superiority`` (the untransformed training block) also holds the tuning
    instance, so both blocks the calibration reads are in it.
    """
    unit, config = calibration_unit(context, superiority)

    def compute(paths: Any, config_hash: str) -> None:
        run_calibration_stage(superiorities, paths, context.scale, config_hash, config)

    return store.resolve(unit, compute)
