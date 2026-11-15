"""Choosing one configuration per algorithm, for the two roles the study compares."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from aas_shortcut.define_algorithm_portfolio import ALGORITHMS

LOGGER = logging.getLogger(__name__)

# Objective the no-selection side is chosen by: each member as generally good as
# its grid allows.
GENERALIST = "generalist"

# Objective the selection side is chosen by: the six members covering the
# functions between them, as well as any binding can.
ORACLE = "oracle"

# Tie-break weight, as a fraction of ``ConfigurationGrid.score_step``. The grid
# lists a member's default first, so a penalty increasing along the grid keeps the
# published configuration when nothing beats it, without changing which binding wins.
TIE_BREAK = 1e-3


@dataclass(frozen=True)
class ConfigurationGrid:
    """The measured grid as a matrix, with the algorithm each row belongs to.

    ``scores`` holds each configuration's mean superiority against the whole grid on
    each function: 1 beats every other run of the grid.
    """

    algorithms: tuple[str, ...]
    config_ids: tuple[str, ...]
    functions: tuple[int, ...]
    scores: np.ndarray
    n_trials: int

    def __post_init__(self) -> None:
        """Reject a grid whose parts do not describe one matrix."""
        rows, columns = self.scores.shape
        if len(self.algorithms) != rows or len(self.config_ids) != rows:
            raise ValueError(
                f"the matrix has {rows} configuration(s) but "
                f"{len(self.algorithms)} algorithm label(s) and "
                f"{len(self.config_ids)} identifier(s)"
            )
        if len(self.functions) != columns:
            raise ValueError(
                f"the matrix has {columns} function(s) but "
                f"{len(self.functions)} label(s)"
            )
        if not np.isfinite(self.scores).all():
            raise ValueError("the score matrix holds a non-finite entry")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be positive, got {self.n_trials}")

    @property
    def score_step(self) -> float:
        """The smallest nonzero difference two scores of this grid can have.

        A score counts half-wins over ``T x T`` trial pairs against each of the other
        ``K - 1`` configurations, so it moves in steps of ``1 / (2 T^2 (K - 1))``.
        """
        pool = max(len(self.config_ids) - 1, 1)
        return 1.0 / (2.0 * self.n_trials**2 * pool)

    @property
    def members(self) -> tuple[str, ...]:
        """The members present, in portfolio order."""
        present = set(self.algorithms)
        return tuple(name for name in ALGORITHMS if name in present)

    def rows_of(self, algorithm: str) -> np.ndarray:
        """Return the row indices belonging to one member."""
        return np.flatnonzero(np.asarray(self.algorithms) == algorithm)

    def row_of(self, algorithm: str, config_id: str) -> int:
        """Return the row index of one configuration."""
        for index in self.rows_of(algorithm):
            if self.config_ids[index] == config_id:
                return int(index)
        raise KeyError(f"{algorithm} has no configuration {config_id!r} in this grid")


@dataclass(frozen=True)
class Selection:
    """One role's chosen binding, and what says it is the right one."""

    objective: str
    chosen: dict[str, str]
    value: float
    bound: float
    optimal: bool

    @property
    def gap(self) -> float:
        """How far the value stands below the bound."""
        return self.bound - self.value

    def describe(self) -> str:
        """One line naming the value and what certifies it."""
        certificate = (
            "proved optimal" if self.optimal else f"gap {self.gap:.4f} above the bound"
        )
        return (
            f"{self.objective}: average mean superiority {self.value:.4f} "
            f"(unconstrained bound {self.bound:.4f}; {certificate})"
        )


def build_grid(
    scores: pd.DataFrame, column: str = "mean_superiority"
) -> ConfigurationGrid:
    """Pivot the long score table into the matrix the two rules read."""
    if scores.empty:
        raise ValueError("the score table is empty; there is nothing to choose from")
    if "budget" in scores.columns and len(set(scores["budget"])) > 1:
        held = sorted(int(value) for value in scores["budget"].unique())
        raise ValueError(
            f"the score table holds budgets {held}; narrow it to one before "
            "choosing, or the two would be mixed into one ranking"
        )
    wide = scores.pivot_table(
        index=["algorithm", "config_id"], columns="function_id", values=column
    )
    if wide.isna().to_numpy().any():
        missing = int(wide.isna().to_numpy().sum())
        raise ValueError(
            f"{missing} (configuration, function) cell(s) are absent from the score "
            "table; every configuration must be measured on every function for the "
            "scores to be comparable"
        )
    # Portfolio order, then the grid's own order within a member, so that the
    # tie-break below prefers the default.
    order = {name: position for position, name in enumerate(ALGORITHMS)}
    positions = {
        (str(algorithm), str(config_id)): index
        for index, (algorithm, config_id) in enumerate(
            scores[["algorithm", "config_id"]].drop_duplicates().itertuples(index=False)
        )
    }
    keys = sorted(
        wide.index, key=lambda key: (order.get(str(key[0]), len(order)), positions[key])
    )
    wide = wide.loc[keys]
    n_trials = int(scores["n_trials"].iloc[0]) if "n_trials" in scores.columns else 1
    return ConfigurationGrid(
        algorithms=tuple(str(key[0]) for key in keys),
        config_ids=tuple(str(key[1]) for key in keys),
        functions=tuple(int(value) for value in wide.columns),
        scores=wide.to_numpy(dtype=np.float64),
        n_trials=n_trials,
    )


def _tie_break(grid: ConfigurationGrid) -> np.ndarray:
    """Return the per-configuration penalty that resolves ties toward the default.

    In units of ``score_step``: summed over every member and every function it
    stays below one step, so it can separate equal bindings and never reorder
    unequal ones.
    """
    n_configurations = len(grid.config_ids)
    if n_configurations < 2:
        return np.zeros(n_configurations)
    within = np.zeros(n_configurations)
    for algorithm in grid.members:
        rows = grid.rows_of(algorithm)
        within[rows] = np.arange(len(rows)) / max(len(rows) - 1, 1)
    return within * TIE_BREAK / len(grid.members)


def unconstrained_bound(grid: ConfigurationGrid) -> float:
    """Return the oracle's value when every function may use any configuration."""
    return float(grid.scores.max(axis=0).mean())


def generalist_bound(grid: ConfigurationGrid) -> float:
    """Return the best mean score any single configuration in the grid reaches."""
    return float(grid.scores.mean(axis=1).max())


# Both bounds above drop the one-configuration-per-member constraint; only the
# oracle's is not attainable.


def oracle_value(grid: ConfigurationGrid, chosen: Mapping[str, str]) -> float:
    """Return the oracle's mean score under one binding."""
    rows = [grid.row_of(name, chosen[name]) for name in grid.members]
    return float(grid.scores[rows].max(axis=0).mean())


def select_generalists(grid: ConfigurationGrid) -> Selection:
    """Choose, for each member, the configuration that is best over all functions."""
    penalty = _tie_break(grid) * grid.score_step
    means = grid.scores.mean(axis=1)
    chosen: dict[str, str] = {}
    for name in grid.members:
        rows = grid.rows_of(name)
        chosen[name] = grid.config_ids[
            int(rows[np.argmax(means[rows] - penalty[rows])])
        ]
    value = max(
        float(means[grid.row_of(name, config_id)]) for name, config_id in chosen.items()
    )
    return Selection(
        objective=GENERALIST,
        chosen=chosen,
        value=value,
        bound=generalist_bound(grid),
        optimal=True,
    )


def _program(
    grid: ConfigurationGrid,
) -> tuple[LinearConstraint, np.ndarray, int]:
    """Build the integer program's constraints, integrality and variable count."""
    n_configurations, n_functions = grid.scores.shape
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0

    # One configuration per member.
    for name in grid.members:
        for index in grid.rows_of(name):
            rows.append(row)
            columns.append(int(index))
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1

    # Every function is served by exactly one configuration.
    for function in range(n_functions):
        for index in range(n_configurations):
            rows.append(row)
            columns.append(n_configurations + function * n_configurations + index)
            values.append(1.0)
        lower.append(1.0)
        upper.append(1.0)
        row += 1

    # A function may only be served by a configuration its member actually runs.
    # Written per pair rather than aggregated: the disaggregated form has a tight
    # linear relaxation, which is what keeps the search at the root node.
    for function in range(n_functions):
        for index in range(n_configurations):
            rows.append(row)
            columns.append(n_configurations + function * n_configurations + index)
            values.append(1.0)
            rows.append(row)
            columns.append(int(index))
            values.append(-1.0)
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1

    n_variables = n_configurations + n_functions * n_configurations
    matrix = sparse.coo_matrix(
        (values, (rows, columns)), shape=(row, n_variables)
    ).tocsr()
    integrality = np.concatenate(
        [np.ones(n_configurations), np.zeros(n_functions * n_configurations)]
    )
    # scipy leaves the bounds unannotated, so they read as their float default.
    constraint = LinearConstraint(matrix, cast(Any, lower), cast(Any, upper))
    return constraint, integrality, n_variables


def select_oracle_optimal(grid: ConfigurationGrid) -> Selection:
    """Choose the binding whose oracle is best, over every combination."""
    n_configurations, n_functions = grid.scores.shape
    constraint, integrality, n_variables = _program(grid)
    # In units of one score step, so the tie-break sits below the granularity and
    # the zero gap proves optimality exactly, not within a relative tolerance.
    loss = (1.0 - grid.scores) / grid.score_step
    cost = np.concatenate([_tie_break(grid) * n_functions, loss.T.reshape(-1)])
    result = milp(
        c=cost,
        constraints=constraint,
        integrality=integrality,
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0},
    )
    if result.x is None:
        raise RuntimeError(
            f"the oracle program returned no binding (status {result.status}: "
            f"{result.message}); every binding is feasible, so this is a solver "
            "failure and not an infeasible design"
        )
    chosen = {
        name: grid.config_ids[int(rows[np.argmax(result.x[rows])])]
        for name in grid.members
        for rows in [grid.rows_of(name)]
    }
    LOGGER.info(
        "oracle program: %d variable(s), %d integer, status %s",
        n_variables,
        n_configurations,
        result.status,
    )
    return Selection(
        objective=ORACLE,
        chosen=chosen,
        # Recomputed from the matrix, so the value carries no tie-break.
        value=oracle_value(grid, chosen),
        bound=unconstrained_bound(grid),
        optimal=result.status == 0,
    )


# The rule each objective is decided by.
RULES = {GENERALIST: select_generalists, ORACLE: select_oracle_optimal}


def select_for_budgets(
    scores: pd.DataFrame,
    objectives: Mapping[int, str],
    column: str = "mean_superiority",
) -> dict[int, Selection]:
    """Choose one binding per budget, each by the objective its role is scored on."""
    unknown = sorted(set(objectives.values()) - set(RULES))
    if unknown:
        raise ValueError(
            f"unknown objective(s) {unknown}; expected a subset of {sorted(RULES)}"
        )
    selections: dict[int, Selection] = {}
    for budget, objective in sorted(objectives.items()):
        at_budget = cast(pd.DataFrame, scores[scores["budget"] == budget])
        if at_budget.empty:
            held = sorted(int(value) for value in scores["budget"].unique())
            raise ValueError(
                f"the score table holds no row at budget {budget}; it has {held}"
            )
        selections[budget] = RULES[objective](build_grid(at_budget, column))
    return selections
