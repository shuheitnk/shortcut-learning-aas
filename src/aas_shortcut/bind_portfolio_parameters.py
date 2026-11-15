"""Which parameters the portfolio actually runs with."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_tunable_portfolio import TUNABLE
from aas_shortcut.optimizers.define_optimizer_base import Runner
from aas_shortcut.optimizers.define_tunable_algorithm import config_id
from aas_shortcut.read_write_tables import read_table

LOGGER = logging.getLogger(__name__)

# Tuned against the KS distance.
CONDITION_E2 = "e2"

# Condition a selector's chosen algorithm runs under (search budget).
CONDITION_SELECTION = "selection"

# Condition the single best solver runs under (total budget).
CONDITION_SBS = "sbs"


# Tuning objective of each condition, stamped on the binding so a condition cannot be
# tuned on one objective and recorded under another. An untuned binding records
# ``"default"``, so it has no entry here.
CONDITION_OBJECTIVES: dict[str, str] = {
    CONDITION_E2: "ks",
    CONDITION_SELECTION: "oracle",
    CONDITION_SBS: "generalist",
}


# The configurations a report names per member and role: published and selected.
ROLE_TUNING_CONFIGURATIONS = ("default", "selected")


# One member's assignment as a sorted tuple of ``(name, value)`` pairs.
Assignment = tuple[tuple[str, Any], ...]


def _freeze(params: Mapping[str, Any]) -> Assignment:
    """Return a hashable, order-independent form of one parameter assignment."""
    return tuple(sorted((str(key), value) for key, value in params.items()))


def _thaw(assignment: Assignment) -> dict[str, Any]:
    """Return the mutable form of a frozen assignment."""
    return dict(assignment)


@dataclass(frozen=True)
class PortfolioBinding:
    """A complete parameter assignment for every portfolio member."""

    condition: str
    objective: str
    budget: int | None
    assignments: tuple[tuple[str, Assignment], ...]

    @classmethod
    def from_pair_tuning(
        cls, tuning: pd.DataFrame, budget: int | None = None
    ) -> PortfolioBinding:
        """Build the pairwise binding from the ``tune_pair`` table.

        Two pairs may not select different configurations for one algorithm: a member
        cannot have two identities in the mean-rank and selection tasks.
        """
        selected = _selected_rows(tuning)
        chosen: dict[str, dict[str, Any]] = {}
        for row in selected.itertuples(index=False):
            algorithm = str(row.tuned)
            params = json.loads(str(row.params))
            previous = chosen.get(algorithm)
            if previous is not None and config_id(previous) != config_id(params):
                raise ValueError(
                    f"{algorithm} was tuned to two different configurations "
                    f"({config_id(previous)} and {config_id(params)}); a member "
                    "cannot have two identities in one portfolio"
                )
            chosen[algorithm] = params
        return cls._with(
            CONDITION_E2, CONDITION_OBJECTIVES[CONDITION_E2], budget, chosen
        )

    @classmethod
    def from_role_tuning(
        cls, tuning: pd.DataFrame, budget: int, condition: str, objective: str
    ) -> PortfolioBinding:
        """Build one role's binding from ``tuning_roles``."""
        at_budget = cast(pd.DataFrame, tuning[tuning["budget"] == budget])
        if at_budget.empty:
            available = sorted(int(value) for value in tuning["budget"].unique())
            raise ValueError(
                f"the role tuning holds no row at budget {budget}; it has {available}"
            )
        return cls._with(condition, objective, budget, _chosen_at(at_budget))

    @classmethod
    def _with(
        cls,
        condition: str,
        objective: str,
        budget: int | None,
        chosen: Mapping[str, Mapping[str, Any]],
    ) -> PortfolioBinding:
        """Overlay a partial selection onto the defaults.

        Unknown members or parameters raise, rather than leaving a member silently at
        its default.
        """
        unknown = sorted(set(chosen) - set(TUNABLE))
        if unknown:
            raise ValueError(
                f"the tuning selected configuration(s) for {unknown}, which "
                f"are not portfolio members; expected a subset of {sorted(TUNABLE)}"
            )
        assignments = []
        for name, entry in sorted(TUNABLE.items()):
            # `resolve` fills missing parameters from the defaults, so a grid that has
            # lost an axis still yields a complete assignment.
            assignments.append((name, _freeze(entry.resolve(chosen.get(name)))))
        return cls(
            condition=condition,
            objective=objective,
            budget=budget,
            assignments=tuple(assignments),
        )

    def params_for(self, algorithm: str) -> dict[str, Any]:
        """Return the complete assignment of one member."""
        for name, assignment in self.assignments:
            if name == algorithm:
                return _thaw(assignment)
        raise KeyError(
            f"{algorithm!r} is not bound by the {self.condition!r} portfolio"
        )

    def runner(self, algorithm: str) -> Runner:
        """Build the solver entry point of one member under this binding."""
        return TUNABLE[algorithm].runner(self.params_for(algorithm))

    def config_id_for(self, algorithm: str) -> str:
        """Return the identifier of one member's assignment."""
        return config_id(self.params_for(algorithm))

    def is_default(self, algorithm: str) -> bool:
        """Whether one member is at its published configuration."""
        return self.config_id_for(algorithm) == TUNABLE[algorithm].default_config_id

    @property
    def tuned_algorithms(self) -> tuple[str, ...]:
        """Members whose assignment differs from the published one."""
        return tuple(name for name, _ in self.assignments if not self.is_default(name))

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description, including every parameter value."""
        return {
            "condition": self.condition,
            "objective": self.objective,
            "budget": self.budget,
            "params": {
                name: _thaw(assignment) for name, assignment in self.assignments
            },
        }

    def describe(self) -> str:
        """One log line naming what moved off its default."""
        tuned = self.tuned_algorithms
        if not tuned:
            return f"portfolio {self.condition!r}: every member at its default"
        moved = ", ".join(f"{name}={self.config_id_for(name)}" for name in tuned)
        return (
            f"portfolio {self.condition!r} ({self.objective}, budget "
            f"{self.budget}): {len(tuned)} member(s) moved off the default: {moved}"
        )


def _chosen_at(tuning: pd.DataFrame) -> dict[str, Any]:
    """Return ``algorithm -> assignment`` for the configurations marked selected."""
    return {
        str(row.algorithm): json.loads(str(row.params))
        for row in _selected_rows(tuning).itertuples(index=False)
    }


def _selected_rows(tuning: pd.DataFrame) -> pd.DataFrame:
    """Return one row per selected configuration."""
    if "is_selected" not in tuning.columns:
        raise ValueError(
            "the tuning table has no 'is_selected' column; it predates the "
            "binding and cannot be applied"
        )
    selected = tuning[tuning["is_selected"].astype(bool)]
    if selected.empty:
        raise ValueError("the tuning table marks no configuration as selected")
    key = [
        column
        for column in ("budget", "pair", "tuned", "algorithm", "config_id")
        if column in selected.columns
    ]
    return selected.drop_duplicates(subset=key)


def read_pair_binding(run_paths: RunPaths, budget: int, step: str) -> PortfolioBinding:
    """Read a run's pair tuning and resolve the binding its benchmark runs under."""
    shard = run_paths.metric("tuning_pairs")
    if not shard.exists():
        raise FileNotFoundError(
            f"{shard.data} does not exist, so there is no tuned portfolio to "
            f"benchmark under; run step {step} first"
        )
    binding = PortfolioBinding.from_pair_tuning(read_table(shard.data), budget)
    LOGGER.info("%s: %s", step, binding.describe())
    return binding


def read_role_bindings(
    run_paths: RunPaths, search_budget: int, total_budget: int
) -> dict[str, PortfolioBinding]:
    """Read the role tuning and resolve the two bindings the selection compares.

    The selection side is at the search budget, chosen on the oracle; the
    no-selection side at the total budget, chosen on the best single member.
    """
    shard = run_paths.metric("tuning_roles")
    if not shard.exists():
        raise FileNotFoundError(
            f"{shard.data} does not exist, so there is no tuned portfolio to select "
            f"among; run the tuning step first"
        )
    table = read_table(shard.data)
    resolved = {
        condition: PortfolioBinding.from_role_tuning(
            table, budget, condition, CONDITION_OBJECTIVES[condition]
        )
        for condition, budget in (
            (CONDITION_SELECTION, search_budget),
            (CONDITION_SBS, total_budget),
        )
    }
    for condition in (CONDITION_SELECTION, CONDITION_SBS):
        LOGGER.info("%s: %s", condition, resolved[condition].describe())
    return resolved


def summarize_role_tuning(
    table: pd.DataFrame, budgets: Mapping[str, int]
) -> pd.DataFrame:
    """Return each member's default and selected configuration under each condition.

    ``mean_superiority`` is the mean over functions of the configuration's mean
    superiority against the whole grid, the quantity both objectives aggregate.
    """
    keys = ["algorithm", "config_id", "params", "is_default", "is_selected"]
    rows: list[dict[str, Any]] = []
    for condition, budget in budgets.items():
        at_budget = table[table["budget"] == budget]
        if at_budget.empty:
            raise ValueError(f"the role tuning holds no row at budget {budget}")
        configurations = (
            at_budget.groupby(keys, sort=False, observed=True)
            .agg(
                mean_superiority=("mean_superiority", "mean"),
                n_functions=("function_id", "nunique"),
            )
            .reset_index()
        )
        for algorithm in ALGORITHMS:
            member = cast(
                pd.DataFrame, configurations[configurations["algorithm"] == algorithm]
            )
            for name, flag in zip(
                ROLE_TUNING_CONFIGURATIONS, ("is_default", "is_selected"), strict=True
            ):
                found = cast(pd.DataFrame, member[member[flag]])
                if len(found) != 1:
                    raise ValueError(
                        f"{condition}, {algorithm}: {len(found)} {name} "
                        "configuration(s)"
                    )
                row = found.iloc[0]
                rows.append(
                    {
                        "condition": condition,
                        "objective": CONDITION_OBJECTIVES[condition],
                        "budget": int(budget),
                        "algorithm": algorithm,
                        "configuration": name,
                        "config_id": str(row["config_id"]),
                        "params": str(row["params"]),
                        "mean_superiority": float(row["mean_superiority"]),
                        "n_functions": int(row["n_functions"]),
                        "n_candidates": len(member),
                    }
                )
    return pd.DataFrame(rows)
