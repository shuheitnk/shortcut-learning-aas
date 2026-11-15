"""Applying the tuning to the portfolio as a fixed binding."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aas_shortcut.bind_portfolio_parameters import (
    CONDITION_SBS,
    CONDITION_SELECTION,
    PortfolioBinding,
    summarize_role_tuning,
)
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_tunable_portfolio import TUNABLE
from aas_shortcut.optimizers.define_tunable_algorithm import config_id

SEARCH_BUDGET = 10_000
TOTAL_BUDGET = 10_500


def pair_table(tuned: str, params: dict) -> pd.DataFrame:
    """A minimal ``tune_pair`` artifact selecting one configuration."""
    rows = [
        {
            "pair": "shade__l-shade",
            "tuned": tuned,
            "config_id": config_id(params),
            "params": json.dumps(params, sort_keys=True),
            "is_selected": selected,
            "function_id": function_id,
        }
        for selected, params in ((True, params),)
        for function_id in (1, 2)
    ]
    return pd.DataFrame(rows)


def role_table(chosen: dict[int, dict[str, dict]]) -> pd.DataFrame:
    """A minimal ``tuning_roles`` artifact: one row per configuration and function."""
    rows = []
    for budget, per_algorithm in chosen.items():
        for algorithm in ALGORITHMS:
            for candidate in TUNABLE[algorithm].candidates()[:2]:
                params = TUNABLE[algorithm].resolve(candidate)
                selected = params == TUNABLE[algorithm].resolve(
                    per_algorithm.get(algorithm)
                )
                for function_id in (1, 2, 3):
                    rows.append(
                        {
                            "budget": budget,
                            "algorithm": algorithm,
                            "config_id": config_id(params),
                            "params": json.dumps(params, sort_keys=True),
                            "is_selected": selected,
                            "function_id": function_id,
                            "mean_superiority": 0.5,
                        }
                    )
    return pd.DataFrame(rows)


class TestRoleTuning:
    """The two role bindings, read back off one table."""

    def test_the_budget_says_which_role_is_read(self):
        moved = TUNABLE["SHADE"].candidates()[1]
        table = role_table({10000: {"SHADE": moved}, 10500: {}})
        oracle = PortfolioBinding.from_role_tuning(table, 10000, "selection", "oracle")
        generalist = PortfolioBinding.from_role_tuning(
            table, 10500, "sbs", "generalist"
        )
        assert oracle.params_for("SHADE") == TUNABLE["SHADE"].resolve(moved)
        assert generalist.params_for("SHADE") == TUNABLE["SHADE"].resolve()

    def test_the_objective_is_stamped_rather_than_looked_up(self):
        """The binding records the objective it is given, not one from the condition."""
        table = role_table({10000: {}})
        binding = PortfolioBinding.from_role_tuning(table, 10000, "selection", "oracle")
        assert binding.objective == "oracle"

    def test_the_repeated_function_rows_collapse_to_one_choice(self):
        """The table carries a row per function; the binding is one per member."""
        table = role_table({10000: {}})
        binding = PortfolioBinding.from_role_tuning(table, 10000, "selection", "oracle")
        assert len(binding.assignments) == len(ALGORITHMS)

    def test_a_budget_the_table_does_not_hold_is_a_clear_error(self):
        table = role_table({10000: {}})
        with pytest.raises(ValueError, match="no row at budget 10500"):
            PortfolioBinding.from_role_tuning(table, 10500, "sbs", "generalist")

    def test_every_member_is_bound_even_when_only_one_moved(self):
        moved = TUNABLE["BOBYQA"].candidates()[1]
        table = role_table({10000: {"BOBYQA": moved}})
        binding = PortfolioBinding.from_role_tuning(table, 10000, "selection", "oracle")
        assert {name for name, _ in binding.assignments} == set(ALGORITHMS)
        assert binding.tuned_algorithms == ("BOBYQA",)


def flagged_role_table(picks: dict[int, dict[str, int]]) -> pd.DataFrame:
    """A ``tuning_roles`` table with three candidates per member, defaults flagged.

    ``picks`` gives, per budget, each member's selected index (default 0). A
    candidate's mean superiority on function ``f`` is ``0.1 * index + 0.01 * f``.
    """
    rows = []
    for budget, chosen in picks.items():
        for algorithm in ALGORITHMS:
            for index, candidate in enumerate(TUNABLE[algorithm].candidates()[:3]):
                params = TUNABLE[algorithm].resolve(candidate)
                for function_id in (1, 2, 3, 4):
                    rows.append(
                        {
                            "budget": budget,
                            "algorithm": algorithm,
                            "config_id": config_id(params),
                            "params": json.dumps(params, sort_keys=True),
                            "is_default": index == 0,
                            "is_selected": index == chosen.get(algorithm, 0),
                            "function_id": function_id,
                            "mean_superiority": 0.1 * index + 0.01 * function_id,
                        }
                    )
    return pd.DataFrame(rows)


BUDGETS = {CONDITION_SELECTION: SEARCH_BUDGET, CONDITION_SBS: TOTAL_BUDGET}


class TestTheRoleTuningSummary:
    """Each member's default and selected configuration, per condition."""

    def test_two_rows_per_member_and_condition_in_portfolio_order(self):
        summary = summarize_role_tuning(
            flagged_role_table({SEARCH_BUDGET: {}, TOTAL_BUDGET: {}}), BUDGETS
        )
        assert len(summary) == 2 * len(ALGORITHMS) * 2
        selection = summary[summary["condition"] == CONDITION_SELECTION]
        assert list(selection["algorithm"]) == [a for a in ALGORITHMS for _ in (0, 1)]
        assert list(selection["configuration"]) == ["default", "selected"] * 6

    def test_each_condition_names_its_objective_and_budget(self):
        summary = summarize_role_tuning(
            flagged_role_table({SEARCH_BUDGET: {}, TOTAL_BUDGET: {}}), BUDGETS
        )
        named = summary.groupby("condition")[["objective", "budget"]].first()
        assert named.loc[CONDITION_SELECTION].tolist() == ["oracle", SEARCH_BUDGET]
        assert named.loc[CONDITION_SBS].tolist() == ["generalist", TOTAL_BUDGET]

    def test_the_selected_configuration_is_the_one_flagged_at_that_budget(self):
        table = flagged_role_table({SEARCH_BUDGET: {"SHADE": 2}, TOTAL_BUDGET: {}})
        summary = summarize_role_tuning(table, BUDGETS).set_index(
            ["condition", "algorithm", "configuration"]
        )
        moved = TUNABLE["SHADE"].resolve(TUNABLE["SHADE"].candidates()[2])
        default = TUNABLE["SHADE"].resolve()
        selected = summary.loc[(CONDITION_SELECTION, "SHADE", "selected")]
        assert selected["config_id"] == config_id(moved)
        assert summary.loc[(CONDITION_SBS, "SHADE", "selected")]["config_id"] == (
            config_id(default)
        )

    def test_the_mean_superiority_is_over_the_functions(self):
        table = flagged_role_table({SEARCH_BUDGET: {"BOBYQA": 1}, TOTAL_BUDGET: {}})
        summary = summarize_role_tuning(table, BUDGETS).set_index(
            ["condition", "algorithm", "configuration"]
        )
        row = summary.loc[(CONDITION_SELECTION, "BOBYQA", "selected")]
        assert row["mean_superiority"] == pytest.approx(0.1 + 0.025)
        assert row["n_functions"] == 4
        assert row["n_candidates"] == 3

    def test_a_budget_the_table_does_not_hold_is_an_error(self):
        with pytest.raises(ValueError, match="no row at budget 10500"):
            summarize_role_tuning(flagged_role_table({SEARCH_BUDGET: {}}), BUDGETS)

    def test_two_selected_configurations_of_one_member_are_an_error(self):
        table = flagged_role_table({SEARCH_BUDGET: {}, TOTAL_BUDGET: {}})
        table.loc[table["algorithm"] == "SHADE", "is_selected"] = True
        with pytest.raises(ValueError, match="SHADE: 3 selected"):
            summarize_role_tuning(table, BUDGETS)


class TestPairTuning:
    def test_the_tuned_member_moves_and_the_others_do_not(self):
        params = {**TUNABLE["SHADE"].resolve(), "population_factor": 5.0}
        binding = PortfolioBinding.from_pair_tuning(pair_table("SHADE", params))

        assert binding.tuned_algorithms == ("SHADE",)
        assert binding.params_for("SHADE")["population_factor"] == 5.0
        assert binding.params_for("L-SHADE") == TUNABLE["L-SHADE"].resolve()

    def test_it_refuses_two_identities_for_one_member(self):
        """A member with two configurations cannot be ranked inside one trial."""
        first = {**TUNABLE["SHADE"].resolve(), "population_factor": 5.0}
        second = {**TUNABLE["SHADE"].resolve(), "population_factor": 30.0}
        table = pd.concat(
            [pair_table("SHADE", first), pair_table("SHADE", second)],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="two different configurations"):
            PortfolioBinding.from_pair_tuning(table)

    def test_an_untuned_table_is_rejected(self):
        table = pair_table("SHADE", TUNABLE["SHADE"].resolve())
        with pytest.raises(ValueError, match="marks no configuration"):
            PortfolioBinding.from_pair_tuning(table.assign(is_selected=False))
