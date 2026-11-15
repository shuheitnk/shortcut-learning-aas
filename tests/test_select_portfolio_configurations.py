r"""Tests for the two selection rules, against enumeration on small grids.

The oracle rule is an integer program standing in for a search over
:math:`\prod_a |\Theta_a|` bindings; the generalist rule claims that search
decomposes. Grid scores are mean superiorities, so larger is better.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.select_portfolio_configurations import (
    GENERALIST,
    ORACLE,
    ConfigurationGrid,
    Selection,
    _tie_break,
    build_grid,
    generalist_bound,
    oracle_value,
    select_generalists,
    select_oracle_optimal,
    unconstrained_bound,
)

# Three members is enough for the ``max`` to couple them, and small enough that
# every binding can be enumerated.
MEMBERS = ALGORITHMS[:3]


def make_grid(scores_by_member: dict[str, list[list[float]]], n_trials: int = 31):
    """Build a grid from per-member lists of per-function score vectors."""
    algorithms, config_ids, rows = [], [], []
    for name, configurations in scores_by_member.items():
        for index, values in enumerate(configurations):
            algorithms.append(name)
            config_ids.append(f"{name[:4]}-{index}")
            rows.append(values)
    matrix = np.array(rows, dtype=np.float64)
    return ConfigurationGrid(
        algorithms=tuple(algorithms),
        config_ids=tuple(config_ids),
        functions=tuple(range(1, matrix.shape[1] + 1)),
        scores=matrix,
        n_trials=n_trials,
    )


def random_grid(rng, sizes=(4, 3, 5), n_functions=6, n_trials=31):
    """A grid of the shape the rules meet, with scores on the study's own lattice."""
    total = sum(sizes)
    steps = 2 * n_trials**2 * (total - 1)
    values = rng.integers(0, steps + 1, size=(total, n_functions)) / steps
    return make_grid(
        {
            name: values[start : start + size].tolist()
            for name, size, start in zip(
                MEMBERS, sizes, np.cumsum((0, *sizes[:-1])), strict=True
            )
        },
        n_trials=n_trials,
    )


def brute_force_oracle(grid: ConfigurationGrid) -> Selection:
    """Return the best binding by enumeration, for checking the program."""
    per_member = [grid.rows_of(name) for name in grid.members]
    total = int(np.prod([len(rows) for rows in per_member]))
    if total > 200_000:
        raise ValueError(
            f"{total} bindings is too many to enumerate; this is for checking "
            "small grids against the program, not for solving the study's"
        )
    penalty = _tie_break(grid) * grid.score_step
    best_rows, best_score = None, -np.inf
    for combination in itertools.product(*per_member):
        rows = list(combination)
        score = float(grid.scores[rows].max(axis=0).mean()) - float(penalty[rows].sum())
        if score > best_score:
            best_rows, best_score = rows, score
    assert best_rows is not None
    chosen = {
        name: grid.config_ids[row]
        for name, row in zip(grid.members, best_rows, strict=True)
    }
    return Selection(
        objective=ORACLE,
        chosen=chosen,
        value=oracle_value(grid, chosen),
        bound=unconstrained_bound(grid),
        optimal=True,
    )


def generalist_values(grid: ConfigurationGrid) -> dict[str, float]:
    """Return each member's mean score over the functions, at its own best."""
    means = grid.scores.mean(axis=1)
    return {name: float(means[grid.rows_of(name)].max()) for name in grid.members}


def selected_flags(grid: ConfigurationGrid, chosen: dict[str, str]) -> list[bool]:
    """Return, per row of the grid, whether that configuration was chosen."""
    wanted = set(chosen.items())
    return [
        (algorithm, config_id) in wanted
        for algorithm, config_id in zip(grid.algorithms, grid.config_ids, strict=True)
    ]


class TestOracleProgram:
    @pytest.mark.parametrize("seed", range(12))
    def test_it_returns_what_enumerating_every_binding_returns(self, seed):
        """The program returns the same value as enumerating every binding."""
        grid = random_grid(np.random.default_rng(seed))
        assert select_oracle_optimal(grid).value == pytest.approx(
            brute_force_oracle(grid).value
        )

    @pytest.mark.parametrize("seed", range(6))
    def test_no_binding_beats_the_one_it_returns(self, seed):
        """No enumerated binding scores above the returned one."""
        grid = random_grid(np.random.default_rng(seed))
        best = select_oracle_optimal(grid).value
        per_member = [grid.rows_of(name) for name in grid.members]
        for combination in itertools.product(*per_member):
            rows = list(combination)
            assert float(grid.scores[rows].max(axis=0).mean()) <= best + 1e-9

    def test_it_reports_optimality(self):
        assert select_oracle_optimal(random_grid(np.random.default_rng(0))).optimal

    def test_it_takes_a_specialist_over_a_uniformly_better_configuration(self):
        """The only configuration reaching function 2 beats a better-on-average one."""
        grid = make_grid(
            {
                MEMBERS[0]: [[0.1, 0.9], [0.6, 0.5]],
                MEMBERS[1]: [[0.8, 0.2]],
                MEMBERS[2]: [[0.7, 0.1]],
            }
        )
        chosen = select_oracle_optimal(grid).chosen
        assert chosen[MEMBERS[0]].endswith("-0")

    def test_the_generalist_rule_would_have_chosen_the_other_one(self):
        """The generalist rule picks the other configuration on the same grid."""
        grid = make_grid(
            {
                MEMBERS[0]: [[0.1, 0.9], [0.6, 0.5]],
                MEMBERS[1]: [[0.8, 0.2]],
                MEMBERS[2]: [[0.7, 0.1]],
            }
        )
        assert select_generalists(grid).chosen[MEMBERS[0]].endswith("-1")

    def test_a_member_whose_configurations_all_tie_does_not_break_it(self):
        """A member whose configurations all tie still gets a valid optimum."""
        grid = make_grid(
            {
                MEMBERS[0]: [[0.9, 0.9], [0.9, 0.9]],
                MEMBERS[1]: [[0.8, 0.7]],
                MEMBERS[2]: [[0.7, 0.8]],
            }
        )
        assert select_oracle_optimal(grid).value == pytest.approx(0.9)

    def test_one_configuration_per_member_is_returned(self):
        grid = random_grid(np.random.default_rng(3))
        assert set(select_oracle_optimal(grid).chosen) == set(grid.members)


class TestSeparability:
    @pytest.mark.parametrize("seed", range(8))
    def test_the_generalist_rule_solves_the_joint_problem(self, seed):
        """Each member's own best equals the best over every combination."""
        grid = random_grid(np.random.default_rng(seed))
        chosen = select_generalists(grid).chosen
        means = grid.scores.mean(axis=1)
        joint = max(
            max(float(means[row]) for row in combination)
            for combination in itertools.product(
                *[grid.rows_of(name) for name in grid.members]
            )
        )
        assert max(
            float(means[grid.row_of(name, config_id)])
            for name, config_id in chosen.items()
        ) == pytest.approx(joint)

    def test_each_member_keeps_its_own_best_whatever_the_others_do(self):
        grid = random_grid(np.random.default_rng(11))
        chosen = select_generalists(grid).chosen
        means = grid.scores.mean(axis=1)
        for name in grid.members:
            rows = grid.rows_of(name)
            assert float(means[grid.row_of(name, chosen[name])]) == pytest.approx(
                float(means[rows].max())
            )


class TestBound:
    @pytest.mark.parametrize("seed", range(8))
    def test_it_is_above_the_optimum(self, seed):
        """The bound is at or above the optimum (it drops a constraint)."""
        grid = random_grid(np.random.default_rng(seed))
        assert unconstrained_bound(grid) >= select_oracle_optimal(grid).value - 1e-9

    def test_it_is_reached_when_one_member_per_function_suffices(self):
        """The bound is reached when each function has its own specialist."""
        grid = make_grid(
            {
                MEMBERS[0]: [[0.9, 0.5, 0.5]],
                MEMBERS[1]: [[0.5, 0.9, 0.5]],
                MEMBERS[2]: [[0.5, 0.5, 0.9]],
            }
        )
        selection = select_oracle_optimal(grid)
        assert selection.gap == pytest.approx(0.0)

    def test_the_gap_is_the_price_of_one_configuration_per_member(self):
        """Two functions wanting opposite configs of one member open a gap."""
        grid = make_grid(
            {
                MEMBERS[0]: [[0.9, 0.4], [0.4, 0.9]],
                MEMBERS[1]: [[0.5, 0.5]],
                MEMBERS[2]: [[0.5, 0.5]],
            }
        )
        selection = select_oracle_optimal(grid)
        assert unconstrained_bound(grid) == pytest.approx(0.9)
        assert selection.value == pytest.approx(0.7)
        assert selection.gap == pytest.approx(0.2)

    def test_the_generalist_selection_is_its_own_bound(self):
        """The generalist selection has zero gap to its own bound."""
        grid = random_grid(np.random.default_rng(5))
        selection = select_generalists(grid)
        assert selection.gap == pytest.approx(0.0)
        assert selection.bound == pytest.approx(generalist_bound(grid))
        assert selection.optimal

    def test_the_two_objectives_are_bounded_by_different_relaxations(self):
        """The oracle and generalist bounds differ on the same grid.

        The oracle's takes the best configuration per function; the generalist's the
        best single configuration over all functions.
        """
        grid = make_grid(
            {
                MEMBERS[0]: [[0.9, 0.1]],
                MEMBERS[1]: [[0.1, 0.9]],
                MEMBERS[2]: [[0.5, 0.5]],
            }
        )
        assert unconstrained_bound(grid) == pytest.approx(0.9)
        assert generalist_bound(grid) == pytest.approx(0.5)

    def test_the_generalist_bound_is_reached_by_a_real_binding(self):
        """The generalist bound is attained by a real binding."""
        grid = random_grid(np.random.default_rng(9))
        assert select_generalists(grid).value == pytest.approx(generalist_bound(grid))


class TestTieBreak:
    def test_an_unimproved_member_keeps_the_configuration_listed_first(self):
        """On a tie the first-listed configuration (the default) is kept."""
        grid = make_grid(
            {
                MEMBERS[0]: [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8]],
                MEMBERS[1]: [[0.7, 0.7]],
                MEMBERS[2]: [[0.7, 0.7]],
            }
        )
        assert select_generalists(grid).chosen[MEMBERS[0]].endswith("-0")
        assert select_oracle_optimal(grid).chosen[MEMBERS[0]].endswith("-0")

    def test_it_cannot_change_which_binding_wins(self):
        """The tie-break never overrides a one-step score difference."""
        # The smallest score step: one half-win against one of three others,
        # over the 31 x 31 trial pairs.
        step = 1.0 / (2 * 31**2 * 3)
        grid = make_grid(
            {
                MEMBERS[0]: [[0.8, 0.8], [0.8 + step, 0.8]],
                MEMBERS[1]: [[0.7, 0.7]],
                MEMBERS[2]: [[0.7, 0.7]],
            }
        )
        assert grid.score_step == pytest.approx(step)
        assert select_generalists(grid).chosen[MEMBERS[0]].endswith("-1")
        assert select_oracle_optimal(grid).chosen[MEMBERS[0]].endswith("-1")


class TestBuildGrid:
    def make_table(self, n_trials=31, budget=10000):
        return pd.DataFrame(
            [
                {
                    "budget": budget,
                    "algorithm": name,
                    "config_id": f"{name[:4]}-{index}",
                    "function_id": function,
                    "mean_superiority": 0.1 * (1 + index + function),
                    "n_trials": n_trials,
                }
                for name in MEMBERS
                for index in range(2)
                for function in (1, 2, 3)
            ]
        )

    def test_the_matrix_matches_the_table(self):
        grid = build_grid(self.make_table())
        assert grid.scores.shape == (6, 3)
        assert grid.functions == (1, 2, 3)

    def test_the_members_come_out_in_portfolio_order(self):
        grid = build_grid(self.make_table())
        assert grid.members == MEMBERS

    def test_a_missing_cell_is_rejected(self):
        """A missing score cell is rejected."""
        table = self.make_table()
        with pytest.raises(ValueError, match="absent from the score table"):
            build_grid(table.iloc[1:])

    def test_two_budgets_at_once_are_rejected(self):
        """A table with two budgets is rejected, as their scores are not comparable."""
        table = pd.concat([self.make_table(), self.make_table(budget=10500)])
        with pytest.raises(ValueError, match="narrow it to one"):
            build_grid(table)

    def test_an_empty_table_is_rejected(self):
        with pytest.raises(ValueError, match="nothing to choose from"):
            build_grid(self.make_table().iloc[:0])


class TestReporting:
    def test_the_flags_mark_exactly_the_chosen_rows(self):
        grid = random_grid(np.random.default_rng(2))
        selection = select_oracle_optimal(grid)
        flags = selected_flags(grid, selection.chosen)
        assert sum(flags) == len(grid.members)

    def test_the_generalist_value_is_the_best_members(self):
        grid = random_grid(np.random.default_rng(4))
        selection = select_generalists(grid)
        assert selection.value == pytest.approx(max(generalist_values(grid).values()))

    def test_the_oracle_value_reads_the_binding_it_is_given(self):
        grid = make_grid(
            {
                MEMBERS[0]: [[0.9, 0.1]],
                MEMBERS[1]: [[0.1, 0.9]],
                MEMBERS[2]: [[0.5, 0.5]],
            }
        )
        chosen = {name: f"{name[:4]}-0" for name in MEMBERS}
        assert oracle_value(grid, chosen) == pytest.approx(0.9)

    def test_the_description_names_the_bound(self):
        selection = select_oracle_optimal(random_grid(np.random.default_rng(7)))
        assert "bound" in selection.describe()
        assert selection.objective == ORACLE

    def test_the_generalist_description_names_its_objective(self):
        assert (
            select_generalists(random_grid(np.random.default_rng(8))).objective
            == GENERALIST
        )
