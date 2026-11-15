"""The pooled mean-superiority reduction over the tuning grid.

The oracle's ``max`` over two members is meaningful only because both are scored
in one pool, so the tests cover the pool as well as the arithmetic.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.rank_configuration_grid import (
    HALVES,
    annotate,
    pooled_mean_superiorities,
)


def make_errors(
    values: dict[tuple[str, str], dict[int, list[float]]], budget: int = 10000
) -> pd.DataFrame:
    """Errors keyed by ``(algorithm, config_id)`` then function, one per trial."""
    return pd.DataFrame(
        [
            {
                "budget": budget,
                "algorithm": algorithm,
                "config_id": identifier,
                "function_id": function_id,
                "instance_id": 11,
                "trial": trial,
                "error": error,
            }
            for (algorithm, identifier), functions in values.items()
            for function_id, series in functions.items()
            for trial, error in enumerate(series, start=1)
        ]
    )


class TestPooledMeanSuperiorities:
    def test_the_pool_spans_the_members(self):
        """A member is scored against the whole grid, not only its own siblings."""
        scores = pooled_mean_superiorities(
            make_errors(
                {
                    ("A", "a0"): {1: [1.0, 1.0]},
                    ("B", "b0"): {1: [2.0, 2.0]},
                }
            )
        ).set_index("config_id")
        assert scores.loc["a0", "mean_superiority"] == pytest.approx(1.0)
        assert scores.loc["b0", "mean_superiority"] == pytest.approx(0.0)

    def test_every_pair_of_trials_is_compared(self):
        """Every pair of trials is compared, not only those sharing an index."""
        scores = pooled_mean_superiorities(
            make_errors(
                {
                    ("A", "a0"): {1: [1.0, 3.0]},
                    ("B", "b0"): {1: [2.0, 4.0]},
                }
            )
        ).set_index("config_id")
        # Trial by trial A wins both; over every pair it wins three of four.
        assert scores.loc["a0", "mean_superiority"] == pytest.approx(0.75)

    def test_a_landslide_counts_once(self):
        """Winning one comparison hugely is worth one comparison, not a margin."""
        scores = pooled_mean_superiorities(
            make_errors(
                {
                    ("A", "a0"): {1: [0.0, 1e9]},
                    ("B", "b0"): {1: [1.0, 1.0]},
                }
            )
        ).set_index("config_id")
        assert scores.loc["a0", "mean_superiority"] == pytest.approx(0.5)

    def test_ties_are_shared(self):
        scores = pooled_mean_superiorities(
            make_errors({("A", "a0"): {1: [1.0]}, ("B", "b0"): {1: [1.0]}})
        )
        assert set(scores["mean_superiority"]) == {0.5}

    def test_the_scores_of_a_pool_sum_to_half_its_size(self):
        """Every pair of configurations hands out one win between them."""
        rng = np.random.default_rng(0)
        errors = make_errors(
            {
                (name, f"{name.lower()}{index}"): {1: list(rng.normal(size=4))}
                for name in ("A", "B")
                for index in range(3)
            }
        )
        assert pooled_mean_superiorities(errors)[
            "mean_superiority"
        ].sum() == pytest.approx(3.0)

    def test_functions_are_scored_separately(self):
        """Two functions are two races; a comparison never crosses them."""
        scores = pooled_mean_superiorities(
            make_errors(
                {
                    ("A", "a0"): {1: [1.0], 2: [9.0]},
                    ("B", "b0"): {1: [9.0], 2: [1.0]},
                }
            )
        ).set_index(["config_id", "function_id"])
        assert scores.loc[("a0", 1), "mean_superiority"] == pytest.approx(1.0)
        assert scores.loc[("a0", 2), "mean_superiority"] == pytest.approx(0.0)

    def test_budgets_are_scored_separately(self):
        """A configuration run to two budgets is two measurements, not one race."""
        errors = pd.concat(
            [
                make_errors({("A", "a0"): {1: [1.0]}, ("B", "b0"): {1: [2.0]}}, 10000),
                make_errors({("A", "a0"): {1: [3.0]}, ("B", "b0"): {1: [1.0]}}, 10500),
            ]
        )
        scores = pooled_mean_superiorities(errors).set_index(["budget", "config_id"])
        assert scores.loc[(10000, "a0"), "mean_superiority"] == pytest.approx(1.0)
        assert scores.loc[(10500, "a0"), "mean_superiority"] == pytest.approx(0.0)

    def test_a_race_missing_a_configuration_is_rejected(self):
        """It would score the rest against a smaller pool and shift that function."""
        errors = make_errors(
            {("A", "a0"): {1: [1.0], 2: [1.0]}, ("B", "b0"): {1: [2.0], 2: [2.0]}}
        )
        with pytest.raises(ValueError, match="fewer than"):
            pooled_mean_superiorities(errors.iloc[1:])

    def test_an_empty_table_is_rejected(self):
        with pytest.raises(ValueError, match="nothing to rank"):
            pooled_mean_superiorities(
                make_errors({}).reindex(columns=["budget"]).iloc[:0]
            )

    def test_the_trial_count_is_recorded(self):
        scores = pooled_mean_superiorities(
            make_errors({("A", "a0"): {1: [1.0] * 5}, ("B", "b0"): {1: [2.0] * 5}})
        )
        assert set(scores["n_trials"]) == {5}


class TestHalves:
    def test_both_halves_are_reduced(self):
        scores = pooled_mean_superiorities(
            make_errors({("A", "a0"): {1: [1.0] * 4}, ("B", "b0"): {1: [2.0] * 4}})
        ).set_index("config_id")
        for name in HALVES:
            assert scores.loc["a0", f"mean_superiority_{name}"] == pytest.approx(1.0)

    def test_the_halves_split_the_trials_and_can_disagree(self):
        """A configuration that wins the odd trials and loses the even ones."""
        scores = pooled_mean_superiorities(
            make_errors(
                {
                    ("A", "a0"): {1: [1.0, 9.0, 1.0, 9.0]},
                    ("B", "b0"): {1: [2.0, 2.0, 2.0, 2.0]},
                }
            )
        ).set_index("config_id")
        assert scores.loc["a0", "mean_superiority_odd"] == pytest.approx(1.0)
        assert scores.loc["a0", "mean_superiority_even"] == pytest.approx(0.0)
        assert scores.loc["a0", "mean_superiority"] == pytest.approx(0.5)

    def test_the_split_is_alternating_rather_than_sequential(self):
        """Trial 1 and trial 3 are one half; trial 2 and trial 4 the other."""
        scores = pooled_mean_superiorities(
            make_errors(
                {
                    ("A", "a0"): {1: [1.0, 1.0, 9.0, 9.0]},
                    ("B", "b0"): {1: [2.0, 2.0, 2.0, 2.0]},
                }
            )
        ).set_index("config_id")
        # A sequential split would give 1.0 and 0.0; an alternating one splits
        # each pair, so both halves see one win and one loss.
        assert scores.loc["a0", "mean_superiority_odd"] == pytest.approx(0.5)
        assert scores.loc["a0", "mean_superiority_even"] == pytest.approx(0.5)


class TestAnnotate:
    def make_entry(self, name, grid):
        from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm

        return TunableAlgorithm(
            name=name,
            slug=name.lower(),
            factory=cast(Any, lambda **params: lambda *a, **k: 0),
            defaults={"x": 1.0},
            grid=grid,
        )

    def test_the_choice_is_marked_at_its_own_budget_only(self):
        """A configuration selected at 1000d is not selected at 1050d."""
        from aas_shortcut.optimizers.define_tunable_algorithm import config_id

        entry = self.make_entry("A", {"x": [1.0, 2.0]})
        ids = [config_id(params) for params in entry.candidates()]
        ranks = pd.DataFrame(
            [
                {
                    "budget": budget,
                    "algorithm": "A",
                    "config_id": identifier,
                    "function_id": 1,
                    "instance_id": 11,
                    "mean_superiority": 1.0,
                    "mean_superiority_odd": 1.0,
                    "mean_superiority_even": 1.0,
                    "n_trials": 31,
                }
                for budget in (10000, 10500)
                for identifier in ids
            ]
        )
        table = annotate(
            ranks, {"A": entry}, {10000: {"A": ids[1]}, 10500: {"A": ids[0]}}, 10
        )
        chosen = table[table["is_selected"]].set_index("budget")["config_id"]
        assert chosen.loc[10000] == ids[1]
        assert chosen.loc[10500] == ids[0]

    def test_the_default_is_flagged(self):
        from aas_shortcut.optimizers.define_tunable_algorithm import config_id

        entry = self.make_entry("A", {"x": [1.0, 2.0]})
        ids = [config_id(params) for params in entry.candidates()]
        ranks = pd.DataFrame(
            [
                {
                    "budget": 10000,
                    "algorithm": "A",
                    "config_id": identifier,
                    "function_id": 1,
                    "instance_id": 11,
                    "mean_superiority": 1.0,
                    "mean_superiority_odd": 1.0,
                    "mean_superiority_even": 1.0,
                    "n_trials": 31,
                }
                for identifier in ids
            ]
        )
        table = annotate(ranks, {"A": entry}, {10000: {"A": ids[0]}}, 10).set_index(
            "config_id"
        )
        assert bool(table.loc[ids[0], "is_default"])
        assert not bool(table.loc[ids[1], "is_default"])
