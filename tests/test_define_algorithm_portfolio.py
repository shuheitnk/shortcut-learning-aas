"""The portfolio, its information-use groups and its pairs."""

from __future__ import annotations

from aas_shortcut.define_algorithm_portfolio import (
    ALGORITHM_GROUP,
    ALGORITHMS,
    FOCUS_PAIR_TYPES,
    FOCUS_PAIRS,
    GROUP_RANK_SEARCH,
    GROUP_RANK_SELECTION,
    GROUP_VALUE_SEARCH,
    GROUPS,
    PAIRS,
    PORTFOLIO,
    SHADE_POPULATION_FACTOR,
    minimum_fe_budget,
    pair_type_of,
)


class TestAlgorithmPortfolio:
    def test_the_portfolio_is_the_one_the_design_specifies(self) -> None:
        """The six members, ordered by group."""
        assert ALGORITHMS == (
            "BIPOP-CMA-ES",
            "Adaptive Nelder-Mead",
            "SHADE",
            "L-SHADE",
            "BOBYQA",
            "Dual Annealing",
        )

    def test_the_groups_partition_the_portfolio_by_information_use(self) -> None:
        members = {
            group: tuple(spec.name for spec in PORTFOLIO if spec.group == group)
            for group in GROUPS
        }
        assert members[GROUP_RANK_SEARCH] == (
            "BIPOP-CMA-ES",
            "Adaptive Nelder-Mead",
        )
        assert members[GROUP_RANK_SELECTION] == ("SHADE", "L-SHADE")
        assert members[GROUP_VALUE_SEARCH] == ("BOBYQA", "Dual Annealing")

    def test_groups_are_ordered_by_value_dependence(self) -> None:
        assert GROUPS == (
            GROUP_RANK_SEARCH,
            GROUP_RANK_SELECTION,
            GROUP_VALUE_SEARCH,
        )

    def test_every_algorithm_has_a_group(self) -> None:
        assert set(ALGORITHM_GROUP) == set(ALGORITHMS)
        assert set(ALGORITHM_GROUP.values()) == set(GROUPS)

    def test_every_runner_is_callable(self) -> None:
        assert all(callable(spec.runner) for spec in PORTFOLIO)

    def test_every_unordered_pair_is_produced(self) -> None:
        n = len(ALGORITHMS)
        assert len(PAIRS) == n * (n - 1) // 2

    def test_pair_type_is_canonical_regardless_of_argument_order(self) -> None:
        assert pair_type_of(GROUP_VALUE_SEARCH, GROUP_RANK_SEARCH) == "G1_G3"
        assert pair_type_of(GROUP_RANK_SEARCH, GROUP_VALUE_SEARCH) == "G1_G3"

    def test_every_reachable_pair_type_is_represented(self) -> None:
        """Every within- and across-group pair type occurs."""
        assert {pair.pair_type for pair in PAIRS} == {
            "G1_G1",
            "G1_G2",
            "G1_G3",
            "G2_G2",
            "G2_G3",
            "G3_G3",
        }

    def test_the_focus_pairs_are_the_within_group_comparisons(self) -> None:
        """The three within-group pairs."""
        assert set(FOCUS_PAIR_TYPES) == {"G1_G1", "G2_G2", "G3_G3"}
        assert {pair.pair_type for pair in FOCUS_PAIRS} == set(FOCUS_PAIR_TYPES)
        assert len(FOCUS_PAIRS) == 3

    def test_pair_slugs_are_unique_and_filename_safe(self) -> None:
        slugs = [pair.slug for pair in PAIRS]
        assert len(set(slugs)) == len(slugs)
        assert all(slug == slug.lower() and " " not in slug for slug in slugs)

    def test_minimum_budget_covers_every_member(self) -> None:
        """Every population-based member must fit one generation in the budget."""
        for dim in (2, 10):
            assert minimum_fe_budget(dim) >= SHADE_POPULATION_FACTOR * dim
