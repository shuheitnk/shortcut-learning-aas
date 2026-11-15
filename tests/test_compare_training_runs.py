"""Tests for the seed-level tests of whether two cells differ beyond their randomness.

Representations are paired by seed (Wilcoxon), training conditions are not
(Mann-Whitney). The unseen-family mean and the known family are separate ranges.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from aas_shortcut.compare_training_runs import (
    ALPHA,
    CONDITIONS,
    EFFECT_COLUMN,
    PAIRED,
    PAIRED_EFFECT,
    SIZE_COLUMN,
    TIED,
    UNPAIRED,
    UNPAIRED_EFFECT,
    compare_paired_runs,
    compare_representations,
    compare_representations_by_scope,
    compare_training_conditions,
    compare_training_conditions_by_scope,
    compare_unpaired_runs,
    holm,
    median_over_seeds,
    per_seed,
)
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_repetitions import MODEL_SEEDS, REPETITIONS
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
    UNSEEN_FAMILIES,
)
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_representation_transfer import VARIANTS
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN

# The known family and the four unseen ones, so the two ranges differ here.
FAMILIES = (CONTROL.label, *UNSEEN_FAMILIES)
POST = TEST_METRICS["post"]

# The lift `TestAugmentationAgainstItsAbsence` puts on one cell; it survives the
# reduction exactly, so the reported size must equal it.
LIFT = 0.5

# The step put under `rank`; constant across seeds, so every paired difference
# against `rank` is exactly this.
STEP = 0.02


def deployment_rows(level, seeds=MODEL_SEEDS, reps=REPETITIONS) -> pd.DataFrame:
    """Build a per-deployment table whose values ``level`` sets per cell and seed."""
    rows = [
        {
            "feature_set": "ela",
            "group": group,
            "feature_variant": variant,
            "training": training,
            "audit": audit,
            "metric": POST,
            "seed": seed,
            "rep": rep,
            "value": level(group, variant, training, seed) + 0.001 * rep,
        }
        for group in GROUPS
        for variant in VARIANTS
        for training in (TRAINING_PLAIN, TRAINING_AUGMENTED)
        for audit in FAMILIES
        for seed in seeds
        for rep in range(1, reps + 1)
    ]
    return pd.DataFrame(rows)


class TestOneValuePerTrainingRun:
    def test_it_takes_the_median_then_the_mean(self):
        """A run's value is the median over repetitions, then the mean over families."""
        runs = per_seed(deployment_rows(lambda *_: 0.5), SCOPE_UNSEEN)
        middle = 0.001 * (REPETITIONS + 1) / 2
        assert runs["value"].to_numpy() == pytest.approx(0.5 + middle)
        assert len(runs) == len(GROUPS) * len(VARIANTS) * 2 * REPETITIONS

    def test_the_unseen_families_are_averaged_not_pooled(self):
        table = deployment_rows(lambda *_: 0.0)
        table.loc[table["audit"] == UNSEEN_FAMILIES[0], "value"] += 1.0
        runs = per_seed(table, SCOPE_UNSEEN)
        assert runs["value"].to_numpy() == pytest.approx(
            1.0 / len(UNSEEN_FAMILIES) + 0.001 * (REPETITIONS + 1) / 2
        )

    def test_the_known_family_is_read_alone_and_outside_the_unseen_mean(self):
        table = deployment_rows(lambda *_: 0.0)
        table.loc[table["audit"] == CONTROL.label, "value"] += 1.0
        middle = 0.001 * (REPETITIONS + 1) / 2
        known = per_seed(table, SCOPE_CONTROL)
        assert known["value"].to_numpy() == pytest.approx(1.0 + middle)
        unseen = per_seed(table, SCOPE_UNSEEN)
        assert unseen["value"].to_numpy() == pytest.approx(middle)

    def test_a_range_missing_one_of_its_families_is_rejected(self):
        """A range missing one of its families is rejected."""
        table = deployment_rows(lambda *_: 0.5)
        short = cast(pd.DataFrame, table[table["audit"] != UNSEEN_FAMILIES[-1]])
        with pytest.raises(ValueError, match="needs the families"):
            per_seed(short, SCOPE_UNSEEN)

    def test_a_cell_missing_a_training_run_is_rejected(self):
        table = deployment_rows(lambda *_: 0.5)
        short = cast(pd.DataFrame, table[table["seed"] != MODEL_SEEDS[-1]])
        with pytest.raises(ValueError, match="every training run"):
            per_seed(short, SCOPE_UNSEEN)

    def test_a_metric_the_table_lacks_is_rejected(self):
        with pytest.raises(ValueError, match="no 'r2_test_pre' row"):
            per_seed(
                deployment_rows(lambda *_: 0.5),
                SCOPE_UNSEEN,
                metric=TEST_METRICS["pre"],
            )


class TestTheTableCell:
    """The main table's cell is the median of the tests' observations."""

    def test_it_is_the_median_of_the_runs(self):
        runs = pd.DataFrame(
            {
                "feature_set": "ela",
                "group": "G1",
                "feature_variant": "value",
                "training": "plain",
                "seed": MODEL_SEEDS,
                "value": [0.1 * index for index in range(len(MODEL_SEEDS))],
            }
        )
        cell = median_over_seeds(runs)
        assert len(cell) == 1
        # Eleven runs: the median is one run's own value, the sixth.
        assert cell["value"].iloc[0] == runs["value"].iloc[5]

    def test_each_cell_is_reduced_apart(self):
        runs = pd.DataFrame(
            {
                "feature_set": "ela",
                "group": ["G1"] * 3 + ["G2"] * 3,
                "feature_variant": "value",
                "training": "plain",
                "seed": [42, 43, 44] * 2,
                "value": [0.1, 0.2, 0.9, 0.5, 0.6, 0.7],
            }
        )
        cells = median_over_seeds(runs).set_index("group")["value"]
        assert cells.to_dict() == {"G1": 0.2, "G2": 0.6}


class TestHolm:
    def test_it_steps_down(self):
        assert holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])

    def test_it_keeps_the_order_it_was_given(self):
        assert holm([0.04, 0.01]) == pytest.approx([0.04, 0.02])

    def test_it_never_lets_a_weaker_result_overtake_a_stronger_one(self):
        adjusted = holm([0.01, 0.011, 0.5])
        assert adjusted[0] <= adjusted[1] <= adjusted[2]

    def test_it_caps_at_one(self):
        assert holm([0.5, 0.6, 0.7]) == pytest.approx([1.0, 1.0, 1.0])


class TestAugmentationAgainstItsAbsence:
    """Unpaired comparison of the training conditions per (group, representation)."""

    @staticmethod
    def level(group, variant, training, seed):
        # Augmentation lifts only G3 x value; the seed wobble is shared.
        lifted = group == GROUPS[-1] and variant == VARIANTS[0]
        rise = 0.5 if (lifted and training == TRAINING_AUGMENTED) else 0.0
        return 0.1 + rise + 0.001 * (seed - MODEL_SEEDS[0])

    @pytest.fixture
    def tested(self):
        runs = per_seed(deployment_rows(self.level), SCOPE_UNSEEN)
        return compare_training_conditions(runs, SCOPE_UNSEEN)

    def test_it_is_unpaired(self, tested):
        assert set(tested["test"]) == {UNPAIRED}
        assert set(tested[f"n_{CONDITIONS[0]}"]) == {REPETITIONS}
        assert set(tested[f"n_{CONDITIONS[1]}"]) == {REPETITIONS}

    def test_one_comparison_per_cell(self, tested):
        assert len(tested) == len(GROUPS) * len(VARIANTS)
        assert not tested.duplicated(["group", "feature_variant"]).any()

    def test_the_lifted_cell_is_found_and_named(self, tested):
        row = tested[
            (tested["group"] == GROUPS[-1]) & (tested["feature_variant"] == VARIANTS[0])
        ].iloc[0]
        assert row["higher"] == TRAINING_AUGMENTED
        assert row["significant"]
        assert row["p_holm"] < ALPHA

    def test_the_untouched_cells_are_not(self, tested):
        rest = tested[
            ~(
                (tested["group"] == GROUPS[-1])
                & (tested["feature_variant"] == VARIANTS[0])
            )
        ]
        assert not rest["significant"].any()
        assert set(rest["higher"]) == {TIED}

    def test_the_correction_is_over_the_nine_cells(self, tested):
        """Holm's multiplier is the number of cells."""
        row = tested[tested["significant"]].iloc[0]
        assert row["p_holm"] == pytest.approx(row["p"] * len(tested))

    def test_the_size_is_the_difference_of_the_two_medians(self, tested):
        """The unpaired size is the difference of the two medians."""
        assert set(tested[EFFECT_COLUMN]) == {UNPAIRED_EFFECT}
        lifted = tested[
            (tested["group"] == GROUPS[-1]) & (tested["feature_variant"] == VARIANTS[0])
        ].iloc[0]
        assert lifted[SIZE_COLUMN] == pytest.approx(LIFT)

    def test_an_untouched_cell_reports_a_size_of_zero(self, tested):
        rest = tested[
            ~(
                (tested["group"] == GROUPS[-1])
                & (tested["feature_variant"] == VARIANTS[0])
            )
        ]
        assert rest[SIZE_COLUMN].abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_the_sign_is_positive_when_augmented_is_the_higher(self, tested):
        """The size is positive when `augmented` (`CONDITIONS[0]`) is higher."""
        higher = tested[tested["higher"] == CONDITIONS[0]]
        assert not higher.empty
        assert (higher[SIZE_COLUMN] > 0.0).all()


class TestTheRepresentationsAgainstEachOther:
    """Paired comparison of the representations per (group, training condition)."""

    @staticmethod
    def level(_group, variant, _training, seed):
        # `rank` sits a step above the others under a shared seed wobble, which
        # pairing removes and which would swamp an unpaired test.
        wobble = 0.3 * ((seed - MODEL_SEEDS[0]) % 3)
        return wobble + (0.02 if variant == VARIANTS[1] else 0.0)

    @pytest.fixture
    def tested(self):
        runs = per_seed(deployment_rows(self.level), SCOPE_UNSEEN)
        return compare_representations(runs, SCOPE_UNSEEN)

    def test_it_is_paired(self, tested):
        assert set(tested["test"]) == {PAIRED}
        assert set(tested["n_pairs"]) == {REPETITIONS}

    def test_three_comparisons_per_group_and_condition(self, tested):
        counted = tested.groupby(["group", "training"], observed=True).size()
        assert set(counted) == {3}
        assert len(tested) == len(GROUPS) * 2 * 3

    def test_the_pairing_sees_a_step_the_wobble_would_hide(self, tested):
        """Pairing detects a step fifteen times smaller than the seed wobble."""
        against_rank = tested[
            (tested["left"] == VARIANTS[1]) | (tested["right"] == VARIANTS[1])
        ]
        assert against_rank["significant"].all()
        assert set(against_rank["higher"]) == {VARIANTS[1]}

    def test_two_cells_that_never_differ_are_tied_rather_than_an_error(self):
        runs = per_seed(deployment_rows(lambda *_: 0.4), SCOPE_UNSEEN)
        tested = compare_representations(runs, SCOPE_UNSEEN)
        assert set(tested["higher"]) == {TIED}
        assert (tested["p"] == 1.0).all()
        assert not tested["significant"].any()

    def test_the_correction_is_over_the_three_pairs(self, tested):
        cell = tested[
            (tested["group"] == GROUPS[0]) & (tested["training"] == TRAINING_PLAIN)
        ]
        strongest = cell.sort_values("p").iloc[0]
        assert strongest["p_holm"] == pytest.approx(min(1.0, strongest["p"] * 3))

    def test_the_size_is_the_median_of_the_paired_differences(self, tested):
        """The paired size is the median of the paired differences."""
        assert set(tested[EFFECT_COLUMN]) == {PAIRED_EFFECT}
        against_rank = tested[
            (tested["left"] == VARIANTS[1]) | (tested["right"] == VARIANTS[1])
        ]
        assert against_rank[SIZE_COLUMN].abs().to_numpy() == pytest.approx(STEP)

    def test_the_sign_points_at_the_side_the_direction_names(self, tested):
        """The size is positive when `left` is higher and negative when `right` is."""
        for row in tested.itertuples():
            size = getattr(row, SIZE_COLUMN)
            if row.higher == row.left:
                assert size > 0.0
            elif row.higher == row.right:
                assert size < 0.0
            else:
                assert row.higher == TIED
                assert size == pytest.approx(0.0, abs=1e-12)

    def test_two_cells_that_never_differ_report_no_size(self):
        runs = per_seed(deployment_rows(lambda *_: 0.4), SCOPE_UNSEEN)
        tested = compare_representations(runs, SCOPE_UNSEEN)
        assert (tested[SIZE_COLUMN].abs() < 1e-12).all()


class TestTheTwoTablesAreAboutOneThing:
    def test_a_direction_is_one_of_the_things_compared(self):
        runs = per_seed(
            deployment_rows(TestAugmentationAgainstItsAbsence.level), SCOPE_UNSEEN
        )
        conditions = compare_training_conditions(runs, SCOPE_UNSEEN)
        assert set(conditions["higher"]) <= {*CONDITIONS, TIED}
        representations = compare_representations(runs, SCOPE_UNSEEN)
        assert set(representations["higher"]) <= {*VARIANTS, TIED}

    def test_the_p_values_are_probabilities(self):
        runs = per_seed(
            deployment_rows(TestAugmentationAgainstItsAbsence.level), SCOPE_UNSEEN
        )
        for tested in (
            compare_training_conditions(runs, SCOPE_UNSEEN),
            compare_representations(runs, SCOPE_UNSEEN),
        ):
            assert ((tested["p"] > 0.0) & (tested["p"] <= 1.0)).all()
            assert (tested["p_holm"] >= tested["p"] - 1e-12).all()
            assert np.isfinite(tested["p"]).all()


class TestTheTwoRangesOfFamilies:
    """The unseen-family mean and the known family are reduced independently."""

    @staticmethod
    def stepped(family: str = CONTROL.label) -> pd.DataFrame:
        """Build a flat table with `rank` a step above the rest in one family."""
        table = deployment_rows(lambda *_: 0.4)
        under_rank = (table["feature_variant"] == VARIANTS[1]) & (
            table["audit"] == family
        )
        table.loc[under_rank, "value"] += STEP
        return table

    @staticmethod
    def against_rank(tested: pd.DataFrame, scope: str) -> pd.DataFrame:
        return tested[
            (tested[SCOPE_COLUMN] == scope)
            & ((tested["left"] == VARIANTS[1]) | (tested["right"] == VARIANTS[1]))
        ]

    @pytest.fixture
    def tested(self):
        return compare_representations_by_scope(self.stepped(), FAMILIES)

    def test_both_ranges_are_reported_and_named_the_main_one_first(self, tested):
        assert set(tested[SCOPE_COLUMN]) == {SCOPE_UNSEEN, SCOPE_CONTROL}
        assert tested[SCOPE_COLUMN].iloc[0] == SCOPE_UNSEEN
        counted = tested.groupby(SCOPE_COLUMN, observed=True).size()
        assert set(counted) == {len(GROUPS) * 2 * 3}

    def test_the_known_range_reads_that_family_alone(self, tested):
        """The known range carries the whole step put under the known family."""
        sizes = self.against_rank(tested, SCOPE_CONTROL)[SIZE_COLUMN]
        assert sizes.abs().to_numpy() == pytest.approx(STEP)

    def test_the_unseen_range_does_not_read_the_known_family(self, tested):
        sizes = self.against_rank(tested, SCOPE_UNSEEN)[SIZE_COLUMN]
        assert sizes.abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_the_unseen_range_averages_its_four_families(self):
        """A step under one unseen family reaches the unseen range quartered."""
        tested = compare_representations_by_scope(
            self.stepped(UNSEEN_FAMILIES[0]), FAMILIES
        )
        sizes = self.against_rank(tested, SCOPE_UNSEEN)[SIZE_COLUMN]
        assert sizes.abs().to_numpy() == pytest.approx(STEP / len(UNSEEN_FAMILIES))

    def test_each_range_is_corrected_on_its_own(self, tested):
        """Holm runs over a cell's three pairs within one range."""
        cell = tested[
            (tested[SCOPE_COLUMN] == SCOPE_CONTROL)
            & (tested["group"] == GROUPS[0])
            & (tested["training"] == TRAINING_PLAIN)
        ]
        assert len(cell) == 3
        strongest = cell.sort_values("p").iloc[0]
        assert strongest["p_holm"] == pytest.approx(min(1.0, strongest["p"] * 3))

    def test_a_run_that_did_not_audit_the_known_family_reports_the_unseen_range(
        self,
    ):
        """Without the known family, only the unseen range is reported."""
        table = self.stepped()
        tested = compare_representations_by_scope(
            cast(pd.DataFrame, table[table["audit"].isin(UNSEEN_FAMILIES)]),
            UNSEEN_FAMILIES,
        )
        assert set(tested[SCOPE_COLUMN]) == {SCOPE_UNSEEN}

    def test_a_run_missing_an_unseen_family_reports_the_known_range_alone(self):
        audited = FAMILIES[:-1]
        table = self.stepped()
        tested = compare_representations_by_scope(
            cast(pd.DataFrame, table[table["audit"].isin(audited)]), audited
        )
        assert set(tested[SCOPE_COLUMN]) == {SCOPE_CONTROL}
        assert not tested.duplicated(["group", "training", "left", "right"]).any()

    def test_a_run_covering_neither_range_reports_nothing(self):
        audited = UNSEEN_FAMILIES[:1]
        table = self.stepped()
        tested = compare_representations_by_scope(
            cast(pd.DataFrame, table[table["audit"].isin(audited)]), audited
        )
        assert tested.empty

    def test_augmentation_is_asked_over_both_ranges(self):
        """The training conditions are compared over the same two ranges."""
        tested = compare_training_conditions_by_scope(
            deployment_rows(TestAugmentationAgainstItsAbsence.level), FAMILIES
        )
        assert set(tested[SCOPE_COLUMN]) == {SCOPE_UNSEEN, SCOPE_CONTROL}
        assert tested[SCOPE_COLUMN].iloc[0] == SCOPE_UNSEEN

    def test_each_range_is_its_own_family_of_nine_cells(self):
        """Holm's multiplier is the nine cells of one range."""
        tested = compare_training_conditions_by_scope(
            deployment_rows(TestAugmentationAgainstItsAbsence.level), FAMILIES
        )
        for _, part in tested.groupby(SCOPE_COLUMN, observed=True):
            assert len(part) == len(GROUPS) * len(VARIANTS)
            row = part[part["significant"]].iloc[0]
            assert row["p_holm"] == pytest.approx(row["p"] * len(part))


def unpaired_by_splits(left, right) -> float:
    """Compute the exact two-sided Mann-Whitney p by enumerating every split.

    U is counted from pairwise comparisons, so the module's midranks are checked
    independently.
    """
    pooled = [*left, *right]

    def u_of(chosen) -> float:
        inside = [pooled[index] for index in chosen]
        outside = [pooled[index] for index in range(len(pooled)) if index not in chosen]
        return sum((x > y) + 0.5 * (x == y) for x in inside for y in outside)

    observed = u_of(range(len(left)))
    counted = [
        u_of(chosen) for chosen in itertools.combinations(range(len(pooled)), len(left))
    ]
    below = sum(u <= observed for u in counted)
    above = sum(u >= observed for u in counted)
    return min(1.0, 2 * min(below, above) / len(counted))


def unpaired_by_recurrence(left, right) -> float:
    """Compute the exact Mann-Whitney p from a rank-sum recurrence.

    Fast enough for eleven runs a side, where enumerating splits is slow.
    """
    ranks = [round(2 * rank) for rank in stats.rankdata([*left, *right])]
    chosen = len(left)
    ways = [Counter() for _ in range(chosen + 1)]
    ways[0][0] = 1
    for rank in ranks:
        for taken in range(chosen, 0, -1):
            for total, count in ways[taken - 1].items():
                ways[taken][total + rank] += count
    observed = sum(ranks[:chosen])
    below = sum(count for total, count in ways[chosen].items() if total <= observed)
    above = sum(count for total, count in ways[chosen].items() if total >= observed)
    return min(1.0, 2 * min(below, above) / math.comb(len(ranks), chosen))


def paired_by_signs(differences) -> float:
    """Compute the exact Wilcoxon p by enumerating signs of non-zero differences."""
    kept = [value for value in differences if value != 0]
    ranks = stats.rankdata(np.abs(kept))
    observed = sum(rank for rank, value in zip(ranks, kept, strict=True) if value > 0)
    counted = [
        sum(rank for rank, positive in zip(ranks, signs, strict=True) if positive)
        for signs in itertools.product((False, True), repeat=len(kept))
    ]
    below = sum(total <= observed for total in counted)
    above = sum(total >= observed for total in counted)
    return min(1.0, 2 * min(below, above) / len(counted))


class TestTheUnpairedTestIsExact:
    """The training conditions' p is exact, with ties at their midrank.

    The exact floor is 2 / C(22, 11) = 2.8e-6; scipy's default normal
    approximation for eleven a side cannot go below 8.2e-5.
    """

    @staticmethod
    def runs(augmented, plain) -> pd.DataFrame:
        """Build one cell's two sets of runs in the shape `per_seed` returns."""
        return pd.DataFrame(
            [
                {
                    "feature_set": "ela",
                    "group": GROUPS[0],
                    "feature_variant": VARIANTS[0],
                    "training": training,
                    "seed": seed,
                    "value": float(value),
                }
                for training, values in (
                    (TRAINING_AUGMENTED, augmented),
                    (TRAINING_PLAIN, plain),
                )
                for seed, value in zip(MODEL_SEEDS, values, strict=True)
            ]
        )

    @staticmethod
    def apart() -> tuple[np.ndarray, np.ndarray]:
        """Return two non-overlapping sets of runs with augmented higher."""
        steps = 0.01 * np.arange(len(MODEL_SEEDS))
        return 0.5 + steps, 0.1 + steps

    def test_samples_that_never_overlap_reach_the_exact_floor(self):
        row = compare_training_conditions(self.runs(*self.apart()), SCOPE_UNSEEN).iloc[
            0
        ]
        assert row["p"] == pytest.approx(2 / math.comb(22, 11), rel=1e-12)
        assert row["higher"] == TRAINING_AUGMENTED

    def test_the_p_value_is_the_exact_one_not_the_approximation(self):
        rng = np.random.default_rng(0)
        augmented = rng.normal(0.5, 0.1, len(MODEL_SEEDS))
        plain = rng.normal(0.45, 0.1, len(MODEL_SEEDS))
        row = compare_training_conditions(
            self.runs(augmented, plain), SCOPE_UNSEEN
        ).iloc[0]
        exact = stats.mannwhitneyu(
            augmented, plain, alternative="two-sided", method="exact"
        ).pvalue
        approximate = stats.mannwhitneyu(
            augmented, plain, alternative="two-sided"
        ).pvalue
        assert row["p"] == pytest.approx(exact, rel=1e-15)
        assert row["p"] != approximate

    def test_a_tie_across_the_conditions_is_counted(self):
        """A tie across conditions lets two splits reach the extreme."""
        augmented, plain = self.apart()
        plain[-1] = augmented[0]
        row = compare_training_conditions(
            self.runs(augmented, plain), SCOPE_UNSEEN
        ).iloc[0]
        assert row["p"] == unpaired_by_recurrence(augmented, plain)
        assert row["p"] == pytest.approx(4 / math.comb(22, 11), rel=1e-12)
        assert row["higher"] == TRAINING_AUGMENTED

    def test_a_tie_within_one_condition_is_counted_too(self):
        augmented, plain = self.apart()
        augmented[1] = augmented[0]
        row = compare_training_conditions(
            self.runs(augmented, plain), SCOPE_UNSEEN
        ).iloc[0]
        assert row["p"] == unpaired_by_recurrence(augmented, plain)
        assert row["p"] == pytest.approx(2 / math.comb(22, 11), rel=1e-12)

    def test_conditions_equal_seed_by_seed_stay_tied(self):
        """Two identical sets of runs are reported tied with p = 1."""
        values = 0.3 + 0.01 * np.arange(len(MODEL_SEEDS))
        row = compare_training_conditions(
            self.runs(values, values.copy()), SCOPE_UNSEEN
        ).iloc[0]
        assert row["higher"] == TIED
        assert row["p"] == 1.0


class TestTheExactPValuesAreCountedOut:
    """Both p-values are counted over every arrangement, ties at their midrank.

    Without ties they match scipy's exact p-values (the paired one bit for bit; the
    unpaired one to the last bit, as scipy sums rounded probabilities). With ties,
    scipy's exact tests do not apply, so independent counts are used instead.
    """

    NAMES = ("left", "right")

    def test_untied_unpaired_runs_get_scipys_exact_p(self):
        rng = np.random.default_rng(1)
        for _ in range(40):
            left = rng.normal(0.5, 0.1, 11)
            right = rng.normal(0.5 + rng.normal(0.0, 0.08), 0.1, 11)
            _, p_value, _ = compare_unpaired_runs(left, right, self.NAMES)
            exact = stats.mannwhitneyu(
                left, right, alternative="two-sided", method="exact"
            ).pvalue
            assert p_value == pytest.approx(exact, rel=1e-15)

    def test_untied_paired_runs_get_scipys_exact_p_to_the_bit(self):
        rng = np.random.default_rng(2)
        for _ in range(40):
            left = rng.normal(0.5, 0.1, 11)
            right = left + rng.normal(rng.normal(0.0, 0.02), 0.03, 11)
            _, p_value, _ = compare_paired_runs(left, right, self.NAMES)
            assert p_value == stats.wilcoxon(left, right).pvalue

    def test_tied_unpaired_runs_are_every_split_counted(self):
        """Tied unpaired runs are counted exactly, unlike scipy's exact test."""
        left = np.array([0.3, 0.4, 0.4, 0.5, 0.7, 0.7])
        right = np.array([0.2, 0.3, 0.4, 0.6, 0.6, 0.65])
        _, p_value, _ = compare_unpaired_runs(left, right, self.NAMES)
        truncated = stats.mannwhitneyu(
            left, right, alternative="two-sided", method="exact"
        ).pvalue
        assert p_value == unpaired_by_splits(left, right)
        assert p_value != truncated

    def test_eleven_tied_runs_a_side_agree_with_the_recurrence(self):
        rng = np.random.default_rng(3)
        for _ in range(5):
            left = rng.integers(0, 6, 11) / 10
            right = rng.integers(1, 7, 11) / 10
            _, p_value, _ = compare_unpaired_runs(left, right, self.NAMES)
            assert p_value == unpaired_by_recurrence(left, right)

    def test_tied_differences_take_their_midrank(self):
        """Tied paired differences take their midrank, unlike scipy's exact test."""
        differences = np.array([0.02, -0.02, 0.03, 0.03, 0.05, -0.01, 0.04])
        _, p_value, _ = compare_paired_runs(
            differences, np.zeros_like(differences), self.NAMES
        )
        assert p_value == paired_by_signs(differences)
        assert p_value != stats.wilcoxon(differences, method="exact").pvalue

    def test_a_zero_difference_is_dropped(self):
        left = np.array([0.50, 0.42, 0.61, 0.55, 0.47, 0.58, 0.44])
        right = np.array([0.45, 0.42, 0.52, 0.51, 0.49, 0.50, 0.41])
        _, p_value, _ = compare_paired_runs(left, right, self.NAMES)
        differs = left != right
        _, without, _ = compare_paired_runs(left[differs], right[differs], self.NAMES)
        assert p_value == without
        assert p_value == paired_by_signs(left - right)

    def test_too_many_arrangements_are_refused(self):
        with pytest.raises(ValueError, match="too many"):
            compare_unpaired_runs(np.arange(15.0), np.arange(15.0) + 0.5, self.NAMES)
        with pytest.raises(ValueError, match="too many"):
            compare_paired_runs(np.arange(1.0, 26.0), np.zeros(25), self.NAMES)


class TestTiesAreDecidedWithinATolerance:
    """Values closer than `TIE_TOLERANCE` tie; an ambiguous gap stops the test.

    Superiorities are rationals on a grid, and summation order moves the float by
    a few bits.
    """

    NAMES = ("left", "right")
    # One rational held as two floats: the sums differ in the last bit.
    ONE = 0.1 + 0.2
    SAME = 0.3
    # The grid of the unseen range at the full scale: four families' mean of
    # 240 problems' mean of superiorities that are multiples of 1/1922.
    STEP = 1 / (1922 * 240 * 4)

    def test_the_two_floats_differ_bit_for_bit(self):
        assert self.ONE != self.SAME
        assert abs(self.ONE - self.SAME) < 1e-15

    def test_two_floats_of_one_value_share_a_rank(self):
        left = np.array([self.ONE, 0.5, 0.6, 0.7])
        right = np.array([self.SAME, 0.1, 0.2, 0.4])
        exact = np.array([self.SAME, 0.5, 0.6, 0.7])
        _, p_value, _ = compare_unpaired_runs(left, right, self.NAMES)
        assert p_value == unpaired_by_splits(exact, right)

    def test_a_pair_holding_one_value_has_no_difference(self):
        left = np.array([self.ONE, 0.52, 0.61, 0.55, 0.47, 0.58, 0.44])
        right = np.array([self.SAME, 0.45, 0.52, 0.51, 0.49, 0.50, 0.41])
        _, p_value, _ = compare_paired_runs(left, right, self.NAMES)
        _, without, _ = compare_paired_runs(left[1:], right[1:], self.NAMES)
        assert p_value == without

    def test_differences_of_one_size_share_a_rank(self):
        """Differences equal up to rounding (0.3 - 0.1, 0.5 - 0.3) share a rank."""
        left = np.array([0.3, 0.5, 0.9, 0.75, 0.62])
        right = np.array([0.1, 0.3, 0.6, 0.7, 0.66])
        assert left[0] - right[0] != left[1] - right[1]
        _, p_value, _ = compare_paired_runs(left, right, self.NAMES)
        assert p_value == paired_by_signs([0.2, 0.2, 0.3, 0.05, -0.04])

    def test_a_size_within_the_tolerance_is_none_and_no_side_is_above(self):
        paired = compare_paired_runs(
            np.array([self.ONE, 0.5, 0.4]), np.array([self.SAME, 0.4, 0.5]), self.NAMES
        )
        unpaired = compare_unpaired_runs(
            np.array([self.ONE, 0.2, 0.4]), np.array([self.SAME, 0.1, 0.5]), self.NAMES
        )
        for higher, _, size in (paired, unpaired):
            assert higher == TIED
            assert size == 0.0

    def test_one_step_of_the_grid_is_a_difference(self):
        """One grid step is a difference, which a 1e-5 relative tolerance would miss."""
        right = np.full(11, 0.5)
        left = right + self.STEP
        higher, p_value, size = compare_paired_runs(left, right, self.NAMES)
        assert higher == "left"
        assert p_value == pytest.approx(2 / 2**11)
        assert size == pytest.approx(self.STEP)
        higher, p_value, _ = compare_unpaired_runs(left, right, self.NAMES)
        assert higher == "left"
        assert p_value == pytest.approx(2 / math.comb(22, 11))

    def test_a_gap_neither_rounding_nor_a_difference_stops_the_test(self):
        """A 1e-10 gap, too large for rounding and too small for a step, raises."""
        with pytest.raises(ValueError, match="cannot be told"):
            compare_paired_runs(
                np.array([0.5 + 1e-10, 0.6, 0.7]), np.array([0.5, 0.4, 0.3]), self.NAMES
            )
        with pytest.raises(ValueError, match="cannot be told"):
            compare_unpaired_runs(
                np.array([0.5, 0.6]), np.array([0.5 + 1e-10, 0.4]), self.NAMES
            )

    def test_differences_whose_sizes_are_that_close_stop_it_too(self):
        with pytest.raises(ValueError, match="cannot be told"):
            compare_paired_runs(
                np.array([0.3, 0.5 + 1e-10, 0.9]), np.array([0.1, 0.3, 0.6]), self.NAMES
            )
