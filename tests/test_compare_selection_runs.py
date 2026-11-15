"""Tests for whether two selectors differ by more than their randomness.

The tests are Experiment 4's; checked here is the selection's own part: a seed's
value is the median of its deployments on the transformed block, the families fold
into two ranges, and the Holm families are the selection's, each range on its own.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.compare_selection_runs import (
    CELL_COLUMNS,
    FAMILY_COLUMN,
    PROBLEM_COLUMNS,
    compare_selection_representations,
    compare_selection_trainings,
    problem_levels,
    seed_values,
    selection_levels,
    selection_runs,
    view_levels,
)
from aas_shortcut.compare_training_runs import (
    CONDITIONS,
    PAIRED,
    SIZE_COLUMN,
    TIED,
    UNPAIRED,
    compare_unpaired_runs,
)
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
    UNSEEN_FAMILIES,
)
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
    VARIANTS,
)

FAMILIES = (CONTROL.label, *UNSEEN_FAMILIES)
REPS = 3

# The step `TestRepresentations` puts under `rank`, the lift `TestTrainings` puts
# on augmented `value`.
STEP = 0.02
LIFT = 0.4


def records(level, seeds=MODEL_SEEDS, families=FAMILIES) -> pd.DataFrame:
    """Every training run's deployments, its value set by ``level`` on the cell.

    Each deployment adds 0.001 x rep, so a seed's median over three deployments
    is its level plus 0.002.
    """
    rows = [
        {
            "feature_set": "ela",
            "feature_variant": variant,
            "training": training,
            FAMILY_COLUMN: family,
            "transform": view,
            "dimension": 10,
            "seed": seed,
            "rep": rep,
            "superiority_vs_sbs": level(variant, training, family, view, seed)
            + 0.001 * rep,
        }
        for variant in VARIANTS
        for training in (TRAINING_PLAIN, TRAINING_AUGMENTED)
        for family in families
        for view in ("pre", "post")
        for seed in seeds
        for rep in range(1, REPS + 1)
    ]
    return pd.DataFrame(rows)


def wobble(seed: int) -> float:
    """The seed-to-seed spread every cell shares."""
    return 0.001 * (seed - MODEL_SEEDS[0])


class TestSeedValues:
    def test_a_seed_is_the_median_of_its_deployments(self):
        values = seed_values(records(lambda *_: 0.3), by=(FAMILY_COLUMN,))
        assert np.allclose(values["value"], 0.302)
        assert len(values) == len(VARIANTS) * 2 * len(FAMILIES) * len(MODEL_SEEDS)

    def test_only_the_transformed_block_is_read(self):
        def level(variant, training, family, view, seed):
            return 0.9 if view == "pre" else 0.3

        assert np.allclose(
            seed_values(records(level), by=(FAMILY_COLUMN,))["value"], 0.302
        )

    def test_a_cell_missing_a_training_run_is_refused(self):
        with pytest.raises(ValueError, match="every training run"):
            seed_values(records(lambda *_: 0.3, seeds=MODEL_SEEDS[:-1]))

    def test_a_table_without_the_transformed_block_is_refused(self):
        table = records(lambda *_: 0.3)
        with pytest.raises(ValueError, match="'post'"):
            seed_values(cast(pd.DataFrame, table[table["transform"] == "pre"]))


class TestRanges:
    @staticmethod
    def level(_variant, _training, family, _view, _seed):
        # Each family a level of its own, so the two ranges are told apart.
        return 0.1 + 0.1 * FAMILIES.index(family)

    def test_the_spline_range_reads_that_family_alone(self):
        runs = selection_runs(records(self.level), SCOPE_CONTROL)
        assert np.allclose(runs["value"], 0.102)

    def test_the_unseen_range_is_the_mean_of_the_four(self):
        runs = selection_runs(records(self.level), SCOPE_UNSEEN)
        expected = np.mean([0.1 + 0.1 * index for index in range(1, 5)]) + 0.002
        assert np.allclose(runs["value"], expected)
        assert len(runs) == len(VARIANTS) * 2 * len(MODEL_SEEDS)

    def test_a_range_missing_a_family_is_refused(self):
        table = records(self.level, families=FAMILIES[:-1])
        with pytest.raises(ValueError, match="needs the families"):
            selection_runs(table, SCOPE_UNSEEN)

    def test_an_unknown_range_is_refused(self):
        with pytest.raises(ValueError, match="unknown range"):
            selection_runs(records(self.level), "family mean")


class TestLevels:
    """The main table's cell: the median of the eleven seeds' values."""

    @staticmethod
    def level(_variant, _training, family, _view, seed):
        # Two unseen families move with the seed, one falling and one rising on a
        # curve, so the median of the seeds' range means (0.021 above the bases)
        # is not the mean of the families' medians (0.01875 above them).
        index = FAMILIES.index(family)
        step = seed - MODEL_SEEDS[0]
        moved = {1: 0.01 * (10 - step), 2: 0.001 * step**2}.get(index, 0.0)
        return 0.1 + 0.1 * index + moved

    def test_a_cell_is_the_median_of_its_seeds_values_over_the_range(self):
        table = records(self.level)
        levels = selection_levels(table, (SCOPE_UNSEEN, SCOPE_CONTROL))
        for scope in (SCOPE_UNSEEN, SCOPE_CONTROL):
            runs = selection_runs(table, scope)
            expected = runs.groupby(list(CELL_COLUMNS))["value"].median()
            found = levels[levels[SCOPE_COLUMN] == scope].set_index(list(CELL_COLUMNS))[
                "value"
            ]
            assert np.array_equal(
                found.sort_index().to_numpy(), expected.sort_index().to_numpy()
            )

    def test_the_range_is_averaged_before_the_seeds_median(self):
        table = records(self.level)
        unseen = selection_levels(table, (SCOPE_UNSEEN,))["value"].to_numpy()
        per_family = seed_values(table, by=(FAMILY_COLUMN,))
        per_family = per_family[per_family[FAMILY_COLUMN].isin(UNSEEN_FAMILIES)]
        medians = per_family.groupby([*CELL_COLUMNS, FAMILY_COLUMN])["value"].median()
        other_order = medians.groupby(list(CELL_COLUMNS)).mean().to_numpy()
        assert not np.allclose(unseen, other_order)

    def test_the_known_family_stays_out_of_the_unseen_range(self):
        def moved(variant, training, family, view, seed):
            shift = 0.5 if family == CONTROL.label else 0.0
            return self.level(variant, training, family, view, seed) + shift

        before = selection_levels(records(self.level), (SCOPE_UNSEEN,))
        after = selection_levels(records(moved), (SCOPE_UNSEEN,))
        assert np.array_equal(before["value"].to_numpy(), after["value"].to_numpy())

    def test_a_range_missing_a_family_is_refused(self):
        table = records(self.level, families=FAMILIES[:-1])
        with pytest.raises(ValueError, match="needs the families"):
            selection_levels(table, (SCOPE_UNSEEN,))


class TestRepresentations:
    """Paired by seed; Holm over the three pairs of one training condition."""

    @staticmethod
    def level(variant, _training, _family, _view, seed):
        return 0.3 + wobble(seed) + (STEP if variant == "rank" else 0.0)

    @pytest.fixture
    def tested(self):
        runs = selection_runs(records(self.level), SCOPE_CONTROL)
        return compare_selection_representations(runs, SCOPE_CONTROL)

    def test_three_pairs_per_training_condition(self, tested):
        assert len(tested) == 3 * 2
        assert set(tested["test"]) == {PAIRED}
        assert set(tested["n_pairs"]) == {len(MODEL_SEEDS)}
        assert set(tested["scope"]) == {SCOPE_CONTROL}

    def test_the_step_is_found_with_its_size(self, tested):
        """The rank step is found with its signed size, in representation order."""
        stepped = tested[(tested["left"] == "rank") | (tested["right"] == "rank")]
        assert len(stepped) == 2 * 2
        assert set(stepped["higher"]) == {"rank"}
        signed = np.where(stepped["left"] == "rank", STEP, -STEP)
        assert np.allclose(stepped[SIZE_COLUMN], signed)
        assert stepped["significant"].all()

    def test_pairs_that_never_differ_are_tied(self, tested):
        flat = tested[(tested["left"] == "concat") & (tested["right"] == "value")]
        assert len(flat) == 2
        assert set(flat["higher"]) == {TIED}
        assert (flat["p"] == 1.0).all()

    def test_the_correction_is_over_the_three_pairs(self, tested):
        for _, family in tested.groupby("training"):
            strongest = family.loc[family["p"].idxmin()]
            assert strongest["p_holm"] == pytest.approx(min(1.0, 3 * strongest["p"]))


class TestTrainings:
    """Unpaired and exact; Holm over the three representations of a feature set."""

    @staticmethod
    def level(variant, training, _family, _view, seed):
        lifted = variant == "value" and training == TRAINING_AUGMENTED
        return 0.3 + wobble(seed) + (LIFT if lifted else 0.0)

    @pytest.fixture
    def tested(self):
        runs = selection_runs(records(self.level), SCOPE_UNSEEN)
        return compare_selection_trainings(runs, SCOPE_UNSEEN)

    def test_one_comparison_per_representation(self, tested):
        assert sorted(tested["feature_variant"]) == sorted(VARIANTS)
        assert set(tested["test"]) == {UNPAIRED}
        assert set(tested[f"n_{TRAINING_AUGMENTED}"]) == {len(MODEL_SEEDS)}

    def test_conditions_that_never_overlap_reach_the_exact_floor(self, tested):
        lifted = tested[tested["feature_variant"] == "value"].iloc[0]
        assert lifted["p"] == pytest.approx(2 / math.comb(22, 11), rel=1e-12)
        assert lifted["higher"] == TRAINING_AUGMENTED
        assert lifted[SIZE_COLUMN] == pytest.approx(LIFT)

    def test_the_correction_is_over_the_three_representations(self, tested):
        lifted = tested[tested["feature_variant"] == "value"].iloc[0]
        assert lifted["p_holm"] == pytest.approx(3 * lifted["p"])
        assert lifted["significant"]
        assert not tested[tested["feature_variant"] != "value"]["significant"].any()

    def test_a_tie_between_the_conditions_is_counted(self):
        """A tie between the conditions is counted at its midrank.

        Ties occur because seeds that choose the same solvers score the same.
        """

        def level(variant, training, family, view, seed):
            # Augmented `value` one step up, so that its runs land on plain
            # runs' values.
            shift = (
                0.001 if variant == "value" and training == TRAINING_AUGMENTED else 0.0
            )
            return 0.3 + wobble(seed) + shift

        runs = selection_runs(records(level), SCOPE_UNSEEN)
        row = compare_selection_trainings(runs, SCOPE_UNSEEN).set_index(
            "feature_variant"
        )
        cell = runs[runs["feature_variant"] == "value"]
        samples = [
            cell[cell["training"] == name].sort_values("seed")["value"].to_numpy()
            for name in CONDITIONS
        ]
        pooled = np.concatenate(samples)
        assert np.unique(pooled).size < pooled.size
        assert row.loc["value", "p"] == compare_unpaired_runs(*samples, CONDITIONS)[1]
        assert 2 / math.comb(22, 11) < row.loc["value", "p"] < 1.0


class TestViewLevels:
    """Fig. 16's rules: each cell's level on both blocks."""

    def test_each_view_is_read_on_its_own(self):
        def level(variant, training, family, view, seed):
            return (0.9 if view == "pre" else 0.3) + wobble(seed)

        levels = view_levels(records(level))
        middle = wobble(MODEL_SEEDS[len(MODEL_SEEDS) // 2])
        for view, expected in (("pre", 0.9), ("post", 0.3)):
            values = levels[levels["transform"] == view]["value"].to_numpy()
            assert values == pytest.approx(expected + middle + 0.002)

    def test_the_post_level_is_the_spline_range_s(self):
        """On one family the post level equals the spline range's level."""
        rows = records(lambda *args: 0.4 + wobble(args[-1]), families=(CONTROL.label,))
        ours = view_levels(rows)
        ours = ours[ours["transform"] == "post"].set_index(list(CELL_COLUMNS))
        theirs = selection_levels(rows, [SCOPE_CONTROL]).set_index(list(CELL_COLUMNS))
        assert ours["value"].sort_index().to_numpy() == pytest.approx(
            theirs["value"].sort_index().to_numpy()
        )


def problems(level, seeds=MODEL_SEEDS, n_problems=4) -> pd.DataFrame:
    """Every training run's problems, each deployment's value set by ``level``."""
    return pd.DataFrame(
        [
            {
                "feature_set": "ela",
                "feature_variant": "rank",
                "training": TRAINING_PLAIN,
                FAMILY_COLUMN: CONTROL.label,
                "transform": "post",
                "seed": seed,
                "rep": rep,
                "problem_class": "bbob",
                "function_id": 1,
                "instance_id": instance,
                "dimension": 10,
                "superiority_vs_sbs": level(instance, seed, rep),
            }
            for seed in seeds
            for rep in range(1, REPS + 1)
            for instance in range(1, n_problems + 1)
        ]
    )


class TestProblemLevels:
    """Fig. 16's violins: a problem read the way a cell is."""

    def test_a_problem_is_the_median_of_its_runs_median_deployment(self):
        def level(instance, seed, rep):
            return 0.1 * instance + wobble(seed) + 0.0001 * rep

        levels = problem_levels(problems(level))
        middle = wobble(MODEL_SEEDS[len(MODEL_SEEDS) // 2])
        by_problem = levels.set_index("instance_id")["superiority_vs_sbs"]
        for instance in range(1, 5):
            assert by_problem[instance] == pytest.approx(
                0.1 * instance + middle + 0.0002
            )

    def test_one_row_per_cell_view_and_problem(self):
        levels = problem_levels(problems(lambda *_: 0.5))
        assert len(levels) == 4
        assert set(PROBLEM_COLUMNS) <= set(levels.columns)
        assert "seed" not in levels.columns

    def test_a_problem_missing_a_training_run_is_refused(self):
        with pytest.raises(ValueError, match="every training run"):
            problem_levels(problems(lambda *_: 0.5, seeds=MODEL_SEEDS[:-1]))
