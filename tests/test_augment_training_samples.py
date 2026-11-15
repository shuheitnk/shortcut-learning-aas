"""Training-time sampling repetitions, and the asymmetry that makes them honest."""

from __future__ import annotations

import pytest

from aas_shortcut.augment_training_samples import (
    AUGMENTED_ID_COLUMNS,
    DEPLOYED_REP,
    REDUCTION_MEAN,
    REDUCTION_ROWS,
    REP_COLUMN,
    available_reps,
    build_matrix,
    validate_reduction,
)
from aas_shortcut.build_feature_matrix import INSTANCE_COLUMNS, join_targets
from tests.conftest import make_features, make_performance, make_superiority

REPS = 4
INSTANCES = 5


@pytest.fixture
def features():
    return make_features("bbob", 1, n_instances=INSTANCES, n_reps=REPS)


class TestAvailableReps:
    def test_it_reports_the_shards_repetitions(self, features):
        assert available_reps(features) == tuple(range(1, REPS + 1))


class TestBuildMatrix:
    def test_every_repetition_becomes_its_own_row(self, features):
        matrix = build_matrix(features, "pre", "value", feature_set="ela")
        assert len(matrix) == INSTANCES * REPS

    def test_the_repetition_is_an_identifier_not_a_feature(self, features):
        matrix = build_matrix(features, "pre", "value", feature_set="ela")
        assert REP_COLUMN in matrix.columns
        assert set(AUGMENTED_ID_COLUMNS) <= set(matrix.columns)

    def test_restricting_the_repetitions_shrinks_the_matrix(self, features):
        matrix = build_matrix(features, "pre", "value", feature_set="ela", reps=(1, 2))
        assert len(matrix) == INSTANCES * 2
        assert set(matrix[REP_COLUMN]) == {1, 2}

    def test_restricting_the_instances_shrinks_the_matrix(self, features):
        matrix = build_matrix(
            features, "pre", "value", feature_set="ela", instances=(1, 2)
        )
        assert set(matrix["instance_id"]) == {1, 2}

    def test_one_repetition_reproduces_the_row_count_of_the_plain_matrix(
        self, features
    ):
        from aas_shortcut.build_feature_matrix import build_feature_matrix

        plain = build_feature_matrix(
            features[features[REP_COLUMN] == 1], "pre", "value", feature_set="ela"
        )
        augmented = build_matrix(features, "pre", "value", feature_set="ela", reps=(1,))
        assert len(augmented) == len(plain)

    def test_an_impossible_restriction_is_rejected(self, features):
        """Silently training on nothing would read as a result."""
        with pytest.raises(ValueError, match="holds no row"):
            build_matrix(features, "pre", "value", feature_set="ela", reps=(99,))


class TestTargets:
    def test_every_repetition_of_an_instance_carries_the_same_target(self, features):
        """The target is a property of the problem, not of the sample drawn."""
        performance = make_performance("bbob", 1, n_instances=INSTANCES)
        superiorities = make_superiority(performance)
        pair = superiorities["pair"].iloc[0]
        matrix = build_matrix(features, "pre", "value", feature_set="ela")
        labelled = join_targets(matrix, superiorities, pair, validate="m:1")

        per_instance = labelled.groupby("instance_id")["superiority"].nunique()
        assert (per_instance == 1).all()
        assert len(labelled) == INSTANCES * REPS

    def test_a_one_to_one_join_refuses_an_augmented_matrix(self, features):
        """The guard stopping an augmented matrix reaching a one-row stage."""
        superiorities = make_superiority(
            make_performance("bbob", 1, n_instances=INSTANCES)
        )
        pair = superiorities["pair"].iloc[0]
        matrix = build_matrix(features, "pre", "value", feature_set="ela")
        with pytest.raises(Exception, match="not a one-to-one merge"):
            join_targets(matrix, superiorities, pair)


class TestMeanReduction:
    """The control: same feature budget, spent on a cleaner input instead of rows."""

    def test_averaging_leaves_one_row_per_instance(self, features):
        matrix = build_matrix(
            features, "pre", "value", feature_set="ela", reduction=REDUCTION_MEAN
        )
        assert len(matrix) == INSTANCES

    def test_averaging_drops_the_repetition_column(self, features):
        """The row no longer belongs to one repetition, so it must not claim to."""
        matrix = build_matrix(
            features, "pre", "value", feature_set="ela", reduction=REDUCTION_MEAN
        )
        assert REP_COLUMN not in matrix.columns

    def test_the_averaged_row_is_the_mean_of_the_repetitions(self, features):
        rows = build_matrix(
            features, "pre", "value", feature_set="ela", reduction=REDUCTION_ROWS
        )
        averaged = build_matrix(
            features, "pre", "value", feature_set="ela", reduction=REDUCTION_MEAN
        )
        column = next(c for c in rows.columns if c.startswith("value__"))
        expected = rows.groupby("instance_id")[column].mean()
        actual = averaged.set_index("instance_id")[column]
        assert actual.round(9).equals(expected.round(9).reindex(actual.index))

    def test_averaging_one_repetition_is_the_identity(self, features):
        """The test side is built through the same call, over a single repetition."""
        rows = build_matrix(
            features,
            "pre",
            "value",
            feature_set="ela",
            reps=(1,),
            reduction=REDUCTION_ROWS,
        )
        averaged = build_matrix(
            features,
            "pre",
            "value",
            feature_set="ela",
            reps=(1,),
            reduction=REDUCTION_MEAN,
        )
        column = next(c for c in averaged.columns if c.startswith("value__"))
        assert len(averaged) == len(rows)
        assert averaged[column].round(9).tolist() == rows[column].round(9).tolist()

    def test_an_unknown_reduction_is_rejected(self):
        with pytest.raises(ValueError, match="unknown reduction"):
            validate_reduction("median")


class TestDeployedRep:
    def test_the_deployed_repetition_is_a_real_one(self, features):
        assert DEPLOYED_REP in available_reps(features)

    def test_the_identifier_columns_extend_the_studys_own(self):
        assert set(INSTANCE_COLUMNS) < set(AUGMENTED_ID_COLUMNS)
