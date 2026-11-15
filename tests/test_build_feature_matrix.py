"""The wide feature matrix a model consumes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.build_feature_matrix import (
    build_feature_matrix,
    feature_columns,
    join_targets,
)


class TestFeatureMatrix:
    def test_value_and_rank_have_the_same_width(self, features):
        value = build_feature_matrix(features, "pre", "value")
        rank = build_feature_matrix(features, "pre", "rank")
        assert len(feature_columns(value)) == len(feature_columns(rank))

    def test_concat_is_the_concatenation(self, features):
        value = build_feature_matrix(features, "pre", "value")
        rank = build_feature_matrix(features, "pre", "rank")
        concat = build_feature_matrix(features, "pre", "concat")
        assert len(feature_columns(concat)) == len(feature_columns(value)) + len(
            feature_columns(rank)
        )

    def test_columns_are_prefixed_per_encoding(self, features):
        concat = build_feature_matrix(features, "pre", "concat")
        columns = feature_columns(concat)
        assert any(c.startswith("value__") for c in columns)
        assert any(c.startswith("rank__") for c in columns)

    def test_one_row_per_instance(self, features):
        matrix = build_feature_matrix(features, "pre", "value")
        assert len(matrix) == features["instance_id"].nunique()

    def test_repetitions_are_averaged(self, features):
        matrix = build_feature_matrix(features, "pre", "value")
        subset = features[
            (features["transform"] == "pre")
            & (features["feature_variant"] == "value")
            & (features["instance_id"] == 1)
            & (features["feature"] == "ela_distr.skewness")
        ]
        assert matrix.loc[matrix["instance_id"] == 1, "value__ela_distr.skewness"].iloc[
            0
        ] == pytest.approx(subset["value"].mean())

    def test_rank_matrix_is_identical_across_transforms(self, features):
        pre = build_feature_matrix(features, "pre", "rank")
        post = build_feature_matrix(features, "post", "rank")
        pd.testing.assert_frame_equal(pre, post)

    def test_unknown_variant_is_rejected(self, features):
        with pytest.raises(ValueError):
            build_feature_matrix(features, "pre", "ordinal")

    def test_join_targets_attaches_the_superiority(self, features, superiorities):
        matrix = build_feature_matrix(features, "pre", "value")
        joined = join_targets(
            matrix, superiorities, "bipop-cmaes__adaptive-nelder-mead"
        )
        assert "superiority" in joined.columns
        assert len(joined) == len(matrix)

    def test_join_targets_selects_the_named_pair(self, features, superiorities):
        matrix = build_feature_matrix(features, "pre", "value")
        first = join_targets(matrix, superiorities, "shade__l-shade")
        second = join_targets(matrix, superiorities, "bipop-cmaes__shade")
        assert not np.allclose(first["superiority"], second["superiority"])
