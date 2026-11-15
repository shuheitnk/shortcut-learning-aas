"""Tests for instance-wise algorithm ranking prediction.

Every member's target must reach the matrix, and a portfolio missing a member is
refused rather than regressed over the rest.
"""

from __future__ import annotations

import pytest

from aas_shortcut.build_feature_matrix import build_feature_matrix
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.predict_ranking import join_mean_superiorities, target_column


class TestJoinMeanSuperiorities:
    def test_one_target_column_per_member(self, features, mean_superiorities):
        matrix = build_feature_matrix(features, "pre", "rank")
        joined = join_mean_superiorities(matrix, mean_superiorities)
        assert len(joined) == len(matrix)
        for algorithm in ALGORITHMS:
            assert target_column(algorithm) in joined.columns

    def test_a_missing_member_raises(self, features, mean_superiorities):
        matrix = build_feature_matrix(features, "pre", "rank")
        partial = mean_superiorities[mean_superiorities["algorithm"] != ALGORITHMS[0]]
        with pytest.raises(ValueError, match="whole portfolio"):
            join_mean_superiorities(matrix, partial)
