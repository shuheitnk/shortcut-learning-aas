"""A feature's class, and the label it is drawn under."""

from __future__ import annotations

import pytest

from aas_shortcut.order_feature_labels import (
    feature_class,
    leaf_label,
)


class TestFeatureClass:
    @pytest.mark.parametrize(
        ("feature", "expected"),
        [
            ("ela_distr.skewness", "ela_distr"),
            ("ela_meta.lin_simple.coef.max_by_min", "ela_meta"),
            ("deep_ela.large_50d_v1.X7", "deep_ela.large_50d_v1"),
        ],
    )
    def test_names_the_class(self, feature, expected):
        assert feature_class(feature) == expected

    def test_leaf_label_drops_only_the_class_prefix(self):
        assert leaf_label("ela_meta.lin_simple.coef.min") == "lin_simple.coef.min"
        assert leaf_label("deep_ela.large_50d_v1.X7") == "X7"
