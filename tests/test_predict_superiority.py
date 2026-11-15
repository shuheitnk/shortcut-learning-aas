"""The superiority prediction: the score and the transfer setting."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.predict_superiority import r2_score, restrict_to_common_features
from tests.conftest import make_features


class TestR2Score:
    def test_r2_is_one_for_a_perfect_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_r2_is_zero_for_the_mean_predictor(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, np.full(3, y.mean())) == pytest.approx(0.0)

    def test_r2_is_negative_for_a_worse_than_mean_predictor(self):
        assert r2_score(np.array([1.0, 2.0, 3.0]), np.array([9.0, 9.0, 9.0])) < 0


class TestCommonFeatureRestriction:
    """The transfer setting needs both suites to share a feature set."""

    def test_unchanged_when_the_suites_agree(self):
        tables = {"ma-bbob": make_features(), "bbob": make_features("bbob", 1)}
        restricted = restrict_to_common_features(tables)
        for key, table in tables.items():
            assert len(restricted[key]) == len(table)

    def test_the_intersection_is_taken(self):
        """A feature a calculator refused on one suite must go from both."""
        mabbob = make_features()
        bbob = make_features("bbob", 1)
        dropped = "ela_distr.skewness"
        bbob = cast(pd.DataFrame, bbob[bbob["feature"] != dropped])

        expected = mabbob["feature"].nunique() - 1
        restricted = restrict_to_common_features({"ma-bbob": mabbob, "bbob": bbob})
        for table in restricted.values():
            assert dropped not in set(table["feature"])
            assert table["feature"].nunique() == expected
