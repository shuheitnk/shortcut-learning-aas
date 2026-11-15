"""Tests for joining a run's two halves from different run roots.

The guards must refuse untransformed rows that disagree with the source, and a join
that lost a view.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.assemble_paired_views import (
    check_source_agreement,
    departures,
    describe_departures,
)

TOLERANCE = 1e-9


def features(**cells: float) -> pd.DataFrame:
    """Build a feature shard from ``transform__name=value`` pairs."""
    return pd.DataFrame(
        [
            {
                "function_id": 1,
                "instance_id": 1,
                "feature_set": "ela",
                "feature_variant": "value",
                "transform": name.split("__")[0],
                "feature": name.split("__")[1],
                "value": value,
            }
            for name, value in cells.items()
        ]
    )


def series(**cells: float) -> pd.Series:
    """Build a value series indexed by feature name."""
    index = pd.MultiIndex.from_tuples([(name,) for name in cells], names=["feature"])
    return pd.Series(list(cells.values()), index=index)


class TestDepartures:
    """The comparison is relative: the features span many orders of magnitude."""

    def test_agreement_is_reported_as_no_departure(self):
        values = series(a=1.0)
        assert departures(values, values, TOLERANCE) == {}

    def test_a_last_digit_difference_in_a_large_feature_is_not_a_departure(self):
        """A ratio of linear coefficients reaches 1e5; its last digit is noise."""
        mine = series(**{"coef.max_by_min": 125_432.000_002_5})
        theirs = series(**{"coef.max_by_min": 125_432.0})
        assert departures(mine, theirs, TOLERANCE) == {}

    def test_the_same_absolute_difference_in_a_small_feature_is_a_departure(self):
        mine = series(**{"ela_distr.skewness": 0.000_002_5})
        theirs = series(**{"ela_distr.skewness": 0.0})
        assert departures(mine, theirs, TOLERANCE) == {"ela_distr.skewness": 1}

    def test_the_cells_of_one_feature_are_counted(self):
        index = pd.MultiIndex.from_tuples([("f",), ("f",), ("g",)], names=["feature"])
        mine = pd.Series([1.0, 2.0, 3.0], index=index)
        theirs = pd.Series([9.0, 9.0, 3.0], index=index)
        assert departures(mine, theirs, TOLERANCE) == {"f": 2}

    def test_the_description_names_the_feature_and_its_count(self):
        assert describe_departures({"f": 2}) == "f (2 cell(s))"


class TestCheckSourceAgreement:
    def test_agreeing_untransformed_features_pass(self):
        mine = features(pre__ela_meta=1.0, post__ela_meta=2.0)
        compared = check_source_agreement(
            mine, features(pre__ela_meta=1.0), (1,), TOLERANCE, "the training block"
        )
        assert compared == 1

    def test_a_moved_untransformed_feature_is_refused(self):
        """The settings would not be sharing one training block."""
        mine = features(pre__ela_meta=1.0, post__ela_meta=2.0)
        with pytest.raises(ValueError, match="disagree"):
            check_source_agreement(
                mine, features(pre__ela_meta=1.5), (1,), TOLERANCE, "the training block"
            )

    def test_the_source_run_is_named_in_the_refusal(self):
        mine = features(pre__ela_meta=1.0)
        with pytest.raises(ValueError, match="the test block"):
            check_source_agreement(
                mine, features(pre__ela_meta=1.5), (1,), TOLERANCE, "the test block"
            )

    def test_no_shared_row_is_refused_rather_than_passing_vacuously(self):
        mine = features(pre__ela_meta=1.0)
        with pytest.raises(ValueError, match="share no row"):
            check_source_agreement(
                mine,
                features(pre__ela_other=1.0),
                (1,),
                TOLERANCE,
                "the training block",
            )

    def test_only_the_named_instances_are_compared(self):
        """The source covers instances this run never computed."""
        mine = features(pre__ela_meta=1.0)
        theirs = pd.concat(
            [
                features(pre__ela_meta=1.0),
                features(pre__ela_meta=9.9).assign(instance_id=42),
            ],
            ignore_index=True,
        )
        assert (
            check_source_agreement(mine, theirs, (1,), TOLERANCE, "the training block")
            == 1
        )
