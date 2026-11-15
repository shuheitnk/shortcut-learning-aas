"""The preanalysis's metric: the standardized shift of a feature or a superiority."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.define_paths import NOT_A_FEATURE
from aas_shortcut.measure_distribution_shift import (
    QUANTITY_KEYS,
    REPORT_KEYS,
    SHIFT_KEYS,
    feature_observations,
    instance_values,
    measure_shift,
    summarize_shift,
    view_superiority_observations,
)


def observations(
    pre: dict[int, list[float]],
    post: dict[int, list[float]],
    quantity: str = "ela_distr.skewness",
    rep: int = 1,
) -> pd.DataFrame:
    """One quantity measured over instances in one deployment, each repeated."""
    rows = []
    for view, values in (("pre", pre), ("post", post)):
        for function_id, repetitions in values.items():
            rows.extend(
                {
                    "problem_class": "bbob",
                    "function_id": function_id,
                    "instance_id": 1,
                    "dimension": 10,
                    "rep": rep,
                    "quantity_kind": "feature",
                    "feature_set": "ela",
                    "feature_variant": "value",
                    "quantity": quantity,
                    "quantity_class": quantity.split(".")[0],
                    "transform": view,
                    "replicate": index,
                    "value": value,
                }
                for index, value in enumerate(repetitions, start=1)
            )
    return pd.DataFrame(rows)


class TestInstanceValues:
    def test_the_repetitions_are_reduced_by_their_mean(self):
        """The instance's value, not one draw of it."""
        table = instance_values(
            observations({1: [1.0, 2.0, 6.0]}, {1: [0.0, 0.0, 0.0]})
        )
        pre = table[table["transform"] == "pre"]
        assert pre["value"].iloc[0] == pytest.approx(3.0)
        assert pre["n_replicates"].iloc[0] == 3

    def test_a_missing_column_is_rejected(self):
        table = observations({1: [1.0]}, {1: [2.0]})
        with pytest.raises(ValueError, match="missing column"):
            instance_values(table.drop(columns=["replicate"]))


class TestMeasureShift:
    def test_the_shift_is_the_difference_over_the_pre_spread(self):
        """|dz| = |x_post - x_pre| / sigma_j, with sigma_j across the instances."""
        pre = {1: [0.0], 2: [2.0], 3: [4.0]}
        post = {1: [1.0], 2: [3.0], 3: [5.0]}
        table = measure_shift(observations(pre, post))
        population = table[table["standardization"] == "population"]

        sigma = float(np.std([0.0, 2.0, 4.0]))
        assert population["sigma_pre"].unique() == pytest.approx(sigma)
        assert population["abs_delta_z"].to_numpy() == pytest.approx(1.0 / sigma)

    def test_the_scale_is_shared_by_every_instance_of_a_quantity(self):
        """One number per quantity, not one per instance."""
        table = measure_shift(
            observations({1: [0.0], 2: [2.0], 3: [9.0]}, {1: [1.0], 2: [2.0], 3: [9.0]})
        )
        population = table[table["standardization"] == "population"]
        assert len(population["sigma_pre"].unique()) == 1

    def test_post_is_standardized_with_the_pre_scale(self):
        """Standardizing post by its own spread would divide out the shift."""
        pre = {1: [0.0], 2: [1.0], 3: [2.0]}
        near = measure_shift(observations(pre, {1: [0.1], 2: [1.1], 3: [2.1]}))
        far = measure_shift(observations(pre, {1: [1.0], 2: [2.0], 3: [3.0]}))
        near_mean = near[near["standardization"] == "population"]["abs_delta_z"].mean()
        far_mean = far[far["standardization"] == "population"]["abs_delta_z"].mean()
        # Post rescaled by its own spread would put these two on the same number.
        assert far_mean == pytest.approx(10.0 * near_mean)

    def test_an_invariant_quantity_measures_exactly_zero(self):
        """The rank encodings and the G1 pair are expected to land here."""
        values = {1: [1.0, 2.0], 2: [3.0, 4.0], 3: [7.0, 1.0]}
        table = measure_shift(observations(values, values))
        assert (table["abs_delta_z"] == 0.0).all()
        assert (table["abs_difference"] == 0.0).all()

    def test_emits_one_row_per_instance_and_quantity(self):
        table = measure_shift(observations({1: [1.0], 2: [2.0]}, {1: [2.0], 2: [4.0]}))
        assert len(table) == 2
        assert set(table["standardization"]) == {"population"}
        assert set(SHIFT_KEYS) <= set(table.columns)

    def test_the_scale_is_the_population_deviation(self):
        """The spread uses ddof = 0, not the sample deviation."""
        table = measure_shift(
            observations({1: [0.0], 2: [1.0], 3: [3.0]}, {1: [1.0], 2: [1.0], 3: [3.0]})
        )
        sigma = float(np.std([0.0, 1.0, 3.0], ddof=0))
        assert table["sigma_pre"].unique() == pytest.approx(sigma)

    def test_a_constant_quantity_that_did_not_move_is_zero_not_undefined(self):
        """No spread and no shift: the answer is zero whatever the unit."""
        flat = {1: [4.0], 2: [4.0], 3: [4.0]}
        table = measure_shift(observations(flat, flat))
        assert (table["abs_delta_z"] == 0.0).all()
        assert table["no_spread"].all()

    def test_a_constant_quantity_that_moved_has_no_unit_to_report_in(self):
        """Only the instance that moved is undefined; the one that did not is zero."""
        table = measure_shift(observations({1: [4.0], 2: [4.0]}, {1: [9.0], 2: [4.0]}))
        assert table["no_spread"].all()
        moved = table[table["function_id"] == 1]
        assert moved["abs_delta_z"].isna().all()
        assert moved["abs_difference"].to_numpy() == pytest.approx(5.0)
        assert (table[table["function_id"] == 2]["abs_delta_z"] == 0.0).all()

    def test_a_spread_at_the_level_of_rounding_is_not_a_spread(self):
        """`np.std` of values that agree to the last bit is not exactly zero.

        Dividing a real shift by 4e-16 would yield 1e14 "standard deviations".
        """
        nine_elevenths = 9.0 / 11.0
        pre = {index: [nine_elevenths] for index in range(1, 25)}
        post = dict(pre)
        post[1] = [nine_elevenths + 0.05]
        table = measure_shift(observations(pre, post))
        assert table["no_spread"].all()
        assert table["abs_delta_z"].isna().sum() == 1  # the instance that moved

    def test_a_missing_view_is_rejected(self):
        table = observations({1: [1.0]}, {1: [2.0]})
        with pytest.raises(ValueError, match="missing transform"):
            measure_shift(cast(pd.DataFrame, table[table["transform"] == "pre"]))

    def test_unequal_repetition_counts_are_rejected(self):
        """An unpaired count would compare sample sizes as well as views."""
        table = observations({1: [1.0, 2.0, 3.0]}, {1: [1.0, 2.0]})
        with pytest.raises(ValueError, match="different numbers of repetitions"):
            measure_shift(table)


class TestSummarizeShift:
    @staticmethod
    def build(delta: list[float], raw: list[float], rep: int = 1) -> pd.DataFrame:
        """One quantity measured on len(delta) instances in one deployment."""
        return pd.DataFrame(
            {
                "problem_class": "bbob",
                "function_id": range(1, len(delta) + 1),
                "instance_id": 1,
                "dimension": 10,
                "rep": rep,
                "quantity_kind": "feature",
                "feature_set": "ela",
                "feature_variant": "value",
                "quantity": "ela_distr.skewness",
                "quantity_class": "ela_distr",
                "standardization": "population",
                "abs_delta_z": delta,
                "abs_difference": raw,
            }
        )

    def test_the_mean_is_taken_over_the_instances(self):
        summary = summarize_shift(self.build([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
        mean = summary[summary["statistic"] == "mean"]["value"].iloc[0]
        assert mean == pytest.approx(2.0)
        assert set(REPORT_KEYS) <= set(summary.columns)
        assert "rep" not in summary.columns

    def test_p_moved_is_read_from_the_unstandardized_shift(self):
        """So that an instance with no unit still counts as having moved."""
        summary = summarize_shift(
            self.build([0.0, float("nan"), 0.0, 0.0], [0.0, 5.0, 0.0, 0.0])
        )
        assert summary["p_moved"].iloc[0] == pytest.approx(0.25)
        assert summary["n_undefined"].iloc[0] == 1
        assert summary["n_instances"].iloc[0] == 4

    def test_the_standardized_statistic_skips_the_undefined_instances(self):
        summary = summarize_shift(self.build([2.0, float("nan")], [2.0, 9.0]))
        mean = summary[summary["statistic"] == "mean"]
        assert mean["value"].iloc[0] == pytest.approx(2.0)
        assert mean["value_raw"].iloc[0] == pytest.approx(5.5)

    def test_a_wholly_undefined_quantity_has_no_standardized_estimate(self):
        """Null, never zero: zero would say it did not move."""
        summary = summarize_shift(self.build([float("nan")] * 3, [1.0, 2.0, 3.0]))
        assert summary["value"].isna().all()
        assert (summary["value_raw"] > 0.0).all()
        assert (summary["p_moved"] == 1.0).all()


class TestObservationAdapters:
    def test_features_keep_every_deployment(self):
        features = pd.DataFrame(
            {
                "problem_class": "bbob",
                "function_id": 1,
                "instance_id": 1,
                "dimension": 10,
                "feature_set": "ela",
                "feature_variant": "value",
                "feature": "ela_distr.skewness",
                "transform": ["pre", "pre", "post", "post"],
                "rep": [1, 2, 1, 2],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        table = feature_observations(features)
        assert len(table) == 4
        # A deployment is kept as one, not averaged with the others.
        assert list(table["rep"]) == [1, 2, 1, 2]
        assert set(table["replicate"]) == {1}
        assert set(table["quantity_class"]) == {"ela_distr"}

    def test_the_superiority_is_taken_as_the_regression_reads_it(self):
        """One observation per view: the rate, not the trials behind it."""
        superiorities = pd.DataFrame(
            {
                "problem_class": "bbob",
                "function_id": 1,
                "instance_id": 1,
                "dimension": 10,
                "transform": ["pre", "post"],
                "pair": "shade__l-shade",
                "algorithm_a": "SHADE",
                "algorithm_b": "L-SHADE",
                "n_trials": 31,
                "superiority": [0.625, 0.25],
            }
        )
        table = instance_values(view_superiority_observations(superiorities, (1,)))
        assert list(table["n_replicates"]) == [1, 1]
        assert table.set_index("transform")["value"].to_dict() == pytest.approx(
            {"pre": 0.625, "post": 0.25}
        )

    def test_superiorities_belong_to_no_representation(self):
        superiorities = pd.DataFrame(
            {
                "problem_class": "bbob",
                "function_id": 1,
                "instance_id": 1,
                "dimension": 10,
                "transform": ["pre", "post"],
                "pair": "shade__l-shade",
                "algorithm_a": "SHADE",
                "algorithm_b": "L-SHADE",
                "n_trials": 31,
                "superiority": [1.0, 0.0],
            }
        )
        table = view_superiority_observations(superiorities, (1,))
        assert set(table["feature_set"]) == {NOT_A_FEATURE}
        assert set(table["feature_variant"]) == {NOT_A_FEATURE}
        # The pair's group, so that the figure can band the axis by it.
        assert set(table["quantity_class"]) == {"G2"}


def superiority_table(values: dict[str, list[float]]) -> pd.DataFrame:
    """One pair's rates over len(values["pre"]) instances, under both views."""
    rows = []
    for view, rates in values.items():
        rows.extend(
            {
                "problem_class": "bbob",
                "function_id": function_id,
                "instance_id": 1,
                "dimension": 10,
                "transform": view,
                "pair": "shade__l-shade",
                "algorithm_a": "SHADE",
                "algorithm_b": "L-SHADE",
                "n_trials": 31,
                "superiority": rate,
            }
            for function_id, rate in enumerate(rates, start=1)
        )
    return pd.DataFrame(rows)


class TestDeployments:
    """Each deployment on its own, then the median of them."""

    @staticmethod
    def three(deployments: dict[int, tuple[dict, dict]]) -> pd.DataFrame:
        """One quantity's observations in several deployments."""
        return pd.concat(
            [
                observations(pre, post, rep=rep)
                for rep, (pre, post) in deployments.items()
            ],
            ignore_index=True,
        )

    def test_the_scale_is_taken_within_each_deployment(self):
        """sigma_{j,r}: the deployment's own pre values over the instances."""
        table = measure_shift(
            self.three(
                {
                    1: ({1: [0.0], 2: [2.0]}, {1: [1.0], 2: [3.0]}),
                    2: ({1: [0.0], 2: [6.0]}, {1: [1.0], 2: [7.0]}),
                    3: ({1: [0.0], 2: [4.0]}, {1: [1.0], 2: [5.0]}),
                }
            ),
        )
        sigma = table.groupby("rep")["sigma_pre"].first().to_dict()
        assert sigma == pytest.approx({1: 1.0, 2: 3.0, 3: 2.0})
        assert set(QUANTITY_KEYS) >= {"rep"}

    def test_the_statistic_is_the_median_over_the_deployments(self):
        shift = pd.concat(
            [
                TestSummarizeShift.build([1.0, 1.0], [1.0, 1.0], rep=1),
                TestSummarizeShift.build([5.0, 5.0], [9.0, 9.0], rep=2),
                TestSummarizeShift.build([2.0, 2.0], [0.0, 3.0], rep=3),
            ],
            ignore_index=True,
        )
        summary = summarize_shift(shift)
        mean = summary[summary["statistic"] == "mean"].iloc[0]
        assert mean["value"] == pytest.approx(2.0)
        assert mean["value_raw"] == pytest.approx(1.5)
        # The deployments moved on 2/2, 2/2 and 1/2 of the instances.
        assert mean["p_moved"] == pytest.approx(1.0)
        assert len(summary) == 1

    def test_one_deployment_is_reported_as_itself(self):
        shift = TestSummarizeShift.build([1.0, 4.0], [1.0, 4.0])
        summary = summarize_shift(shift)
        assert summary[summary["statistic"] == "mean"]["value"].iloc[0] == 2.5

    def test_a_deployment_without_a_standardized_estimate_leaves_none(self):
        """The median is not taken over a subset of the deployments."""
        shift = pd.concat(
            [
                TestSummarizeShift.build([1.0], [1.0], rep=1),
                TestSummarizeShift.build([float("nan")], [2.0], rep=2),
                TestSummarizeShift.build([3.0], [3.0], rep=3),
            ],
            ignore_index=True,
        )
        summary = summarize_shift(shift)
        assert summary["value"].isna().all()
        assert summary[summary["statistic"] == "mean"]["value_raw"].iloc[0] == 2.0

    def test_an_even_number_of_deployments_is_refused(self):
        shift = pd.concat(
            [
                TestSummarizeShift.build([1.0], [1.0], rep=1),
                TestSummarizeShift.build([3.0], [3.0], rep=2),
            ],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="even number"):
            summarize_shift(shift)

    def test_a_superiority_reads_the_same_in_every_deployment(self):
        rates = superiority_table({"pre": [0.25, 0.5, 1.0], "post": [0.5, 0.5, 0.0]})
        once = summarize_shift(
            measure_shift(view_superiority_observations(rates, (1,)))
        )
        eleven = measure_shift(
            view_superiority_observations(rates, tuple(range(1, 12)))
        )
        assert sorted(eleven["rep"].unique()) == list(range(1, 12))
        by_deployment = eleven.drop(columns="rep").groupby(
            ["function_id", "standardization"]
        )
        assert (by_deployment["abs_delta_z"].nunique() == 1).all()
        pd.testing.assert_frame_equal(summarize_shift(eleven), once)
