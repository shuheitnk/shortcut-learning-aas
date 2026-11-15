"""The appendix's levels: medians and quartiles over the seeds, and Spearman's rho."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.measure_lofo_regression import METRIC
from aas_shortcut.summarize_lofo_levels import (
    function_rho_by_group,
    level_table,
    lofo_runs,
    prediction_runs,
    rrt_correlations,
    spread_correlations,
    summary_table,
)


def _scores(values_by_seed: dict[int, list[float]], variant: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set": "ela",
                "group": "G1",
                "feature_variant": variant,
                "training": "plain",
                "seed": seed,
                "rep": rep,
                "metric": METRIC,
                "value": value,
            }
            for seed, values in values_by_seed.items()
            for rep, value in enumerate(values, start=1)
        ]
    )


class TestLevels:
    def test_a_seed_is_its_median_deployment(self):
        runs = lofo_runs(_scores({42: [0.1, 0.5, 0.3]}, "value"))
        assert runs["value"].tolist() == [0.3]

    def test_the_level_is_the_median_with_quartiles_over_the_seeds(self):
        scores = _scores({s: [0.1 * i] for i, s in enumerate(range(42, 47))}, "value")
        row = level_table(scores).iloc[0]
        assert row["lofo_median"] == pytest.approx(0.2)
        assert row["lofo_q1"] == pytest.approx(0.1)
        assert row["lofo_q3"] == pytest.approx(0.3)
        assert row["lofo_n_seeds"] == 5


def _predictions(scale: float, variant: str) -> pd.DataFrame:
    """Two functions of two problems; predictions are the targets times ``scale``."""
    observed = [0.1, 0.2, 0.7, 0.9]
    return pd.DataFrame(
        {
            "feature_set": "ela",
            "group": "G1",
            "feature_variant": variant,
            "training": "plain",
            "seed": 42,
            "rep": 1,
            "function_id": [1, 1, 2, 2],
            "observed": observed,
            "predicted": [scale * value for value in observed],
        }
    )


class TestPredictions:
    def test_spread_and_function_rho_come_from_one_deployment(self):
        runs = prediction_runs(_predictions(2.0, "value"))
        row = runs.iloc[0]
        assert row["prediction_sd"] == pytest.approx(2 * row["observed_sd"])
        assert row["function_rho"] == pytest.approx(1.0)

    def test_the_summary_joins_the_r2_and_the_spread(self):
        scores = _scores({42: [-0.5]}, "value")
        row = summary_table(scores, _predictions(2.0, "value")).iloc[0]
        assert row["lofo_r2"] == pytest.approx(-0.5)
        assert row["sd_ratio"] == pytest.approx(2.0)


class TestSpreadCorrelations:
    def test_rho_per_group_and_over_all_groups(self):
        summary = pd.DataFrame(
            {
                "group": ["G1"] * 3 + ["G2"] * 3,
                "prediction_sd": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                "sd_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "lofo_r2": [-0.1, -0.2, -0.3, -0.3, -0.2, -0.1],
            }
        )
        table = spread_correlations(summary).set_index("scope")
        assert table.loc["G1", "rho"] == pytest.approx(-1.0)
        assert table.loc["G2", "rho"] == pytest.approx(1.0)
        assert table.loc["all groups", "spread"] == "sd_ratio"
        assert table.loc["all groups", "n_points"] == 6


class TestPerGroupTables:
    def test_rrt_rho_is_taken_per_group(self):
        levels = pd.DataFrame(
            {
                "group": ["G1"] * 3 + ["G2"] * 3,
                "rrt_median": [0.1, 0.2, 0.3] * 2,
                "lofo_median": [-0.9, -0.5, -0.1, -0.1, -0.5, -0.9],
            }
        )
        table = rrt_correlations(levels).set_index("group")
        assert table.loc["G1", "rho"] == pytest.approx(1.0)
        assert table.loc["G2", "rho"] == pytest.approx(-1.0)

    def test_the_function_rho_is_summarized_over_the_conditions(self):
        summary = pd.DataFrame({"group": ["G3"] * 3, "function_rho": [0.0, 0.3, 0.1]})
        row = function_rho_by_group(summary).iloc[0]
        assert row["n_conditions"] == 3
        assert row["function_rho_median"] == pytest.approx(0.1)
        assert row["function_rho_min"] == pytest.approx(0.0)
        assert row["function_rho_max"] == pytest.approx(0.3)
