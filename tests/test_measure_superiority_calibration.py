"""Tests for the calibration check of the target the regression is fitted on.

It reads the benchmark's superiorities over the regression's block and reports how
uniform the block is and in what shape it is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.measure_superiority_calibration import (
    CALIBRATION_SHARD,
    DECILE_SHARD,
    DECILES,
    CalibrationConfig,
    calibration_deciles,
    calibration_summary,
    run_calibration_stage,
)
from aas_shortcut.read_write_tables import read_table
from aas_shortcut.tune_algorithms import comparisons_per_rate, floor_for, ks_distance

TUNING = (11,)
TRAINING = tuple(range(11, 21))
N_TRIALS = 31


def shard(values_for) -> pd.DataFrame:
    """A view-superiority shard whose rates are set by ``values_for(pair, f, i)``."""
    rows = [
        {
            "problem_class": "bbob",
            "function_id": function_id,
            "instance_id": instance_id,
            "dimension": 10,
            "transform": transform,
            "pair": pair.slug,
            "algorithm_a": pair.first,
            "algorithm_b": pair.second,
            "pair_type": pair.pair_type,
            "n_trials": N_TRIALS,
            "superiority": values_for(pair, function_id, instance_id),
        }
        for pair in FOCUS_PAIRS
        for function_id in range(1, 25)
        for instance_id in TRAINING
        for transform in ("pre", "post")
    ]
    return pd.DataFrame(rows)


def read_calibration(run_paths: RunPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both tables as the stage wrote them into the run root."""
    return (
        read_table(run_paths.metric(CALIBRATION_SHARD).data),
        read_table(run_paths.metric(DECILE_SHARD).data),
    )


def spread(pair, function_id, instance_id) -> float:
    """Rates fanned over [0, 1] so no block is degenerate."""
    del pair, instance_id
    return (function_id - 1) / 23.0


CONFIG = CalibrationConfig(blocks={"tuning": TUNING, "training": TRAINING})


class TestTheSummaryReadsTheBlockItNames:
    @pytest.fixture
    def summary(self):
        return calibration_summary(shard(spread), CONFIG)

    def test_one_row_per_block_and_pair(self, summary):
        assert len(summary) == 2 * len(FOCUS_PAIRS)
        assert not summary.duplicated(["block", "pair"]).any()

    def test_the_blocks_carry_the_sizes_they_name(self, summary):
        sizes = summary.set_index(["block", "group"])["n_problems"]
        for pair in FOCUS_PAIRS:
            assert sizes[("tuning", group_of(pair))] == 24
            assert sizes[("training", group_of(pair))] == 24 * len(TRAINING)

    def test_it_reads_the_untransformed_view(self):
        """The binding was chosen on `pre`; `post` is a different question."""

        def by_view(pair, function_id, instance_id):
            del pair, instance_id
            return (function_id - 1) / 23.0

        moved = shard(by_view).copy()
        moved.loc[moved["transform"] == "post", "superiority"] = 1.0
        summary = calibration_summary(moved, CONFIG)
        assert set(summary["transform"]) == {"pre"}
        # Untouched by the `post` rows being saturated.
        assert (summary["saturated_share"] < 0.1).all()

    def test_the_ks_distance_is_the_tunings_own_definition(self, summary):
        block = shard(spread)
        rates = block[
            (block["pair"] == FOCUS_PAIRS[0].slug)
            & (block["transform"] == "pre")
            & (block["instance_id"] == TUNING[0])
        ]["superiority"].to_numpy(float)
        row = summary[
            (summary["block"] == "tuning") & (summary["pair"] == FOCUS_PAIRS[0].slug)
        ].iloc[0]
        assert row["ks_distance"] == pytest.approx(ks_distance(rates))

    def test_the_floor_falls_as_the_block_grows(self, summary):
        """2400 problems admit a far smaller D_KS than 24; the floor says so."""
        floors = summary.set_index("block")["ks_floor"]
        assert floors["training"].max() < floors["tuning"].min()

    def test_the_floor_is_at_the_estimators_granularity(self, summary):
        """The tuning's floor: A12 over every pair of the two members' runs."""
        floors = summary.set_index("block")["ks_floor"]
        comparisons = comparisons_per_rate(N_TRIALS)
        assert (floors["tuning"] == floor_for(24, comparisons)).all()
        assert (floors["training"] == floor_for(24 * len(TRAINING), comparisons)).all()
        assert floor_for(24, comparisons) < floor_for(24, N_TRIALS)

    def test_the_unit_names_its_estimator(self):
        """The unit's fingerprint names its estimator."""
        assert CONFIG.as_dict()["estimator"] == "vargha-delaney"

    def test_the_variance_is_the_population_one(self, summary):
        block = shard(spread)
        rates = block[
            (block["pair"] == FOCUS_PAIRS[0].slug)
            & (block["transform"] == "pre")
            & (block["instance_id"].isin(TRAINING))
        ]["superiority"].to_numpy(float)
        row = summary[
            (summary["block"] == "training") & (summary["pair"] == FOCUS_PAIRS[0].slug)
        ].iloc[0]
        assert row["variance"] == pytest.approx(rates.var(ddof=0))
        assert row["variance"] != pytest.approx(rates.var(ddof=1))

    def test_saturation_counts_only_the_ends(self):
        def ends(pair, function_id, instance_id):
            del pair, instance_id
            return 0.0 if function_id <= 6 else (1.0 if function_id >= 19 else 0.5)

        summary = calibration_summary(shard(ends), CONFIG)
        row = summary[summary["block"] == "tuning"].iloc[0]
        assert row["n_saturated"] == 12
        assert row["saturated_share"] == pytest.approx(0.5)
        # 0.5 is inside the band, 0 and 1 are not.
        assert row["informative_share"] == pytest.approx(0.5)

    def test_a_block_the_shard_does_not_hold_is_rejected(self):
        with pytest.raises(ValueError, match="no pre row"):
            calibration_summary(shard(spread), CalibrationConfig(blocks={"x": (999,)}))


class TestTheDecilesShowTheShape:
    def test_ten_bins_per_block_and_pair(self):
        deciles = calibration_deciles(shard(spread), CONFIG)
        assert len(deciles) == 2 * len(FOCUS_PAIRS) * DECILES
        counted = deciles.groupby(["block", "pair"], observed=True).size()
        assert set(counted) == {DECILES}

    def test_the_counts_add_up_to_the_block(self):
        deciles = calibration_deciles(shard(spread), CONFIG)
        totals = deciles.groupby(["block", "pair"], observed=True)["n_problems"].sum()
        assert totals[("tuning", FOCUS_PAIRS[0].slug)] == 24
        assert totals[("training", FOCUS_PAIRS[0].slug)] == 24 * len(TRAINING)

    def test_both_saturated_ends_land_in_a_bin(self):
        """A rate of exactly 1 belongs to the last bin, not outside the range."""

        def ends(pair, function_id, instance_id):
            del pair, instance_id
            return 0.0 if function_id <= 12 else 1.0

        deciles = calibration_deciles(shard(ends), CONFIG)
        one = deciles[
            (deciles["block"] == "tuning") & (deciles["pair"] == FOCUS_PAIRS[0].slug)
        ].sort_values("decile")
        assert one["n_problems"].sum() == 24
        assert one.iloc[0]["n_problems"] == 12
        assert one.iloc[-1]["n_problems"] == 12
        assert one.iloc[1:-1]["n_problems"].sum() == 0

    def test_a_u_shape_and_a_flat_block_are_told_apart(self):
        """The summary cannot separate them; this is why the deciles exist."""

        def flat(pair, function_id, instance_id):
            del pair, instance_id
            return (function_id - 0.5) / 24.0

        def u_shaped(pair, function_id, instance_id):
            del pair, instance_id
            return 0.0 if function_id % 2 else 1.0

        shapes = {}
        for name, values in (("flat", flat), ("u", u_shaped)):
            page = calibration_deciles(shard(values), CONFIG)
            page = page[
                (page["block"] == "tuning") & (page["pair"] == FOCUS_PAIRS[0].slug)
            ].sort_values("decile")
            shapes[name] = page["n_problems"].to_numpy()
        assert shapes["flat"].max() - shapes["flat"].min() <= 1
        assert shapes["u"][0] > 0 and shapes["u"][-1] > 0
        assert shapes["u"][1:-1].sum() == 0

    def test_the_expected_count_is_the_uniform_one(self):
        deciles = calibration_deciles(shard(spread), CONFIG)
        training = deciles[deciles["block"] == "training"]
        assert set(training["expected_uniform"]) == {24 * len(TRAINING) / DECILES}


class TestTheStageWritesBothTables:
    def test_it_writes_and_reads_back(self, tmp_path):
        run_paths = RunPaths.from_root(tmp_path / "experiment2")
        run_paths.mkdirs()
        summary, deciles = run_calibration_stage(
            shard(spread), run_paths, "full", "abc123", CONFIG
        )
        assert run_paths.metric(CALIBRATION_SHARD).data.exists()
        assert run_paths.metric(DECILE_SHARD).data.exists()
        assert set(summary["scale"]) == {"full"}
        assert set(deciles["config_hash"]) == {"abc123"}

        read_summary, read_deciles = read_calibration(run_paths)
        assert len(read_summary) == len(summary)
        assert len(read_deciles) == len(deciles)
        assert read_summary["ks_distance"].to_numpy() == pytest.approx(
            summary["ks_distance"].to_numpy()
        )


class TestTheConfigRejectsWhatItCannotRead:
    def test_an_unknown_view(self):
        with pytest.raises(ValueError, match="unknown view"):
            CalibrationConfig(transform="sideways")

    def test_an_empty_block(self):
        with pytest.raises(ValueError, match="names no instance"):
            CalibrationConfig(blocks={"training": ()})

    def test_no_pair(self):
        with pytest.raises(ValueError, match="at least one pair"):
            CalibrationConfig(pairs=())

    def test_a_mixed_trial_count_is_rejected(self):
        """The floor is defined against one granularity."""
        mixed = shard(spread)
        mixed.loc[mixed["function_id"] == 3, "n_trials"] = 15
        with pytest.raises(ValueError, match="granularity floor"):
            calibration_summary(mixed, CONFIG)


def test_the_fingerprint_moves_with_the_blocks():
    """A table built from another block is another measurement."""
    one = CalibrationConfig(blocks={"training": TRAINING}).as_dict()
    other = CalibrationConfig(blocks={"training": TRAINING[:5]}).as_dict()
    assert one != other
    assert one["stage"] == "superiority_calibration"


def test_the_summary_and_the_deciles_agree_on_the_block_size():
    summary = calibration_summary(shard(spread), CONFIG)
    deciles = calibration_deciles(shard(spread), CONFIG)
    counted = deciles.groupby(["block", "pair"], observed=True)["n_problems"].sum()
    for row in summary.itertuples():
        assert counted[(row.block, row.pair)] == row.n_problems
        assert np.isclose(
            row.n_problems / DECILES,
            deciles[(deciles["block"] == row.block) & (deciles["pair"] == row.pair)][
                "expected_uniform"
            ].iloc[0],
        )
