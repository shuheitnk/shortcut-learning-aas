"""Tests for Experiment 1's judgement, read from two run roots.

Each predictor set is fitted by its own stage call into its own run root, differing
only in the columns, so the comparison narrows Experiment 4's conditions.
"""

from __future__ import annotations

from typing import ClassVar, cast

import pandas as pd
import pytest

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.compare_predictor_sets import (
    ALL_PREDICTORS,
    INVARIANT_PREDICTORS,
    LEVEL_COLUMNS,
    PREDICTOR_SETS,
    REPORTED_PAIR,
    REPORTED_VARIANT,
    Cell,
    SourceCells,
    compare,
    hero_frame,
    invariant_cell_config,
    invariant_columns,
    median_levels,
    per_seed_runs,
    shortcut_detected,
)
from aas_shortcut.compare_training_runs import PAIRED
from aas_shortcut.define_paths import ELA_FEATURE_SET, RunPaths
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_augmented_regression import (
    DEPLOYMENT_SHARD,
    AugmentedRegressionConfig,
    run_augmented_regression_stage,
)
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.read_write_tables import read_table
from tests.conftest import FEATURE_NAMES, make_features
from tests.test_measure_representation_transfer import G1, make_view_superiorities

# The invariant set is a subset of the full one, as the experiment's 36 are of 59.
ALL_COLUMNS = FEATURE_NAMES
INVARIANT_COLUMNS = FEATURE_NAMES[:2]

# Three training runs rather than eleven, to keep the test fast.
SEEDS = MODEL_SEEDS[:3]

INSTANCES = 22
TRAIN_INSTANCES = tuple(range(1, 21))
TEST_INSTANCES = (21, 22)


def _config(columns, candidates=("random-forest",)) -> AugmentedRegressionConfig:
    """One narrowed cell, as Experiment 1 narrows Experiment 4's configuration."""
    return AugmentedRegressionConfig(
        feature_set=ELA_FEATURE_SET,
        variants=("value",),
        pairs=(G1,),
        trainings=(TRAINING_PLAIN,),
        audits=(CONTROL.label,),
        features=tuple(sorted(columns)),
        train_instances=TRAIN_INSTANCES,
        test_instances=TEST_INSTANCES,
        candidates=candidates,
        model=ModelConfig(kind="random-forest"),
        n_seeds=len(SEEDS),
    )


def _run(root, columns, predictors: str) -> RunPaths:
    """Fit one predictor set's cell into its own run root."""
    features = make_features(n_instances=INSTANCES, n_reps=3)
    features = cast(pd.DataFrame, features[features["feature"].isin(list(columns))])
    superiorities = make_view_superiorities(n_instances=INSTANCES)

    paths = RunPaths.from_root(root)
    paths.mkdirs()
    config = _config(columns)
    run_augmented_regression_stage(
        cast(pd.DataFrame, features[features["instance_id"].isin(TRAIN_INSTANCES)]),
        superiorities,
        {CONTROL.label: features},
        {CONTROL.label: superiorities},
        paths,
        "debug",
        f"hash-{predictors}",
        config,
        deployment_name=DEPLOYMENT_SHARD,
    )
    return paths


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """Both halves of the comparison, each in its own run root."""
    root = tmp_path_factory.mktemp("compare")
    _run(root / "experiment4", ALL_COLUMNS, ALL_PREDICTORS)
    _run(root / "experiment1_invariant", INVARIANT_COLUMNS, INVARIANT_PREDICTORS)
    return root


@pytest.fixture(scope="module")
def source() -> SourceCells:
    """Where the two halves are read from."""
    return SourceCells(
        all_cell=Cell(ALL_PREDICTORS, "experiment4"),
        invariant_cell=Cell(INVARIANT_PREDICTORS, "experiment1_invariant"),
        seeds=SEEDS,
    )


@pytest.fixture(scope="module")
def levels(roots, source) -> pd.DataFrame:
    return median_levels(roots, source)


class TestTheReportedLevels:
    """The unit the paper's table carries: the median of the training runs."""

    def test_it_holds_one_row_per_predictor_set(self, levels):
        assert list(levels["predictors"]) == list(PREDICTOR_SETS)

    def test_the_invariant_set_is_the_narrower_one(self, levels):
        counted = levels.set_index("predictors")["n_features"]
        assert counted[INVARIANT_PREDICTORS] < counted[ALL_PREDICTORS]
        assert counted[ALL_PREDICTORS] == len(ALL_COLUMNS)
        assert counted[INVARIANT_PREDICTORS] == len(INVARIANT_COLUMNS)

    def test_every_level_says_which_unit_it_is_in(self, levels):
        """A column that did not would let the two units be read as one."""
        scored = set(levels.columns) - {"predictors", "n_features"}
        assert all(name.endswith("_median") for name in scored), scored

    def test_a_level_is_the_median_over_the_runs_of_each_run_s_median(
        self, roots, source, levels
    ):
        """The test's observation, reduced the way the table is."""
        scored = levels.set_index("predictors")
        for cell in source.cells:
            shard = cell.paths(roots).metric(cell.deployments)
            runs = read_table(shard.data)
            for metric, name in LEVEL_COLUMNS.items():
                rows = runs[runs["metric"] == metric]
                expected = rows.groupby("seed")["value"].median().median()
                assert scored.loc[cell.predictors, name] == expected

    def test_the_table_carries_the_levels_and_nothing_else(self, levels):
        """The table carries the levels only, not their differences."""
        assert set(levels.columns) == {
            "predictors",
            "n_features",
            *LEVEL_COLUMNS.values(),
        }

    @staticmethod
    def judged(all_pre, all_post, invariant_pre, invariant_post) -> bool:
        pre, post = (LEVEL_COLUMNS[TEST_METRICS[view]] for view in ("pre", "post"))
        frame = pd.DataFrame(
            [
                {"predictors": ALL_PREDICTORS, pre: all_pre, post: all_post},
                {
                    "predictors": INVARIANT_PREDICTORS,
                    pre: invariant_pre,
                    post: invariant_post,
                },
            ]
        )
        return shortcut_detected(frame)

    def test_the_verdict_needs_both_conditions_of_the_definition(self):
        """`all` must be above before the transformation and below after."""
        assert self.judged(0.9, 0.2, 0.7, 0.6)
        assert not self.judged(0.6, 0.2, 0.7, 0.6), "below before: no shortcut"
        assert not self.judged(0.9, 0.8, 0.7, 0.6), "above after: no shortcut"

    def test_the_verdict_reads_the_levels_of_the_two_cells(self, levels):
        scored = levels.set_index("predictors")
        pre, post = (LEVEL_COLUMNS[TEST_METRICS[view]] for view in ("pre", "post"))
        expected = (
            scored.loc[ALL_PREDICTORS, pre] > scored.loc[INVARIANT_PREDICTORS, pre]
            and scored.loc[INVARIANT_PREDICTORS, post]
            > scored.loc[ALL_PREDICTORS, post]
        )
        assert shortcut_detected(levels) == bool(expected)

    def test_a_missing_side_is_refused(self, levels):
        with pytest.raises(ValueError, match="nothing to compare"):
            shortcut_detected(levels[levels["predictors"] == ALL_PREDICTORS])


class TestTheTestedRuns:
    """The other unit: every training run, paired by seed number."""

    def test_it_holds_every_run_of_both_cells(self, roots, source):
        runs = per_seed_runs(roots, source)
        held = runs.groupby("predictors")["seed"].agg(set)
        for predictors in PREDICTOR_SETS:
            assert held[predictors] == set(SEEDS)

    def test_its_column_says_which_unit_it_is_in(self, roots, source):
        runs = per_seed_runs(roots, source)
        assert "r2_post_seed" in runs.columns

    def test_a_cell_short_of_a_run_is_refused(self, roots, source):
        """A test over a subset of the runs is not the comparison that was fixed."""
        with pytest.raises(ValueError, match="missing training run"):
            per_seed_runs(roots, SourceCells(*source.cells, seeds=(*SEEDS, 999)))

    def test_the_comparison_pairs_the_two_cells_by_seed(self, roots, source):
        tested = compare(per_seed_runs(roots, source), SEEDS)
        assert len(tested) == 1
        row = tested.iloc[0]
        assert row["n_pairs"] == len(SEEDS)
        assert row["test"] == PAIRED
        assert row["unit"] == "model_seed"
        assert row["left"] == INVARIANT_PREDICTORS
        assert row["right"] == ALL_PREDICTORS
        assert 0.0 <= row["p"] <= 1.0


class TestTheFigureFrame:
    """Fig. 14 reads the table's levels, one row per predictor set and block."""

    EXPECTED: ClassVar[set[str]] = set(TEST_METRICS.values())

    def test_the_long_frame_holds_one_row_per_set_and_block(self, levels):
        frame = hero_frame(levels)
        assert len(frame) == len(PREDICTOR_SETS) * len(TEST_METRICS)
        assert set(frame["metric"]) == self.EXPECTED


class TestTheInvariantCell:
    def test_its_columns_are_the_fixed_list_under_the_name_rule(self):
        """36 invariant columns at the paper's scale, 33 at the debug one."""
        assert len(invariant_columns("full")) == 36
        assert len(invariant_columns("debug")) == 33
        assert set(invariant_columns("debug")) < set(invariant_columns("full"))

    def test_its_configuration_is_one_cell_scored_on_the_control_family(self):
        config = invariant_cell_config(
            ModelConfig(), ["realmlp"], invariant_columns("full")
        )
        assert config.pairs == (REPORTED_PAIR,)
        assert config.variants == (REPORTED_VARIANT,)
        assert config.trainings == (TRAINING_PLAIN,)
        assert config.audits == (CONTROL.label,)
        assert config.features == invariant_columns("full")
