"""Experiment 4's entry point: what it measures, and where it writes it.

The measurement itself is tested in ``tests/test_measure_augmented_regression.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiment4"))

from run_experiment4 import (
    SPEC_PATH,
    TUNING_STEP,
    draw_conditions,
    report_audits,
)

from aas_shortcut.compare_training_runs import (
    EFFECT_COLUMN,
    PAIRED_EFFECT,
    SIZE_COLUMN,
    TIED,
    UNPAIRED_EFFECT,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS, GROUPS
from aas_shortcut.define_paths import ELA_FEATURE_SET, TRANSFORMS, RunPaths
from aas_shortcut.define_repetitions import MODEL_SEEDS, REPETITIONS
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
    SCOPES,
)
from aas_shortcut.define_warp_audits import AUDITS, CONTROL
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_augmented_regression import (
    AugmentedRegressionConfig,
    board_columns,
    seed_levels,
)
from aas_shortcut.measure_representation_transfer import VARIANTS, group_of
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_CONDITIONS,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    compose_config,
    resolve_run_paths,
)
from aas_shortcut.sample_monotone_transform import (
    AUGMENTATION_DOMAIN,
    HELD_OUT_DOMAIN,
    SPLINE_FAMILY,
    WarpConfig,
)


@pytest.fixture
def spec() -> ExperimentSpec:
    return ExperimentSpec.load(SPEC_PATH)


class TestSpec:
    def test_the_spec_is_experiment_4(self, spec):
        assert spec.id == "experiment4"
        assert spec.preset == "full"

    def test_it_writes_its_report_to_its_own_run_root(self, spec):
        assert "paths.run_name=experiment4" in spec.overrides

    def test_it_binds_from_experiment_2s_tuning_step(self):
        """It binds from Experiment 2's tuning step, so `plain` matches Experiment 2."""
        assert TUNING_STEP == "E2-1"


class TestAuditPoints:
    def test_every_point_names_a_warp_that_can_be_drawn(self):
        for audit in AUDITS:
            WarpConfig(family=audit.family, domain=audit.domain, knots=audit.knots)

    def test_the_control_is_the_family_the_augmentation_was_drawn_from(self):
        assert CONTROL.family == SPLINE_FAMILY
        assert CONTROL.domain == HELD_OUT_DOMAIN

    def test_the_training_side_keeps_the_augmentation_domain(self, spec):
        """Its features and its transformed runs are drawn from that domain."""
        cfg = compose_config(spec.hydra_overrides(spec.preset))
        assert cfg.task.warp_domain == AUGMENTATION_DOMAIN

    def test_a_rehearsal_writes_beside_the_reported_report(self, spec, tmp_path):
        cfg = compose_config(
            spec.hydra_overrides("debug", [f"paths.output_root={tmp_path}"])
        )
        assert resolve_run_paths(cfg).root == tmp_path / "experiment4_debug"


def make_deployments(labels: tuple[str, ...], lifted: bool = False) -> pd.DataFrame:
    """A per-deployment table whose medians are known.

    Runs are centred on 0.8 untransformed and on 0.40 less 0.02 per family
    transformed. With ``lifted``, only G3 x `value` is lifted by augmentation.
    """
    middle_seed = MODEL_SEEDS[len(MODEL_SEEDS) // 2]
    middle_rep = (REPETITIONS + 1) // 2
    rows = []
    for pair in FOCUS_PAIRS:
        for variant in VARIANTS:
            for training in TRAINING_CONDITIONS:
                lift = (
                    0.5
                    if lifted
                    and group_of(pair) == GROUPS[-1]
                    and variant == VARIANTS[0]
                    and training == TRAINING_AUGMENTED
                    else 0.0
                )
                for index, label in enumerate(labels):
                    for transform, metric in TEST_METRICS.items():
                        base = 0.8 if transform == "pre" else 0.4 - 0.02 * index
                        for seed in MODEL_SEEDS:
                            for rep in range(1, REPETITIONS + 1):
                                rows.append(
                                    {
                                        "feature_set": ELA_FEATURE_SET,
                                        "feature_variant": variant,
                                        "training": training,
                                        "audit": label,
                                        "pair": pair.slug,
                                        "group": group_of(pair),
                                        "model": "realmlp",
                                        "n_features": 107,
                                        "seed": seed,
                                        "rep": rep,
                                        "metric": metric,
                                        "value": base
                                        + (lift if transform == "post" else 0.0)
                                        + 0.001 * (seed - middle_seed)
                                        + 0.0001 * (rep - middle_rep),
                                        "n_test": 240,
                                        "scale": "debug",
                                        "config_hash": "0123456789abcdef",
                                    }
                                )
    return pd.DataFrame(rows)


class TestReportAudits:
    """The tables one feature set's reading is written as."""

    def make(self, tmp_path, lifted: bool = False):
        labels = tuple(audit.label for audit in AUDITS)
        deployments = make_deployments(labels, lifted)
        table = seed_levels(deployments)
        config = AugmentedRegressionConfig(feature_set=ELA_FEATURE_SET, audits=labels)
        run_paths = RunPaths.from_root(tmp_path / "experiment4")
        run_paths.mkdirs()
        written = report_audits(table, deployments, config, run_paths)
        return run_paths, written

    def make_with_runs(self, tmp_path):
        """The same reading, one cell lifted by augmentation."""
        return self.make(tmp_path, lifted=True)

    def test_fig_27_is_drawn_from_the_long_table_one_file_per_condition(self, tmp_path):
        """The two training conditions, unseen families, transformed block."""
        run_paths, _ = self.make(tmp_path)
        written = draw_conditions({ELA_FEATURE_SET: pd.DataFrame()}, run_paths)
        assert sorted(path.name for path in written if path.suffix == ".png") == [
            "fig27_augmentation_regression_augmented.png",
            "fig27_augmentation_regression_plain.png",
        ]

    @staticmethod
    def levels(run_paths, scope: str, transform: str = "post") -> pd.DataFrame:
        """The rows of one held-out block and range of the table of levels."""
        levels = pd.read_csv(run_paths.root / "group_regression_levels.csv")
        return levels[
            (levels[SCOPE_COLUMN] == scope) & (levels["transform"] == transform)
        ]

    def test_the_levels_and_the_two_comparisons_are_the_tables(self, tmp_path):
        run_paths, written = self.make(tmp_path)
        assert sorted(path.name for path in written) == [
            "group_regression_levels.csv",
            "representation_tests.csv",
            "training_tests.csv",
        ]
        assert all(path.exists() for path in written)
        levels = pd.read_csv(run_paths.root / "group_regression_levels.csv")
        assert len(levels) == len(TRANSFORMS) * len(SCOPES) * len(GROUPS) * 6

    def test_a_cell_is_the_median_of_its_runs_not_a_run_s_own_score(self, tmp_path):
        """Lifting the lowest seed's runs moves the median one step, to 0.401."""
        labels = tuple(audit.label for audit in AUDITS)
        deployments = make_deployments(labels)
        first = deployments["seed"] == MODEL_SEEDS[0]
        deployments.loc[first, "value"] += 0.3
        config = AugmentedRegressionConfig(feature_set=ELA_FEATURE_SET, audits=labels)
        run_paths = RunPaths.from_root(tmp_path / "experiment4")
        run_paths.mkdirs()
        table = seed_levels(deployments)
        report_audits(table, deployments, config, run_paths)
        level = self.levels(run_paths, SCOPE_CONTROL)["level"]
        assert level.to_numpy() == pytest.approx(0.401)

    def test_each_range_is_three_groups_by_six_cells(self, tmp_path):
        run_paths, _ = self.make(tmp_path)
        for scope in SCOPES:
            rows = self.levels(run_paths, scope)
            assert list(dict.fromkeys(rows["group"])) == list(GROUPS)
            cells = list(
                dict.fromkeys(
                    zip(rows["feature_variant"], rows["training"], strict=True)
                )
            )
            assert cells == board_columns()

    def test_an_unseen_cell_carries_the_mean_and_the_spread_of_the_four(self, tmp_path):
        """The four unseen families (0.38 down to 0.32) give mean 0.35, spread 0.06."""
        run_paths, _ = self.make(tmp_path)
        rows = self.levels(run_paths, SCOPE_UNSEEN)
        assert rows["level"].to_numpy() == pytest.approx(0.35)
        assert rows["spread"].to_numpy() == pytest.approx(0.06)

    def test_a_known_cell_carries_the_known_family_s_level_alone(self, tmp_path):
        """One family: nothing to spread over, so no spread is written."""
        run_paths, _ = self.make(tmp_path)
        rows = self.levels(run_paths, SCOPE_CONTROL)
        assert rows["level"].to_numpy() == pytest.approx(0.40)
        assert rows["spread"].isna().all()

    def test_the_untransformed_block_has_its_own_rows(self, tmp_path):
        """The untransformed block has its own rows, with zero spread over families."""
        run_paths, _ = self.make(tmp_path)
        unseen = self.levels(run_paths, SCOPE_UNSEEN, "pre")
        assert unseen["level"].to_numpy() == pytest.approx(0.8)
        assert unseen["spread"].to_numpy() == pytest.approx(0.0, abs=1e-12)
        known = self.levels(run_paths, SCOPE_CONTROL, "pre")
        assert known["level"].to_numpy() == pytest.approx(0.8)

    def test_it_draws_no_figure(self, tmp_path):
        """Fig. 27 is drawn by :func:`draw_conditions`, from the table of levels."""
        _, written = self.make(tmp_path)
        assert all(path.suffix == ".csv" for path in written)


class TestThePlannedComparisons:
    """The two comparisons reach the artifacts, each carrying its own size and sign."""

    def test_both_tables_are_written_when_the_runs_are_there(self, tmp_path):
        run_paths, written = TestReportAudits().make_with_runs(tmp_path)
        assert len(written) == 3
        assert (run_paths.root / "representation_tests.csv").exists()
        assert (run_paths.root / "training_tests.csv").exists()

    def test_the_lifted_cell_is_the_one_the_comparison_names(self, tmp_path):
        run_paths, _ = TestReportAudits().make_with_runs(tmp_path)
        tested = pd.read_csv(run_paths.root / "training_tests.csv")
        found = tested[tested["significant"]]
        assert set(
            zip(found["group"], found["feature_variant"], found["higher"], strict=True)
        ) == {(GROUPS[-1], VARIANTS[0], TRAINING_AUGMENTED)}
        assert set(found[SCOPE_COLUMN]) == set(SCOPES)

    def test_each_table_names_the_size_it_carries(self, tmp_path):
        """The two tests' sizes are different quantities, so each says which."""
        run_paths, _ = TestReportAudits().make_with_runs(tmp_path)
        for name, kind in (
            ("representation_tests.csv", PAIRED_EFFECT),
            ("training_tests.csv", UNPAIRED_EFFECT),
        ):
            table = pd.read_csv(run_paths.root / name)
            assert SIZE_COLUMN in table.columns
            assert set(table[EFFECT_COLUMN]) == {kind}

    def test_the_representations_reach_the_table_under_both_ranges(self, tmp_path):
        """The unseen families' mean and the known family alone."""
        run_paths, _ = TestReportAudits().make_with_runs(tmp_path)
        tested = pd.read_csv(run_paths.root / "representation_tests.csv")
        assert set(tested[SCOPE_COLUMN]) == {SCOPE_UNSEEN, SCOPE_CONTROL}

    def test_augmentation_reaches_the_table_under_both_ranges(self, tmp_path):
        """The training conditions too, each range its own Holm family."""
        run_paths, _ = TestReportAudits().make_with_runs(tmp_path)
        tested = pd.read_csv(run_paths.root / "training_tests.csv")
        assert set(tested[SCOPE_COLUMN]) == {SCOPE_UNSEEN, SCOPE_CONTROL}

    def test_the_size_agrees_with_the_direction_it_is_reported_beside(self, tmp_path):
        """A sign that disagreed with `higher` would make the pair unreadable."""
        run_paths, _ = TestReportAudits().make_with_runs(tmp_path)
        tested = pd.read_csv(run_paths.root / "training_tests.csv")
        lifted = tested[tested["higher"] == TRAINING_AUGMENTED]
        assert not lifted.empty
        assert (lifted[SIZE_COLUMN] > 0.0).all()
        tied = tested[tested["higher"] == TIED][SIZE_COLUMN]
        assert max((abs(float(value)) for value in tied), default=0.0) < 1e-12
