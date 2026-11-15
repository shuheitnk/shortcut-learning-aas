"""Tests for Experiment 5's audit points and the tables they are collected into.

The control is a draw of the training family and no other point is; the report goes
to the run's own root; the gain table reads the oracle's ceiling and the cells'
levels over each range.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiment5"))

from run_experiment5 import (
    AUDITS,
    GAIN_COLUMNS,
    GAIN_TABLE,
    REPRESENTATION_TEST_TABLE,
    SPEC_PATH,
    TRAINING_TEST_TABLE,
    augmentation_gain,
    oracle_ceilings,
    parse_args,
    report_gain,
    report_seed_tests,
    report_split,
)

from aas_shortcut.compare_selection_runs import FAMILY_COLUMN
from aas_shortcut.define_paths import ORACLE_VARIANT
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
    SCOPES,
    UNSEEN_FAMILIES,
)
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    compose_config,
    resolve_run_paths,
)
from aas_shortcut.sample_monotone_transform import (
    AUGMENTATION_DOMAIN,
    HELD_OUT_DOMAIN,
    SPLINE_FAMILY,
    SPLINE_KNOTS,
    WarpConfig,
)


@pytest.fixture
def spec() -> ExperimentSpec:
    return ExperimentSpec.load(SPEC_PATH)


def make_oracle(levels: dict[str, float]) -> pd.DataFrame:
    """The oracle's rows on every family: two problems, both views and conditions.

    The transformed problems sit 0.1 either side of the family's level; the
    untransformed block sits far off it, so a ceiling that read it would show.
    """
    return pd.DataFrame(
        [
            {
                FAMILY_COLUMN: family,
                "feature_set": feature_set,
                "feature_variant": ORACLE_VARIANT,
                "training": training,
                "transform": transform,
                "instance_id": instance,
                "superiority_vs_sbs": (level + offset) if transform == "post" else 0.99,
            }
            for family, level in levels.items()
            for feature_set in ("ela", "deep-ela")
            for training in ("plain", "augmented")
            for transform in ("pre", "post")
            for instance, offset in ((1, -0.1), (2, 0.1))
        ]
    )


ORACLE_LEVELS = {
    CONTROL.label: 0.80,
    **{family: 0.70 + 0.01 * index for index, family in enumerate(UNSEEN_FAMILIES)},
}


class TestAuditPoints:
    def test_the_control_is_the_training_family(self):
        """Without it, a family effect cannot be told from an unseen draw."""
        assert CONTROL is AUDITS[0]
        assert CONTROL.family == SPLINE_FAMILY
        assert CONTROL.knots == SPLINE_KNOTS

    def test_the_control_is_still_a_draw_the_training_never_saw(self):
        """A second draw of the same family is the point of the control."""
        assert CONTROL.domain == HELD_OUT_DOMAIN

    def test_every_other_point_is_outside_the_training_family(self):
        outside = [audit for audit in AUDITS if audit is not CONTROL]
        assert outside
        for audit in outside:
            assert (audit.family, audit.knots) != (SPLINE_FAMILY, SPLINE_KNOTS)

    def test_the_knot_points_keep_the_family_and_move_only_the_count(self):
        knotted = [
            audit
            for audit in AUDITS
            if audit.family == SPLINE_FAMILY and audit is not CONTROL
        ]
        assert len(knotted) == 2
        counts = {audit.knots for audit in knotted}
        # One below the training range and one above it.
        assert any(high <= SPLINE_KNOTS[0] for _, high in counts)
        assert any(low >= SPLINE_KNOTS[1] for low, _ in counts)

    def test_every_point_names_a_warp_that_can_be_drawn(self):
        for audit in AUDITS:
            WarpConfig(family=audit.family, domain=audit.domain, knots=audit.knots)


class TestSpec:
    def test_the_spec_is_experiment_5(self, spec):
        assert spec.id == "experiment5"
        assert spec.preset == "full"

    def test_it_writes_its_report_to_its_own_run_root(self, spec):
        assert "paths.run_name=experiment5" in spec.overrides

    def test_a_rehearsal_writes_beside_the_reported_report(self, spec, tmp_path):
        cfg = compose_config(
            spec.hydra_overrides("debug", [f"paths.output_root={tmp_path}"])
        )
        assert resolve_run_paths(cfg).root == tmp_path / "experiment5_debug"

    def test_the_grid_covers_the_training_block(self, spec):
        """Its transformed half is what the augmented condition is fitted on."""
        cfg = compose_config(spec.hydra_overrides(spec.preset))
        assert int(cfg.preset.max_instances) == 110
        assert list(cfg.task.tuning_instance_ids) == [11]

    def test_the_training_side_keeps_the_augmentation_domain(self, spec):
        """Its features and its transformed runs are drawn from that domain."""
        cfg = compose_config(spec.hydra_overrides(spec.preset))
        assert cfg.task.warp_domain == AUGMENTATION_DOMAIN


class TestCommandLine:
    def test_the_store_can_be_read_without_computing(self):
        assert parse_args(["--require-cached"]).require_cached
        assert parse_args(["--plan"]).plan


class TestOracleCeilings:
    """The perfect selector's level over each range, read off the oracle units."""

    def test_a_family_s_ceiling_is_its_transformed_problems_mean(self):
        ceilings = oracle_ceilings(
            make_oracle(ORACLE_LEVELS), [CONTROL.label], [SCOPE_CONTROL]
        )
        assert set(ceilings["feature_set"]) == {"ela", "deep-ela"}
        assert ceilings["value"].to_numpy() == pytest.approx(0.80)

    def test_the_unseen_ceiling_is_the_mean_over_the_four_unseen_families(self):
        labels = list(ORACLE_LEVELS)
        ceilings = oracle_ceilings(make_oracle(ORACLE_LEVELS), labels, [SCOPE_UNSEEN])
        expected = sum(ORACLE_LEVELS[family] for family in UNSEEN_FAMILIES) / 4
        assert ceilings["value"].to_numpy() == pytest.approx(expected)

    def test_each_range_is_one_row_per_representation_set(self):
        labels = list(ORACLE_LEVELS)
        ceilings = oracle_ceilings(
            make_oracle(ORACLE_LEVELS), labels, [SCOPE_UNSEEN, SCOPE_CONTROL]
        )
        assert len(ceilings) == 2 * 2
        assert set(ceilings[SCOPE_COLUMN]) == {SCOPE_UNSEEN, SCOPE_CONTROL}


class TestTheGainTable:
    """What augmentation buys over each range, in one table for both sets."""

    def test_the_table_holds_both_ranges_and_nothing_is_drawn(self, tmp_path):
        from aas_shortcut.define_paths import RunPaths
        from tests.test_compare_selection_runs import FAMILIES, records, wobble

        run_paths = RunPaths.from_root(tmp_path)
        rows = records(lambda *args: 0.6 + wobble(args[-1]), families=FAMILIES)
        oracle = make_oracle(ORACLE_LEVELS)
        oracle = cast(pd.DataFrame, oracle[oracle["feature_set"] == "ela"])
        written = report_gain(rows, oracle, list(FAMILIES), run_paths)
        gain = pd.read_csv(tmp_path / GAIN_TABLE)
        assert written == [tmp_path / GAIN_TABLE]
        assert set(gain[SCOPE_COLUMN]) == {SCOPE_UNSEEN, SCOPE_CONTROL}
        assert set(gain["augmentation"]) == {"plain", "augmented"}


# Two sets with different levels, and a known family apart from the unseen range, so
# that pooling the sets or mixing the ranges is caught.
GAIN_LEVELS: dict[str, dict[str, tuple[float, float]]] = {
    "ela": {"value": (0.59, 0.65), "rank": (0.63, 0.66), "concat": (0.59, 0.66)},
    "deep-ela": {"value": (0.54, 0.61), "rank": (0.60, 0.64), "concat": (0.53, 0.65)},
}
KNOWN_SHIFT = -0.03


def make_gain(**kwargs) -> pd.DataFrame:
    """The gain table over both ranges, as `selection_levels` feeds it."""
    levels = pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "feature_variant": variant,
                "training": training,
                "value": level + (KNOWN_SHIFT if scope == SCOPE_CONTROL else 0.0),
                SCOPE_COLUMN: scope,
            }
            for feature_set, cells in GAIN_LEVELS.items()
            for scope in SCOPES
            for variant, pair in cells.items()
            for training, level in zip(("plain", "augmented"), pair, strict=True)
        ]
    )
    ceilings = pd.DataFrame(
        [
            {"feature_set": feature_set, SCOPE_COLUMN: scope, "value": 0.78}
            for feature_set in GAIN_LEVELS
            for scope in SCOPES
        ]
    )
    return augmentation_gain(levels, ceilings, **kwargs)


class TestGainRows:
    def test_its_columns(self):
        assert list(make_gain().columns) == list(GAIN_COLUMNS)

    def test_both_ranges_are_reported_the_main_one_first(self):
        assert list(dict.fromkeys(make_gain()[SCOPE_COLUMN])) == list(SCOPES)

    def test_a_row_carries_its_own_range_s_level_at_full_precision(self):
        gain = make_gain().set_index(
            ["feature_set", SCOPE_COLUMN, "representation", "augmentation"]
        )
        for feature_set, cells in GAIN_LEVELS.items():
            for variant, (plain, _) in cells.items():
                unseen = (feature_set, SCOPE_UNSEEN, variant, "plain")
                known = (feature_set, SCOPE_CONTROL, variant, "plain")
                assert gain.loc[unseen, "superiority_vs_sbs_mean"] == plain
                assert gain.loc[known, "superiority_vs_sbs_mean"] == plain + KNOWN_SHIFT

    def test_only_the_ranges_asked_for_are_reported(self):
        assert set(make_gain(scopes=(SCOPE_CONTROL,))[SCOPE_COLUMN]) == {SCOPE_CONTROL}


class TestTheSplitFigure:
    """Fig. 28: the two training conditions in Fig. 16's design."""

    @staticmethod
    def problem(instance: int) -> dict:
        return {
            "problem_class": "bbob",
            "function_id": 1,
            "instance_id": instance,
            "dimension": 10,
        }

    def make(self):
        from aas_shortcut.define_repetitions import MODEL_SEEDS

        problems = pd.DataFrame(
            [
                {
                    FAMILY_COLUMN: family,
                    "feature_set": "ela",
                    "feature_variant": variant,
                    "training": training,
                    "transform": transform,
                    "seed": seed,
                    "rep": 1,
                    **self.problem(instance),
                    "superiority_vs_sbs": 0.5
                    + 0.1 * (training == "augmented")
                    + 0.01 * instance,
                }
                for family in (CONTROL.label, *UNSEEN_FAMILIES)
                for variant in ("value", "rank", "concat")
                for training in ("plain", "augmented")
                for transform in ("pre", "post")
                for seed in MODEL_SEEDS
                for instance in (1, 2, 3)
            ]
        )
        oracle = pd.DataFrame(
            [
                {
                    FAMILY_COLUMN: family,
                    "feature_set": "ela",
                    "feature_variant": ORACLE_VARIANT,
                    "training": training,
                    "transform": "post",
                    **self.problem(instance),
                    "superiority_vs_sbs": 0.8,
                }
                for family in (CONTROL.label, *UNSEEN_FAMILIES)
                for training in ("plain", "augmented")
                for instance in (1, 2, 3)
            ]
        )
        gain = pd.DataFrame(
            [
                {
                    "feature_set": "ela",
                    SCOPE_COLUMN: SCOPE_UNSEEN,
                    "representation": variant,
                    "augmentation": training,
                    "superiority_vs_sbs_mean": 0.52 + 0.1 * (training == "augmented"),
                }
                for variant in ("value", "rank", "concat")
                for training in ("plain", "augmented")
            ]
        )
        units = SimpleNamespace(
            labels=(CONTROL.label, *UNSEEN_FAMILIES),
            problems=lambda labels: problems[problems[FAMILY_COLUMN].isin(labels)],
        )
        return cast(Any, units), oracle, gain

    def test_one_figure_per_representation_set_it_holds(self, tmp_path):
        from aas_shortcut.define_paths import RunPaths

        units, oracle, gain = self.make()
        written = report_split(units, oracle, gain, RunPaths.from_root(tmp_path))
        assert sorted(path.name for path in written) == [
            "fig28_augmentation_selection_split.pdf",
            "fig28_augmentation_selection_split.png",
            "fig28_augmentation_selection_split.svg",
        ]

    def test_a_run_without_every_unseen_family_draws_nothing(self, tmp_path):
        from aas_shortcut.define_paths import RunPaths

        units, oracle, gain = self.make()
        units.labels = (CONTROL.label,)
        assert report_split(units, oracle, gain, RunPaths.from_root(tmp_path)) == []


class TestTheSeedLevelTests:
    """The seed-level tests reach the run root: both tables, both ranges."""

    @staticmethod
    def records(families=None) -> pd.DataFrame:
        from tests.test_compare_selection_runs import FAMILIES, records, wobble

        return records(
            lambda variant, training, family, view, seed: 0.3 + wobble(seed),
            families=families or FAMILIES,
        )

    def test_both_tables_are_written_over_both_ranges(self, tmp_path):
        from aas_shortcut.define_paths import RunPaths
        from aas_shortcut.define_reporting_scopes import SCOPE_CONTROL, SCOPE_UNSEEN

        run_paths = RunPaths.from_root(tmp_path)
        run_paths.mkdirs()
        written = report_seed_tests(self.records(), run_paths)
        assert sorted(path.name for path in written) == sorted(
            [REPRESENTATION_TEST_TABLE.name, TRAINING_TEST_TABLE.name]
        )
        for name in (REPRESENTATION_TEST_TABLE, TRAINING_TEST_TABLE):
            table = pd.read_csv(tmp_path / name)
            assert set(table["scope"]) == {SCOPE_UNSEEN, SCOPE_CONTROL}

    def test_a_range_the_run_did_not_audit_is_left_out(self, tmp_path, caplog):
        from aas_shortcut.define_paths import RunPaths
        from aas_shortcut.define_reporting_scopes import SCOPE_CONTROL

        run_paths = RunPaths.from_root(tmp_path)
        run_paths.mkdirs()
        with caplog.at_level(logging.WARNING):
            report_seed_tests(self.records(families=(CONTROL.label,)), run_paths)
        table = pd.read_csv(tmp_path / REPRESENTATION_TEST_TABLE)
        assert set(table["scope"]) == {SCOPE_CONTROL}
        assert "did not audit" in caplog.text
