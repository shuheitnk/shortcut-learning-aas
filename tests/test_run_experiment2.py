"""Experiment 2's entry point: the data it asks for Experiment 4's plain cells."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiment2"))

import run_experiment2
from run_experiment2 import Blocks

from aas_shortcut.create_figures.draw_hero_figures import HERO_FORMATS
from aas_shortcut.define_algorithm_portfolio import ALGORITHM_GROUP, FOCUS_PAIRS
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.resolve_tuning_stages import CALIBRATION_TABLES, PAIR_TABLE


class TestTheBlocks:
    """Experiment 4's data, resolved by Experiment 4's functions.

    A plain cell is named by every family it is scored on, so cells are shared only
    if the blocks are.
    """

    @pytest.fixture
    def asked(self, monkeypatch) -> dict[str, object]:
        calls: dict[str, object] = {}
        pre = SimpleNamespace(
            superiority="superiority-unit", superiorities="superiorities"
        )

        def training(store, context, binding, tuning, step, **kwargs):
            calls["trainings"] = kwargs.get("trainings")
            return SimpleNamespace(pre=pre)

        def held_out(store, context, binding, tuning, step):
            calls["families"] = "every family in range"
            return ["blocks"]

        def calibration(store, context, superiority, superiorities):
            calls["calibration"] = (superiority, superiorities)
            return "calibration-unit"

        monkeypatch.setattr(
            run_experiment2,
            "resolve_pair_binding",
            lambda store, context, step: ("binding", "tuning"),
        )
        monkeypatch.setattr(run_experiment2, "resolve_training_block", training)
        monkeypatch.setattr(run_experiment2, "resolve_held_out_in_range", held_out)
        monkeypatch.setattr(run_experiment2, "resolve_calibration", calibration)
        calls["blocks"] = run_experiment2.resolve_blocks(None, SimpleNamespace())  # type: ignore[arg-type]
        return calls

    def test_the_training_block_is_resolved_for_plain_cells_alone(self, asked):
        assert asked["trainings"] == (TRAINING_PLAIN,)

    def test_every_family_in_range_is_resolved(self, asked):
        assert asked["families"] == "every family in range"

    def test_the_calibration_reads_the_untransformed_training_runs(self, asked):
        assert asked["calibration"] == ("superiority-unit", "superiorities")

    def test_the_pair_tuning_is_kept_for_fig5(self, asked):
        assert asked["blocks"].tuning == "tuning"

    def test_the_calibration_is_kept_for_its_tables(self, asked):
        assert asked["blocks"].calibration == "calibration-unit"

    def test_a_run_without_the_control_family_says_so(self):
        blocks = Blocks(None, [SimpleNamespace(label="power/warpheld")], None, None)  # type: ignore[arg-type, list-item]
        with pytest.raises(ValueError, match=CONTROL.label):
            _ = blocks.control


class TestFigure5:
    """Fig. 5 is drawn from the pair tuning the run resolved."""

    @staticmethod
    def _tuning_table() -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = []
        for pair in FOCUS_PAIRS:
            for flag in ("is_default", "is_selected"):
                for function_id in range(1, 25):
                    rows.append(
                        {
                            "group": ALGORITHM_GROUP[pair.first],
                            "pair": pair.slug,
                            "is_default": flag == "is_default",
                            "is_selected": flag == "is_selected",
                            "function_id": function_id,
                            "superiority": float(rng.uniform()),
                        }
                    )
        return pd.DataFrame(rows)

    def test_the_tuning_unit_table_is_what_is_drawn(self, monkeypatch, run_paths):
        table = self._tuning_table()
        read: list[str] = []

        def read_table(name):
            read.append(name)
            return table

        drawn: dict[str, object] = {}

        def figure(tuning, paths, formats):
            drawn.update(tuning=tuning, paths=paths, formats=formats)
            return ["fig5"]

        monkeypatch.setattr(run_experiment2, "figure_tuning_histogram", figure)
        blocks = Blocks(None, [], SimpleNamespace(read_table=read_table), None)  # type: ignore[arg-type]

        assert run_experiment2.draw_tuning(blocks, run_paths) == ["fig5"]
        assert read == [PAIR_TABLE]
        assert drawn["tuning"] is table
        assert drawn["paths"] is run_paths
        assert drawn["formats"] == HERO_FORMATS

    def test_the_histogram_is_written_beside_the_other_figures(self, run_paths):
        tuning = SimpleNamespace(read_table=lambda name: self._tuning_table())
        written = run_experiment2.draw_tuning(Blocks(None, [], tuning, None), run_paths)  # type: ignore[arg-type]

        assert sorted(path.name for path in written) == [
            "fig5_tuning_histogram.png",
            "fig5_tuning_histogram.svg",
        ]
        assert all(path.parent == run_paths.figures_dir for path in written)
        assert not (run_paths.figures_dir / "fig5_tuning.png").exists()


class TestTheCalibrationTables:
    """The calibration is written beside Fig. 5, as its unit holds it."""

    @staticmethod
    def _tables() -> dict[str, pd.DataFrame]:
        summary = pd.DataFrame(
            {
                "block": ["tuning", "training"],
                "group": ["G1", "G1"],
                "n_problems": [24, 2400],
                "ks_distance": [0.260753, 0.211919],
            }
        )
        deciles = pd.DataFrame(
            {"block": ["training"] * 2, "decile": [0, 9], "n_problems": [671, 12]}
        )
        return dict(zip(CALIBRATION_TABLES, (summary, deciles), strict=True))

    def test_each_table_is_written_as_the_unit_holds_it(self, run_paths):
        tables = self._tables()
        calibration = SimpleNamespace(read_table=lambda name: tables[name])
        blocks = Blocks(None, [], None, calibration)  # type: ignore[arg-type]

        written = run_experiment2.write_calibration(blocks, run_paths)

        assert [path.name for path in written] == [
            "superiority_calibration.csv",
            "superiority_calibration_deciles.csv",
        ]
        assert all(path.parent == run_paths.root for path in written)
        for path, name in zip(written, CALIBRATION_TABLES, strict=True):
            pd.testing.assert_frame_equal(pd.read_csv(path), tables[name])


class TestThePairTuningTable:
    """Each pair's default and selected configuration, beside Fig. 5."""

    def test_is_the_tuning_unit_s_two_configurations(self, run_paths):
        tuning_table = pd.DataFrame(
            {
                "group": ["G1"] * 2,
                "pair": ["a__b"] * 2,
                "tuned": ["B"] * 2,
                "fixed": ["A"] * 2,
                "config_id": ["published", "chosen"],
                "params": ["{}"] * 2,
                "is_default": [True, False],
                "is_selected": [False, True],
                "function_id": [1, 1],
                "instance_id": [11, 11],
                "score": [0.401, 0.261],
                "floor": [0.021, 0.021],
                "n_saturated": [13, 7],
            }
        )
        tables = {PAIR_TABLE: tuning_table}
        tuning = SimpleNamespace(read_table=lambda name: tables[name])
        blocks = Blocks(None, [], tuning, None)  # type: ignore[arg-type]

        written = run_experiment2.write_pair_tuning(blocks, run_paths)

        assert written == run_paths.root / "pair_tuning.csv"
        table = pd.read_csv(written)
        assert list(table["configuration"]) == ["default", "selected"]
        assert list(table["score"]) == [0.401, 0.261]
        assert list(table["n_saturated"]) == [13, 7]
