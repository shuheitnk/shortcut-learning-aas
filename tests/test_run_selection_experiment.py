"""Tests for Experiment 3's entry point, which reads Experiment 5's plain cells.

Checked here: the command line, the units it asks for, the levels it writes and the
seed-level test it computes from their records.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from aas_shortcut import run_selection_experiment as entry
from aas_shortcut.compare_selection_runs import (
    compare_selection_representations,
    selection_runs,
)
from aas_shortcut.compare_training_runs import PAIRED, SIZE_COLUMN
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_paths import TRANSFORMS, RunPaths
from aas_shortcut.define_reporting_scopes import SCOPE_CONTROL
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
    VARIANTS,
)
from aas_shortcut.run_selection_experiment import (
    LEVEL_COLUMNS,
    LEVEL_TABLE,
    REPRESENTATION_TEST_TABLE,
    ROLE_TUNING_TABLE,
    SBS_TABLE,
    SelectionUnits,
    level_board,
    level_table,
    parse_args,
    report_seed_tests,
    resolve_selection_units,
    write_levels,
    write_portfolio_tables,
)
from aas_shortcut.select_algorithms import SelectionConfig
from tests.test_bind_portfolio_parameters import (
    SEARCH_BUDGET,
    TOTAL_BUDGET,
    flagged_role_table,
)
from tests.test_compare_selection_runs import FAMILY_COLUMN, records, wobble


class TestParseArgs:
    def test_the_defaults_run_the_experiment_whole(self):
        args = parse_args([], "experiment3")
        assert args.preset is None  # the spec's own `preset: full`
        assert args.overrides == []

    def test_hydra_overrides_are_passed_through(self):
        args = parse_args(["task.n_jobs=26", "preset.dim=2"], "experiment3")
        assert args.overrides == ["task.n_jobs=26", "preset.dim=2"]


class TestTheUnitsAskedFor:
    """Experiment 5's plain cells and data, asked for as Experiment 5 asks.

    A cell is named by every family it is scored on, so the experiments share cells
    only if Experiment 3 resolves families and oracles by Experiment 5's functions.
    """

    @pytest.fixture
    def asked(self, monkeypatch) -> dict[str, object]:
        calls: dict[str, object] = {}

        def training(store, context, bindings, **kwargs):
            calls["trainings"] = kwargs.get("trainings")
            return "training"

        def cells(store, config, asked_cells, data, setting, scale):
            calls["cells"] = list(asked_cells)
            return {}

        def oracles(store, config, data, setting, scale, *rest, **kwargs):
            calls["oracles"] = (rest, kwargs)
            return {}

        def held_out(store, context, bindings):
            calls["families"] = "every family in range"
            return ["blocks"]

        monkeypatch.setattr(
            entry, "resolve_portfolio_bindings", lambda store, context: "bindings"
        )
        monkeypatch.setattr(entry, "resolve_selection_training", training)
        monkeypatch.setattr(entry, "resolve_selection_held_out_in_range", held_out)
        monkeypatch.setattr(entry, "selection_data", lambda *args: "data")
        monkeypatch.setattr(entry, "resolve_selection_cells", cells)
        monkeypatch.setattr(entry, "resolve_oracles", oracles)
        monkeypatch.setattr(entry, "paper_columns", lambda sets, scale: ("a", "b"))
        config = SelectionConfig(feature_variants=("value", "rank", "concat"))
        context = SimpleNamespace(
            selection_config=lambda: config,
            feature_sets=("ela",),
            scale="debug",
            dim=2,
            seed=42,
            feature_config=lambda problem_class: SimpleNamespace(sample_size=100),
        )
        resolve_selection_units(None, context)  # type: ignore[arg-type]
        return calls

    def test_the_training_block_is_resolved_for_plain_cells_alone(self, asked):
        assert asked["trainings"] == (TRAINING_PLAIN,)

    def test_every_family_in_range_is_resolved(self, asked):
        assert asked["families"] == "every family in range"

    def test_only_the_plain_cells_are_asked_for(self, asked):
        cells = asked["cells"]
        assert isinstance(cells, list)
        assert len(cells) == 3
        assert {cell.training for cell in cells} == {TRAINING_PLAIN}

    def test_the_oracles_are_experiment_5_s_units(self, asked):
        """The oracles are asked for as Experiment 5 asks: one unit, both conditions."""
        assert asked["oracles"] == ((), {})


class TestTheControlBlock:
    def test_a_run_without_the_control_family_says_so(self):
        units = SelectionUnits(
            None,  # type: ignore[arg-type]
            [SimpleNamespace(label="power/warpheld")],  # type: ignore[list-item]
            None,  # type: ignore[arg-type]
            {},
            {},
            None,  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match=CONTROL.label):
            _ = units.control


class TestTheLevelTable:
    """The levels the report prints, one table per representation set."""

    @staticmethod
    def _levels() -> pd.DataFrame:
        # Distinct per set, representation and block, with a tail rounding would drop.
        return pd.DataFrame(
            [
                {
                    "feature_set": feature_set,
                    "feature_variant": variant,
                    "training": TRAINING_PLAIN,
                    FAMILY_COLUMN: CONTROL.label,
                    "transform": view,
                    "value": 0.6 + 0.1 * shift + 0.01 * index - 0.05 * block + 3e-7,
                }
                for shift, feature_set in enumerate(("ela", "deep-ela"))
                for index, variant in enumerate(VARIANTS)
                for block, view in enumerate(TRANSFORMS)
            ]
        )

    @staticmethod
    def _oracle() -> pd.DataFrame:
        # Two problems per block; the augmented rows must not count.
        means = {"pre": (0.7, 0.8), "post": (0.75, 0.76)}
        return pd.DataFrame(
            [
                {
                    "feature_set": feature_set,
                    "training": training,
                    "transform": view,
                    "instance_id": problem,
                    "superiority_vs_sbs": value if training == TRAINING_PLAIN else 0.0,
                }
                for feature_set in ("ela", "deep-ela")
                for training in (TRAINING_PLAIN, TRAINING_AUGMENTED)
                for view, values in means.items()
                for problem, value in enumerate(values)
            ]
        )

    def _table(self, feature_set: str = "ela") -> pd.DataFrame:
        board = level_board(self._levels(), self._oracle(), feature_set)
        return level_table(board, feature_set)

    def test_a_row_per_representation_and_block(self):
        table = self._table()
        assert list(table.columns) == list(LEVEL_COLUMNS)
        assert list(zip(table["representation"], table["transform"], strict=True)) == [
            (variant, view) for variant in VARIANTS for view in TRANSFORMS
        ]
        assert set(table["scope"]) == {SCOPE_CONTROL}
        assert set(table["feature_set"]) == {"ela"}

    def test_each_level_is_its_cell_s_at_full_precision(self):
        """Each level is its cell's value at full precision."""
        levels = self._levels()
        mine = levels[levels["feature_set"] == "deep-ela"]
        for _, row in self._table("deep-ela").iterrows():
            level = mine[
                (mine["feature_variant"] == row["representation"])
                & (mine["transform"] == row["transform"])
            ]["value"].item()
            assert row["superiority_vs_sbs_mean"] == level

    def test_the_oracle_is_its_mean_over_the_problems_of_each_block(self):
        table = self._table()
        assert set(table[table["transform"] == "pre"]["oracle"]) == {0.75}
        assert set(table[table["transform"] == "post"]["oracle"]) == {0.755}

    def test_one_table_per_representation_set(self, tmp_path):
        run_paths = RunPaths.from_root(tmp_path)
        run_paths.mkdirs()
        written = write_levels(self._levels(), self._oracle(), run_paths)
        assert [path.name for path in written] == [
            LEVEL_TABLE.name,
            "selection_levels_deep-ela.csv",
        ]
        for path, feature_set in zip(written, ("ela", "deep-ela"), strict=True):
            pd.testing.assert_frame_equal(pd.read_csv(path), self._table(feature_set))

    def test_a_set_the_run_did_not_fit_writes_nothing(self, tmp_path):
        run_paths = RunPaths.from_root(tmp_path)
        run_paths.mkdirs()
        levels = self._levels()
        written = write_levels(
            levels.loc[levels["feature_set"] == "ela"], self._oracle(), run_paths
        )
        assert [path.name for path in written] == [LEVEL_TABLE.name]


class TestTheSeedLevelTest:
    """Experiment 3 compares its representations by seed."""

    @staticmethod
    def level(variant, _training, _family, _view, seed):
        return 0.3 + wobble(seed) + (0.02 if variant == "rank" else 0.0)

    def test_the_table_is_experiment_5_s_rows_for_its_plain_cells(self, tmp_path):
        every = records(self.level)
        mine = cast(
            pd.DataFrame,
            every[
                (every["training"] == TRAINING_PLAIN)
                & (every[FAMILY_COLUMN] == CONTROL.label)
            ],
        )
        run_paths = RunPaths.from_root(tmp_path)
        run_paths.mkdirs()
        written = report_seed_tests(mine, run_paths)
        assert [path.name for path in written] == [REPRESENTATION_TEST_TABLE.name]
        tested = pd.read_csv(written[0], float_precision=cast(Any, "round_trip"))
        assert set(tested["scope"]) == {SCOPE_CONTROL}
        assert set(tested["test"]) == {PAIRED}

        eight = compare_selection_representations(
            selection_runs(every, SCOPE_CONTROL), SCOPE_CONTROL
        )
        eight = eight[eight["training"] == TRAINING_PLAIN].reset_index(drop=True)
        for column in ("left", "right", "higher", "p", "p_holm", SIZE_COLUMN):
            assert list(tested[column]) == list(eight[column])

    def test_no_record_writes_nothing(self, tmp_path):
        run_paths = RunPaths.from_root(tmp_path)
        run_paths.mkdirs()
        assert report_seed_tests(records(self.level).iloc[0:0], run_paths) == []


class TestThePortfolioTables:
    """The role tuning's configurations and the SBS, in the run root."""

    @staticmethod
    def units(recorded_sbs: str) -> Any:
        """Units whose training block favours BIPOP-CMA-ES and nothing else does."""
        rows = []
        for instance_id in (1, 2, 11, 12):
            for view in ("pre", "post"):
                for algorithm in ALGORITHMS:
                    training_pre = instance_id >= 11 and view == "pre"
                    favoured = "BIPOP-CMA-ES" if training_pre else "SHADE"
                    rows.append(
                        {
                            "instance_id": instance_id,
                            "transform": view,
                            "algorithm": algorithm,
                            "mean_superiority": 0.9 if algorithm == favoured else 0.4,
                        }
                    )
        table = flagged_role_table({SEARCH_BUDGET: {}, TOTAL_BUDGET: {}})
        return SimpleNamespace(
            bindings=SimpleNamespace(
                roles=SimpleNamespace(read_table=lambda relative: table)
            ),
            training=SimpleNamespace(
                mean_superiorities=lambda condition: pd.DataFrame(rows)
            ),
            config=SimpleNamespace(algorithms=ALGORITHMS),
            oracle=lambda: pd.DataFrame({"sbs_algorithm": [recorded_sbs] * 3}),
        )

    @staticmethod
    def context() -> Any:
        return SimpleNamespace(search_budget=SEARCH_BUDGET, total_budget=TOTAL_BUDGET)

    def test_both_tables_are_written(self, tmp_path):
        run_paths = RunPaths.from_root(tmp_path)
        written = write_portfolio_tables(
            self.units("BIPOP-CMA-ES"), self.context(), run_paths
        )
        assert written == [tmp_path / ROLE_TUNING_TABLE, tmp_path / SBS_TABLE]
        roles = pd.read_csv(tmp_path / ROLE_TUNING_TABLE)
        assert len(roles) == 2 * len(ALGORITHMS) * 2
        ranking = pd.read_csv(tmp_path / SBS_TABLE)
        assert list(ranking["algorithm"]) == list(ALGORITHMS)

    def test_the_sbs_is_read_off_the_untransformed_training_block(self, tmp_path):
        """The SBS is read off the untransformed training block alone."""
        write_portfolio_tables(
            self.units("BIPOP-CMA-ES"), self.context(), RunPaths.from_root(tmp_path)
        )
        ranking = pd.read_csv(tmp_path / SBS_TABLE)
        assert ranking.loc[ranking["is_sbs"], "algorithm"].tolist() == ["BIPOP-CMA-ES"]
        assert set(ranking["n_problems"]) == {2}

    def test_a_ranking_the_cells_disagree_with_stops_the_run(self, tmp_path):
        with pytest.raises(ValueError, match="scored against"):
            write_portfolio_tables(
                self.units("SHADE"), self.context(), RunPaths.from_root(tmp_path)
            )
