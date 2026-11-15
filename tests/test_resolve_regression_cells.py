"""Regression cells: units, data, and which cells a call fits."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, ClassVar

import pandas as pd
import pytest

from aas_shortcut import measure_augmented_regression as regression_module
from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.measure_augmented_regression import (
    DEPLOYMENT_SHARD,
    AugmentedRegressionConfig,
    fingerprint_variants,
)
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN
from aas_shortcut.record_manifests import stage_fingerprint
from aas_shortcut.resolve_pair_blocks import HeldOutBlock, TrainingBlock
from aas_shortcut.resolve_regression_cells import (
    RegressionCell,
    RegressionData,
    cell_inputs,
    cell_table,
    cells_of,
    learning_settings,
    regression_cell_unit,
    regression_data,
    resolve_regression_cells,
    rows_on_family,
)
from aas_shortcut.resolve_stage_artifacts import StageStore, unit_fingerprint
from aas_shortcut.split_bbob_instances import TEST_INSTANCES, TRAIN_INSTANCES
from tests.conftest import make_features
from tests.test_measure_representation_transfer import make_view_superiorities

CONFIG = AugmentedRegressionConfig(
    audits=("spline/warpheld", "power/warpheld"),
    features=("a", "b"),
    candidates=("realmlp",),
)
INPUTS = {
    "train_features": "f",
    "train_superiority_pre": "p",
    "train_superiority_post": "q",
}
# What a plain cell reads of them: not the transformed training superiorities.
PLAIN_INPUTS = {"train_features": "f", "train_superiority_pre": "p"}


class TestWhatACellReads:
    """A cell names the training views its condition fits on, and no other.

    A plain cell fits on untransformed rows alone, so it is not refitted when only
    the transformed training runs change. The training features are one unit.
    """

    @staticmethod
    def cell(training: str) -> RegressionCell:
        return next(c for c in cells_of(CONFIG) if c.training == training)

    def test_a_plain_cell_does_not_name_the_transformed_training_superiorities(self):
        unit = regression_cell_unit(CONFIG, self.cell(TRAINING_PLAIN), INPUTS)
        assert "train_superiority_post" not in unit.inputs
        assert unit.inputs["train_features"] == "f"

    def test_a_plain_cell_is_the_same_unit_with_or_without_them(self):
        cell = self.cell(TRAINING_PLAIN)
        fingerprint = regression_cell_unit(CONFIG, cell, INPUTS).fingerprint
        assert (
            fingerprint == regression_cell_unit(CONFIG, cell, PLAIN_INPUTS).fingerprint
        )
        moved = {**INPUTS, "train_superiority_post": "other"}
        assert fingerprint == regression_cell_unit(CONFIG, cell, moved).fingerprint

    def test_an_augmented_cell_keeps_every_input_and_its_fingerprint(self):
        unit = regression_cell_unit(CONFIG, self.cell(TRAINING_AUGMENTED), INPUTS)
        assert unit.inputs == INPUTS
        assert unit.fingerprint == unit_fingerprint(unit.settings, INPUTS)

    def test_an_augmented_cell_without_the_transformed_view_is_refused(self):
        with pytest.raises(ValueError, match="post"):
            regression_cell_unit(CONFIG, self.cell(TRAINING_AUGMENTED), PLAIN_INPUTS)

    def test_the_other_inputs_pass_through(self):
        inputs = {**INPUTS, "test_superiority_pre": "t", "test_features:power": "d"}
        assert cell_inputs(inputs, TRAINING_PLAIN) == {
            **PLAIN_INPUTS,
            "test_superiority_pre": "t",
            "test_features:power": "d",
        }


class TestCells:
    def test_every_cell_of_a_configuration_in_the_stage_s_order(self):
        cells = cells_of(CONFIG)
        assert len(cells) == len(FOCUS_PAIRS) * 3 * 2
        assert cells[0] == RegressionCell(
            FOCUS_PAIRS[0], "value", TRAINING_PLAIN, "realmlp"
        )
        assert cells[1].training == TRAINING_AUGMENTED

    def test_a_cell_s_unit_is_its_configuration_narrowed_to_it(self):
        cell = cells_of(CONFIG)[0]
        unit = regression_cell_unit(CONFIG, cell, INPUTS)
        settings = unit.settings
        assert settings["pairs"] == [cell.pair.slug]
        assert settings["variants"] == fingerprint_variants(("value",))
        assert settings["trainings"] == [TRAINING_PLAIN]
        assert settings["audits"] == list(CONFIG.audits)
        assert unit.inputs == PLAIN_INPUTS

    def test_the_unit_records_what_the_fit_depends_on_beyond_the_configuration(self):
        cell = cells_of(CONFIG)[1]
        learning = regression_cell_unit(CONFIG, cell, INPUTS).settings["learning"]
        assert learning == learning_settings(CONFIG, cell)
        # The augmented condition spends both views, so its epochs are halved.
        assert learning["epoch_divisor"] == 2
        # No cross-validation chooses a run to report.
        assert "seed_selection_cv" not in learning
        settings = regression_cell_unit(CONFIG, cell, INPUTS).settings
        assert settings["model_seeds"] == list(MODEL_SEEDS)
        assert "predictions" not in settings

    def test_a_learner_records_only_what_its_fit_depends_on(self):
        """The forest has no epochs and no split; XGBoost stops on the split."""
        config = replace(CONFIG, candidates=("realmlp", "random-forest", "xgboost"))
        augmented = {
            cell.kind: learning_settings(config, cell)
            for cell in cells_of(config)
            if cell.training == TRAINING_AUGMENTED
        }
        split = {"unit": "problem", "fraction": 0.2}
        assert augmented == {
            "realmlp": {"epoch_divisor": 2, "early_stopping_split": split},
            "random-forest": {},
            "xgboost": {"early_stopping_split": split},
        }

    def test_two_cells_are_two_units(self):
        first, second = cells_of(CONFIG)[:2]
        assert (
            regression_cell_unit(CONFIG, first, INPUTS).fingerprint
            != regression_cell_unit(CONFIG, second, INPUTS).fingerprint
        )

    def test_another_seed_count_is_another_unit(self):
        cell = cells_of(CONFIG)[0]
        assert (
            regression_cell_unit(CONFIG, cell, INPUTS).fingerprint
            != regression_cell_unit(
                replace(CONFIG, n_seeds=3), cell, INPUTS
            ).fingerprint
        )

    def test_other_inputs_are_another_unit(self):
        cell = cells_of(CONFIG)[0]
        assert (
            regression_cell_unit(CONFIG, cell, INPUTS).fingerprint
            != regression_cell_unit(
                CONFIG, cell, {**INPUTS, "train_features": "g"}
            ).fingerprint
        )

    def test_the_worker_count_is_not_part_of_a_unit(self):
        cell = cells_of(CONFIG)[0]
        assert (
            regression_cell_unit(CONFIG, cell, INPUTS).fingerprint
            == regression_cell_unit(
                replace(CONFIG, n_jobs=26), cell, INPUTS
            ).fingerprint
        )

    def test_a_one_cell_configuration_narrows_to_itself(self):
        """The fingerprint the stage gave a one-cell configuration is its own."""
        one = replace(
            CONFIG,
            pairs=(FOCUS_PAIRS[0],),
            variants=("value",),
            trainings=(TRAINING_PLAIN,),
        )
        (cell,) = cells_of(one)
        settings = regression_cell_unit(one, cell, INPUTS).settings
        narrowed = {k: v for k, v in settings.items() if k != "learning"}
        assert stage_fingerprint(narrowed) == stage_fingerprint(one.as_dict())

    def test_the_table_sits_where_the_stage_writes_it(self):
        assert cell_table(DEPLOYMENT_SHARD) == "metrics/deployment_scores.parquet"


def _features(instances, label: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"instance_id": i, "feature": name, "source": label}
            for i in instances
            for name in ("a", "b", "c")
        ]
    )


class TestRegressionData:
    def test_the_training_rows_and_every_block_narrowed_to_the_columns(self):
        unit: Any = SimpleNamespace(fingerprint="x")
        pair: Any = SimpleNamespace(superiority=unit, superiorities=pd.DataFrame())
        training = TrainingBlock(
            unit, _features((*TEST_INSTANCES, *TRAIN_INSTANCES), "train"), pair, pair
        )  # type: ignore[arg-type]
        blocks = [
            HeldOutBlock(label, unit, _features(TEST_INSTANCES, label), pair, pair)  # type: ignore[arg-type]
            for label in ("spline/warpheld", "power/warpheld")
        ]
        data = regression_data(training, blocks, ("a", "b"))
        assert set(data.train_features["instance_id"]) == set(TRAIN_INSTANCES)
        assert set(data.train_features["feature"]) == {"a", "b"}
        assert list(data.test_features) == ["spline/warpheld", "power/warpheld"]
        assert set(data.test_features["power/warpheld"]["feature"]) == {"a", "b"}
        assert set(data.test_features["power/warpheld"]["source"]) == {"power/warpheld"}


# Two blocks small enough to fit in a unit test, and two families.
SMALL_TRAIN = tuple(range(3, 23))
SMALL_TEST = (1, 2)
SMALL_LABELS = ("spline/warpheld", "power/warpheld")
SMALL_CONFIG = AugmentedRegressionConfig(
    audits=SMALL_LABELS,
    train_instances=SMALL_TRAIN,
    test_instances=SMALL_TEST,
    candidates=("random-forest",),
    model=ModelConfig(kind="random-forest"),
    n_seeds=1,
)
# Three cells that share no pair, encoding or condition pair: three of a 3 x 3 x 2
# grid.
OFF_THE_GRID = [
    RegressionCell(FOCUS_PAIRS[0], "value", TRAINING_PLAIN, "random-forest"),
    RegressionCell(FOCUS_PAIRS[2], "concat", TRAINING_AUGMENTED, "random-forest"),
    RegressionCell(FOCUS_PAIRS[1], "rank", TRAINING_PLAIN, "random-forest"),
]


@pytest.fixture
def small_data() -> RegressionData:
    instances = len(SMALL_TRAIN) + len(SMALL_TEST)
    rates = make_view_superiorities(n_instances=instances)
    return RegressionData(
        train_features=make_features(n_instances=instances, n_reps=1),
        train_superiorities=rates,
        test_features={
            label: make_features(n_instances=instances, n_reps=3, value_shift=shift)
            for label, shift in zip(SMALL_LABELS, (1.5, 4.0), strict=True)
        },
        test_superiorities=dict.fromkeys(SMALL_LABELS, rates),
        inputs=INPUTS,
    )


def _keys(cells) -> list[tuple[str, str, str, str]]:
    return sorted(
        (cell.pair.slug, cell.variant, cell.training, cell.kind) for cell in cells
    )


class TestOnlyTheMissingCellsAreFitted:
    """The cells a call is missing are fitted, not the grid their axes span."""

    @pytest.fixture
    def searched(self, monkeypatch) -> list[list[tuple[str, str, str, str]]]:
        """The cells each pass of the stage fitted, pass by pass."""
        passes: list[list[tuple[str, str, str, str]]] = []
        stage = regression_module.run_augmented_regression_stage

        def spy(*args, cells, **kwargs):
            passes.append(
                sorted(
                    (pair.slug, variant, training, kind)
                    for pair, variant, training, kind in cells
                )
            )
            return stage(*args, cells=cells, **kwargs)

        import aas_shortcut.resolve_regression_cells as cells_module

        monkeypatch.setattr(cells_module, "run_augmented_regression_stage", spy)
        return passes

    def test_three_cells_off_the_grid_are_three_fits(
        self, tmp_path, small_data, searched
    ):
        resolved = resolve_regression_cells(
            StageStore.under(tmp_path), SMALL_CONFIG, OFF_THE_GRID, small_data, "debug"
        )
        assert searched == [_keys(OFF_THE_GRID)]
        assert list(resolved) == OFF_THE_GRID

    def test_a_stored_cell_is_not_fitted_again(self, tmp_path, small_data, searched):
        store = StageStore.under(tmp_path)
        resolve_regression_cells(
            store, SMALL_CONFIG, OFF_THE_GRID[:1], small_data, "debug"
        )
        resolve_regression_cells(store, SMALL_CONFIG, OFF_THE_GRID, small_data, "debug")
        assert searched == [_keys(OFF_THE_GRID[:1]), _keys(OFF_THE_GRID[1:])]

    def test_a_cell_s_rows_do_not_depend_on_the_cells_fitted_beside_it(
        self, tmp_path, small_data
    ):
        alone = resolve_regression_cells(
            StageStore.under(tmp_path / "alone"),
            SMALL_CONFIG,
            OFF_THE_GRID,
            small_data,
            "debug",
        )
        grid = resolve_regression_cells(
            StageStore.under(tmp_path / "grid"),
            SMALL_CONFIG,
            cells_of(SMALL_CONFIG),
            small_data,
            "debug",
        )
        for cell in OFF_THE_GRID:
            assert alone[cell].fingerprint == grid[cell].fingerprint
            pd.testing.assert_frame_equal(
                alone[cell].read_table(cell_table(DEPLOYMENT_SHARD)),
                grid[cell].read_table(cell_table(DEPLOYMENT_SHARD)),
            )


class TestTheRowsOnOneFamily:
    """Experiment 2's table, read off Experiment 4's plain cells.

    No column names the training condition or the family. A level is the median over
    the training runs of each run's median deployment.
    """

    # The level the runs behind each family's score sit around.
    RUN_LEVELS: ClassVar[dict[tuple[str, str], float]] = {
        ("spline/warpheld", "r2_test_pre"): 0.65,
        ("spline/warpheld", "r2_test_post"): 0.45,
        ("power/warpheld", "r2_test_post"): 0.15,
    }

    @classmethod
    def cell_runs(cls, cell: RegressionCell) -> pd.DataFrame:
        middle = MODEL_SEEDS[len(MODEL_SEEDS) // 2]
        return pd.DataFrame(
            [
                {
                    "feature_set": "ela",
                    "feature_variant": cell.variant,
                    "training": cell.training,
                    "audit": audit,
                    "pair": cell.pair.slug,
                    "group": cell.pair.pair_type,
                    "model": cell.kind,
                    "n_features": 2,
                    "seed": seed,
                    "rep": rep,
                    "metric": metric,
                    "value": level + 0.001 * (seed - middle) + 0.0001 * (rep - 2),
                    "n_test": 30,
                    "scale": "debug",
                    "config_hash": "f" * 16,
                }
                for (audit, metric), level in cls.RUN_LEVELS.items()
                for seed in MODEL_SEEDS
                for rep in (1, 2, 3)
            ]
        )

    def artifacts(self, cells) -> dict:
        def reader(cell):
            def read_table(path):
                assert DEPLOYMENT_SHARD in str(path), path
                return self.cell_runs(cell)

            return read_table

        return {cell: SimpleNamespace(read_table=reader(cell)) for cell in cells}

    @staticmethod
    def plain() -> list[RegressionCell]:
        return [cell for cell in cells_of(CONFIG) if cell.training == TRAINING_PLAIN]

    def test_the_family_s_two_levels_are_kept(self):
        table = rows_on_family(self.artifacts(self.plain()), "spline/warpheld")
        assert set(table["metric"]) == {"r2_test_pre", "r2_test_post"}
        assert len(table) == 2 * len(self.plain())
        assert "training" not in table.columns
        assert "audit" not in table.columns

    def test_another_family_s_rows_are_left_out(self):
        table = rows_on_family(self.artifacts(self.plain()), "spline/warpheld")
        assert not {0.15} & set(table["value"])

    def test_a_level_is_the_median_of_its_runs(self):
        table = rows_on_family(self.artifacts(self.plain()), "spline/warpheld")
        values = table.groupby("metric")["value"].agg(set).to_dict()
        assert values["r2_test_pre"] == {0.65}
        assert values["r2_test_post"] == {0.45}

    def test_a_cell_without_the_family_is_refused(self):
        with pytest.raises(ValueError, match="knots-dense"):
            rows_on_family(self.artifacts(self.plain()), "knots-dense/warpheld")

    def test_cells_of_two_conditions_are_refused(self):
        with pytest.raises(ValueError, match="training conditions"):
            rows_on_family(self.artifacts(cells_of(CONFIG)[:2]), "spline/warpheld")
