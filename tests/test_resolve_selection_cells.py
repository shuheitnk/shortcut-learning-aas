"""Tests for selection cells: what names a cell's unit, and what a cell keeps.

A cell fits each training run once and scores it on every block it is deployed on;
nothing is chosen between the runs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import aas_shortcut.resolve_selection_cells as cells_module
from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.compare_selection_runs import FAMILY_COLUMN, PROBLEM_COLUMNS
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    ORACLE_VARIANT,
    TRANSFORMS,
)
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_table_schema import (
    SELECTION_DEPLOYMENT_SCORE_SCHEMA,
    SELECTION_PROBLEM_SCORE_SCHEMA,
    conform,
)
from aas_shortcut.measure_selection_transfer import (
    PROBLEM_SHARD,
    SEED_DEPLOYMENT_SHARD,
    TRAINING_AUGMENTED,
    TRAINING_CONDITIONS,
    TRAINING_PLAIN,
    TRAINING_VIEWS,
)
from aas_shortcut.resolve_selection_cells import (
    FamilyData,
    SelectionCell,
    SelectionData,
    SelectionSetting,
    assemble_oracles,
    assemble_problems,
    assemble_records,
    cell_config,
    cell_inputs,
    cells_of,
    family_table,
    oracle_table,
    resolve_oracles,
    resolve_selection_cells,
    score_cells,
    selection_cell_unit,
    selection_learning,
)
from aas_shortcut.resolve_stage_artifacts import StageStore
from aas_shortcut.select_algorithms import SelectionConfig, SelectionSplit
from tests.conftest import identical_tables
from tests.test_measure_selection_transfer import (
    TEST,
    TRAIN,
    make_view_features,
    make_view_mean_superiorities,
    make_view_performance,
)
from tests.test_select_algorithms import DIM, SAMPLE_SIZE, SEARCH_BUDGET, TOTAL_BUDGET

LABEL = "spline/warpheld"
SEEDS = 3
REPS = 3
SPLIT = SelectionSplit(
    train_problem_class="bbob",
    test_problem_class="bbob",
    train_instances=TRAIN,
    test_instances=TEST,
)
CONFIG = SelectionConfig(
    feature_variants=("value", "rank"),
    feature_sets=(ELA_FEATURE_SET,),
    model=ModelConfig(kind="random-forest"),
    n_seeds=SEEDS,
)
SETTING = SelectionSetting(
    features=("a", "b"), dim=DIM, base_seed=42, sample_size=SAMPLE_SIZE
)
# The training block's units a cell may name, by role: the features, both
# conditions untransformed, and the selection condition transformed.
TRAINING_INPUTS = {
    "train_features": "y",
    "train_mean_superiority_pre:selection": "rp",
    "train_mean_superiority_pre:sbs": "sp",
    "train_mean_superiority_post:selection": "rq",
}


@pytest.fixture(scope="module")
def split():
    """The test's own split, for the cells."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cells_module, "SPLIT", SPLIT)
        yield SPLIT


@pytest.fixture(scope="module")
def tables():
    features = make_view_features(n_reps=REPS)
    selected = make_view_performance(SEARCH_BUDGET)
    sbs = make_view_performance(TOTAL_BUDGET, seed_offset=7)
    return (
        features,
        make_view_mean_superiorities(selected),
        selected,
        sbs,
        make_view_mean_superiorities(sbs),
    )


@pytest.fixture(scope="module")
def data(tables, split) -> SelectionData:
    """The same data, split the way the cells read it."""
    features, ranks, selected, sbs, sbs_ranks = tables
    train, test = (
        features["instance_id"].isin(TRAIN),
        features["instance_id"].isin(TEST),
    )

    def held(table: pd.DataFrame) -> pd.DataFrame:
        return table[table["instance_id"].isin(TEST)]

    return SelectionData(
        features=features[train],
        mean_superiorities=ranks[ranks["instance_id"].isin(TRAIN)],
        sbs_mean_superiorities=sbs_ranks[sbs_ranks["instance_id"].isin(TRAIN)],
        blocks={
            LABEL: FamilyData(
                features=features[test],
                mean_superiorities=held(ranks),
                selected_performance=held(selected),
                sbs_performance=held(sbs),
                inputs={"test": "x"},
            )
        },
        inputs=TRAINING_INPUTS,
    )


class TestWhatACellReads:
    """A cell names the training runs it reads, and no other.

    No cell names the transformed run at the total budget (the SBS is named from the
    untransformed one), and a plain cell names no transformed run at all.
    """

    @staticmethod
    def unit(training: str, inputs=TRAINING_INPUTS):
        cell = next(c for c in cells_of(CONFIG) if c.training == training)
        return selection_cell_unit(CONFIG, cell, SETTING, [LABEL], inputs)

    def test_a_plain_cell_names_no_transformed_run(self):
        named = set(self.unit(TRAINING_PLAIN).inputs)
        assert named == {
            "train_features",
            "train_mean_superiority_pre:selection",
            "train_mean_superiority_pre:sbs",
        }

    def test_an_augmented_cell_adds_the_selection_s_transformed_run(self):
        named = set(self.unit(TRAINING_AUGMENTED).inputs)
        assert "train_mean_superiority_post:selection" in named
        assert "train_mean_superiority_post:sbs" not in named

    def test_the_transformed_single_best_run_moves_no_cell(self):
        moved = {**TRAINING_INPUTS, "train_mean_superiority_post:sbs": "other"}
        for training in (TRAINING_PLAIN, TRAINING_AUGMENTED):
            assert (
                self.unit(training).fingerprint
                == self.unit(training, moved).fingerprint
            )

    def test_an_augmented_cell_without_the_transformed_run_is_refused(self):
        plain_only = {
            role: value
            for role, value in TRAINING_INPUTS.items()
            if not role.startswith("train_mean_superiority_post")
        }
        with pytest.raises(ValueError, match="post:selection"):
            self.unit(TRAINING_AUGMENTED, plain_only)

    def test_the_other_inputs_pass_through(self):
        inputs = {**TRAINING_INPUTS, "test_features:power": "d"}
        assert cell_inputs(inputs, TRAINING_PLAIN)["test_features:power"] == "d"


class TestCells:
    def test_every_cell_in_the_order_the_stage_scored_them(self):
        cells = cells_of(CONFIG)
        assert cells == [
            SelectionCell(ELA_FEATURE_SET, variant, training)
            for training in TRAINING_CONDITIONS
            for variant in ("value", "rank")
        ]

    def test_a_cell_s_configuration_is_narrowed_to_it(self):
        cell = SelectionCell(ELA_FEATURE_SET, "rank", TRAINING_AUGMENTED)
        narrowed = cell_config(CONFIG, cell)
        assert narrowed.feature_variants == ("rank",)
        assert narrowed.train_transforms == TRAINING_VIEWS[TRAINING_AUGMENTED]

    def test_the_unit_records_what_the_fit_depends_on(self):
        cell = SelectionCell(ELA_FEATURE_SET, "value", TRAINING_AUGMENTED)
        settings = selection_cell_unit(
            CONFIG, cell, SETTING, [LABEL], TRAINING_INPUTS
        ).settings
        assert settings["learning"] == selection_learning(CONFIG, TRAINING_AUGMENTED)
        # The forest the tests fit has no epochs to divide and no split to stop on.
        assert settings["learning"] == {}
        realmlp = replace(CONFIG, model=ModelConfig())
        assert selection_learning(realmlp, TRAINING_AUGMENTED) == {
            "epoch_divisor": 2,
            "early_stopping_split": {"unit": "problem", "fraction": 0.2},
        }
        assert settings["families"] == [LABEL]
        assert settings["features"] == ["a", "b"]
        assert settings["model_seeds"] == list(MODEL_SEEDS[:SEEDS])
        assert "seed_selection_cv" not in settings["learning"]

    def test_the_unit_names_the_records_it_keeps(self):
        """The unit names both records: deployments and their problems."""
        cell = cells_of(CONFIG)[0]
        settings = selection_cell_unit(
            CONFIG, cell, SETTING, [LABEL], TRAINING_INPUTS
        ).settings
        assert settings["records"] == [SEED_DEPLOYMENT_SHARD, PROBLEM_SHARD]

    def test_another_seed_count_is_another_unit(self):
        cell = cells_of(CONFIG)[0]
        assert (
            selection_cell_unit(
                CONFIG, cell, SETTING, [LABEL], TRAINING_INPUTS
            ).fingerprint
            != selection_cell_unit(
                replace(CONFIG, n_seeds=SEEDS + 1),
                cell,
                SETTING,
                [LABEL],
                TRAINING_INPUTS,
            ).fingerprint
        )

    def test_two_cells_are_two_units(self):
        one, two = cells_of(CONFIG)[:2]
        assert (
            selection_cell_unit(
                CONFIG, one, SETTING, [LABEL], TRAINING_INPUTS
            ).fingerprint
            != selection_cell_unit(
                CONFIG, two, SETTING, [LABEL], TRAINING_INPUTS
            ).fingerprint
        )

    def test_another_family_list_is_another_unit(self):
        cell = cells_of(CONFIG)[0]
        assert (
            selection_cell_unit(
                CONFIG, cell, SETTING, [LABEL], TRAINING_INPUTS
            ).fingerprint
            != selection_cell_unit(
                CONFIG, cell, SETTING, [LABEL, "power/warpheld"], TRAINING_INPUTS
            ).fingerprint
        )

    def test_the_worker_count_is_not_part_of_a_unit(self):
        cell = cells_of(CONFIG)[0]
        assert (
            selection_cell_unit(
                CONFIG, cell, SETTING, [LABEL], TRAINING_INPUTS
            ).fingerprint
            == selection_cell_unit(
                replace(CONFIG, n_jobs=8), cell, SETTING, [LABEL], TRAINING_INPUTS
            ).fingerprint
        )

    def test_a_family_s_tables_sit_in_its_own_directory(self):
        assert family_table("power/warpheld", PROBLEM_SHARD) == (
            "families/power_warpheld/metrics/problem_scores.parquet"
        )


@pytest.fixture(scope="module")
def scored(data):
    """Every cell, fitted and scored by the cells' own path."""
    return score_cells(data, CONFIG, SETTING, cells_of(CONFIG), "debug")


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    return tmp_path_factory.mktemp("store")


@pytest.fixture(scope="module")
def stored(data, store_root):
    """Every cell and the oracle, resolved into a store."""
    store = StageStore.under(store_root)
    cells = resolve_selection_cells(
        store, CONFIG, cells_of(CONFIG), data, SETTING, "debug"
    )
    oracles = resolve_oracles(store, CONFIG, data, SETTING, "debug")
    return cells, oracles


class TestTheCellsKeepEveryRun:
    """Every training run, every deployment, both views: nothing is chosen."""

    def test_every_run_is_recorded_in_every_deployment(self, scored):
        for result in scored.values():
            records = result.records[LABEL]
            assert sorted(records["seed"].unique()) == list(MODEL_SEEDS[:SEEDS])
            assert len(records) == SEEDS * REPS * len(TRANSFORMS)
            assert "deployed" not in records.columns

    def test_every_problem_of_every_deployment_is_kept(self, scored):
        for result in scored.values():
            problems = result.problems[LABEL]
            per_deployment = problems.groupby(["seed", "transform", "rep"]).size()
            assert len(per_deployment) == SEEDS * REPS * len(TRANSFORMS)
            assert set(per_deployment) == {len(TEST) * 3}

    def test_a_deployment_s_record_is_the_mean_of_its_problems(self, scored):
        """A deployment's record is bit for bit the mean of its problems."""
        keys = ["seed", "transform", "rep"]
        for cell, result in scored.items():
            records = result.records[LABEL].set_index(keys)["superiority_vs_sbs"]
            means = (
                result.problems[LABEL]
                .groupby(keys)["superiority_vs_sbs"]
                .agg(lambda values: float(np.mean(values.tolist())))
            )
            ours = records.sort_index().to_numpy()
            theirs = means.reindex(records.sort_index().index).to_numpy()
            assert np.array_equal(ours.view(np.uint64), theirs.view(np.uint64)), cell

    def test_both_tables_conform(self, scored):
        for result in scored.values():
            conform(result.records[LABEL], SELECTION_DEPLOYMENT_SCORE_SCHEMA)
            conform(result.problems[LABEL], SELECTION_PROBLEM_SCORE_SCHEMA)

    def test_a_cell_does_not_depend_on_the_cells_beside_it(self, data, scored):
        """A cell fitted alone gives the same rows as fitted with the others."""
        cell = cells_of(CONFIG)[-1]
        alone = score_cells(data, CONFIG, SETTING, [cell], "debug")[cell]
        for table in ("records", "problems"):
            same, detail = identical_tables(
                getattr(alone, table)[LABEL], getattr(scored[cell], table)[LABEL]
            )
            assert same, f"{table}: {detail}"

    def test_a_single_seed_is_a_cell(self, data):
        """A single seed is a valid cell."""
        cell = cells_of(CONFIG)[0]
        result = score_cells(data, replace(CONFIG, n_seeds=1), SETTING, [cell], "x")
        assert set(result[cell].records[LABEL]["seed"]) == {MODEL_SEEDS[0]}


class TestTheUnits:
    def test_every_unit_carries_its_own_fingerprint(self, stored):
        cells, oracles = stored
        for artifact in cells.values():
            for shard in (SEED_DEPLOYMENT_SHARD, PROBLEM_SHARD):
                rows = artifact.read_table(family_table(LABEL, shard))
                assert set(rows["config_hash"]) == {artifact.fingerprint}
        for artifact in oracles.values():
            rows = artifact.read_table(oracle_table())
            assert set(rows["config_hash"]) == {artifact.fingerprint}

    def test_a_cell_keeps_no_table_of_one_chosen_run(self, stored):
        cells, _ = stored
        for artifact in cells.values():
            names = {path.name for path in artifact.path.rglob("*.parquet")}
            assert names == {
                f"{SEED_DEPLOYMENT_SHARD}.parquet",
                f"{PROBLEM_SHARD}.parquet",
            }

    def test_the_records_assemble_with_their_family_named(self, scored, stored):
        cells, _ = stored
        records = assemble_records(cells, [LABEL])
        assert set(records[FAMILY_COLUMN]) == {LABEL}
        assert len(records) == sum(
            len(result.records[LABEL]) for result in scored.values()
        )

    def test_the_problems_assemble_with_their_family_named(self, scored, stored):
        cells, _ = stored
        problems = assemble_problems(cells, [LABEL])
        assert set(problems[FAMILY_COLUMN]) == {LABEL}
        assert len(problems) == sum(
            len(result.problems[LABEL]) for result in scored.values()
        )
        assert set(PROBLEM_COLUMNS) <= set(problems.columns)

    def test_the_oracles_assemble_with_their_family_named(self, stored):
        _, oracles = stored
        oracle = assemble_oracles(oracles, [LABEL])
        assert set(oracle[FAMILY_COLUMN]) == {LABEL}
        assert set(oracle["feature_variant"].astype(str)) == {ORACLE_VARIANT}
        assert set(oracle["training"].astype(str)) == set(TRAINING_CONDITIONS)
        assert set(oracle["transform"].astype(str)) == set(TRANSFORMS)

    def test_a_second_call_reuses_every_unit(self, data, stored, store_root):
        store = StageStore.under(store_root)
        resolve_selection_cells(store, CONFIG, cells_of(CONFIG), data, SETTING, "debug")
        resolve_oracles(store, CONFIG, data, SETTING, "debug")
        assert {entry["action"] for entry in store.resolved} == {"reused"}
