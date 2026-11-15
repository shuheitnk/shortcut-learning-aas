"""The portfolio benchmark's blocks: budgets, mean-superiority units, their inputs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from aas_shortcut import resolve_portfolio_blocks as blocks_module
from aas_shortcut.bind_portfolio_parameters import CONDITION_SBS, CONDITION_SELECTION
from aas_shortcut.calculate_features import OrderPreservationError
from aas_shortcut.calculate_mean_superiority import (
    VIEW_GROUP_COLUMNS,
    compute_mean_superiorities,
)
from aas_shortcut.define_table_schema import VIEW_MEAN_SUPERIORITY_SCHEMA, conform
from aas_shortcut.define_warp_audits import AUDITS
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN
from aas_shortcut.resolve_benchmark_units import POST, PRE
from aas_shortcut.resolve_portfolio_blocks import (
    CONDITIONS,
    SelectionHeldOut,
    SelectionTraining,
    condition_budget,
    mean_superiority_settings,
    mean_superiority_table,
    resolve_mean_superiorities,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
    selection_inputs,
)
from aas_shortcut.resolve_stage_artifacts import StageStore, StageUnit
from aas_shortcut.sample_monotone_transform import (
    DEFAULT_WARP,
    HELD_OUT_DOMAIN,
    WarpConfig,
)
from tests.conftest import identical_tables
from tests.test_measure_selection_transfer import make_view_performance
from tests.test_select_algorithms import DIM, SEARCH_BUDGET

CONTEXT: Any = SimpleNamespace(
    search_budget=200, total_budget=300, dim=DIM, scale="debug", warp=WarpConfig()
)


def _unit(fingerprint: str) -> Any:
    return SimpleNamespace(fingerprint=fingerprint)


def _portfolio(name: str) -> SimpleNamespace:
    return SimpleNamespace(benchmark=_unit(f"{name}-runs"), rank=_unit(f"{name}-rank"))


class TestConditions:
    def test_the_selection_searches_with_what_the_sample_leaves(self):
        assert condition_budget(CONTEXT, CONDITION_SELECTION) == 200

    def test_the_single_best_solver_is_given_the_whole_budget(self):
        assert condition_budget(CONTEXT, CONDITION_SBS) == 300

    def test_an_unknown_condition_is_rejected(self):
        with pytest.raises(ValueError, match="unknown condition"):
            condition_budget(CONTEXT, "oracle")

    def test_the_selection_is_read_first(self):
        assert CONDITIONS == (CONDITION_SELECTION, CONDITION_SBS)


class TestMeanSuperiorityUnits:
    def test_the_table_sits_where_the_stage_writes_it(self):
        assert mean_superiority_table(10, CONDITION_SBS) == (
            "mean_superiority/view_mean_superiority_sbs_bbob_10d.parquet"
        )

    def test_the_family_is_not_a_setting(self):
        """The family is not a setting, so the untransformed units are shared."""
        settings = mean_superiority_settings(CONTEXT, CONDITION_SELECTION)
        assert "warp" not in settings
        assert settings["condition"] == CONDITION_SELECTION

    def test_a_unit_s_rows_are_the_stage_s_and_a_second_call_reuses_it(self, tmp_path):
        performance = make_view_performance(SEARCH_BUDGET)
        source: Any = SimpleNamespace(fingerprint="0123456789abcdef")
        store = StageStore.under(tmp_path)
        artifact, rows = resolve_mean_superiorities(
            store, CONTEXT, CONDITION_SELECTION, (source, performance)
        )
        expected = conform(
            compute_mean_superiorities(
                performance, group_columns=VIEW_GROUP_COLUMNS
            ).assign(scale="debug", config_hash=""),
            VIEW_MEAN_SUPERIORITY_SCHEMA,
        )
        same, detail = identical_tables(rows, expected)
        assert same, detail
        assert set(rows["config_hash"]) == {artifact.fingerprint}
        assert artifact.unit.inputs == {"performance": source.fingerprint}

        again = StageStore.under(tmp_path)
        resolve_mean_superiorities(
            again, CONTEXT, CONDITION_SELECTION, (source, performance)
        )
        assert [entry["action"] for entry in again.resolved] == ["reused"]


class TestInputs:
    def training(self) -> SelectionTraining:
        return SelectionTraining(
            _unit("features"),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            {c: _portfolio(f"pre-{c}") for c in CONDITIONS},  # type: ignore[misc]
            {c: _portfolio(f"post-{c}") for c in CONDITIONS},  # type: ignore[misc]
        )

    def block(self, label: str, pre: str = "pre") -> SelectionHeldOut:
        return SelectionHeldOut(
            label,
            _unit(f"features-{label}"),  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            {c: _portfolio(f"{pre}-test-{c}") for c in CONDITIONS},  # type: ignore[misc]
            {c: _portfolio(f"post-{label}-{c}") for c in CONDITIONS},  # type: ignore[misc]
        )

    def test_the_training_block_names_its_features_and_mean_superiorities(self):
        inputs = self.training().inputs
        assert inputs["train_features"] == "features"
        assert (
            inputs[f"train_mean_superiority_post:{CONDITION_SBS}"]
            == f"post-{CONDITION_SBS}-rank"
        )

    def test_a_block_names_its_family_s_units_under_its_label(self):
        inputs = self.block("power/warpheld").inputs
        assert inputs["test_features:power/warpheld"] == "features-power/warpheld"
        assert (
            inputs[f"test_performance_post:{CONDITION_SELECTION}:power/warpheld"]
            == f"post-power/warpheld-{CONDITION_SELECTION}-runs"
        )
        assert inputs[f"test_mean_superiority_pre:{CONDITION_SBS}"] == (
            f"pre-test-{CONDITION_SBS}-rank"
        )

    def test_the_shared_untransformed_units_are_named_once(self):
        inputs = selection_inputs(
            self.training(),
            [self.block("spline/warpheld"), self.block("power/warpheld")],
        )
        assert "test_features:spline/warpheld" in inputs
        assert "test_features:power/warpheld" in inputs

    def test_two_blocks_that_disagree_on_a_shared_unit_are_refused(self):
        with pytest.raises(ValueError, match="another block"):
            selection_inputs(
                self.training(),
                [self.block("spline/warpheld"), self.block("power/warpheld", "other")],
            )


class TestTheTrainingBlockResolvesWhatIsRead:
    """Only the training runs a cell reads are benchmarked.

    The SBS is named from the untransformed block, so its transformed run is never
    benchmarked.
    """

    @staticmethod
    def resolved(monkeypatch, **kwargs) -> list[tuple[str, str]]:
        runs: list[tuple[str, str]] = []

        def unit(store, context, bindings, condition, *, view, block, warp=None):
            runs.append((condition, view))
            return _portfolio(f"{view}-{condition}")

        monkeypatch.setattr(
            blocks_module,
            "resolve_training_features",
            lambda store, context: (_unit("features"), pd.DataFrame()),
        )
        monkeypatch.setattr(blocks_module, "resolve_portfolio_unit", unit)
        context = SimpleNamespace(warp=DEFAULT_WARP)
        training = resolve_selection_training(None, context, None, **kwargs)  # type: ignore[arg-type]
        named = {
            role
            for role in training.inputs
            if role.startswith("train_mean_superiority_")
        }
        assert named == {
            f"train_mean_superiority_{view}:{condition}" for condition, view in runs
        }
        return runs

    def test_plain_cells_alone_run_nothing_transformed(self, monkeypatch):
        runs = self.resolved(monkeypatch, trainings=(TRAINING_PLAIN,))
        assert sorted(runs) == sorted(
            [(CONDITION_SELECTION, PRE), (CONDITION_SBS, PRE)]
        )

    def test_augmented_cells_add_the_selection_s_transformed_run_alone(
        self, monkeypatch
    ):
        runs = self.resolved(monkeypatch, trainings=(TRAINING_AUGMENTED,))
        assert (CONDITION_SELECTION, POST) in runs
        assert (CONDITION_SBS, POST) not in runs
        assert len(runs) == 3

    def test_by_default_both_conditions_are_resolved_for(self, monkeypatch):
        assert sorted(self.resolved(monkeypatch)) == sorted(
            [
                (CONDITION_SELECTION, PRE),
                (CONDITION_SBS, PRE),
                (CONDITION_SELECTION, POST),
            ]
        )


def test_the_training_block_is_drawn_under_the_study_s_own_warp(tmp_path):
    held_out = SimpleNamespace(warp=WarpConfig(domain=HELD_OUT_DOMAIN))
    with pytest.raises(ValueError, match="features are computed under"):
        resolve_selection_training(StageStore.under(tmp_path), held_out, None)  # type: ignore[arg-type]


def test_a_stage_unit_is_what_the_resolver_builds():
    """The mean-superiority unit is named by its settings and its benchmark alone."""
    unit = StageUnit(
        "view_mean_superiority",
        mean_superiority_settings(CONTEXT, CONDITION_SBS),
        inputs={"performance": "a"},
    )
    other = StageUnit(
        "view_mean_superiority",
        mean_superiority_settings(CONTEXT, CONDITION_SBS),
        inputs={"performance": "b"},
    )
    assert unit.fingerprint != other.fingerprint


def test_a_family_out_of_its_range_is_left_out_and_the_rest_are_scored(monkeypatch):
    """A family that cannot preserve the sample's order is skipped."""

    def resolve(store, context, bindings, audit, pre):
        if audit.name == "power":
            raise OrderPreservationError("out of range")
        return audit.label

    monkeypatch.setattr(
        blocks_module, "resolve_selection_held_out_pre", lambda *a: None
    )
    monkeypatch.setattr(blocks_module, "resolve_selection_held_out", resolve)
    blocks = resolve_selection_held_out_in_range(None, None, None)  # type: ignore[arg-type]
    assert blocks == [audit.label for audit in AUDITS if audit.name != "power"]
