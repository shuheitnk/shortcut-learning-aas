"""The fingerprints a regression cell records of the blocks it is fitted on."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from aas_shortcut import resolve_pair_blocks as blocks_module
from aas_shortcut.calculate_features import OrderPreservationError
from aas_shortcut.define_warp_audits import AUDITS
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED, TRAINING_PLAIN
from aas_shortcut.resolve_benchmark_units import POST, PRE
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    block_inputs,
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.sample_monotone_transform import (
    DEFAULT_WARP,
    HELD_OUT_DOMAIN,
    WarpConfig,
)


def _unit(fingerprint: str) -> Any:
    return SimpleNamespace(fingerprint=fingerprint)


def _pair(superiority: str) -> Any:
    return SimpleNamespace(superiority=_unit(superiority))


def training() -> TrainingBlock:
    return TrainingBlock(
        _unit("features"), pd.DataFrame(), _pair("train-pre"), _pair("train-post")
    )  # type: ignore[arg-type]


def held_out(label: str, pre: str = "test-pre") -> HeldOutBlock:
    return HeldOutBlock(
        label,
        _unit(f"deployments-{label}"),
        pd.DataFrame(),
        _pair(pre),
        _pair(f"post-{label}"),
    )  # type: ignore[arg-type]


class TestBlockInputs:
    def test_the_training_block_names_its_three_units(self):
        assert block_inputs(training(), []) == {
            "train_features": "features",
            "train_superiority_pre": "train-pre",
            "train_superiority_post": "train-post",
        }

    def test_a_block_resolved_for_plain_cells_names_no_transformed_run(self):
        plain = TrainingBlock(_unit("features"), pd.DataFrame(), _pair("train-pre"))  # type: ignore[arg-type]
        assert block_inputs(plain, []) == {
            "train_features": "features",
            "train_superiority_pre": "train-pre",
        }

    def test_each_family_adds_its_own_units_and_shares_the_untransformed_one(self):
        inputs = block_inputs(training(), [held_out("a"), held_out("b")])
        assert inputs["test_superiority_pre"] == "test-pre"
        assert inputs["test_features:a"] == "deployments-a"
        assert inputs["test_superiority_post:b"] == "post-b"
        # Three for the training block, the shared one, two per family.
        assert len(inputs) == 3 + 1 + 2 * 2

    def test_the_order_of_the_families_does_not_change_the_inputs(self):
        forward = block_inputs(training(), [held_out("a"), held_out("b")])
        backward = block_inputs(training(), [held_out("b"), held_out("a")])
        assert forward == backward

    def test_two_untransformed_units_are_refused(self):
        with pytest.raises(ValueError, match="test_superiority_pre"):
            block_inputs(training(), [held_out("a"), held_out("b", pre="other")])


class TestTrainingBlock:
    def test_another_draw_than_the_study_s_is_refused(self):
        """Its features are computed under the study's draw; its runs would not be."""
        context = SimpleNamespace(warp=WarpConfig(domain=HELD_OUT_DOMAIN))
        with pytest.raises(ValueError, match="features are computed under"):
            resolve_training_block(None, context, None, None, "E2-1")  # type: ignore[arg-type]

    @staticmethod
    def resolved_views(monkeypatch, **kwargs) -> list[str]:
        """The views of the training instances the resolver benchmarks."""
        views: list[str] = []

        def unit(store, context, binding, tuning, *, view, block, step, warp=None):
            views.append(view)
            return _pair(f"train-{view}")

        monkeypatch.setattr(
            blocks_module,
            "resolve_training_features",
            lambda store, context: (_unit("features"), pd.DataFrame()),
        )
        monkeypatch.setattr(blocks_module, "resolve_pair_unit", unit)
        context = SimpleNamespace(warp=DEFAULT_WARP)
        block = resolve_training_block(None, context, None, None, "E1-1", **kwargs)  # type: ignore[arg-type]
        assert list(block.units) == views
        return views

    def test_plain_cells_alone_do_not_run_the_transformed_training_benchmark(
        self, monkeypatch
    ):
        views = self.resolved_views(monkeypatch, trainings=(TRAINING_PLAIN,))
        assert views == [PRE]

    def test_augmented_cells_run_both_views(self, monkeypatch):
        assert self.resolved_views(monkeypatch, trainings=(TRAINING_AUGMENTED,)) == [
            PRE,
            POST,
        ]

    def test_by_default_both_conditions_are_resolved_for(self, monkeypatch):
        assert self.resolved_views(monkeypatch) == [PRE, POST]


class TestTheFamiliesInRange:
    def test_a_family_out_of_its_range_is_left_out_and_the_rest_are_scored(
        self, monkeypatch
    ):
        """A family that cannot preserve the sample's order is skipped."""

        def resolve(store, context, binding, tuning, audit, pre, step):
            if audit.name == "power":
                raise OrderPreservationError("out of range")
            return audit.label

        monkeypatch.setattr(blocks_module, "resolve_held_out_pre", lambda *a: None)
        monkeypatch.setattr(blocks_module, "resolve_held_out_block", resolve)
        blocks = resolve_held_out_in_range(None, None, None, None, "E2-1")  # type: ignore[arg-type]
        assert blocks == [audit.label for audit in AUDITS if audit.name != "power"]
