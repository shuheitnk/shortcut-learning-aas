"""The preanalysis's G1 control, its deployments, and the block it reads.

The rank-based pair reads only the ordering of the objective and both views share
trial seeds, so a moved superiority is a bug, not a finding.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "preanalysis"))

from run_preanalysis import (
    DEPLOYMENTS,
    _check_control,
    _check_deployments,
    _deployed,
    _held_out,
)

from aas_shortcut.calculate_delta_z import PRIMARY_STANDARDIZATION
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_paths import FEATURE_QUANTITY, SUPERIORITY_QUANTITY
from aas_shortcut.define_repetitions import REPETITIONS
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.split_bbob_instances import TEST_INSTANCES

FUNCTIONS = (1, 2)


def summary(g1_shift: float = 0.0, g1_raw: float = 0.0) -> pd.DataFrame:
    """The reported summary of the G1 pair and of one feature beside it."""
    common = {"statistic": "mean", "standardization": PRIMARY_STANDARDIZATION}
    return pd.DataFrame(
        [
            {
                **common,
                "quantity_kind": SUPERIORITY_QUANTITY,
                "feature_variant": "none",
                "quantity": "bipop-cmaes__adaptive-nelder-mead",
                "quantity_class": GROUPS[0],
                "value": g1_shift,
                "value_raw": g1_raw,
                "p_moved": 1.0 if g1_raw else 0.0,
                "n_instances": 240,
            },
            {
                **common,
                "quantity_kind": FEATURE_QUANTITY,
                "feature_variant": "value",
                "quantity": "ic.eps_s",
                "quantity_class": "ic",
                "value": 0.9,
                "value_raw": 0.1,
                "p_moved": 1.0,
                "n_instances": 240,
            },
        ]
    )


def block(
    transforms: tuple[str, ...] = ("pre", "post"),
    instances: tuple[int, ...] = TEST_INSTANCES,
    extra_instances: tuple[int, ...] = (),
) -> pd.DataFrame:
    """A superiority shard covering the given views and instances."""
    return pd.DataFrame(
        [
            {
                "problem_class": "bbob",
                "function_id": function,
                "instance_id": instance,
                "dimension": 10,
                "transform": transform,
                "pair": "shade__l-shade",
                "superiority": 0.5,
            }
            for transform in transforms
            for function in FUNCTIONS
            for instance in (
                instances if transform == "post" else (*instances, *extra_instances)
            )
        ]
    )


class TestControl:
    def test_a_pair_that_did_not_move_is_readable(self):
        assert _check_control(summary()) is True

    def test_a_moved_rank_based_pair_is_not(self, caplog):
        assert _check_control(summary(g1_shift=0.01, g1_raw=0.004)) is False
        assert "rank-based pair's superiority moved" in caplog.text

    def test_a_pair_that_moved_below_its_own_scale_is_not_either(self, caplog):
        """A raw move too small to show once standardized is still a move."""
        assert _check_control(summary(g1_shift=0.0, g1_raw=1e-9)) is False
        assert "the G1 control failed" in caplog.text


class TestHeldOutBlock:
    def test_the_training_block_is_left_out(self):
        """Training instances in the shard are dropped from the held-out block."""
        kept = _held_out(block(extra_instances=(11, 12)), shard="superiority")
        assert set(kept["instance_id"]) == set(TEST_INSTANCES)

    def test_a_view_missing_a_problem_is_refused(self):
        partial = block(instances=TEST_INSTANCES[:-1])
        with pytest.raises(ValueError, match="is missing 2 problem"):
            _held_out(partial, shard="superiority")

    def test_a_shard_holding_one_view_is_refused(self):
        with pytest.raises(ValueError, match="'post' view is missing"):
            _held_out(block(transforms=("pre",)), shard="superiority")


def features() -> pd.DataFrame:
    """A deployment feature table holding two samples and both encodings."""
    return pd.DataFrame(
        {
            "rep": [1, 1, 2, 1],
            "feature_variant": ["value", "rank", "value", "value"],
            "feature": ["ic.eps_s", "ic.eps_s", "ic.eps_s", "ic.eps_ratio"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def every_deployment(drop: tuple[str, int] | None = None) -> pd.DataFrame:
    """One problem's column under both views in every deployment, but ``drop``."""
    rows = [
        {
            "function_id": 1,
            "instance_id": 1,
            "transform": view,
            "feature": "ic.eps_s",
            "rep": rep,
        }
        for view in ("pre", "post")
        for rep in DEPLOYMENTS
        if (view, rep) != drop
    ]
    return pd.DataFrame(rows)


class TestDeployedSample:
    def test_every_deployment_and_one_encoding_reach_the_measurement(self):
        """Every deployment and only the value encoding are kept."""
        assert tuple(range(1, REPETITIONS + 1)) == DEPLOYMENTS
        assert list(_deployed(features())["value"]) == [1.0, 3.0, 4.0]

    def test_a_table_holding_every_deployment_passes(self):
        _check_deployments(every_deployment())

    def test_a_deployment_missing_from_one_view_is_refused(self):
        """A deployment missing from one view is refused."""
        with pytest.raises(ValueError, match=r"post ic\.eps_s"):
            _check_deployments(every_deployment(drop=("post", 3)))

    def test_a_column_the_regression_does_not_fit_on_is_not_measured(self):
        """Only the columns the regression fits on are measured."""
        kept = _deployed(features(), {"ic.eps_s"})
        assert set(kept["feature"]) == {"ic.eps_s"}
        assert list(kept["rep"]) == [1, 2]


class TestTheBlock:
    """The control family's held-out block, the one Experiment 4 is scored on."""

    def test_the_control_family_s_block_is_resolved(self, monkeypatch):
        import run_preanalysis

        asked: dict[str, object] = {}

        def block(store, context, binding, tuning, audit, pre, step):
            asked["audit"] = audit
            return "block"

        monkeypatch.setattr(
            run_preanalysis,
            "resolve_pair_binding",
            lambda store, context, step: ("binding", "tuning"),
        )
        monkeypatch.setattr(run_preanalysis, "resolve_held_out_pre", lambda *a: "pre")
        monkeypatch.setattr(run_preanalysis, "resolve_held_out_block", block)
        assert run_preanalysis._resolve(cast(Any, None), None) == "block"
        assert asked["audit"] == CONTROL
