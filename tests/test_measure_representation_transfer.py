"""Tests for Experiment 2: reading value, rank and concat superiority models.

The regression is Experiment 4's plain cells; tested here is how their table is read
and its premise: the G1 target is identical on both views while G3's is not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.define_table_schema import (
    VIEW_SUPERIORITY_SCHEMA,
    conform,
)
from aas_shortcut.measure_representation_transfer import (
    ANCHOR_CHECK,
    PREMISE_COLUMNS,
    TARGET_CHECK,
    describe_target_invariance,
    invariance_holds,
    premise_table,
)
from tests.conftest import CONFIG_HASH

# Two of the seed-choosing folds of ten training instances, and two held out.
INSTANCES = 22

# G1's pair, whose superiority a strictly increasing transformation cannot move, and
# G3's, whose it can.
G1 = next(pair for pair in FOCUS_PAIRS if pair.pair_type == "G1_G1")
G3 = next(pair for pair in FOCUS_PAIRS if pair.pair_type == "G3_G3")


def make_view_superiorities(
    n_instances: int = INSTANCES,
    dim: int = 2,
    *,
    moved: tuple[str, ...] = (G3.slug,),
    scale: str = "debug",
) -> pd.DataFrame:
    """Superiorities of the three focus pairs, per instance and view.

    ``moved`` names the pairs whose rate differs between the views; G1 must not be
    one, since its members search by comparison alone.
    """
    rows = []
    for pair in FOCUS_PAIRS:
        for instance_id in range(1, n_instances + 1):
            base = 0.1 + 0.8 * (instance_id % 5) / 4.0
            for transform in TRANSFORMS:
                rate = base
                if pair.slug in moved and transform == "post":
                    rate = min(1.0, base + 0.1)
                rows.append(
                    {
                        "problem_class": "bbob",
                        "function_id": 1,
                        "instance_id": instance_id,
                        "dimension": dim,
                        "transform": transform,
                        "pair": pair.slug,
                        "algorithm_a": pair.first,
                        "algorithm_b": pair.second,
                        "pair_type": pair.pair_type,
                        "n_trials": 3,
                        "superiority": rate,
                        "scale": scale,
                        "config_hash": CONFIG_HASH,
                    }
                )
    return conform(pd.DataFrame(rows), VIEW_SUPERIORITY_SCHEMA)


@pytest.fixture
def superiorities() -> pd.DataFrame:
    return make_view_superiorities()


class TestTargetInvariance:
    def test_g1_does_not_move_and_g3_does(self, superiorities):
        moved = describe_target_invariance(superiorities).set_index("group")
        assert int(moved.loc["G1", "n_moved"]) == 0
        assert int(moved.loc["G3", "n_moved"]) > 0
        assert invariance_holds(superiorities)

    def test_a_g1_that_moved_is_reported_as_broken(self):
        """Its members reproduce each trial bit for bit; any difference is a bug."""
        broken = make_view_superiorities(moved=(G1.slug,))
        assert not invariance_holds(broken)
        moved = describe_target_invariance(broken).set_index("group")
        assert float(moved.loc["G1", "max_difference"]) > 0.0

    def test_a_missing_view_is_rejected(self, superiorities):
        with pytest.raises(ValueError, match="missing view"):
            describe_target_invariance(
                superiorities[superiorities["transform"] == "pre"]
            )

    def test_instances_the_post_view_never_ran_are_skipped(self, superiorities):
        """Instances measured under one view only are skipped, not read as unmoved."""
        held_out = (1, 2)
        partial = superiorities[
            (superiorities["transform"] == "pre")
            | (superiorities["instance_id"].isin(held_out))
        ]
        moved = describe_target_invariance(partial).set_index("group")
        assert (moved["n_instances"] == len(held_out)).all()
        assert int(moved.loc["G3", "n_moved"]) > 0

    def test_a_table_with_nothing_paired_does_not_satisfy_the_premise(
        self, superiorities
    ):
        """Nothing moved and nothing was compared must not be the same answer."""
        unpaired = superiorities[superiorities["transform"] == "pre"].copy()
        post = unpaired.copy()
        post["transform"] = "post"
        post["instance_id"] = post["instance_id"] + INSTANCES
        assert not invariance_holds(pd.concat([unpaired, post], ignore_index=True))


class TestThePremiseTable:
    """A run's two checks as counts, written into its root."""

    def test_one_row_per_pair_and_block_then_one_per_anchor(self, superiorities):
        moved = describe_target_invariance(superiorities)
        table = premise_table(
            {"spline/warpheld": moved, "training block": moved},
            {"spline/warpheld": 14400},
            1e-9,
        )
        assert list(table.columns) == list(PREMISE_COLUMNS)
        assert list(table["check"]) == [TARGET_CHECK] * 6 + [ANCHOR_CHECK]
        targets = table[table["check"] == TARGET_CHECK].set_index(["block", "group"])
        g1 = targets.loc[("spline/warpheld", "G1")]
        assert (g1["n_compared"], g1["n_differing"]) == (INSTANCES, 0)
        assert targets.loc[("training block", "G3"), "n_differing"] > 0
        assert (targets["tolerance"] == 0.0).all()

    def test_an_anchor_counts_what_agreed_within_its_tolerance(self, superiorities):
        table = premise_table({}, {"spline/warpheld": 14400}, 1e-9)
        anchor = table.iloc[0]
        assert (anchor["n_compared"], anchor["n_differing"]) == (14400, 0)
        assert anchor["tolerance"] == 1e-9
        assert pd.isna(anchor["max_difference"])
