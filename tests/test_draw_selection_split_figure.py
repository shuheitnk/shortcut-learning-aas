"""Tests that Fig. 16 and Fig. 28 say what the selection table says.

The two halves of a violin are paired on the problem key, never on row order; each
half's rule and number are the table's level (or mean); each representation has its
own file.
"""

from __future__ import annotations

from typing import cast

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from aas_shortcut.create_figures.draw_selection_split_figure import (
    AUGMENTATION_SPLIT,
    PROBLEM_KEYS,
    SPLIT_STEM,
    SPLIT_VALUE_SIZE,
    figure_selection_split_pooled,
    paired_superiorities,
)
from aas_shortcut.create_figures.render_experiment_figures import (
    render_selection_split,
)
from aas_shortcut.define_paths import ORACLE_VARIANT, RunPaths

# Chosen so that two methods fall by more than `SPLIT_EMPHASIS_THRESHOLD` and two do
# not.
LEVELS: dict[str, tuple[float, float]] = {
    "value": (0.70, 0.55),
    "rank": (0.66, 0.63),
    "concat": (0.72, 0.57),
    ORACLE_VARIANT: (0.78, 0.78),
}

# Spread around each method's mean and problems per block. Symmetric, so the mean is
# exactly the level above.
SPREAD = 0.06
PROBLEMS = 12


def make_rows(
    feature_set: str = "ela",
    levels: dict[str, tuple[float, float]] | None = None,
    training: str = "plain",
) -> pd.DataFrame:
    """One selection table, column for column as `view_selection` holds it."""
    levels = levels or LEVELS
    offsets = np.linspace(-SPREAD, SPREAD, PROBLEMS)
    rows = []
    for variant, (pre, post) in levels.items():
        for transform, mean in (("pre", pre), ("post", post)):
            for index, offset in enumerate(offsets):
                rows.append(
                    {
                        "problem_class": "bbob",
                        "function_id": 1 + index // 4,
                        "instance_id": 1 + index % 4,
                        "dimension": 10,
                        "transform": transform,
                        "training": training,
                        "feature_set": feature_set,
                        "feature_variant": variant,
                        "superiority_vs_sbs": mean + offset,
                    }
                )
    return pd.DataFrame(rows)


def drawn_figure(table: pd.DataFrame, tmp_path, **kwargs):
    """Render once and hand back the figure before the canvas is closed."""
    import matplotlib.pyplot as plt

    seen = []
    original = plt.close
    # The figure is kept open so its artists can be measured; drop the last test's.
    original("all")
    plt.close = seen.append
    try:
        figure_selection_split_pooled(
            table, RunPaths.from_root(tmp_path), ("png",), **kwargs
        )
    finally:
        plt.close = original
    return seen[0]


def drawn_axes(table: pd.DataFrame, tmp_path, **kwargs):
    """The panel of one render."""
    return drawn_figure(table, tmp_path, **kwargs).axes[0]


def by_gid(ax, prefix: str) -> dict:
    """Every named mark whose name starts with ``prefix``, keyed by method."""
    marks = {}
    for artist in [*ax.lines, *ax.patches, *ax.texts]:
        gid = artist.get_gid() or ""
        if gid.startswith(f"{prefix}-"):
            marks[gid[len(prefix) + 1 :]] = artist
    return marks


class TestPairing:
    def test_the_two_halves_are_the_same_problems(self):
        original, transformed = paired_superiorities(
            make_rows(), "feature_variant", "value"
        )
        assert original.size == transformed.size == PROBLEMS

    def test_a_block_written_in_another_order_pairs_the_same(self):
        """The halves are joined on the problem key, never on position."""
        table = make_rows()
        shuffled = cast(
            pd.DataFrame,
            pd.concat(
                [
                    table[table["transform"] == "pre"],
                    table[table["transform"] == "post"].iloc[::-1],
                ],
                ignore_index=True,
            ),
        )
        straight = paired_superiorities(table, "feature_variant", "value")
        crossed = paired_superiorities(shuffled, "feature_variant", "value")
        for one, other in zip(straight, crossed, strict=True):
            assert one == pytest.approx(other)

    def test_a_problem_only_one_block_holds_is_left_out_of_both(self):
        table = make_rows()
        dropped = cast(
            pd.DataFrame,
            table[
                (table["transform"] == "post")
                & (table["feature_variant"] == "value")
                & (table["instance_id"] == 1)
            ],
        )
        short = table.drop(list(dropped.index))
        original, transformed = paired_superiorities(short, "feature_variant", "value")
        assert original.size == transformed.size
        assert original.size == PROBLEMS - len(range(0, PROBLEMS, 4))

    def test_a_table_without_a_problem_key_is_refused(self):
        """Pairing by row order would report a difference no problem has."""
        table = make_rows().drop(columns=list(PROBLEM_KEYS))
        with pytest.raises(KeyError):
            paired_superiorities(table, "feature_variant", "value")


def make_training_rows(families: tuple[str, ...] = ("a", "b")) -> pd.DataFrame:
    """Fig. 28's table: the transformed block of each family, both conditions."""
    lifted = {name: (post, post + 0.05) for name, (_, post) in LEVELS.items()}
    lifted[ORACLE_VARIANT] = (LEVELS[ORACLE_VARIANT][1], LEVELS[ORACLE_VARIANT][1])
    frames = []
    for family in families:
        for training, side in (("plain", 0), ("augmented", 1)):
            levels = {name: (pair[side], pair[side]) for name, pair in lifted.items()}
            rows = make_rows(levels=levels, training=training)
            frames.append(rows[rows["transform"] == "post"].assign(family=family))
    return pd.concat(frames, ignore_index=True)


class TestTrainingSplit:
    """Fig. 28: the halves are the training conditions, pooled over families."""

    KEYS = (*PROBLEM_KEYS, "family")

    def test_every_family_s_problems_are_pooled_and_paired(self):
        plain, augmented = paired_superiorities(
            make_training_rows(),
            "feature_variant",
            "value",
            split="training",
            blocks=AUGMENTATION_SPLIT,
            keys=self.KEYS,
        )
        assert plain.size == augmented.size == 2 * PROBLEMS
        assert augmented - plain == pytest.approx(np.full(2 * PROBLEMS, 0.05))

    def test_the_halves_and_the_key_are_named_after_the_conditions(self, tmp_path):
        ax = drawn_axes(
            make_training_rows(),
            tmp_path,
            split="training",
            blocks=AUGMENTATION_SPLIT,
            keys=self.KEYS,
        )
        assert set(by_gid(ax, "plain")) == {*LEVELS}
        assert set(by_gid(ax, "augmented")) == {*LEVELS}
        words = {text.get_text() for text in ax.texts}
        assert {"plain", "augmented"} <= words


class TestLevels:
    def test_each_half_is_marked_at_its_own_mean(self, tmp_path):
        """Each half draws one rule, at its mean."""
        ax = drawn_axes(make_rows(), tmp_path)
        for block, index in (("original", 0), ("transformed", 1)):
            rules = by_gid(ax, f"mean-{block}")
            assert set(rules) == set(LEVELS)
            for variant, rule in rules.items():
                assert float(rule.get_ydata()[0]) == pytest.approx(
                    LEVELS[variant][index]
                )
                assert rule.get_ydata()[0] == rule.get_ydata()[1]

    def test_each_block_writes_its_own_mean_beside_its_own_rule(self, tmp_path):
        ax = drawn_axes(make_rows(), tmp_path)
        for block, index in (("original", 0), ("transformed", 1)):
            written = by_gid(ax, f"level-{block}")
            assert set(written) == set(LEVELS)
            for variant, text in written.items():
                level = LEVELS[variant][index]
                assert text.get_text() == f"{level:.3f}"
                assert text.get_fontsize() == SPLIT_VALUE_SIZE
                assert text.get_position()[1] == pytest.approx(level)
                assert text.get_va() == "center"

    def test_a_level_given_is_where_the_rule_and_the_number_go(self, tmp_path):
        """The table's level, not the violin's mean."""
        given = {("value", "pre"): 0.61, ("value", "post"): 0.43}
        ax = drawn_axes(make_rows(), tmp_path, levels=given)
        for block, transform in (("original", "pre"), ("transformed", "post")):
            rule = by_gid(ax, f"mean-{block}")["value"]
            text = by_gid(ax, f"level-{block}")["value"]
            level = given[("value", transform)]
            assert float(rule.get_ydata()[0]) == pytest.approx(level)
            assert text.get_text() == f"{level:.3f}"

    def test_a_method_without_a_level_is_drawn_at_its_mean(self, tmp_path):
        """The oracle has no training run, so it keeps its own mean."""
        given = {("value", "pre"): 0.61, ("value", "post"): 0.43}
        ax = drawn_axes(make_rows(), tmp_path, levels=given)
        rule = by_gid(ax, "mean-transformed")[ORACLE_VARIANT]
        assert float(rule.get_ydata()[0]) == pytest.approx(LEVELS[ORACLE_VARIANT][1])


class TestFiles:
    def test_it_is_written_under_its_own_name(self, tmp_path):
        paths = figure_selection_split_pooled(
            make_rows(), RunPaths.from_root(tmp_path), ("png",)
        )
        assert [path.name for path in paths] == [f"{SPLIT_STEM}.png"]
        assert paths[0].exists()

    def test_each_representation_is_its_own_file(self, tmp_path):
        table = pd.concat([make_rows("ela"), make_rows("deep-ela")], ignore_index=True)
        names = {
            path.name
            for path in render_selection_split(
                table, RunPaths.from_root(tmp_path), ("png",)
            )
        }
        assert names == {f"{SPLIT_STEM}.png", f"{SPLIT_STEM}_deep-ela.png"}

    def test_only_the_named_training_condition_is_drawn(self, tmp_path):
        table = pd.concat(
            [make_rows(training="plain"), make_rows(training="augmented")],
            ignore_index=True,
        )
        assert (
            render_selection_split(
                table, RunPaths.from_root(tmp_path), ("png",), training="absent"
            )
            == []
        )
        assert render_selection_split(table, RunPaths.from_root(tmp_path), ("png",))

    def test_a_representation_the_table_lacks_writes_nothing(self, tmp_path):
        assert (
            figure_selection_split_pooled(
                make_rows(),
                RunPaths.from_root(tmp_path),
                ("png",),
                feature_set="absent",
            )
            == []
        )
