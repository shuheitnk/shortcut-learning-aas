"""The presentation figures: Experiment 1's G1 pair, and the representations.

A missing cell is refused rather than drawn as a zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from aas_shortcut.create_figures.draw_hero_figures import (
    AUGMENTATION_BLOCKS,
    BLOCKS,
    HERO_PAIR,
    HERO_STEM,
    REPRESENTATION_BLOCKS,
    REPRESENTATIONS,
    SERIES,
    figure_representation_hero,
    figure_shortcut_hero,
    hero_scores,
    representation_scores,
)
from aas_shortcut.define_paths import RunPaths


def make_report(
    scores: dict[str, tuple[float, float]] | None = None, pair: str = HERO_PAIR
) -> pd.DataFrame:
    """A shortcut report holding one score per predictor set and block."""
    scores = scores or {
        "all": (0.86, 0.06),
        "invariant": (0.48, 0.48),
        # The report carries a third set the figure does not draw.
        "varying": (0.87, -0.08),
    }
    return pd.DataFrame(
        [
            {
                "pair": pair,
                "predictors": predictors,
                "metric": metric,
                "value": value,
                "fold": "held-out",
            }
            for predictors, values in scores.items()
            for (metric, _), value in zip(BLOCKS, values, strict=True)
        ]
    )


class TestHeroScores:
    def test_it_compares_the_two_predictor_sets_the_claim_is_about(self):
        """`all` against `invariant`: what a model may read, and what survives."""
        assert [name for name, _, _ in SERIES] == ["all", "invariant"]

    def test_it_reads_the_pair_the_figure_is_about(self):
        """G1: the pair whose superiority the transformation cannot move."""
        assert HERO_PAIR == "bipop-cmaes__adaptive-nelder-mead"

    def test_every_predictor_set_and_block_is_read(self):
        table = hero_scores(make_report())
        for name, _, _ in SERIES:
            for metric, _ in BLOCKS:
                assert not pd.isna(table.loc[name, metric])

    def test_another_pair_is_not_drawn_in_its_place(self):
        """A report of the other pairs would otherwise be drawn as if it were G1."""
        with pytest.raises(ValueError, match="no rows"):
            hero_scores(make_report(pair="shade__l-shade"))

    def test_a_missing_cell_is_refused(self):
        """A hole in a line would be drawn at zero and read as a collapse."""
        report = make_report()
        thinned = cast(pd.DataFrame, report[report["predictors"] != "invariant"])
        with pytest.raises(ValueError, match="missing"):
            hero_scores(thinned)

    def test_the_rows_of_several_folds_are_reduced(self):
        """The report carries a row per fold; the figure wants one number."""
        doubled = pd.concat([make_report(), make_report()], ignore_index=True)
        table = hero_scores(doubled)
        assert table.loc["invariant", BLOCKS[0][0]] == pytest.approx(0.48)


class TestFigure:
    def test_one_file_per_format(self, tmp_path: Path):
        """The figure is written once per format."""
        written = figure_shortcut_hero(
            make_report(), RunPaths.from_root(tmp_path), formats=("png", "svg")
        )
        assert [path.name for path in written] == [
            f"{HERO_STEM}.png",
            f"{HERO_STEM}.svg",
        ]
        assert all(path.stat().st_size > 0 for path in written)

    def test_a_score_below_zero_stays_inside_the_frame(self, tmp_path: Path):
        """A model worse than the training mean is the strongest reading there is."""
        report = make_report({"all": (0.9, -0.3), "invariant": (0.5, 0.5)})
        written = figure_shortcut_hero(
            report, RunPaths.from_root(tmp_path), formats=("png",)
        )
        assert written[0].exists()


def make_representation(
    scores: dict[str, dict[str, tuple[float, float]]] | None = None,
    feature_set: str = "ela",
) -> pd.DataFrame:
    """A representation report holding one score per group, variant and block."""
    scores = scores or {
        "G1": {"value": (0.86, 0.06), "rank": (0.65, 0.65), "concat": (0.87, 0.09)},
        "G2": {"value": (0.79, 0.17), "rank": (0.66, 0.66), "concat": (0.76, 0.26)},
        "G3": {"value": (0.85, 0.05), "rank": (0.73, 0.05), "concat": (0.86, 0.05)},
    }
    return pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "group": group,
                "feature_variant": variant,
                "metric": metric,
                "value": value,
            }
            for group, series in scores.items()
            for variant, values in series.items()
            for (metric, _, _), value in zip(REPRESENTATION_BLOCKS, values, strict=True)
        ]
    )


class TestRepresentationHero:
    def test_it_draws_the_three_representations_the_study_compares(self):
        assert [name for name, _, _ in REPRESENTATIONS] == ["value", "rank", "concat"]

    def test_every_group_is_read_in_the_order_it_is_named(self):
        table = representation_scores(
            make_representation(), REPRESENTATION_BLOCKS[0][0]
        )
        assert list(table.index) == ["G1", "G2", "G3"]

    def test_a_missing_representation_is_refused(self):
        """A group drawn with two bars would read as a third scoring zero."""
        report = make_representation()
        thinned = cast(pd.DataFrame, report[report["feature_variant"] != "concat"])
        with pytest.raises(ValueError, match="missing representation"):
            representation_scores(thinned, REPRESENTATION_BLOCKS[0][0])

    def test_a_block_the_report_does_not_hold_is_refused(self):
        with pytest.raises(ValueError, match="no 'r2_test_missing' row"):
            representation_scores(make_representation(), "r2_test_missing")

    def test_one_file_per_block_and_representation(self, tmp_path: Path):
        written = figure_representation_hero(
            make_representation(), RunPaths.from_root(tmp_path), formats=("png",)
        )
        assert [path.name for path in written] == [
            f"{stem}.png" for _, _, stem in REPRESENTATION_BLOCKS
        ]

    def test_deep_ela_is_written_beside_ela_and_never_pooled(self, tmp_path: Path):
        """ELA and Deep-ELA are two figures."""
        report = pd.concat(
            [make_representation(), make_representation(feature_set="deep-ela")],
            ignore_index=True,
        )
        written = figure_representation_hero(
            report, RunPaths.from_root(tmp_path), formats=("png",)
        )
        names = [path.name for path in written]
        assert any(name.endswith("_deep-ela.png") for name in names)
        assert len(names) == 2 * len(REPRESENTATION_BLOCKS)


class TestAugmentationConditions:
    """Fig. 27: Experiment 4's two training conditions in Fig. 15's design."""

    def test_one_file_per_condition(self, tmp_path: Path):
        report = make_representation().replace(
            {"metric": {"r2_test_pre": "plain", "r2_test_post": "augmented"}}
        )
        written = figure_representation_hero(
            report,
            RunPaths.from_root(tmp_path),
            formats=("png",),
            blocks=AUGMENTATION_BLOCKS,
        )
        assert [path.name for path in written] == [
            "fig27_augmentation_regression_plain.png",
            "fig27_augmentation_regression_augmented.png",
        ]

    def test_the_default_is_still_fig_15(self, tmp_path: Path):
        written = figure_representation_hero(
            make_representation(), RunPaths.from_root(tmp_path), formats=("png",)
        )
        assert all(path.name.startswith("fig15_") for path in written)
