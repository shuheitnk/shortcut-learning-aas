"""Bar figures of R^2 sized for a projector as well as for the page.

Experiment 1: G1 (whose superiority is invariant by construction) before and after
the transformation, all versus invariant predictors. Experiment 2 and 4: the three
representations over the three groups.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aas_shortcut.define_figure_theme import FigureStyle, default_style, save_figure
from aas_shortcut.define_paths import RunPaths

LOGGER = logging.getLogger(__name__)


# G1: both members are rank-based, so the transformation cannot move its superiority.
HERO_PAIR = "bipop-cmaes__adaptive-nelder-mead"

HERO_FORMATS: tuple[str, ...] = ("png", "svg")

# Predictor set, label and hue: "all" takes the value-based hue, "invariant" the
# rank-based one.
SERIES: tuple[tuple[str, str, str], ...] = (
    ("all", "all", "#D55E00"),
    ("invariant", "invariant", "#0072B2"),
)

BLOCKS: tuple[tuple[str, str], ...] = (
    ("r2_test_pre", "original"),
    ("r2_test_post", "transformed"),
)

HERO_STEM = "fig14_shortcut_hero"

# Canvas, in inches.
FIGURE_SIZE = (8.4, 5.6)

# Type sizes, larger than the paper's scale so the figure also reads on a projector.
BLOCK_SIZE = 26.0
NAME_SIZE = 20.0
VALUE_SIZE = 24.0
AXIS_SIZE = 24.0

MINUS = "\u2212"

# R^2 of a model that predicts the training mean and of a perfect one, ruled in ink.
REFERENCES: tuple[float, ...] = (0.0, 1.0)

# Room above the upper reference, so it reads as a line inside the frame.
HEADROOM = 0.06

BAR_WIDTH = 0.74
BLOCK_GAP = 1.0


def hero_scores(report: pd.DataFrame, pair: str = HERO_PAIR) -> pd.DataFrame:
    """Return one row per predictor set with its score on each block.

    A missing cell raises, since a bar left out would read as a score of zero.
    """
    rows = report[report["pair"] == pair]
    if rows.empty:
        held = sorted(str(value) for value in report["pair"].unique())
        raise ValueError(f"the report holds no rows for {pair!r}; it covers {held}")

    table = rows.pivot_table(
        index="predictors", columns="metric", values="value", aggfunc="mean"
    )
    missing = [
        f"{name}/{metric}"
        for name, _, _ in SERIES
        for metric, _ in BLOCKS
        if name not in table.index or metric not in table.columns
    ]
    if missing:
        raise ValueError(f"the report is missing {len(missing)} cell(s): {missing}")
    return table


def _layout() -> tuple[list[float], list[float]]:
    """Where the bars sit, and where each block's tick belongs."""
    positions = [
        block * (len(SERIES) + BLOCK_GAP) + slot
        for block in range(len(BLOCKS))
        for slot in range(len(SERIES))
    ]
    centres = [
        block * (len(SERIES) + BLOCK_GAP) + (len(SERIES) - 1) / 2.0
        for block in range(len(BLOCKS))
    ]
    return positions, centres


def _draw(table: pd.DataFrame, style: FigureStyle):
    """Draw both blocks."""
    import matplotlib.pyplot as plt

    from aas_shortcut.define_figure_theme import apply_theme

    apply_theme(plt, style)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    values = [
        float(table.loc[name, metric]) for metric, _ in BLOCKS for name, _, _ in SERIES
    ]
    hues = [hue for _ in BLOCKS for _, _, hue in SERIES]
    names = [name for _ in BLOCKS for _, name, _ in SERIES]
    positions, centres = _layout()

    floor = _floor(min(values))
    ceiling = max(max(REFERENCES), max(values) + 0.08)
    span = ceiling - floor

    ax.bar(
        positions,
        values,
        width=BAR_WIDTH,
        color=[style.tint(hue) for hue in hues],
        edgecolor=hues,
        linewidth=1.8,
        zorder=3,
    )
    ax.set_ylim(floor, ceiling + HEADROOM)
    ax.set_xlim(positions[0] - 1.0, positions[-1] + 1.0)
    ax.set_xticks(centres)
    ax.set_xticklabels([label for _, label in BLOCKS])
    ax.set_xticks(positions, minor=True)
    ax.set_xticklabels(names, minor=True)
    ax.tick_params(axis="x", which="minor", labelsize=NAME_SIZE, length=0, pad=6)
    ax.set_ylabel("$R^2$", fontsize=AXIS_SIZE, labelpad=10)
    ax.tick_params(axis="x", labelsize=BLOCK_SIZE, length=0, pad=38)
    ax.tick_params(axis="y", labelsize=AXIS_SIZE * 0.8, length=0)
    ax.yaxis.grid(True, color=style.grid, linewidth=1.1)
    for reference in REFERENCES:
        ax.axhline(reference, color=style.ink, linewidth=1.4, zorder=4)
    ax.set_axisbelow(True)

    for position, value, hue in zip(positions, values, hues, strict=True):
        gap = 0.02 * span
        text = f"{value:.2f}".replace("-", MINUS)
        above = value >= 0.0
        ax.text(
            position,
            value + gap if above else value - gap,
            text,
            fontsize=VALUE_SIZE,
            color=hue,
            va="bottom" if above else "top",
            ha="center",
            fontweight="bold",
            zorder=5,
        )

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def figure_shortcut_hero(
    report: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = HERO_FORMATS,
    style: FigureStyle | None = None,
    pair: str = HERO_PAIR,
) -> list[Path]:
    """Draw G1's two blocks and write the figure."""
    style = style or default_style()
    table = hero_scores(report, pair)
    run_paths.figures_dir.mkdir(parents=True, exist_ok=True)

    fig = _draw(table, style)
    written = [
        save_figure(fig, run_paths.figure(HERO_STEM, extension), style)
        for extension in formats
    ]
    LOGGER.info("figures: wrote %s", ", ".join(str(path) for path in written))
    return written


# ---------------------------------------------------------------------------
# Experiment 2, v2: the three representations over the three groups
# ---------------------------------------------------------------------------

# Representation, label and hue (the same hues as in every other figure).
REPRESENTATIONS: tuple[tuple[str, str, str], ...] = (
    ("value", "value-based", "#D55E00"),
    ("rank", "rank-based", "#0072B2"),
    ("concat", "concat", "#009E73"),
)

# The report's `metric` value, the condition's label and the file stem; one figure
# per block.
REPRESENTATION_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("r2_test_pre", "original", "fig15_representation_original_v2"),
    ("r2_test_post", "transformed", "fig15_representation_transformed_v2"),
)

# Experiment 4's two training conditions on the four unseen families, in the same
# design. Here the report's `metric` column carries the training condition.
AUGMENTATION_BLOCKS: tuple[tuple[str, str, str], ...] = (
    ("plain", "without augmentation", "fig27_augmentation_regression_plain"),
    ("augmented", "with augmentation", "fig27_augmentation_regression_augmented"),
)

# The value range shared by Experiments 2 and 4, so a bar's height means the same in
# both. Fixed here so neither experiment reads the other's results; the floor is
# :func:`_floor` of the lowest score of both. A score outside it widens it.
REPRESENTATION_LIMITS: tuple[float, float] = (-1.08, 1.0)

REPRESENTATION_SIZE = (11.0, 6.0)

# Cut of a line under a score, in points: the gap either side of the score, and the
# reach past the line's thickness so its anti-aliased edge goes too.
VALUE_KNOCKOUT_PAD = 1.0
KNOCKOUT_BLEED = 0.25


def representation_scores(report: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one row per group with each representation's score on one block.

    A missing representation raises, since a bar left out would read as zero.
    """
    rows = report[report["metric"] == metric]
    if rows.empty:
        held = sorted(str(value) for value in report["metric"].unique())
        raise ValueError(f"the report holds no {metric!r} row; it covers {held}")

    table = rows.pivot_table(
        index="group", columns="feature_variant", values="value", aggfunc="mean"
    )
    missing = [name for name, _, _ in REPRESENTATIONS if name not in table.columns]
    if missing:
        raise ValueError(f"the report is missing representation(s) {missing}")
    return table.sort_index()


def _floor(lowest: float) -> float:
    """Return where the value axis starts: zero unless a score is negative."""
    return 0.0 if lowest >= 0.0 else lowest - 0.08


def representation_limits(
    report: pd.DataFrame,
    blocks: Sequence[tuple[str, str, str]] = REPRESENTATION_BLOCKS,
) -> tuple[float, float]:
    """Return :data:`REPRESENTATION_LIMITS`, widened if a score falls outside it.

    One range for every copy, since these figures are read against each other.
    """
    wanted = [metric for metric, _, _ in blocks]
    values = cast(pd.Series, report[report["metric"].isin(wanted)]["value"]).astype(
        float
    )
    lowest, highest = float(cast(float, values.min())), float(cast(float, values.max()))
    floor, ceiling = REPRESENTATION_LIMITS
    return min(floor, _floor(lowest)), max(ceiling, highest + 0.08)


def _draw_representation(
    table: pd.DataFrame, style: FigureStyle, limits: tuple[float, float]
):
    """Draw the three representations of every group, in one block."""
    import matplotlib.pyplot as plt

    from aas_shortcut.define_figure_theme import apply_theme

    apply_theme(plt, style)
    fig, ax = plt.subplots(figsize=REPRESENTATION_SIZE)

    groups = list(table.index)
    width = len(REPRESENTATIONS)
    positions = [
        group * (width + BLOCK_GAP) + slot
        for group in range(len(groups))
        for slot in range(width)
    ]
    centres = [
        group * (width + BLOCK_GAP) + (width - 1) / 2.0 for group in range(len(groups))
    ]
    values = [
        float(table.loc[group, name])
        for group in groups
        for name, _, _ in REPRESENTATIONS
    ]
    hues = [hue for _ in groups for _, _, hue in REPRESENTATIONS]

    floor, ceiling = limits
    span = ceiling - floor

    ax.bar(
        positions,
        values,
        width=BAR_WIDTH,
        color=[style.tint(hue) for hue in hues],
        edgecolor=hues,
        linewidth=1.8,
        zorder=3,
        label="_nolegend_",
    )
    ax.set_ylim(floor, ceiling + HEADROOM)
    ax.set_xlim(positions[0] - 1.0, positions[-1] + 1.0)
    ax.set_xticks(centres)
    ax.set_xticklabels(groups)
    ax.set_ylabel("$R^2$", fontsize=AXIS_SIZE, labelpad=10)
    ax.tick_params(axis="x", labelsize=BLOCK_SIZE, length=0, pad=14)
    ax.tick_params(axis="y", labelsize=AXIS_SIZE * 0.8, length=0)
    ax.yaxis.grid(True, color=style.grid, linewidth=1.1)
    for reference in REFERENCES:
        ax.axhline(reference, color=style.ink, linewidth=1.4, zorder=4)
    ax.set_axisbelow(True)

    scores = []
    for position, value, hue in zip(positions, values, hues, strict=True):
        # Outside the bar: above a negative bar it would sit on the zero rule.
        above = value >= 0.0
        gap = 0.02 * span
        scores.append(
            ax.text(
                position,
                value + gap if above else value - gap,
                f"{value:.2f}".replace("-", MINUS),
                fontsize=VALUE_SIZE * 0.86,
                color=hue,
                va="bottom" if above else "top",
                ha="center",
                fontweight="bold",
                zorder=5,
            )
        )

    # Legend above the frame rather than a name under every bar.
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=style.tint(hue), edgecolor=hue, linewidth=1.8
        )
        for _, _, hue in REPRESENTATIONS
    ]
    ax.legend(
        handles,
        [label for _, label, _ in REPRESENTATIONS],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(REPRESENTATIONS),
        frameon=False,
        fontsize=NAME_SIZE,
        handlelength=1.4,
        handleheight=1.2,
        columnspacing=2.0,
    )

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    _cut_lines_under_scores(ax, scores, style)
    return fig


def _cut_lines_under_scores(ax, scores: Sequence[Any], style: FigureStyle) -> None:
    """Cut every horizontal line where it runs through a score.

    The cut spans the line's whole thickness (a partial cut reads as a dent). It is
    measured on the final layout, so it must be called last.
    """
    from matplotlib.patches import Rectangle

    figure = ax.figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    frame = ax.get_window_extent(renderer)
    (left, right), (low, high) = ax.get_xlim(), ax.get_ylim()
    pixels_per_point = figure.dpi / 72.0
    x_per_point = (right - left) / frame.width * pixels_per_point
    y_per_point = (high - low) / frame.height * pixels_per_point
    # Every horizontal line with its zorder: references over the bars, grid under.
    lines = [
        (float(line.get_ydata()[0]), line.get_linewidth(), line.get_zorder())
        for line in ax.lines
    ]
    lines += [
        (float(tick.get_loc()), tick.gridline.get_linewidth(), ax.yaxis.get_zorder())
        for tick in ax.yaxis.get_major_ticks()
        if tick.gridline.get_visible() and low <= tick.get_loc() <= high
    ]
    to_data = ax.transData.inverted()
    pad = VALUE_KNOCKOUT_PAD * x_per_point
    for score in scores:
        (x0, y0), (x1, y1) = to_data.transform(
            score.get_window_extent(renderer).get_points()
        )
        for level, width, zorder in lines:
            half = (0.5 * width + KNOCKOUT_BLEED) * y_per_point
            if not (y0 < level + half and level - half < y1):
                continue
            # Over the line, under the score.
            cut = Rectangle(
                (x0 - pad, level - half),
                x1 - x0 + 2.0 * pad,
                2.0 * half,
                facecolor=style.paper,
                edgecolor="none",
                linewidth=0.0,
                zorder=zorder + 0.5,
            )
            cut.set_gid(f"cut-{level:g}")
            ax.add_patch(cut)


def figure_representation_hero(
    report: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = HERO_FORMATS,
    style: FigureStyle | None = None,
    blocks: Sequence[tuple[str, str, str]] = REPRESENTATION_BLOCKS,
) -> list[Path]:
    """Draw two conditions (test blocks or training conditions), one figure each."""
    from aas_shortcut.create_figures.define_figure_vocabulary import feature_set_stem

    style = style or default_style()
    run_paths.figures_dir.mkdir(parents=True, exist_ok=True)

    limits = representation_limits(report, blocks)
    written: list[Path] = []
    for feature_set in sorted(report["feature_set"].unique()):
        rows = cast(pd.DataFrame, report[report["feature_set"] == feature_set])
        for metric, _, stem in blocks:
            fig = _draw_representation(
                representation_scores(rows, metric), style, limits
            )
            name = feature_set_stem(stem, str(feature_set))
            written.extend(
                save_figure(fig, run_paths.figure(name, extension), style)
                for extension in formats
            )
    LOGGER.info("figures: wrote %s", ", ".join(str(path) for path in written))
    return written
