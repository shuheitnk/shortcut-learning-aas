"""Fig. 16: each method's violin split into the original and transformed test blocks.

Fig. 28 is the same figure with the halves being Experiment 5's two training
conditions, pooled over the four unseen families' transformed problems.

A violin shows per-problem medians over the training runs; its rule and number are
the table's level (median over runs of the mean over problems). The two reductions
differ in order, so the rule need not sit at the violin's mean.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.create_figures.define_figure_vocabulary import (
    LOGGER,
    VARIANT_LABELS,
    VARIANT_ORDER,
    SeriesStyle,
    _restrict,
    _save,
    variant_colors,
)
from aas_shortcut.define_figure_theme import (
    FigureStyle,
    apply_theme,
    default_style,
    strip_axes,
)
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    ORACLE_VARIANT,
    RunPaths,
)

# The paper takes the SVG and the PDF; the PNG is for checking the layout.
SPLIT_STEM = "fig16_selection_split_pooled"
SPLIT_FORMATS: tuple[str, ...] = ("svg", "pdf", "png")

# Left half first: the value of ``transform`` and its label.
SPLIT_BLOCKS: tuple[tuple[str, str], ...] = (
    ("pre", "original"),
    ("post", "transformed"),
)

# Fig. 28's halves, left first: the values of ``training`` and their labels.
AUGMENTATION_SPLIT: tuple[tuple[str, str], ...] = (
    ("plain", "plain"),
    ("augmented", "augmented"),
)
AUGMENTATION_SPLIT_STEM = "fig28_augmentation_selection_split"

# Halves are paired on these, never on row order, so both sides are the same problems.
PROBLEM_KEYS: tuple[str, ...] = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
)

# Inches; wider than one ACM column so the numbers on both sides of each violin fit.
SPLIT_SIZE = (5.0, 2.34)

# X units between methods; the gap holds two methods' numbers.
SPLIT_PITCH = 1.25

# Point sizes, set for the page rather than the project's slide scale.
SPLIT_NAME_SIZE = 12.0
SPLIT_TICK_SIZE = 10.5
SPLIT_VALUE_SIZE = 8.0
SPLIT_KEY_SIZE = 8.0
SPLIT_TYPE: dict[str, float] = {
    "axis": SPLIT_TICK_SIZE,
    "tick": SPLIT_NAME_SIZE,
    "annotation": SPLIT_VALUE_SIZE,
}

SPLIT_NAME_INK = "#222222"
SPLIT_TICK_INK = "#6b6b6b"

# The scale is three hairlines; the middle one, the no-selection baseline, is darker.
SPLIT_RULE_FAINT = "#e6e6e6"
SPLIT_RULE_MID = "#cfcfcf"
SPLIT_RULE_WIDTH = 0.4
SPLIT_RULE_LEVELS: tuple[float, ...] = (0.0, 0.5, 1.0)
SPLIT_BASELINE = 0.5

# Width of a whole violin, both halves together.
SPLIT_VIOLIN_WIDTH = 0.44

# KDE grid points on [0, 1]; a coarser grid shows facets at this size.
SPLIT_KDE_POINTS = 256

# Fill opacity of the transformed and original halves. Only the transformed half is
# outlined, so the weight is on the block the figure is about.
SPLIT_ALPHA = 0.34
SPLIT_ALPHA_ORIGINAL = 0.12
SPLIT_EDGE_WIDTH = 0.6
# White gap at the split, in points, so the halves read as two measurements.
SPLIT_HALF_GAP = 0.5

# Mean rules start this far from the centre, leaving room for the slope between them.
SPLIT_MEAN_INNER = 0.08
SPLIT_MEAN_WIDTH = 1.2
SPLIT_MEAN_ALPHA_ORIGINAL = 0.5
# Drawn for every method, however small the difference.
SPLIT_SLOPE_WIDTH = 0.6

# Offset of a written mean from the method's centre, just past its rule.
SPLIT_LABEL_OFFSET = 0.24

# Headroom and height of the split key, high enough to clear the top tick label.
SPLIT_HEADROOM = 0.15
SPLIT_KEY_HEIGHT = 1.05
SPLIT_KEY_GAP = 0.035

# How far the axis reaches past the outermost violin.
SPLIT_MARGIN = 0.64


def split_style(style: FigureStyle | None = None) -> FigureStyle:
    """Return ``style`` resized for the wide slot this figure is written for."""
    style = style or default_style()
    return replace(
        style,
        type={**style.type, **SPLIT_TYPE},
        label_pad=2.5,
        tick_length=0.0,
        spine_width=SPLIT_RULE_WIDTH,
        reference_width=SPLIT_RULE_WIDTH,
        formats=SPLIT_FORMATS,
        pad_inches=0.02,
    )


def split_series(style: FigureStyle) -> tuple[SeriesStyle, ...]:
    """Return the four methods of one panel, in drawing order, each with a hue only."""
    colors = variant_colors(style)
    return tuple(
        SeriesStyle(
            key=variant,
            label=VARIANT_LABELS[variant],
            color=colors[variant],
            marker="",
        )
        for variant in (*VARIANT_ORDER, ORACLE_VARIANT)
    )


def paired_superiorities(
    rows: pd.DataFrame,
    column: str,
    key: str,
    *,
    split: str = "transform",
    blocks: Sequence[tuple[str, str]] = SPLIT_BLOCKS,
    keys: Sequence[str] = PROBLEM_KEYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one method's superiorities on the two halves, paired on ``keys``."""
    side = rows[rows[column] == key]
    present = [name for name in keys if name in side.columns]
    if not present:
        raise KeyError(f"none of {tuple(keys)} is in the selection table")
    halves = [
        pd.Series(
            pd.DataFrame(side[side[split] == value]).set_index(present)[
                "superiority_vs_sbs"
            ]
        )
        for value, _ in blocks
    ]
    shared = cast(pd.Index, halves[0].index.intersection(halves[1].index))
    first, second = (
        half.loc[sorted(shared)].to_numpy(dtype=np.float64) for half in halves
    )
    return first, second


def half_density(
    values: np.ndarray,
    *,
    width: float = SPLIT_VIOLIN_WIDTH,
    bandwidth: float | None = None,
    points: int = SPLIT_KDE_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return one block's density on [0, 1] scaled to half a violin, or ``None``."""
    from matplotlib.mlab import GaussianKDE

    if values.size < 2 or float(np.ptp(values)) <= 0.0:
        return None
    # A superiority cannot leave [0, 1], so no density is drawn outside it.
    level = np.linspace(0.0, 1.0, points)
    kernel = GaussianKDE(values, bw_method=bandwidth)
    density = np.asarray(kernel.evaluate(level), dtype=np.float64)
    peak = float(density.max())
    if peak <= 0.0:
        return None
    return level, 0.5 * width * density / peak


def _points_per_unit(ax) -> float:
    """Return the page points one x-axis unit is worth, once laid out."""
    figure = ax.figure
    figure.canvas.draw()
    (left, _), (right, _) = ax.transData.transform([(0.0, 0.0), (1.0, 0.0)])
    return float(right - left) * 72.0 / float(figure.dpi)


def _half_violin(
    ax,
    position: float,
    values: np.ndarray,
    color: str,
    *,
    name: str,
    block: str,
    left: bool,
    gap: float,
    alpha: float,
    outlined: bool,
) -> None:
    """Draw one block's density as half a violin on one side of the split.

    Built by hand rather than with ``violinplot`` so the outline can leave out the
    top, where the density is cut by the ceiling of 1 rather than by the data.
    """
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Polygon

    drawn = half_density(values)
    if drawn is None:
        return
    level, width = drawn
    side = -1.0 if left else 1.0
    inner = position + side * gap
    outer = inner + side * width

    patch = Polygon(
        np.column_stack(
            [
                np.r_[np.full_like(level, inner), outer[::-1]],
                np.r_[level, level[::-1]],
            ]
        ),
        closed=True,
        # Alpha on the fill, not the artist: matplotlib writes an artist's alpha on
        # the SVG group, where it cannot be edited apart from the outline.
        facecolor=to_rgba(color, alpha),
        edgecolor="none",
        linewidth=0.0,
        zorder=1,
    )
    # Named marks make the SVG editable by element.
    patch.set_gid(f"{block}-{name}")
    ax.add_patch(patch)
    if not outlined:
        return
    # Open at the top: the edge at 1 is the ceiling, not a boundary of the data.
    (outline,) = ax.plot(
        np.r_[outer[::-1], inner, inner],
        np.r_[level[::-1], 0.0, 1.0],
        color=color,
        lw=SPLIT_EDGE_WIDTH,
        solid_capstyle="butt",
        solid_joinstyle="round",
        zorder=2,
    )
    outline.set_gid(f"outline-{name}")


def _slope(
    ax,
    position: float,
    original: float,
    transformed: float,
    color: str,
    name: str,
    blocks: Sequence[tuple[str, str]] = SPLIT_BLOCKS,
) -> None:
    """Draw one method's two mean rules and the slope that joins them."""
    from matplotlib.colors import to_rgba

    outer = 0.5 * SPLIT_VIOLIN_WIDTH
    for (level, sign, alpha), (_, block) in zip(
        (
            (original, -1.0, SPLIT_MEAN_ALPHA_ORIGINAL),
            (transformed, 1.0, 1.0),
        ),
        blocks,
        strict=True,
    ):
        (rule,) = ax.plot(
            [position + sign * SPLIT_MEAN_INNER, position + sign * outer],
            [level, level],
            color=to_rgba(color, alpha),
            lw=SPLIT_MEAN_WIDTH,
            solid_capstyle="round",
            zorder=4,
        )
        rule.set_gid(f"mean-{block}-{name}")
    (join,) = ax.plot(
        [position - SPLIT_MEAN_INNER, position + SPLIT_MEAN_INNER],
        [original, transformed],
        color=color,
        lw=SPLIT_SLOPE_WIDTH,
        solid_capstyle="round",
        zorder=3,
    )
    join.set_gid(f"slope-{name}")


def _levels(
    ax,
    position: float,
    original: float,
    transformed: float,
    color: str,
    name: str,
    blocks: Sequence[tuple[str, str]] = SPLIT_BLOCKS,
) -> None:
    """Write each block's mean beside its own rule, on its own side of the split."""
    for level, (_, block) in zip((original, transformed), blocks, strict=True):
        left = block == blocks[0][1]
        text = ax.text(
            position - SPLIT_LABEL_OFFSET if left else position + SPLIT_LABEL_OFFSET,
            level,
            f"{level:.3f}",
            ha="right" if left else "left",
            va="center",
            fontsize=SPLIT_VALUE_SIZE,
            color=color,
            # Bold marks the transformed block.
            fontweight="normal" if left else "bold",
            zorder=5,
        )
        text.set_gid(f"level-{block}-{name}")


def figure_selection_split_pooled(
    table: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = SPLIT_FORMATS,
    style: FigureStyle | None = None,
    *,
    series: Sequence[SeriesStyle] | None = None,
    column: str = "feature_variant",
    stem: str = SPLIT_STEM,
    feature_set: str = ELA_FEATURE_SET,
    levels: Mapping[tuple[str, str], float] | None = None,
    split: str = "transform",
    blocks: Sequence[tuple[str, str]] = SPLIT_BLOCKS,
    keys: Sequence[str] = PROBLEM_KEYS,
) -> list[Path]:
    """Draw both halves' superiorities in one panel, one split violin per method.

    ``levels`` maps ``(method, block)`` to the table's level, which need not be the
    violin's mean; a method it does not name is drawn at its violin's mean.
    """
    import matplotlib.pyplot as plt

    style = split_style(style)
    apply_theme(plt, style)
    rows = _restrict(table, feature_set)
    entries = series if series is not None else split_series(style)
    drawn = [entry for entry in entries if not rows[rows[column] == entry.key].empty]
    if not drawn:
        LOGGER.warning("figures: no drawable series; skipping %s", stem)
        return []

    fig, ax = plt.subplots(figsize=SPLIT_SIZE, layout="constrained")
    for level in SPLIT_RULE_LEVELS:
        rule = ax.axhline(
            level,
            color=SPLIT_RULE_MID if level == SPLIT_BASELINE else SPLIT_RULE_FAINT,
            lw=SPLIT_RULE_WIDTH,
            zorder=0,
        )
        rule.set_gid(f"rule-{level:g}")

    ax.set_xticks([index * SPLIT_PITCH for index in range(len(drawn))])
    ax.set_xticklabels([entry.label for entry in drawn])
    ax.set_xlim(-SPLIT_MARGIN, (len(drawn) - 1) * SPLIT_PITCH + SPLIT_MARGIN)
    ax.set_ylim(0.0, 1.0 + SPLIT_HEADROOM)
    ax.set_yticks(list(SPLIT_RULE_LEVELS))
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.tick_params(axis="x", length=0, pad=style.label_pad + 2.0)
    ax.tick_params(axis="y", length=0)
    for label in ax.get_xticklabels():
        label.set_fontsize(SPLIT_NAME_SIZE)
        label.set_color(SPLIT_NAME_INK)
    for label in ax.get_yticklabels():
        label.set_fontsize(SPLIT_TICK_SIZE)
        label.set_color(SPLIT_TICK_INK)
    strip_axes(ax, keep=())
    ax.set_ylabel(style.label["fig4_y"], fontsize=SPLIT_TICK_SIZE, color=SPLIT_TICK_INK)

    gap = 0.5 * SPLIT_HALF_GAP / _points_per_unit(ax)
    for index, entry in enumerate(drawn):
        position = index * SPLIT_PITCH
        halves = paired_superiorities(
            rows, column, entry.key, split=split, blocks=blocks, keys=keys
        )
        for index, values in enumerate(halves):
            left = index == 0
            _half_violin(
                ax,
                position,
                values,
                entry.color,
                name=entry.key,
                block=blocks[index][1],
                left=left,
                gap=gap,
                alpha=SPLIT_ALPHA_ORIGINAL if left else SPLIT_ALPHA,
                outlined=not left,
            )
        if not all(values.size for values in halves):
            continue
        original, transformed = (
            (levels or {}).get((entry.key, value), float(np.mean(values)))
            for (value, _), values in zip(blocks, halves, strict=True)
        )
        _slope(ax, position, original, transformed, entry.color, entry.key, blocks)
        _levels(ax, position, original, transformed, entry.color, entry.key, blocks)

    # The key names the halves once, over the first violin's split.
    for (_, block), side in zip(blocks, ("right", "left"), strict=True):
        ax.text(
            -SPLIT_KEY_GAP if side == "right" else SPLIT_KEY_GAP,
            SPLIT_KEY_HEIGHT,
            block,
            ha=side,
            va="bottom",
            fontsize=SPLIT_KEY_SIZE,
            color=SPLIT_TICK_INK,
        )
    (divider,) = ax.plot(
        [0.0, 0.0],
        [1.0, SPLIT_KEY_HEIGHT + SPLIT_KEY_GAP],
        color=SPLIT_RULE_MID,
        lw=SPLIT_RULE_WIDTH,
        solid_capstyle="butt",
        zorder=3,
    )
    divider.set_gid("key-split")

    written = _save(fig, run_paths, stem, formats, style)
    plt.close(fig)
    return written
