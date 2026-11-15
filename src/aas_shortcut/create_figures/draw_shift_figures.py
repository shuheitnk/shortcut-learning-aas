"""Fig. 13: class-banded bars of what a transformation moves."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from aas_shortcut.calculate_delta_z import PRIMARY_STANDARDIZATION
from aas_shortcut.create_figures.define_figure_vocabulary import (
    FIGURE_SUFFIXES,
    LOGGER,
    _save,
    variant_colors,
)
from aas_shortcut.define_algorithm_portfolio import (
    ALGORITHMS,
    GROUP_RANK_SEARCH,
    GROUP_RANK_SELECTION,
    GROUP_VALUE_SEARCH,
    GROUPS,
    PAIRS,
)
from aas_shortcut.define_figure_theme import (
    FigureStyle,
    apply_theme,
    bound_spine,
    default_style,
    hairline_grid,
    strip_axes,
)
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    FEATURE_QUANTITY,
    RunPaths,
)
from aas_shortcut.order_feature_labels import (
    ClassSpan,
    class_label,
    class_spans,
    leaf_label,
    order_features,
)

# ---------------------------------------------------------------------------
# Feature shift under the transformation
# ---------------------------------------------------------------------------

# Blank axis positions inserted between two feature classes.
CLASS_GAP = 1.6
# Heights of the class band, as multiples of the top tick.
CLASS_RULE = 1.06
CLASS_LABEL = 1.085
CLASS_TOP = 1.24


def _positions(spans: Sequence[ClassSpan], total: int) -> np.ndarray:
    """Return one x position per bar, with a gap between classes."""
    if not spans:
        return np.arange(total, dtype=np.float64)

    positions = np.zeros(total, dtype=np.float64)
    cursor = 0.0
    for index, span in enumerate(spans):
        if index:
            cursor += CLASS_GAP
        positions[span.start : span.stop] = cursor + np.arange(span.size)
        cursor += span.size
    return positions


def _ticks_to(high: int) -> list[float]:
    """Return evenly spaced ticks from zero to ``high`` inclusive."""
    for count in (4, 5, 3, 6, 2):
        if high % count == 0:
            return [high * index / count for index in range(count + 1)]
    # A prime above six: unit steps.
    return [float(index) for index in range(high + 1)]


# Mantissas a sub-unit axis may end on, with interval counts that keep steps round.
SUBUNIT_TOPS: tuple[tuple[float, int], ...] = (
    (1.0, 4),
    (2.0, 4),
    (2.5, 5),
    (5.0, 5),
    (10.0, 4),
)


def _ticks_for(maximum: float) -> list[float]:
    """Return evenly spaced ticks from zero to a round number above ``maximum``."""
    if maximum <= 0.0:
        return [0.0, 1.0]
    if maximum >= 1.0:
        return _ticks_to(math.ceil(maximum))
    decade = 10.0 ** math.floor(math.log10(maximum))
    for mantissa, count in SUBUNIT_TOPS:
        top = mantissa * decade
        # Keeps a maximum that is already round from being pushed to the next one.
        if top >= maximum - 1e-12 * decade:
            return [top * index / count for index in range(count + 1)]
    return [0.0, 1.0]


@dataclass(frozen=True)
class BarPanel:
    """One class-banded bar chart."""

    labels: Sequence[str]
    heights: np.ndarray
    spans: Sequence[ClassSpan]
    colors: Any
    size_key: str
    width_key: str
    y_label: str
    rotation: float = 90.0
    tick_size: str = "micro"


def _draw_class_bars(panel: BarPanel, style: FigureStyle):
    """Draw one class-banded bar chart and return its figure."""
    import matplotlib.pyplot as plt

    positions = _positions(panel.spans, len(panel.labels))
    fixed, height = style.size[panel.size_key]
    per_column = style.width_per_column[panel.width_key]
    # Wide enough that each rotated tick label gets its own column.
    width = max(fixed, per_column * float(positions[-1] + 1) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    ax.bar(positions, panel.heights, width=0.72, color=panel.colors, zorder=3)

    # The class band sits inside the axes; outside, the layout engine would clip it.
    banded = bool(panel.spans)
    reach = float(panel.heights.max())
    ticks = _ticks_for(reach)
    top = ticks[-1]
    ax.set_ylim(0.0, top * (CLASS_TOP if banded else 1.04))
    ax.set_yticks(ticks)

    rule_y = top * CLASS_RULE
    for index, span in enumerate(panel.spans):
        left = float(positions[span.start])
        right = float(positions[span.stop - 1])
        if index:
            ax.axvline(
                (float(positions[span.start - 1]) + left) / 2.0,
                color=style.rule,
                lw=0.5,
                ls=(0, (1, 2.5)),
                zorder=1,
            )
        ax.plot(
            [left - 0.4, right + 0.4],
            [rule_y, rule_y],
            color=style.ink_faint,
            lw=0.7,
            solid_capstyle="butt",
            zorder=3,
        )
        ax.text(
            (left + right) / 2.0,
            top * CLASS_LABEL,
            class_label(span.name),
            ha="center",
            va="bottom",
            fontsize=style.type["group"],
            color=style.ink,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        panel.labels, rotation=panel.rotation, fontsize=style.type[panel.tick_size]
    )
    # Fade labels of quantities that did not move, so invariant blocks read as blocks.
    for label, value in zip(ax.get_xticklabels(), panel.heights, strict=True):
        label.set_color(style.ink if value > 0.0 else style.ink_faint)
    ax.set_xlim(-0.9, float(positions[-1]) + 0.9)
    ax.set_ylabel(panel.y_label)
    ax.tick_params(axis="x", length=0)
    hairline_grid(ax, axis="y", style=style)
    strip_axes(ax, keep=("left",))
    bound_spine(ax, 0.0, max(ticks))
    return fig


# ---------------------------------------------------------------------------
# Fig. 13 - distribution shift under a monotone warp (the preanalysis)
# ---------------------------------------------------------------------------

# No PDF: the paper takes the SVG.
SHIFT_FORMATS: tuple[str, ...] = ("png", "svg")

# Filename suffix of the superiority figure.
ALGORITHM_SUFFIX = "_algorithm"


def group_colors(style: FigureStyle) -> dict[str, str]:
    """Return the hue per group bar of the superiority figure."""
    hue = variant_colors(style)["value"]
    return {
        GROUP_RANK_SEARCH: hue,
        GROUP_RANK_SELECTION: hue,
        GROUP_VALUE_SEARCH: hue,
    }


def _shift_rows(
    summary: pd.DataFrame,
    quantity_kind: str,
    statistic: str,
    standardization: str,
) -> pd.DataFrame:
    """Return the rows of one reading of the shift summary."""
    return summary[
        (summary["quantity_kind"] == quantity_kind)
        & (summary["statistic"] == statistic)
        & (summary["standardization"] == standardization)
    ]


def _feature_panel(rows: pd.DataFrame, style: FigureStyle, variant: str) -> BarPanel:
    """Build the panel of a feature figure, in extraction order."""
    scores = dict(zip(rows["quantity"].astype(str), rows["value"], strict=True))
    names = order_features(scores)
    return BarPanel(
        labels=[leaf_label(name) for name in names],
        heights=np.array([float(scores[name]) for name in names], dtype=np.float64),
        # Drawn even for one class, so the side-by-side figures look alike.
        spans=class_spans(names),
        colors=variant_colors(style)[variant],
        size_key="fig13_distribution_shift",
        width_key="fig13_distribution_shift",
        y_label=style.label["fig13_y"],
    )


# Band label of the superiority figure, in the slot of the feature class.
ALGORITHM_QUANTITY = "probability of superiority"


def _algorithm_panel(rows: pd.DataFrame, style: FigureStyle) -> BarPanel:
    """Build the panel of the superiority figure, in portfolio order."""
    pairs = {pair.slug: pair for pair in PAIRS}
    scores = dict(zip(rows["quantity"].astype(str), rows["value"], strict=True))
    classes = dict(
        zip(rows["quantity"].astype(str), rows["quantity_class"], strict=True)
    )

    def order(slug: str) -> tuple[int, int, str]:
        group = str(classes[slug])
        pair = pairs.get(slug)
        member = ALGORITHMS.index(pair.first) if pair else len(ALGORITHMS)
        return (GROUPS.index(group) if group in GROUPS else len(GROUPS), member, slug)

    slugs = sorted(scores, key=order)
    hues = group_colors(style)
    return BarPanel(
        labels=[str(classes[slug]) for slug in slugs],
        heights=np.array([float(scores[slug]) for slug in slugs], dtype=np.float64),
        spans=[ClassSpan(ALGORITHM_QUANTITY, 0, len(slugs))],
        colors=[hues.get(str(classes[slug]), style.ink_muted) for slug in slugs],
        size_key="fig13_algorithm",
        width_key="fig13_algorithm",
        y_label=style.label["fig13_y"],
        rotation=0.0,
        tick_size="axis",
    )


def figure_distribution_shift(
    summary: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = SHIFT_FORMATS,
    *,
    quantity_kind: str = FEATURE_QUANTITY,
    feature_set: str = ELA_FEATURE_SET,
    variant: str = "value",
    statistic: str = "mean",
    standardization: str = PRIMARY_STANDARDIZATION,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Draw Fig. 13: the mean standardized shift per quantity."""
    import matplotlib.pyplot as plt

    style = style or default_style()
    apply_theme(plt, style)
    rows = _shift_rows(summary, quantity_kind, statistic, standardization)
    if quantity_kind == FEATURE_QUANTITY:
        rows = cast(
            pd.DataFrame,
            rows[
                (rows["feature_set"] == feature_set)
                & (rows["feature_variant"] == variant)
            ],
        )
    if rows.empty:
        LOGGER.warning(
            "figures: no shift rows for kind=%s set=%s variant=%s; skipping",
            quantity_kind,
            feature_set,
            variant,
        )
        return []

    # An undefined shift (no spread before) is left out and logged: a zero bar would
    # claim it did not move.
    undefined = rows[rows["value"].isna()]
    if not undefined.empty:
        LOGGER.warning(
            "figures: %d quantity/quantities have no standardized shift and are "
            "not drawn: %s",
            len(undefined),
            ", ".join(sorted(undefined["quantity"].astype(str))),
        )
        rows = cast(pd.DataFrame, rows[rows["value"].notna()])
        if rows.empty:
            return []

    feature_side = quantity_kind == FEATURE_QUANTITY
    panel = (
        _feature_panel(rows, style, variant)
        if feature_side
        else _algorithm_panel(rows, style)
    )
    fig = _draw_class_bars(panel, style)

    # The caption's numbers, logged rather than drawn.
    moved = int((panel.heights > 0.0).sum())
    LOGGER.info(
        "figures: %s shift, %d of %d quantities move, max %.3f sd",
        feature_set if feature_side else "superiority",
        moved,
        len(panel.heights),
        float(panel.heights.max()),
    )
    suffix = (
        FIGURE_SUFFIXES.get(feature_set, f"_{feature_set}")
        if feature_side
        else ALGORITHM_SUFFIX
    )
    written = _save(fig, run_paths, f"fig13_distribution_shift{suffix}", formats, style)
    plt.close(fig)
    return written
