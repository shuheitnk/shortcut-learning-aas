"""The appendix's figure: LOFO against the RRT, one panel per group.

One file per feature set, on shared axis ranges. Marker: representation; fill:
training condition (filled is augmented). Bars are the seed quartiles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from aas_shortcut.create_figures.define_figure_vocabulary import (
    FEATURE_SET_LABELS,
    VARIANT_LABELS,
    VARIANT_ORDER,
    _save,
    feature_set_stem,
    variant_colors,
)
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_figure_theme import (
    DEFAULT_FORMATS,
    FigureStyle,
    apply_theme,
    default_style,
    hairline_grid,
    strip_axes,
)
from aas_shortcut.define_paths import RunPaths

STEM = "appendix_lofo_against_rrt"
SIZE = (7.0, 2.6)
MARKERS = {"value": "o", "rank": "s", "concat": "^"}
MARKER_SIZE = 5.5
FILLED = "augmented"
# Font sizes for a page-width figure.
TYPE = {"panel": 9.5, "axis": 9.5, "legend": 8.5, "tick": 8.0}
GRID = "#ECECEC"
# Share of the data range added on each side of the shared axes.
MARGIN = 0.05

Limits = tuple[tuple[float, float], tuple[float, float]]


def lofo_style(style: FigureStyle | None = None) -> FigureStyle:
    """Return ``style`` sized and weighted for this page-width figure."""
    style = style or default_style()
    return replace(
        style,
        type={**style.type, **TYPE},
        label_pad=3.0,
        tick_length=3.0,
        spine_width=0.8,
        grid=GRID,
        grid_width=0.6,
        rule=style.ink_muted,
        pad_inches=0.02,
    )


def _signed(value: float) -> str:
    """``value`` to two decimals with a typographic minus."""
    return f"{value:.2f}".replace("-", "\u2212")


def shared_limits(levels: Sequence[pd.DataFrame]) -> Limits:
    """Axis ranges that hold every point and bar of every feature set's levels."""
    table = pd.concat(levels, ignore_index=True)

    def padded(low: float, high: float) -> tuple[float, float]:
        pad = MARGIN * (high - low)
        return low - pad, high + pad

    def column(name: str) -> np.ndarray:
        return table[name].to_numpy(dtype=np.float64)

    return (
        padded(float(column("rrt_q1").min()), float(column("rrt_q3").max())),
        padded(float(column("lofo_q1").min()), float(column("lofo_q3").max())),
    )


def figure_lofo_against_rrt(
    levels: pd.DataFrame,
    correlations: pd.DataFrame,
    feature_set: str,
    limits: Limits,
    run_paths: RunPaths,
    formats: Sequence[str] = DEFAULT_FORMATS,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Draw one feature set's three panels and return the paths written."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = lofo_style(style)
    apply_theme(plt, style)
    colors = variant_colors(style)
    rho: Mapping[str, float] = correlations.set_index("group")["rho"].to_dict()

    fig, axes = plt.subplots(1, len(GROUPS), figsize=SIZE, layout="constrained")
    for ax, group in zip(axes, GROUPS, strict=True):
        rows = levels[levels["group"] == group]
        for row in rows.itertuples(index=False):
            variant = str(row.feature_variant)
            color = colors[variant]
            ax.errorbar(
                row.rrt_median,
                row.lofo_median,
                xerr=[[row.rrt_median - row.rrt_q1], [row.rrt_q3 - row.rrt_median]],
                yerr=[
                    [row.lofo_median - row.lofo_q1],
                    [row.lofo_q3 - row.lofo_median],
                ],
                marker=MARKERS[variant],
                markersize=MARKER_SIZE,
                markerfacecolor=color if row.training == FILLED else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                linestyle="none",
                zorder=3,
            )
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
        strip_axes(ax)
        hairline_grid(ax, "both", style)
        ax.set_title(rf"{group}  ($\rho$ = {_signed(rho[group])})")
    fig.supxlabel("RRT R² (unseen families)", fontsize=style.type["axis"])
    axes[0].set_ylabel(f"LOFO R² ({FEATURE_SET_LABELS[feature_set]})")

    handles = [
        Line2D(
            [],
            [],
            marker=MARKERS[variant],
            color=colors[variant],
            linestyle="none",
            markersize=MARKER_SIZE,
            label=VARIANT_LABELS[variant],
        )
        for variant in VARIANT_ORDER
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=MARKER_SIZE,
            markerfacecolor=fill,
            markeredgecolor=style.ink_muted,
            label=label,
        )
        for fill, label in ((style.ink_muted, "augmented"), ("white", "plain"))
    ]
    fig.legend(
        handles=handles,
        loc="outside upper center",
        ncol=len(handles),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    written = _save(fig, run_paths, feature_set_stem(STEM, feature_set), formats, style)
    plt.close(fig)
    return written
