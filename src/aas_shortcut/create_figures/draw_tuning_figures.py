"""Fig. 5: what the tuning did."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aas_shortcut.create_figures.define_figure_vocabulary import (
    LOGGER,
    _save,
)
from aas_shortcut.define_algorithm_portfolio import (
    FOCUS_PAIRS,
)
from aas_shortcut.define_figure_theme import (
    DEFAULT_FORMATS,
    FigureStyle,
    apply_theme,
    bound_spine,
    default_style,
    hairline_grid,
    strip_axes,
)
from aas_shortcut.define_paths import (
    RunPaths,
)


def tuning_styles(style: FigureStyle) -> dict[str, dict[str, Any]]:
    """Return a pen per tuning condition; the dash keeps them apart in greyscale."""
    return {
        "before": {
            "color": style.hue["baseline"],
            "label": "default",
            "dash": (0, (3.2, 1.7)),
        },
        "after": {
            "color": style.hue["value"],
            "label": "tuned",
            "dash": (0, ()),
        },
    }


# Bins over [0, 1]; with 24 instances, finer bins hold only 0, 1 or 2.
SUPERIORITY_BINS = 10


def figure_tuning_histogram(
    tuning: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = DEFAULT_FORMATS,
    style: FigureStyle | None = None,
) -> list[Path]:
    """Draw Fig. 5: each pair's superiorities before and after tuning, as histograms."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    style = style or default_style()
    apply_theme(plt, style)
    pens = tuning_styles(style)
    if tuning.empty:
        LOGGER.warning("figures: no tuning rows; skipping fig5")
        return []

    pairs = [
        pair for pair in FOCUS_PAIRS if not tuning[tuning["pair"] == pair.slug].empty
    ]
    if not pairs:
        return []

    edges = np.linspace(0.0, 1.0, SUPERIORITY_BINS + 1)
    grid = np.linspace(0.0, 1.0, 200)

    fig, axes = plt.subplots(
        1,
        len(pairs),
        figsize=(
            style.width_per_column["fig5_tuning_histogram"] * len(pairs)
            + style.size["fig5_tuning_histogram"][0],
            style.size["fig5_tuning_histogram"][1],
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    samples: dict[int, dict[str, np.ndarray]] = {}
    top = 0.0
    for index, pair in enumerate(pairs):
        panel = tuning[tuning["pair"] == pair.slug]
        drawn = {}
        for condition, mask in (
            ("before", panel["is_default"]),
            ("after", panel["is_selected"]),
        ):
            rows = panel[mask.astype(bool)]
            if not rows.empty:
                values = rows["superiority"].to_numpy(dtype=np.float64)
                drawn[condition] = values
                top = max(top, float(np.histogram(values, bins=edges)[0].max()))
        samples[index] = drawn

    top = max(top, 1.0) * 1.12

    for index, (ax, pair) in enumerate(zip(axes.ravel(), pairs, strict=True)):
        panel = tuning[tuning["pair"] == pair.slug]
        for condition, values in samples[index].items():
            pen = pens[condition]
            ax.hist(
                values,
                bins=edges,
                histtype="stepfilled",
                facecolor=pen["color"],
                edgecolor=pen["color"],
                linestyle=pen["dash"],
                linewidth=1.2,
                alpha=0.30,
                zorder=2,
            )
            if float(np.ptp(values)) > 0.0:
                density = gaussian_kde(values)(grid) * values.size / SUPERIORITY_BINS
                ax.plot(
                    grid,
                    density,
                    color=pen["color"],
                    linestyle=pen["dash"],
                    lw=style.line_width,
                    zorder=4,
                )

        # The uniform level: every bin holding the same count.
        counts = sum(values.size for values in samples[index].values()) / max(
            len(samples[index]), 1
        )
        ax.axhline(
            counts / SUPERIORITY_BINS,
            color=style.ink_muted,
            lw=style.reference_width,
            ls=(0, (1.2, 2.0)),
            zorder=3,
        )
        if index == 0:
            for offset, (_, pen) in enumerate(pens.items()):
                ax.text(
                    0.98,
                    top * (0.97 - 0.09 * offset),
                    pen["label"],
                    ha="right",
                    va="top",
                    fontsize=style.type["annotation"],
                    color=pen["color"],
                )

        ax.set_title(
            str(panel["group"].iloc[0]), fontsize=style.type["panel"], loc="left"
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, top)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel(style.label["fig5_histogram_x"])
        hairline_grid(ax, axis="y", style=style)
        strip_axes(ax, keep=("left",) if index == 0 else ())
        if index:
            ax.tick_params(axis="y", length=0)
        else:
            ticks = [tick for tick in ax.get_yticks() if 0.0 <= tick <= top]
            if ticks:
                ax.set_yticks(ticks)
                bound_spine(ax, 0.0, max(ticks))

    axes.ravel()[0].set_ylabel(style.label["fig5_histogram_y"])
    written = _save(fig, run_paths, "fig5_tuning_histogram", formats, style)
    plt.close(fig)
    return written
