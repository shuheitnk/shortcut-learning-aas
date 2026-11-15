"""Appendix B's figure: the strictly increasing maps the transformed view applies."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from aas_shortcut.create_figures.define_figure_vocabulary import (
    LOGGER,
    _save,
    variant_colors,
)
from aas_shortcut.define_figure_theme import (
    FigureStyle,
    apply_theme,
    default_style,
    hairline_grid,
    strip_axes,
)
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.sample_monotone_transform import WarpConfig, transform_for_instance

STEM = "fig18_monotone_warp"

# Part of each map's key, so the curves are the maps the experiments apply, not
# maps sampled for the picture. Must equal ``seed`` in configs/config.yaml.
WARP_SEED = 42
WARP_PROBLEM_CLASS = "bbob"

# The test instances' maps (spline/warpheld), as Section 4.2 defines them. The
# training augmentation's domain draws from the same distribution, but instance 1
# is a test instance, so only this domain gives maps a run actually applied.
WARP_CONFIG: WarpConfig = CONTROL.warp

# One map per BBOB function, on its first instance at the study's dimension.
WARP_FUNCTIONS: tuple[int, ...] = tuple(range(1, 25))
WARP_INSTANCE = 1
WARP_DIM = 10

# Opacity of one draw. Twenty-four curves in one hue should read as a band, which
# is what a family is, rather than as twenty-four series.
WARP_ALPHA = 0.55

# PNG to look at, SVG for the document.
WARP_FORMATS: tuple[str, ...] = ("png", "svg")


def figure_monotone_warps(
    run_paths: RunPaths,
    formats: Sequence[str] = WARP_FORMATS,
    style: FigureStyle | None = None,
    *,
    function_ids: Sequence[int] = WARP_FUNCTIONS,
    instance_id: int = WARP_INSTANCE,
    dim: int = WARP_DIM,
    base_seed: int = WARP_SEED,
    problem_class: str = WARP_PROBLEM_CLASS,
    config: WarpConfig = WARP_CONFIG,
    stem: str = STEM,
) -> list[Path]:
    """Draw one map per function against the identity and write the figure."""
    import matplotlib.pyplot as plt

    style = style or default_style()
    apply_theme(plt, style)
    # The maps act on values, so they take the value-based representation's hue.
    hue = variant_colors(style)["value"]

    warps = [
        transform_for_instance(
            base_seed, problem_class, function_id, instance_id, dim, config
        )
        for function_id in function_ids
    ]

    fig, ax = plt.subplots(figsize=style.size[stem], layout="constrained")
    # The identity first and underneath: it is the reference the curves are read
    # against, not one of them.
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        color=style.ink_muted,
        lw=style.reference_width,
        ls=(0, (4.0, 2.4)),
        zorder=1,
    )
    # `values` is the map's own array on its 2,001-point grid, not a pandas accessor.
    curves = [(np.asarray(warp.grid), np.asarray(warp.values)) for warp in warps]
    for inputs, outputs in curves:
        ax.plot(
            inputs,
            outputs,
            color=hue,
            lw=style.line_width * 0.62,
            alpha=WARP_ALPHA,
            solid_capstyle="round",
            zorder=2,
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel(style.label["fig18_x"])
    ax.set_ylabel(style.label["fig18_y"])
    hairline_grid(ax, axis="both", style=style)
    strip_axes(ax)

    deviations = [float(np.max(np.abs(outputs - inputs))) for inputs, outputs in curves]
    LOGGER.info(
        "figures: %d warp(s) drawn, sup distance from the identity in "
        "[%.3f, %.3f], median %.3f",
        len(warps),
        min(deviations),
        max(deviations),
        float(np.median(deviations)),
    )
    written = _save(fig, run_paths, stem, formats, style)
    plt.close(fig)
    return written
