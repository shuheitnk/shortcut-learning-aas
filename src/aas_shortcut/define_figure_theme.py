"""The one visual language every paper figure obeys."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Ink
# ---------------------------------------------------------------------------

# Neutral greys (``R == G == B``) for references, kept free of any hue a reader
# could mistake for a representation's.

# Primary text: axis labels, the value a figure is read for.
INK = "#1B1B1B"
# Secondary text: tick labels, annotations, the neutral baseline series.
INK_MUTED = "#5C5C5C"
# Tertiary marks: class rules, whiskers, labels of quantities that measured zero.
INK_FAINT = "#9B9B9B"
# Spines and separators.
RULE = "#C4C4C4"
# Grid hairlines, one step lighter than the spines.
GRID = "#E0E0E0"
# The page.
PAPER = "#FFFFFF"

# PDF and SVG for the paper, PNG for checking a run.
DEFAULT_FORMATS: tuple[str, ...] = ("pdf", "png", "svg")

# ---------------------------------------------------------------------------
# Hues
# ---------------------------------------------------------------------------

# Hue per representation, fixed across every figure. None lies in the purple and
# magenta band that protanopia folds onto blue and grey.
VALUE_HUE = "#D55E00"
RANK_HUE = "#0072B2"
CONCAT_HUE = "#009E73"

# A baseline series takes neutral ink rather than a hue.
BASELINE = INK_MUTED


def _blend(color: str, other: str, amount: float) -> tuple[float, float, float]:
    """Mix two hex colours in sRGB, ``amount`` of ``other``."""
    from matplotlib.colors import to_rgb

    left, right = to_rgb(color), to_rgb(other)
    return tuple(  # type: ignore[return-value]
        (1.0 - amount) * a + amount * b for a, b in zip(left, right, strict=True)
    )


# How much paper :meth:`FigureStyle.tint` mixes into a hue: the least that keeps
# ink on the fill above 5:1 contrast, so the fill still reads as its hue.
TINT_AMOUNT = 0.35


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

# One type scale, in points, sized to be read from across a room. No title role:
# titles and captions belong to the LaTeX document. ``micro`` is for Fig. 13's
# dense feature axis.
TYPE: dict[str, float] = {
    "panel": 17.0,
    "group": 20.0,
    "axis": 21.0,
    "legend": 21.0,
    "tick": 17.0,
    "annotation": 17.0,
    "micro": 16.5,
}


# Face per maths role, for ``mathtext.fontset = "custom"``: all sans, so maths is
# set in the text face. ``cal`` is mapped because matplotlib resolves it anyway and
# its generic ``cursive`` family is absent from a headless image.
MATH_FACES: dict[str, str] = {
    "rm": "sans",
    "it": "sans:italic",
    "bf": "sans:bold",
    "sf": "sans",
    "cal": "sans:italic",
    "tt": "monospace",
}

# Tick-to-label and label-to-axis-label distance, in points. At this type size the
# default 3.5 pt leaves the numbers sitting on the axis.
LABEL_PAD = 10.0

# ---------------------------------------------------------------------------
# Strokes
# ---------------------------------------------------------------------------

# Strokes are scaled with the type, so rules do not vanish beside large text.

# Axis spines and the rules under a group label.
SPINE_WIDTH = 1.4
# Wider than a hairline so the grid survives projection.
GRID_WIDTH = 1.1
# A reference the reader measures against (``R^2 = 0``, ``w = 0.5``, ``U(0,1)``).
REFERENCE_WIDTH = 2.2
LINE_WIDTH = 2.6


# Canvas per figure, in inches. A content-sized figure takes only its height from
# here and its width from :data:`WIDTH_PER_COLUMN`. Figures not listed keep their
# canvas beside their drawing code.
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "fig5_tuning_histogram": (1.1, 4.8),
    "fig13_distribution_shift": (13.0, 7.6),
    # Three bars, so narrower than the feature panel.
    "fig13_algorithm": (6.4, 7.6),
    "fig18_monotone_warp": (6.4, 6.4),
}

# Inches of width per column of a content-sized figure, added to the fixed part in
# ``FIGURE_SIZES``.
WIDTH_PER_COLUMN: dict[str, float] = {
    "fig5_tuning_histogram": 3.7,
    # One pitch for the preanalysis's figures, which report one measurement.
    "fig13_distribution_shift": 0.30,
    "fig13_algorithm": 1.5,
}

# Axis words keyed by figure, so changing one changes one figure only. Only the
# first word is capitalised.
AXIS_LABELS: dict[str, str] = {
    # Fig. 16's; two lines to fit its 2.3-inch canvas.
    "fig4_y": "Probability of superiority\nover the SBS",
    "fig5_histogram_x": "Probability of superiority",
    "fig5_histogram_y": "Count",
    # The mean over instances of |z_post - z_pre|, defined in the caption.
    "fig13_y": "Shift",
    "fig18_x": "Original value",
    "fig18_y": "Transformed value",
}


@dataclass
class FigureStyle:
    """Every visual decision the figures make, in one editable object."""

    # Faces to try, in order; each should share Arial's metrics so the layout holds.
    font_family: list[str] = field(
        default_factory=lambda: ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
    )
    # ``custom`` with :attr:`math_faces` sets maths in :attr:`font_family`.
    # ``stixsans`` only resembles it, and matplotlib writes the resolved face into
    # an SVG.
    math_fontset: str = "custom"
    # ``regular``: unstyled maths in the upright text face, as a grotesque has no
    # slanted-variable convention.
    math_default: str = "regular"
    math_faces: dict[str, str] = field(
        default_factory=lambda: dict(MATH_FACES),
    )

    type: dict[str, float] = field(default_factory=lambda: dict(TYPE))
    label_pad: float = LABEL_PAD
    # Tick lengths, in pt.
    tick_length: float = 4.2
    minor_tick_length: float = 2.2

    ink: str = INK
    ink_muted: str = INK_MUTED
    ink_faint: str = INK_FAINT
    rule: str = RULE
    grid: str = GRID
    paper: str = PAPER

    # Hue per entity. ``oracle`` and ``baseline`` are references, not entities,
    # so they take ink rather than a hue of their own.
    hue: dict[str, str] = field(
        default_factory=lambda: {
            "value": VALUE_HUE,
            "rank": RANK_HUE,
            "concat": CONCAT_HUE,
            "baseline": BASELINE,
        }
    )
    # How much paper :meth:`tint` mixes into a hue that has to carry text.
    tint_amount: float = TINT_AMOUNT

    spine_width: float = SPINE_WIDTH
    grid_width: float = GRID_WIDTH
    reference_width: float = REFERENCE_WIDTH
    line_width: float = LINE_WIDTH

    size: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(FIGURE_SIZES)
    )
    width_per_column: dict[str, float] = field(
        default_factory=lambda: dict(WIDTH_PER_COLUMN)
    )

    label: dict[str, str] = field(default_factory=lambda: dict(AXIS_LABELS))

    # Output. ``formats`` is which files are written, ``save_dpi`` the raster
    # resolution, ``pad_inches`` the white kept around the cropped figure.
    formats: tuple[str, ...] = DEFAULT_FORMATS
    dpi: float = 150
    save_dpi: float = 600
    pad_inches: float = 0.06

    def tint(self, hue: str) -> tuple[float, float, float]:
        """``hue`` mixed toward the page by this style's :attr:`tint_amount`."""
        return _blend(hue, self.paper, self.tint_amount)


def default_style() -> FigureStyle:
    """A fresh style holding the values the paper's figures are drawn with."""
    return FigureStyle()


def apply_theme(plt: Any, style: FigureStyle | None = None) -> None:
    """Install ``style``'s rcParams on ``plt``."""
    style = style or default_style()
    plt.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "savefig.dpi": style.save_dpi,
            "savefig.bbox": "tight",
            # Not zero, because a rotated mathtext label under-reports its extent.
            "savefig.pad_inches": style.pad_inches,
            # Embed text as text, not outlines.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # Arial, then Liberation Sans (in the image), Helvetica (macOS) and
            # DejaVu Sans (matplotlib's own, so a bare checkout renders).
            "font.family": "sans-serif",
            "font.sans-serif": list(style.font_family),
            "mathtext.fontset": style.math_fontset,
            "mathtext.default": style.math_default,
            **{f"mathtext.{role}": face for role, face in style.math_faces.items()},
            "font.size": style.type["axis"],
            # Pinned so that a stray title stays on the scale.
            "figure.titlesize": style.type["panel"],
            "axes.titlesize": style.type["panel"],
            "axes.labelsize": style.type["axis"],
            "legend.fontsize": style.type["legend"],
            "xtick.labelsize": style.type["tick"],
            "ytick.labelsize": style.type["tick"],
            "axes.edgecolor": style.rule,
            "axes.linewidth": style.spine_width,
            "axes.labelcolor": style.ink,
            "axes.titlecolor": style.ink,
            "text.color": style.ink,
            "xtick.color": style.rule,
            "ytick.color": style.rule,
            "xtick.labelcolor": style.ink,
            "ytick.labelcolor": style.ink,
            "xtick.major.pad": style.label_pad,
            "ytick.major.pad": style.label_pad,
            "axes.labelpad": style.label_pad,
            "axes.titlepad": style.label_pad,
            "xtick.major.width": style.spine_width,
            "ytick.major.width": style.spine_width,
            "xtick.major.size": style.tick_length,
            "ytick.major.size": style.tick_length,
            "xtick.minor.size": style.minor_tick_length,
            "ytick.minor.size": style.minor_tick_length,
            "axes.grid": False,
            "grid.color": style.grid,
            "grid.linewidth": style.grid_width,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.handlelength": 1.1,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.6,
            "legend.borderpad": 0.0,
            "figure.facecolor": style.paper,
            "axes.facecolor": style.paper,
            "savefig.facecolor": style.paper,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
        }
    )


def font_stack(style: FigureStyle | None = None) -> str:
    """The CSS family list a text object is written with, as one string."""
    style = style or default_style()
    return ", ".join([f"'{name}'" for name in style.font_family] + ["sans-serif"])


# A ``font-family`` naming exactly one face. Ordinary text never looks like this,
# because matplotlib writes the whole family list for it.
_ONE_FAMILY = re.compile(r"font-family: '([^']+)'(?=[;\"])")


def restack_svg(markup: str, style: FigureStyle | None = None) -> str:
    """Give every mark in an SVG the same font stack the ordinary text has."""
    style = style or default_style()
    stack = font_stack(style)
    requested = set(style.font_family)

    def replace(match: re.Match[str]) -> str:
        return (
            f"font-family: {stack}" if match.group(1) in requested else match.group(0)
        )

    return _ONE_FAMILY.sub(replace, markup)


# Timestamps are dropped so that redrawing an unchanged table writes a
# byte-identical file.
TIMESTAMP_METADATA = {".svg": {"Date": None}, ".pdf": {"CreationDate": None}}

# Without a fixed salt, matplotlib's SVG element ids are random per process. Any
# fixed string will do.
SVG_HASH_SALT = "svg-element-ids"


def save_figure(fig: Any, path: Any, style: FigureStyle | None = None) -> Any:
    """Write one figure, in the one way every figure of this project is written."""
    from pathlib import Path

    import matplotlib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASH_SALT}):
        fig.savefig(path, metadata=TIMESTAMP_METADATA.get(path.suffix.lower()))
    if path.suffix.lower() == ".svg":
        path.write_text(restack_svg(path.read_text(encoding="utf-8"), style), "utf-8")
    return path


def strip_axes(ax: Any, keep: Sequence[str] = ("left", "bottom")) -> None:
    """Leave only the spines that carry a scale."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)


def hairline_grid(ax: Any, axis: str = "y", style: FigureStyle | None = None) -> None:
    """Put a grid behind the marks on one axis only, at the faintest weight."""
    style = style or default_style()
    ax.grid(axis=axis, color=style.grid, linewidth=style.grid_width, zorder=0)
    ax.set_axisbelow(True)


def bound_spine(ax: Any, low: float, high: float, side: str = "left") -> None:
    """Shorten a spine to the range the ticks actually cover."""
    ax.spines[side].set_bounds(low, high)
