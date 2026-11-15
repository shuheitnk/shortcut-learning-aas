"""Names, hues and helpers every paper figure is built from."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from aas_shortcut.define_figure_theme import (
    FigureStyle,
    save_figure,
)
from aas_shortcut.define_paths import (
    DEEP_ELA_FEATURE_SET,
    ELA_FEATURE_SET,
    ORACLE_VARIANT,
    REPRESENTATION_SUFFIXES,
    RunPaths,
)

# Named for the package: every figure module logs through this one logger.
LOGGER = logging.getLogger("aas_shortcut.create_figures")


def variant_colors(style: FigureStyle) -> dict[str, str]:
    """Hue per representation. The Oracle is a reference, so it takes neutral ink."""
    return {
        "value": style.hue["value"],
        "rank": style.hue["rank"],
        "concat": style.hue["concat"],
        ORACLE_VARIANT: style.ink_muted,
    }


# The representations, in the order every figure draws them.
VARIANT_ORDER: tuple[str, ...] = ("value", "rank", "concat")


class SeriesStyle(NamedTuple):
    """How one violin inside a panel is drawn."""

    key: str
    label: str
    color: str
    marker: str
    filled: bool = True


# Axis labels. ``concat`` is the two matrices side by side, never averaged.
VARIANT_LABELS = {
    "value": "value-based",
    "rank": "rank-based",
    "concat": "concat",
    ORACLE_VARIANT: "oracle",
}


# How each representation is named when a figure reports both.
FEATURE_SET_LABELS = {
    ELA_FEATURE_SET: "ELA",
    DEEP_ELA_FEATURE_SET: "Deep-ELA",
}

# Filename suffix for a representation other than handcrafted ELA; tables use the same.
FIGURE_SUFFIXES = REPRESENTATION_SUFFIXES


def feature_set_stem(stem: str, feature_set: str) -> str:
    """Return the file name one representation's copy of a figure is saved under.

    The representations are never pooled: ELA takes the bare name, Deep-ELA a suffix.
    """
    return f"{stem}{FIGURE_SUFFIXES.get(feature_set, f'_{feature_set}')}"


def _save(
    fig,
    run_paths: RunPaths,
    name: str,
    formats: Sequence[str],
    style: FigureStyle | None = None,
) -> list[Path]:
    """Write one figure in every requested format."""
    run_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    written = [
        save_figure(fig, run_paths.figure(name, extension), style)
        for extension in formats
    ]
    LOGGER.info("figures: wrote %s", ", ".join(str(path) for path in written))
    return written


def _restrict(table: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Keep the rows belonging to one representation."""
    if "feature_set" not in table.columns:
        return table
    return table[table["feature_set"] == feature_set]


def _present_feature_sets(table: pd.DataFrame) -> list[str]:
    """Return the representations a table actually holds, in reporting order."""
    if "feature_set" in table.columns:
        present = set(table["feature_set"].unique())
    elif "feature_set" in (table.index.names or ()):
        present = set(table.index.get_level_values("feature_set").unique())
    else:
        return [ELA_FEATURE_SET]
    return [name for name in FEATURE_SET_LABELS if name in present]
