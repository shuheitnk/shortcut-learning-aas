"""One entry point per experiment, each drawing that experiment's figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.create_figures.define_figure_vocabulary import (
    LOGGER,
    _present_feature_sets,
    feature_set_stem,
)
from aas_shortcut.create_figures.draw_selection_split_figure import (
    SPLIT_FORMATS,
    SPLIT_STEM,
    figure_selection_split_pooled,
)
from aas_shortcut.create_figures.draw_shift_figures import (
    SHIFT_FORMATS,
    figure_distribution_shift,
)
from aas_shortcut.define_figure_theme import FigureStyle, default_style
from aas_shortcut.define_paths import (
    FEATURE_QUANTITY,
    SUPERIORITY_QUANTITY,
    RunPaths,
)


def render_distribution_shift(
    summary: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = SHIFT_FORMATS,
    variant: str = "value",
    style: FigureStyle | None = None,
) -> list[Path]:
    """Draw the preanalysis's three distribution-shift figures.

    Each keeps its own vertical axis: a saturated Deep-ELA coordinate shifts by
    several standard deviations, which would flatten the superiority panel.
    """
    style = style or default_style()
    features = cast(pd.DataFrame, summary[summary["quantity_kind"] == FEATURE_QUANTITY])
    written: list[Path] = []
    for feature_set in _present_feature_sets(features):
        written += figure_distribution_shift(
            summary,
            run_paths,
            formats,
            quantity_kind=FEATURE_QUANTITY,
            feature_set=feature_set,
            variant=variant,
            style=style,
        )
    written += figure_distribution_shift(
        summary,
        run_paths,
        formats,
        quantity_kind=SUPERIORITY_QUANTITY,
        style=style,
    )
    return written


def render_selection_split(
    table: pd.DataFrame,
    run_paths: RunPaths,
    formats: Sequence[str] = SPLIT_FORMATS,
    style: FigureStyle | None = None,
    *,
    stem: str = SPLIT_STEM,
    training: str = "plain",
    levels: pd.DataFrame | None = None,
) -> list[Path]:
    """Draw Experiment 3's selection figure, one file per representation.

    ``levels`` gives the level drawn per ``(feature_set, feature_variant, transform)``;
    a method without one (the oracle) is drawn at its violin's mean.
    """
    rows = cast(
        pd.DataFrame,
        table[table["training"] == training] if "training" in table.columns else table,
    )
    if rows.empty:
        LOGGER.warning("figures: no %s rows to draw; skipping %s", training, stem)
        return []
    written: list[Path] = []
    for feature_set in _present_feature_sets(rows):
        drawn_at = None
        if levels is not None:
            own = levels[levels["feature_set"] == feature_set]
            drawn_at = {
                (str(variant), str(transform)): float(value)
                for variant, transform, value in zip(
                    own["feature_variant"], own["transform"], own["value"], strict=True
                )
            }
        written += figure_selection_split_pooled(
            rows,
            run_paths,
            formats,
            style,
            stem=feature_set_stem(stem, feature_set),
            feature_set=feature_set,
            levels=drawn_at,
        )
    return written
