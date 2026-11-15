"""Tests for what Fig. 13's files hold, independently of how they look.

An unmeasured quantity is not drawn as zero, an empty table writes no file, and each
figure is one basename in every format.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.calculate_delta_z import PRIMARY_STANDARDIZATION
from aas_shortcut.create_figures.draw_shift_figures import (
    SHIFT_FORMATS,
    figure_distribution_shift,
)
from aas_shortcut.create_figures.render_experiment_figures import (
    render_distribution_shift,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    FEATURE_QUANTITY,
    NOT_A_FEATURE,
    SUPERIORITY_QUANTITY,
    RunPaths,
)

# Given out of order, with a value ranking that disagrees with the extraction order.
FEATURES: dict[str, float] = {
    "ic.m0": 0.0,
    "ela_distr.number_of_peaks": 1.0,
    "fitness_distance.fd_cov": 6.0,
    "ela_distr.skewness": 2.0,
    "ic.h_max": 0.5,
    "ela_distr.kurtosis": 0.8,
    "fitness_distance.fd_correlation": 0.3,
}


@pytest.fixture
def run_paths(tmp_path):
    return RunPaths(root=tmp_path)


def _shift_summary(
    feature_values: dict[str, float] | None = None,
    win_values: dict[str, float] | None = None,
) -> pd.DataFrame:
    """A ``shift_summary`` table with the columns the preanalysis figure reads."""
    feature_values = FEATURES if feature_values is None else feature_values
    win_values = (
        {pair.slug: 0.1 * index for index, pair in enumerate(FOCUS_PAIRS)}
        if win_values is None
        else win_values
    )
    rows = [
        {
            "quantity_kind": FEATURE_QUANTITY,
            "feature_set": ELA_FEATURE_SET,
            "feature_variant": "value",
            "quantity": name,
            "quantity_class": name.split(".")[0],
            "standardization": PRIMARY_STANDARDIZATION,
            "statistic": "mean",
            "value": value,
            "value_raw": 0.5,
            "p_moved": 1.0,
            "n_instances": 24,
            "n_undefined": 0,
        }
        for name, value in feature_values.items()
    ]
    by_slug = {pair.slug: pair for pair in FOCUS_PAIRS}
    rows += [
        {
            "quantity_kind": SUPERIORITY_QUANTITY,
            "feature_set": NOT_A_FEATURE,
            "feature_variant": NOT_A_FEATURE,
            "quantity": slug,
            "quantity_class": by_slug[slug].pair_type.split("_")[0],
            "standardization": PRIMARY_STANDARDIZATION,
            "statistic": "mean",
            "value": value,
            "value_raw": 0.2,
            "p_moved": 1.0,
            "n_instances": 24,
            "n_undefined": 0,
        }
        for slug, value in win_values.items()
    ]
    return pd.DataFrame(rows)


class TestDistributionShift:
    def test_an_undefined_quantity_is_not_drawn_as_a_zero(self, run_paths, monkeypatch):
        """Zero would say it did not move, which is the opposite of undefined."""
        captured: dict[str, list[str]] = {}
        _capture_ticks(monkeypatch, captured)

        figure_distribution_shift(
            _shift_summary({"ic.m0": float("nan"), "ic.h_max": 2.0}),
            run_paths,
            formats=("png",),
        )

        assert captured["labels"] == ["h_max"]

    def test_a_wholly_undefined_table_writes_nothing(self, run_paths):
        summary = _shift_summary({"ic.m0": float("nan")})
        assert figure_distribution_shift(summary, run_paths, formats=("png",)) == []

    def test_the_three_figures_are_three_files(self, run_paths):
        written = render_distribution_shift(_shift_summary(), run_paths, SHIFT_FORMATS)

        assert {path.stem for path in written} == {
            "fig13_distribution_shift",
            "fig13_distribution_shift_algorithm",
        }
        # PNG and SVG, as the preanalysis asks for, and no PDF.
        assert {path.suffix.lstrip(".") for path in written} == {"png", "svg"}

    def test_an_empty_table_writes_nothing(self, run_paths):
        empty = _shift_summary().iloc[0:0]
        assert figure_distribution_shift(empty, run_paths, formats=("png",)) == []


class TestOutputFormats:
    def test_a_figure_is_one_basename_in_every_format(self, run_paths):
        written = figure_distribution_shift(_shift_summary(), run_paths)

        assert {path.suffix.lstrip(".") for path in written} == set(SHIFT_FORMATS)
        assert {path.stem for path in written} == {"fig13_distribution_shift"}


def _capture_ticks(monkeypatch, captured: dict) -> None:
    """Record the x tick labels the figure sets, in axis order."""
    from matplotlib.axes import Axes

    original = Axes.set_xticklabels

    def record(self, labels, *args, **kwargs):
        captured["labels"] = [str(label) for label in labels]
        return original(self, labels, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xticklabels", record)
