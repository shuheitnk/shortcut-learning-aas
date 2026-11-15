"""Fig. 5: each focus pair's superiorities before and after the pair tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.create_figures.draw_tuning_figures import (
    SUPERIORITY_BINS,
    figure_tuning_histogram,
)
from aas_shortcut.define_algorithm_portfolio import ALGORITHM_GROUP, FOCUS_PAIRS

STEM = "fig5_tuning_histogram"

# Problems the tuning measures each configuration on: the 24 functions.
PROBLEMS = 24


def tuning_table(pairs=FOCUS_PAIRS, seed: int = 0) -> pd.DataFrame:
    """Return the `tuning_pairs.parquet` columns the figure reads, per problem."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "group": ALGORITHM_GROUP[pair.first],
            "pair": pair.slug,
            "is_default": condition == "default",
            "is_selected": condition == "selected",
            "function_id": function_id,
            "superiority": float(rng.uniform()),
        }
        for pair in pairs
        for condition in ("default", "selected")
        for function_id in range(1, PROBLEMS + 1)
    ]
    return pd.DataFrame(rows)


def _capture(monkeypatch, method: str) -> list:
    """Record the first positional argument of every call to an ``Axes`` method."""
    from matplotlib.axes import Axes

    original = getattr(Axes, method)
    calls: list = []

    def record(self, *args, **kwargs):
        calls.append(args[0] if args else None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, method, record)
    return calls


class TestTuningHistogram:
    def test_one_panel_per_pair_named_by_its_group(self, run_paths, monkeypatch):
        titles = _capture(monkeypatch, "set_title")

        figure_tuning_histogram(tuning_table(), run_paths, formats=("png",))

        assert titles == [ALGORITHM_GROUP[pair.first] for pair in FOCUS_PAIRS]

    def test_each_panel_draws_the_default_and_the_tuned_configuration(
        self, run_paths, monkeypatch
    ):
        histograms = _capture(monkeypatch, "hist")
        table = tuning_table()

        figure_tuning_histogram(table, run_paths, formats=("png",))

        assert len(histograms) == 2 * len(FOCUS_PAIRS)
        first = table[table["pair"] == FOCUS_PAIRS[0].slug]
        default = first.loc[first["is_default"], "superiority"]
        selected = first.loc[first["is_selected"], "superiority"]
        np.testing.assert_array_equal(histograms[0], np.asarray(default))
        np.testing.assert_array_equal(histograms[1], np.asarray(selected))

    def test_the_uniform_line_is_the_count_each_bin_would_hold(
        self, run_paths, monkeypatch
    ):
        lines = _capture(monkeypatch, "axhline")

        figure_tuning_histogram(tuning_table(), run_paths, formats=("png",))

        assert lines == pytest.approx([PROBLEMS / SUPERIORITY_BINS] * len(FOCUS_PAIRS))

    def test_it_is_written_once_per_format(self, run_paths):
        written = figure_tuning_histogram(
            tuning_table(), run_paths, formats=("png", "svg")
        )

        assert sorted(path.name for path in written) == [f"{STEM}.png", f"{STEM}.svg"]
        assert all(path.exists() for path in written)

    def test_a_pair_the_table_lacks_is_left_out(self, run_paths, monkeypatch):
        titles = _capture(monkeypatch, "set_title")

        figure_tuning_histogram(
            tuning_table(FOCUS_PAIRS[:2]), run_paths, formats=("png",)
        )

        assert titles == [ALGORITHM_GROUP[pair.first] for pair in FOCUS_PAIRS[:2]]

    def test_an_empty_table_writes_nothing(self, run_paths):
        written = figure_tuning_histogram(
            tuning_table().iloc[0:0], run_paths, formats=("png",)
        )

        assert written == []
        assert not list(run_paths.figures_dir.glob(f"{STEM}.*"))
