"""Appendix B's figure draws the maps a run applies, not maps drawn for the figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.axes import Axes
from omegaconf import OmegaConf

from aas_shortcut.create_figures.draw_warp_figure import (
    STEM,
    WARP_CONFIG,
    WARP_PROBLEM_CLASS,
    WARP_SEED,
    figure_monotone_warps,
)
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.sample_monotone_transform import (
    HELD_OUT_DOMAIN,
    SPLINE_FAMILY,
    transform_for_instance,
)

CONFIG = Path(__file__).parents[1] / "src" / "aas_shortcut" / "configs" / "config.yaml"


def _warp(function_id: int, dim: int = 2):
    return transform_for_instance(
        WARP_SEED, WARP_PROBLEM_CLASS, function_id, 1, dim, WARP_CONFIG
    )


def test_the_figure_keys_its_maps_by_the_experiments_seed():
    assert OmegaConf.load(CONFIG).seed == WARP_SEED


def test_the_maps_are_the_test_instances_spline_warpheld():
    """Instance 1 is a test instance, so only the held-out domain was applied to it."""
    assert CONTROL.warp == WARP_CONFIG
    assert WARP_CONFIG.family == SPLINE_FAMILY
    assert WARP_CONFIG.domain == HELD_OUT_DOMAIN


def test_every_curve_is_a_warp_of_a_real_instance(run_paths, monkeypatch):
    drawn: list[tuple] = []
    original = Axes.plot

    def record(self, *args, **kwargs):
        drawn.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", record)
    figure_monotone_warps(run_paths, ("png",), function_ids=(1, 2, 3), dim=2)

    # Three warps and the identity they are read against.
    assert len(drawn) == 4
    for (inputs, outputs), function_id in zip(drawn[1:], (1, 2, 3), strict=True):
        warp = _warp(function_id)
        np.testing.assert_array_equal(outputs, np.asarray(warp.values))
        np.testing.assert_array_equal(inputs, np.asarray(warp.grid))


def test_a_curve_is_strictly_increasing_and_pinned_at_both_ends():
    for function_id in (1, 5, 12):
        knots = np.asarray(_warp(function_id).values)
        assert np.all(np.diff(knots) > 0.0)
        assert knots[0] == pytest.approx(0.0)
        assert knots[-1] == pytest.approx(1.0)


def test_the_figure_is_written_in_every_format(run_paths):
    written = figure_monotone_warps(run_paths, ("png", "svg"), function_ids=(1,), dim=2)
    assert sorted(path.name for path in written) == [f"{STEM}.png", f"{STEM}.svg"]
    assert all(path.stat().st_size > 0 for path in written)
