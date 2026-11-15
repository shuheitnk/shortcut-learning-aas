"""The appendix's figure: one file per feature set, on shared axis ranges."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.create_figures.draw_lofo_figure import (
    STEM,
    figure_lofo_against_rrt,
    shared_limits,
)
from aas_shortcut.define_paths import RunPaths


def _levels(shift: float) -> pd.DataFrame:
    rows = []
    for group in ("G1", "G2", "G3"):
        for variant in ("value", "rank", "concat"):
            for training in ("plain", "augmented"):
                rows.append(
                    {
                        "group": group,
                        "feature_variant": variant,
                        "training": training,
                        "rrt_median": 0.5 + shift,
                        "rrt_q1": 0.4 + shift,
                        "rrt_q3": 0.6 + shift,
                        "lofo_median": -0.5,
                        "lofo_q1": -0.6,
                        "lofo_q3": -0.4,
                    }
                )
    return pd.DataFrame(rows)


CORRELATIONS = pd.DataFrame({"group": ["G1", "G2", "G3"], "rho": [0.1, 0.2, 0.3]})


def test_the_limits_hold_both_feature_sets_bars():
    (x_low, x_high), (y_low, y_high) = shared_limits([_levels(0.0), _levels(1.0)])
    assert x_low < 0.4 and x_high > 1.6
    assert y_low < -0.6 and y_high > -0.4


@pytest.mark.parametrize(
    ("feature_set", "name"), [("ela", STEM), ("deep-ela", f"{STEM}_deep-ela")]
)
def test_each_feature_set_is_its_own_file(tmp_path, feature_set, name):
    limits = shared_limits([_levels(0.0)])
    written = figure_lofo_against_rrt(
        _levels(0.0), CORRELATIONS, feature_set, limits, RunPaths(tmp_path), ("png",)
    )
    assert [path.name for path in written] == [f"{name}.png"]
    assert written[0].exists()
