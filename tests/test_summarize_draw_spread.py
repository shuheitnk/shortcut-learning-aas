"""The spread of a level over the control family's draws."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.summarize_draw_spread import draw_spread

DRAWS = ("spline/warpheld", "spline-draw1/warpheld", "spline-draw2/warpheld")


def make_levels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["G1"] * 4 + ["G2"] * 4,
            "audit": [*DRAWS, "spline-alpha0/warpheld"] * 2,
            "value": [0.1, 0.4, 0.2, 0.9, 0.5, 0.5, 0.6, 0.0],
        }
    )


def test_each_level_is_summarized_over_the_draws_only():
    spread = draw_spread(make_levels(), "audit", DRAWS).set_index("group")
    assert spread.loc["G1", "n_draws"] == 3
    assert spread.loc["G1", ["min", "median", "max"]].tolist() == [0.1, 0.2, 0.4]
    assert spread.loc["G2", ["min", "median", "max"]].tolist() == [0.5, 0.5, 0.6]


def test_a_missing_draw_stops():
    with pytest.raises(ValueError, match="draws missing"):
        draw_spread(make_levels(), "audit", (*DRAWS, "spline-draw3/warpheld"))
