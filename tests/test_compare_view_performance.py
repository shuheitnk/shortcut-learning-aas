"""Whether a member's runs are the same in both views."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.compare_view_performance import trajectory_identity


def _runs(transform: str, best: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "problem_class": "bbob",
            "function_id": 1,
            "instance_id": 1,
            "dimension": 10,
            "algorithm": "BIPOP-CMA-ES",
            "trial": [1, 2],
            "transform": transform,
            "seed": [7, 8],
            "best_y": [1.0, best],
            "evaluations_used": 100,
            "restarts": 0,
        }
    )


def test_identical_runs_differ_in_nothing():
    table = trajectory_identity(pd.concat([_runs("pre", 2.0), _runs("post", 2.0)]))
    assert table[["runs", "best_y_differ", "runs_differ"]].iloc[0].tolist() == [2, 0, 0]


def test_a_run_that_ends_elsewhere_is_counted():
    table = trajectory_identity(
        pd.concat([_runs("pre", 2.0), _runs("post", 2.0 + 1e-15)])
    )
    assert table[["best_y_differ", "evaluations_used_differ", "runs_differ"]].iloc[
        0
    ].tolist() == [1, 0, 1]


def test_views_with_other_runs_are_refused():
    with pytest.raises(ValueError, match="same runs"):
        trajectory_identity(pd.concat([_runs("pre", 2.0), _runs("post", 2.0).iloc[1:]]))
