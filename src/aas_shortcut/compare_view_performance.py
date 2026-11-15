"""Appendix: whether a member's runs are the same before and after the transformation.

Both views run each trial from one seed, and `best_y` is read from the untransformed
problem in both, so a member whose search reads only comparisons must end every run
with the same `best_y`, evaluations and restarts.
"""

from __future__ import annotations

import pandas as pd

RUN_KEYS: tuple[str, ...] = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
    "algorithm",
    "trial",
)

# What a run reports that the transformation must not move for a G1 member.
COMPARED: tuple[str, ...] = ("best_y", "evaluations_used", "restarts", "seed")


def trajectory_identity(performance: pd.DataFrame) -> pd.DataFrame:
    """Count, per member, the runs whose reported values differ between the views.

    ``performance`` holds both views' runs of one block; a run must be in both.
    """
    views = {
        view: performance[performance["transform"].astype(str) == view].set_index(
            list(RUN_KEYS)
        )[list(COMPARED)]
        for view in ("pre", "post")
    }
    pre, post = views["pre"], views["post"]
    if not pre.index.sort_values().equals(post.index.sort_values()):
        raise ValueError("the two views do not hold the same runs")
    post = post.reindex(pre.index)
    rows = []
    for algorithm, before in pre.groupby(level="algorithm", observed=True):
        after = post.loc[before.index]
        row: dict = {"algorithm": str(algorithm), "runs": len(before)}
        for column in COMPARED:
            row[f"{column}_differ"] = int(
                (before[column].to_numpy() != after[column].to_numpy()).sum()
            )
        row["runs_differ"] = int(
            (before.to_numpy() != after.to_numpy()).any(axis=1).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)
