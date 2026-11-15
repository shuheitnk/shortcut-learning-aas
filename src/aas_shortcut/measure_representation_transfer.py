"""Experiment 2: which representation survives a strictly increasing transformation?"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS, AlgorithmPair
from aas_shortcut.define_paths import TRANSFORMS

# The representations, in the order every table and figure of the study uses.
VARIANTS: tuple[str, ...] = ("value", "rank", "concat")


def group_of(pair: AlgorithmPair) -> str:
    """Return the algorithm group a within-group pair belongs to, e.g. ``"G1"``."""
    return pair.pair_type.split("_")[0]


def describe_target_invariance(
    superiorities: pd.DataFrame, pairs: Sequence[AlgorithmPair] = FOCUS_PAIRS
) -> pd.DataFrame:
    """Report how many instances each pair's superiority moved on, and how far."""
    from aas_shortcut.calculate_superiority import views_of

    rows = []
    for pair in pairs:
        views = views_of(superiorities, pair.slug)
        missing = [name for name in TRANSFORMS if name not in views.columns]
        if missing:
            raise ValueError(
                f"the superiority table is missing view(s) {missing}; the target's "
                "invariance cannot be checked"
            )
        paired = views.dropna(subset=list(TRANSFORMS))
        difference = (paired["post"] - paired["pre"]).abs()
        rows.append(
            {
                "group": group_of(pair),
                "pair": pair.slug,
                "n_instances": len(paired),
                "n_moved": int((difference > 0.0).sum()),
                "max_difference": float(difference.max()) if len(paired) else 0.0,
            }
        )
    return pd.DataFrame(rows)


# The premises a run checked, written so the counts the paper quotes are in a file.
# ``target`` rows: instances on which a pair's superiority differs between views
# (exact). ``anchor`` rows: untransformed feature values of a held-out block that
# agreed with the training block's within ``tolerance`` (a disagreement stops the run).
PREMISE_TABLE = Path("premises.csv")
TARGET_CHECK = "target"
ANCHOR_CHECK = "anchor"
PREMISE_COLUMNS = (
    "check",
    "block",
    "group",
    "pair",
    "n_compared",
    "n_differing",
    "max_difference",
    "tolerance",
)


def premise_table(
    movements: Mapping[str, pd.DataFrame],
    anchors: Mapping[str, int],
    tolerance: float,
) -> pd.DataFrame:
    """Return a run's premises, one row per pair and block, then per anchor.

    ``movements`` maps a block's label to :func:`describe_target_invariance` of
    its superiorities, and ``anchors`` a held-out block's label to how many
    untransformed feature values agreed with the training block's.
    """
    rows = [
        {
            "check": TARGET_CHECK,
            "block": block,
            "group": record["group"],
            "pair": record["pair"],
            "n_compared": int(record["n_instances"]),
            "n_differing": int(record["n_moved"]),
            "max_difference": float(record["max_difference"]),
            "tolerance": 0.0,
        }
        for block, moved in movements.items()
        for record in moved.to_dict("records")
    ]
    rows += [
        {
            "check": ANCHOR_CHECK,
            "block": block,
            "group": "",
            "pair": "",
            "n_compared": int(n_compared),
            "n_differing": 0,
            "max_difference": float("nan"),
            "tolerance": tolerance,
        }
        for block, n_compared in anchors.items()
    ]
    return pd.DataFrame(rows, columns=list(PREMISE_COLUMNS))


def invariance_holds(superiorities: pd.DataFrame, group: str = "G1") -> bool:
    """Whether the group whose target cannot move indeed did not move."""
    moved = describe_target_invariance(superiorities)
    rows = moved[moved["group"] == group]
    if rows.empty or int(rows["n_instances"].iloc[0]) == 0:
        return False
    return int(rows["n_moved"].iloc[0]) == 0
