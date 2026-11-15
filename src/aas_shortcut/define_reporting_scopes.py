"""The two ranges of families a transformed block is reported over.

The four families the training never saw form RQ3's main range; spline, the known
family, is read on its own. The five-family mean is not reported, since it would fold
the known family into the claim about the unknown ones. Experiments 4 and 5 both read
the ranges from here, so they cannot read one range two ways.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from aas_shortcut.define_warp_audits import AUDITS, CONTROL

# The two ranges are different quantities, so they are never averaged together.
SCOPE_COLUMN = "scope"

# The known family on its own, named after the control audit so the two cannot drift.
SCOPE_CONTROL = f"{CONTROL.name} only"

SCOPE_UNSEEN = "unseen-family mean"
UNSEEN_FAMILIES: tuple[str, ...] = tuple(
    audit.label for audit in AUDITS if audit.label != CONTROL.label
)

# Each range's families, the main range first (the reporting order).
SCOPES: dict[str, tuple[str, ...]] = {
    SCOPE_UNSEEN: UNSEEN_FAMILIES,
    SCOPE_CONTROL: (CONTROL.label,),
}


def scope_families(scope: str) -> tuple[str, ...]:
    """Return the families one range is read over."""
    if scope not in SCOPES:
        raise ValueError(f"unknown range {scope!r}; expected one of {list(SCOPES)}")
    return SCOPES[scope]


def measurable_scopes(
    families: Iterable[str],
) -> tuple[list[str], dict[str, list[str]]]:
    """Return the ranges a run's families cover, and the families the others lack.

    A range needs every one of its families: a mean over fewer is another quantity.
    """
    held = {str(name) for name in families}
    covered: list[str] = []
    lacking: dict[str, list[str]] = {}
    for scope, needed in SCOPES.items():
        missing = [label for label in needed if label not in held]
        if missing:
            lacking[scope] = missing
        else:
            covered.append(scope)
    return covered, lacking


def scope_rows(table: pd.DataFrame, scope: str, family_column: str) -> pd.DataFrame:
    """Return the rows of one range's families, raising if one is missing."""
    wanted = scope_families(scope)
    named = table[family_column].astype(str)
    present = set(named)
    missing = [label for label in wanted if label not in present]
    if missing:
        raise ValueError(f"the {scope!r} range needs the families {missing}")
    return table[named.isin(wanted)]


def scope_mean(
    table: pd.DataFrame,
    keys: Sequence[str],
    scope: str,
    family_column: str,
    value_column: str = "value",
) -> pd.DataFrame:
    """Average one value per family over one range's families.

    ``table`` holds one row per ``(*keys, family)``, already reduced over deployments.
    """
    rows = scope_rows(table, scope, family_column)
    return rows.groupby(list(keys), observed=True)[value_column].mean().reset_index()
