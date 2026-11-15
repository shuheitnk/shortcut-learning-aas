"""Appendix: the preanalysis's shift of the regression's target, on any test block.

The preanalysis measures it on the control family's block; the appendix's
further blocks are measured the same way, so their rows read alike.
"""

from __future__ import annotations

import pandas as pd

from aas_shortcut.calculate_delta_z import PRIMARY_STANDARDIZATION, STATISTIC
from aas_shortcut.define_paths import SUPERIORITY_QUANTITY
from aas_shortcut.define_repetitions import REPETITIONS
from aas_shortcut.measure_distribution_shift import (
    measure_shift,
    summarize_shift,
    view_superiority_observations,
)

DEPLOYMENTS: tuple[int, ...] = tuple(range(1, REPETITIONS + 1))


def superiority_shift(superiorities: pd.DataFrame, label: str) -> pd.DataFrame:
    """Each pair's standardized shift on one block, as the preanalysis reports it."""
    summary = summarize_shift(
        measure_shift(view_superiority_observations(superiorities, DEPLOYMENTS))
    )
    rows = summary[
        (summary["quantity_kind"] == SUPERIORITY_QUANTITY)
        & (summary["statistic"] == STATISTIC)
        & (summary["standardization"] == PRIMARY_STANDARDIZATION)
    ]
    return rows[
        ["quantity", "quantity_class", "p_moved", "n_instances", "value", "value_raw"]
    ].assign(audit=label)
