"""The preanalysis's metric: the standardized feature shift."""

from __future__ import annotations

import numpy as np
import pandas as pd

# ``V_j`` aggregates |Δz| over the instances by the mean.
STATISTIC = "mean"

# ``sigma_j`` is the population standard deviation (ddof = 0) of the ``pre`` values.
PRIMARY_STANDARDIZATION = "population"


def pre_scale(pre_values: pd.Series) -> float:
    """Return the population standard deviation of a quantity's ``pre`` values."""
    return float(np.std(np.asarray(pre_values, dtype=np.float64), ddof=0))
