"""Scoring one regression fit against both views of the objective.

A cell is fitted once per model seed on the whole training block and scored on each
deployment sample of every test block; nothing is chosen or cross-validated.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from aas_shortcut.augment_training_samples import REP_COLUMN
from aas_shortcut.build_feature_matrix import INSTANCE_COLUMNS
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.predict_superiority import score_on

# The test scores, one per view of the objective.
TEST_METRICS: dict[str, str] = {
    transform: f"r2_test_{transform}" for transform in TRANSFORMS
}

# Groups for the early-stopping split, so an augmented matrix never puts one copy
# of a problem in the validation part and its twin in the fitted part.
GROUP_COLUMNS: tuple[str, ...] = INSTANCE_COLUMNS

# Key of a block with no deployment axis; deployments are numbered from one.
SINGLE_SAMPLE = 0


def deployment_scores(
    model, block: pd.DataFrame, columns: list[str]
) -> dict[int, float]:
    """Score one test block once per deployment sample, keyed by the sample.

    Samples are not pooled: that would score a model that saw the problem eleven
    times. A block without a deployment axis gives one score under
    :data:`SINGLE_SAMPLE`.
    """
    if REP_COLUMN not in block.columns or block[REP_COLUMN].nunique() < 2:
        return {SINGLE_SAMPLE: score_on(model, block, columns)}
    return {
        int(cast(int, rep)): score_on(model, rows, columns)
        for rep, rows in block.groupby(REP_COLUMN, sort=True)
    }
