"""Experiments 3 and 5: the selection cells' training conditions, split and tables."""

from __future__ import annotations

from collections.abc import Sequence

from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.select_algorithms import SelectionSplit
from aas_shortcut.split_bbob_instances import (
    PROBLEM_CLASS,
    TEST_INSTANCES,
    TRAIN_INSTANCES,
)

# Fitted on the untransformed training block alone.
TRAINING_PLAIN = "plain"

# Fitted on the untransformed and the transformed training block, each row carrying
# the mean superiorities its own view earned.
TRAINING_AUGMENTED = "augmented"

# In report order.
TRAINING_CONDITIONS: tuple[str, ...] = (TRAINING_PLAIN, TRAINING_AUGMENTED)

TRAINING_VIEWS: dict[str, tuple[str, ...]] = {
    TRAINING_PLAIN: (TRANSFORMS[0],),
    TRAINING_AUGMENTED: TRANSFORMS,
}


def training_views(trainings: Sequence[str]) -> tuple[str, ...]:
    """Every view the given training conditions fit on, in the views' order."""
    unknown = [name for name in trainings if name not in TRAINING_VIEWS]
    if unknown:
        raise ValueError(
            f"unknown training condition(s) {unknown}; expected a subset of "
            f"{list(TRAINING_VIEWS)}"
        )
    read = {view for name in trainings for view in TRAINING_VIEWS[name]}
    return tuple(view for view in TRANSFORMS if view in read)


# The oracle is scored beside these as a ceiling, not a candidate.
VARIANTS: tuple[str, ...] = ("value", "rank", "concat")

# Fit on BBOB instances 11-110, select on 1-10 of the same suite.
SPLIT = SelectionSplit(
    train_problem_class=PROBLEM_CLASS,
    test_problem_class=PROBLEM_CLASS,
    train_instances=TRAIN_INSTANCES,
    test_instances=TEST_INSTANCES,
)


# Each training run's mean superiority over the held-out problems, per deployment.
SEED_DEPLOYMENT_SHARD = "deployment_scores"

# Each training run's score on every held-out problem of every deployment.
PROBLEM_SHARD = "problem_scores"
