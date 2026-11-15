"""The instance-level train/test splits of BBOB."""

from __future__ import annotations

from collections.abc import Sequence

# The instance ids the models are fitted on.
TRAIN_INSTANCES: tuple[int, ...] = tuple(range(11, 111))

# The instance ids they are scored on, disjoint from :data:`TRAIN_INSTANCES`.
TEST_INSTANCES: tuple[int, ...] = tuple(range(1, 11))

# Highest instance id the feature and benchmark grids cover.
MAX_INSTANCE: int = max(TRAIN_INSTANCES)

# One tuning problem per function, from the training block: a held-out problem must
# take no part in choosing the parameters its own target is measured under.
TUNING_INSTANCES: tuple[int, ...] = (TRAIN_INSTANCES[0],)

PROBLEM_CLASS = "bbob"


def check_tuning_instances(instance_ids: Sequence[int]) -> tuple[int, ...]:
    """Return the tuning instance ids after checking they stay off the test block."""
    ids = tuple(int(value) for value in instance_ids)
    if not ids:
        raise ValueError("the portfolio needs at least one tuning instance")
    leaked = sorted(set(ids) & set(TEST_INSTANCES))
    if leaked:
        raise ValueError(
            f"tuning instance(s) {leaked} lie in the held-out block "
            f"{TEST_INSTANCES[0]}-{TEST_INSTANCES[-1]}: a held-out problem must take "
            f"no part in choosing the parameters its own target is measured under; "
            f"tune on {list(TUNING_INSTANCES)}"
        )
    return ids
