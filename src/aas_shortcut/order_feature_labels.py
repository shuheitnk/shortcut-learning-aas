"""The order and the class hierarchy of the feature labels, for the figures."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from aas_shortcut.calculate_deep_ela_features import (
    FEATURE_PREFIX as DEEP_ELA_PREFIX,
)
from aas_shortcut.calculate_deep_ela_features import (
    is_deep_ela_feature,
)

LOGGER = logging.getLogger(__name__)

# Probe sample; ``calculate_ela_level`` refuses fewer than 100 points.
PROBE_SIZE = 200
PROBE_DIM = 2
PROBE_SEED = 0

# Trailing index of a Deep-ELA embedding coordinate, e.g. ``X7``.
_EMBEDDING_INDEX = re.compile(r"^X(\d+)$")


@dataclass(frozen=True)
class ClassSpan:
    """One feature class and the half-open range of axis positions it occupies."""

    name: str
    start: int
    stop: int

    @property
    def size(self) -> int:
        """How many features the class contributes."""
        return self.stop - self.start

    @property
    def centre(self) -> float:
        """Axis position the class label is centred on."""
        return (self.start + self.stop - 1) / 2.0


def feature_class(feature: str) -> str:
    """Return the class a feature name belongs to."""
    if is_deep_ela_feature(feature):
        model = feature[len(DEEP_ELA_PREFIX) :].split(".")[0]
        return f"{DEEP_ELA_PREFIX}{model}"
    return feature.split(".")[0]


# Display labels only: the class name keys the artifacts and orders the axis, so it
# must not be renamed.
CLASS_LABELS: dict[str, str] = {"fitness_distance": "fdc"}


def class_label(name: str) -> str:
    """Return how a feature class is written on a figure."""
    if name.startswith(DEEP_ELA_PREFIX):
        return name[len(DEEP_ELA_PREFIX) :]
    return CLASS_LABELS.get(name, name)


def leaf_label(feature: str) -> str:
    """Return the part of a feature name its class does not already say."""
    prefix = f"{feature_class(feature)}."
    return feature[len(prefix) :] if feature.startswith(prefix) else feature


@lru_cache(maxsize=1)
def extraction_order() -> tuple[str, ...]:
    """Return the handcrafted feature names in the order the calculators emit them."""
    import numpy as np
    import pandas as pd

    from aas_shortcut.calculate_features import ELA_CALCULATORS, RUNTIME_SUFFIX

    rng = np.random.default_rng(PROBE_SEED)
    sample = rng.uniform(-5.0, 5.0, size=(PROBE_SIZE, PROBE_DIM))
    columns = [f"x{index}" for index in range(PROBE_DIM)]
    X = pd.DataFrame(sample, columns=columns)
    # Smooth and asymmetric, so no calculator declines the probe.
    y = pd.Series(np.sum(sample**2, axis=1) + np.sin(sample[:, 0]))

    # The calculators may touch NumPy's legacy global RNG; restore it afterwards.
    state = np.random.get_state()
    names: list[str] = []
    try:
        for calculator in ELA_CALCULATORS:
            try:
                emitted = calculator(X, y, PROBE_SEED)
            except Exception as error:
                LOGGER.warning(
                    "feature order: calculator %s refused the probe (%s); its "
                    "features keep the artifact's order",
                    calculator.name,
                    error,
                )
                continue
            names += [
                str(key) for key in emitted if not str(key).endswith(RUNTIME_SUFFIX)
            ]
    finally:
        np.random.set_state(state)
    return tuple(names)


@lru_cache(maxsize=1)
def class_order() -> tuple[str, ...]:
    """Return the feature classes in the order their calculators run."""
    from aas_shortcut.calculate_features import ELA_CALCULATORS

    return tuple(calculator.group for calculator in ELA_CALCULATORS)


def _sort_key(feature: str) -> tuple[int, int, str]:
    """Rank one feature by class, then by position within its class."""
    classes = class_order()
    name = feature_class(feature)
    # Deep-ELA and unknown classes sort after the handcrafted ones.
    class_rank = classes.index(name) if name in classes else len(classes)
    if is_deep_ela_feature(feature):
        match = _EMBEDDING_INDEX.match(leaf_label(feature))
        # Embedding coordinates are numbered, so ``X10`` follows ``X9``.
        return (class_rank, int(match.group(1)) if match else 0, feature)
    order = extraction_order()
    within = order.index(feature) if feature in order else len(order)
    return (class_rank, within, feature)


def order_features(features: Iterable[str]) -> list[str]:
    """Return the features in extraction order, classes kept contiguous."""
    return sorted(dict.fromkeys(features), key=_sort_key)


def spans_of(classes: Sequence[str]) -> list[ClassSpan]:
    """Return where each run of equal class names begins and ends."""
    spans: list[ClassSpan] = []
    for position, name in enumerate(classes):
        if spans and spans[-1].name == name:
            spans[-1] = ClassSpan(name, spans[-1].start, position + 1)
            continue
        spans.append(ClassSpan(name, position, position + 1))
    return spans


def class_spans(features: Sequence[str]) -> list[ClassSpan]:
    """Return where each class begins and ends along an ordered feature axis."""
    return spans_of([feature_class(feature) for feature in features])
