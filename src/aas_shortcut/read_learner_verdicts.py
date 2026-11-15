"""RQ1 to RQ3's verdicts under one learner, and whether they are RealMLP's.

Each verdict is broken into readings, one per item a rule names, read off the
same levels and tests the paper's tables report, so a learner that disagrees with
RealMLP is told by the items it reads differently:

* RQ1: whether both shortcut conditions hold, and the test on the transformed block;
* RQ2 (spline family, no augmentation): which representation pairs differ after
  Holm and in which direction, and which representation is highest;
* RQ3 (mean over the four unseen families): in which cells augmentation is
  significantly above its absence after Holm.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from aas_shortcut.compare_predictor_sets import shortcut_detected
from aas_shortcut.compare_selection_runs import (
    compare_selection_representations,
    compare_selection_trainings,
    selection_levels,
    selection_runs,
)
from aas_shortcut.compare_training_runs import (
    compare_representations_by_scope,
    compare_training_conditions_by_scope,
    median_over_seeds,
    per_seed,
)
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
)
from aas_shortcut.measure_augmented_regression import best_variant
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
    VARIANTS,
)

# The columns of the table.
QUESTIONS: tuple[str, ...] = ("RQ1", "RQ2", "RQ3")

# The two tasks RQ2 and RQ3 are read on.
REGRESSION, SELECTION = "regression", "selection"

# Deep-ELA's RQ1 is not measured, because Experiment 1 uses ELA alone.
AGREE = "agree"
DISAGREE = "disagree"
REFERENCE = "reference"
NOT_MEASURED = "—"

# What a pair reads when neither side is significantly above the other.
NOT_SIGNIFICANT = "not significant"
# Whether augmentation is significantly above its absence in a cell.
AUGMENTED_ABOVE = "augmented above"
NOT_ABOVE = "not above"

# What an item reads in a learner that has no reading of it.
ABSENT = "absent"


@dataclass(frozen=True)
class Reading:
    """One item a verdict rule reads off one learner's numbers."""

    question: str
    part: str
    item: str
    value: str


@dataclass
class Measured:
    """One learner's readings in one representation, and the numbers behind them."""

    readings: list[Reading] = field(default_factory=list)
    levels: list[pd.DataFrame] = field(default_factory=list)
    tests: list[pd.DataFrame] = field(default_factory=list)

    def extend(self, other: Measured) -> None:
        """Add another experiment's readings and numbers after these."""
        self.readings += other.readings
        self.levels += other.levels
        self.tests += other.tests


def _direction(row: pd.Series) -> str:
    """A pair's direction as ``higher>other`` when significant after Holm."""
    if not bool(row["significant"]):
        return NOT_SIGNIFICANT
    higher = str(row["higher"])
    other = str(row["right"]) if higher == str(row["left"]) else str(row["left"])
    return f"{higher}>{other}"


def _pair_readings(
    tested: pd.DataFrame, question: str, part: str, prefix: str = ""
) -> list[Reading]:
    """One reading per pair of representations a test table holds."""
    return [
        Reading(
            question, part, f"{prefix}{row['left']} vs {row['right']}", _direction(row)
        )
        for _, row in tested.iterrows()
    ]


def _highest(levels: pd.Series) -> str:
    """The representation with the highest level, a tie to the first in order.

    Ties break as `measure_augmented_regression.best_variant` breaks them.
    """
    best = levels.max()
    tied = [str(name) for name, value in levels.items() if float(value) == best]
    return min(tied, key=VARIANTS.index)


def _augmentation_readings(
    tested: pd.DataFrame, part: str, item: Callable[[pd.Series], str]
) -> list[Reading]:
    """One reading per cell: whether augmentation is significantly above."""
    return [
        Reading(
            "RQ3",
            part,
            item(row),
            AUGMENTED_ABOVE
            if bool(row["significant"]) and str(row["higher"]) == TRAINING_AUGMENTED
            else NOT_ABOVE,
        )
        for _, row in tested.iterrows()
    ]


def read_regression(deployments: pd.DataFrame, labels: Sequence[str]) -> Measured:
    """RQ2's and RQ3's regression halves, from Experiment 4's cells of one set."""
    representations = compare_representations_by_scope(deployments, labels)
    known = cast(
        pd.DataFrame,
        representations[
            (representations[SCOPE_COLUMN] == SCOPE_CONTROL)
            & (representations["training"] == TRAINING_PLAIN)
        ],
    )
    best = best_variant(deployments, TRAINING_PLAIN, SCOPE_CONTROL)
    readings: list[Reading] = []
    for group in GROUPS:
        readings += _pair_readings(
            cast(pd.DataFrame, known[known["group"] == group]),
            "RQ2",
            REGRESSION,
            f"{group} ",
        )
        readings.append(Reading("RQ2", REGRESSION, f"{group} highest", best[group]))
    trainings = compare_training_conditions_by_scope(deployments, labels)
    unseen = cast(pd.DataFrame, trainings[trainings[SCOPE_COLUMN] == SCOPE_UNSEEN])
    readings += _augmentation_readings(
        unseen, REGRESSION, lambda row: f"{row['group']} x {row['feature_variant']}"
    )
    # The main table's cells (`group_board`), one row each.
    levels = [
        median_over_seeds(per_seed(deployments, scope)).assign(
            question=question, part=REGRESSION, **{SCOPE_COLUMN: scope}
        )
        for question, scope in (("RQ2", SCOPE_CONTROL), ("RQ3", SCOPE_UNSEEN))
    ]
    tests = [
        known.assign(question="RQ2", part=REGRESSION),
        unseen.assign(question="RQ3", part=REGRESSION),
    ]
    return Measured(readings, levels, tests)


def read_selection(records: pd.DataFrame) -> Measured:
    """RQ2's and RQ3's selection halves, from Experiment 5's cells of one set."""
    representations = compare_selection_representations(
        selection_runs(records, SCOPE_CONTROL), SCOPE_CONTROL
    )
    known = cast(
        pd.DataFrame, representations[representations["training"] == TRAINING_PLAIN]
    )
    levels = selection_levels(records, [SCOPE_CONTROL, SCOPE_UNSEEN])
    known_levels = cast(
        pd.Series,
        levels[
            (levels[SCOPE_COLUMN] == SCOPE_CONTROL)
            & (levels["training"] == TRAINING_PLAIN)
        ].set_index("feature_variant")["value"],
    )
    readings = _pair_readings(known, "RQ2", SELECTION)
    readings.append(Reading("RQ2", SELECTION, "highest", _highest(known_levels)))
    unseen = compare_selection_trainings(
        selection_runs(records, SCOPE_UNSEEN), SCOPE_UNSEEN
    )
    readings += _augmentation_readings(
        unseen, SELECTION, lambda row: str(row["feature_variant"])
    )
    question = levels[SCOPE_COLUMN].map({SCOPE_CONTROL: "RQ2", SCOPE_UNSEEN: "RQ3"})
    tests = [
        known.assign(question="RQ2", part=SELECTION),
        unseen.assign(question="RQ3", part=SELECTION),
    ]
    return Measured(readings, [levels.assign(question=question, part=SELECTION)], tests)


def read_predictors(levels: pd.DataFrame, tested: pd.DataFrame) -> Measured:
    """RQ1, from Experiment 1's levels and its test (`compare_predictor_sets`)."""
    row = tested.iloc[0]
    test = (
        f"{row['higher']} higher, significant"
        if bool(row["significant"])
        else NOT_SIGNIFICANT
    )
    readings = [
        Reading(
            "RQ1",
            REGRESSION,
            "both shortcut conditions",
            str(shortcut_detected(levels)),
        ),
        Reading("RQ1", REGRESSION, f"{row['left']} vs {row['right']}", test),
    ]
    return Measured(
        readings,
        [levels.assign(question="RQ1", part=REGRESSION)],
        [tested.assign(question="RQ1", part=REGRESSION)],
    )


def _differences(own: Iterable[Reading], reference: Iterable[Reading]) -> list[str]:
    """Every item one learner reads differently from the reference, in order.

    An item only one of the two reads is a difference too.
    """
    mine = {one.item: one.value for one in own}
    theirs = {one.item: one.value for one in reference}
    return [
        f"{item}: {mine.get(item, ABSENT)} (RealMLP: {theirs.get(item, ABSENT)})"
        for item in dict.fromkeys([*theirs, *mine])
        if mine.get(item) != theirs.get(item)
    ]


def _of(readings: Iterable[Reading], question: str, part: str) -> list[Reading]:
    return [one for one in readings if one.question == question and one.part == part]


def agreement(
    measured: Mapping[str, Mapping[str, Measured]], main: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The verdict table, and the long table of every reading beside the main's."""
    order = [main, *(learner for learner in measured if learner != main)]
    summary: list[dict] = []
    detail: list[dict] = []
    for learner in order:
        for feature_set, own in measured[learner].items():
            reference = measured[main][feature_set].readings
            row: dict[str, str] = {"learner": learner, "feature_set": feature_set}
            for question in QUESTIONS:
                parts = []
                for part in (REGRESSION, SELECTION):
                    read = _of(own.readings, question, part)
                    if not read:
                        continue
                    if learner == main:
                        parts.append(f"{part}: {REFERENCE}")
                        continue
                    differ = _differences(read, _of(reference, question, part))
                    parts.append(
                        f"{part}: {DISAGREE} ({'; '.join(differ)})"
                        if differ
                        else f"{part}: {AGREE}"
                    )
                row[question] = " / ".join(parts) if parts else NOT_MEASURED
            summary.append(row)
            theirs = {
                (one.question, one.part, one.item): one.value for one in reference
            }
            for one in own.readings:
                main_value = theirs.get((one.question, one.part, one.item), ABSENT)
                detail.append(
                    {
                        "learner": learner,
                        "feature_set": feature_set,
                        "question": one.question,
                        "part": one.part,
                        "item": one.item,
                        "value": one.value,
                        main: main_value,
                        "agrees": main_value == one.value,
                    }
                )
    return pd.DataFrame(summary), pd.DataFrame(detail)


def long_table(
    measured: Mapping[str, Mapping[str, Measured]], kind: str
) -> pd.DataFrame:
    """Every learner's levels or tests (``kind``), with the learner named per row."""
    return pd.concat(
        [
            table.assign(learner=learner, feature_set=feature_set)
            for learner, sets in measured.items()
            for feature_set, own in sets.items()
            for table in getattr(own, kind)
        ],
        ignore_index=True,
    )
