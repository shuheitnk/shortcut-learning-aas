"""The comparisons the appendix's bootstrap reports, and the levels they compare.

A side is one reported level: a regression cell's R² or a selection cell's Θ, on one
view, over one range of families. A comparison is the left level minus the right
one. Representations share their seeds (the paper pairs them by seed); training
conditions draw theirs apart (the paper does not pair them).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from aas_shortcut.compare_predictor_sets import ALL_PREDICTORS, INVARIANT_PREDICTORS
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_paths import DEEP_ELA_FEATURE_SET, ELA_FEATURE_SET
from aas_shortcut.define_reporting_scopes import SCOPE_CONTROL, SCOPE_UNSEEN
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
)

REGRESSION = "regression"
SELECTION = "selection"

# The selection is reported over the whole portfolio, not a group.
MIXED = "Mixed"

PRE, POST = "pre", "post"


@dataclass(frozen=True)
class Side:
    """One reported level."""

    part: str
    feature_set: str
    group: str
    variant: str
    training: str
    scope: str
    view: str
    predictors: str = ALL_PREDICTORS


@dataclass(frozen=True)
class Comparison:
    """A row of the table: the left level minus the right one."""

    question: str
    left: Side
    right: Side

    @property
    def paired(self) -> bool:
        """Whether both sides read the same drawn seeds."""
        return self.left.training == self.right.training

    @property
    def label(self) -> str:
        """The difference in the words of the paper's tables."""
        left, right = self.left, self.right
        if left.predictors != right.predictors:
            return f"{left.predictors} - {right.predictors}"
        if left.training != right.training:
            return f"{left.training} - {right.training}"
        if left.view != right.view:
            return f"{left.view} - {right.view}"
        return f"{left.variant} - {right.variant}"


def _pairs(
    question: str,
    part: str,
    feature_set: str,
    group: str,
    training: str,
    scope: str,
    view: str,
    differences: tuple[tuple[str, str], ...],
) -> list[Comparison]:
    return [
        Comparison(
            question,
            Side(part, feature_set, group, left, training, scope, view),
            Side(part, feature_set, group, right, training, scope, view),
        )
        for left, right in differences
    ]


def comparisons(
    feature_sets: tuple[str, ...] = (ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET),
) -> list[Comparison]:
    """Every row of the appendix's table, in the order it is written."""
    rows: list[Comparison] = []
    if ELA_FEATURE_SET in feature_sets:
        invariant = Side(
            REGRESSION,
            ELA_FEATURE_SET,
            GROUPS[0],
            "value",
            TRAINING_PLAIN,
            SCOPE_CONTROL,
            POST,
            INVARIANT_PREDICTORS,
        )
        every = replace(invariant, predictors=ALL_PREDICTORS)
        rows.append(Comparison("RQ1", invariant, every))
        rows += [
            Comparison("RQ1", replace(side, view=PRE), side)
            for side in (every, invariant)
        ]
        rows.append(
            Comparison("RQ1", replace(every, view=PRE), replace(invariant, view=PRE))
        )
    for feature_set in feature_sets:
        for group in GROUPS:
            rows += _pairs(
                "RQ2",
                REGRESSION,
                feature_set,
                group,
                TRAINING_PLAIN,
                SCOPE_CONTROL,
                POST,
                (("rank", "value"), ("rank", "concat")),
            )
            rows += _pairs(
                "RQ2",
                REGRESSION,
                feature_set,
                group,
                TRAINING_PLAIN,
                SCOPE_CONTROL,
                PRE,
                (("rank", "value"),),
            )
        rows += [
            Comparison(
                "RQ2",
                Side(
                    REGRESSION,
                    feature_set,
                    group,
                    variant,
                    TRAINING_PLAIN,
                    SCOPE_CONTROL,
                    PRE,
                ),
                Side(
                    REGRESSION,
                    feature_set,
                    group,
                    variant,
                    TRAINING_PLAIN,
                    SCOPE_CONTROL,
                    POST,
                ),
            )
            for group in GROUPS[:2]
            for variant in ("value", "rank", "concat")
        ]
        rows += _pairs(
            "RQ2",
            SELECTION,
            feature_set,
            MIXED,
            TRAINING_PLAIN,
            SCOPE_CONTROL,
            POST,
            (("rank", "value"), ("rank", "concat")),
        )
        for view in (POST, PRE):
            for group in GROUPS:
                for variant in ("value", "rank", "concat"):
                    rows.append(
                        Comparison(
                            "RQ3",
                            Side(
                                REGRESSION,
                                feature_set,
                                group,
                                variant,
                                TRAINING_AUGMENTED,
                                SCOPE_UNSEEN,
                                view,
                            ),
                            Side(
                                REGRESSION,
                                feature_set,
                                group,
                                variant,
                                TRAINING_PLAIN,
                                SCOPE_UNSEEN,
                                view,
                            ),
                        )
                    )
        for variant in ("value", "rank", "concat"):
            rows.append(
                Comparison(
                    "RQ3",
                    Side(
                        SELECTION,
                        feature_set,
                        MIXED,
                        variant,
                        TRAINING_AUGMENTED,
                        SCOPE_UNSEEN,
                        POST,
                    ),
                    Side(
                        SELECTION,
                        feature_set,
                        MIXED,
                        variant,
                        TRAINING_PLAIN,
                        SCOPE_UNSEEN,
                        POST,
                    ),
                )
            )
        after = (("concat", "rank"), ("concat", "value"), ("rank", "value"))
        for group in GROUPS:
            rows += _pairs(
                "RQ3",
                REGRESSION,
                feature_set,
                group,
                TRAINING_AUGMENTED,
                SCOPE_UNSEEN,
                POST,
                after,
            )
        rows += _pairs(
            "RQ3",
            SELECTION,
            feature_set,
            MIXED,
            TRAINING_AUGMENTED,
            SCOPE_UNSEEN,
            POST,
            after,
        )
    return rows
