"""Tests for the appendix's learner verdicts.

`rank` stands a step above the other representations on every seed, and augmentation
lifts one cell on every seed, so the tests are significant exactly there.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.compare_predictor_sets import (
    ALL_PREDICTORS,
    INVARIANT_PREDICTORS,
    LEVEL_COLUMNS,
)
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_reporting_scopes import SCOPE_COLUMN
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_selection_transfer import TRAINING_AUGMENTED
from aas_shortcut.read_learner_verdicts import (
    AGREE,
    DISAGREE,
    NOT_MEASURED,
    NOT_SIGNIFICANT,
    REFERENCE,
    REGRESSION,
    SELECTION,
    Measured,
    Reading,
    agreement,
    long_table,
    read_predictors,
    read_regression,
    read_selection,
)
from tests.test_compare_selection_runs import records
from tests.test_compare_training_runs import FAMILIES, deployment_rows

# The step `rank` stands above the others, and augmentation's lift on one cell.
STEP = 0.02
LIFT = 0.3


def wobble(seed: int) -> float:
    """The seed-to-seed spread every cell shares."""
    return 0.001 * (seed - MODEL_SEEDS[0])


def regression_level(group, variant, training, seed) -> float:
    lifted = (
        training == TRAINING_AUGMENTED and group == GROUPS[1] and variant == "value"
    )
    return (
        0.5
        + (STEP if variant == "rank" else 0.0)
        + (LIFT if lifted else 0.0)
        + (wobble(seed))
    )


def selection_level(variant, training, family, view, seed) -> float:
    lifted = training == TRAINING_AUGMENTED and variant == "concat"
    return (
        0.5
        + (STEP if variant == "rank" else 0.0)
        + (LIFT if lifted else 0.0)
        + (wobble(seed))
    )


def values(measured: Measured) -> dict[tuple[str, str, str], str]:
    return {(one.question, one.part, one.item): one.value for one in measured.readings}


@pytest.fixture(scope="module")
def regression() -> Measured:
    return read_regression(deployment_rows(regression_level), FAMILIES)


@pytest.fixture(scope="module")
def selection() -> Measured:
    return read_selection(records(selection_level))


class TestTheRegressionHalf:
    def test_rq2_reads_every_pair_of_every_group_and_its_highest(self, regression):
        """Pairs are named as the test tables order them."""
        read = values(regression)
        for group in GROUPS:
            assert read[("RQ2", REGRESSION, f"{group} rank vs value")] == "rank>value"
            assert read[("RQ2", REGRESSION, f"{group} concat vs rank")] == (
                "rank>concat"
            )
            assert read[("RQ2", REGRESSION, f"{group} concat vs value")] == (
                NOT_SIGNIFICANT
            )
            assert read[("RQ2", REGRESSION, f"{group} highest")] == "rank"

    def test_rq3_reads_the_nine_cells_on_the_unseen_families(self, regression):
        read = {
            key[2]: value
            for key, value in values(regression).items()
            if key[0] == "RQ3"
        }
        assert len(read) == 9
        assert {item for item, value in read.items() if value == "augmented above"} == {
            f"{GROUPS[1]} x value"
        }

    def test_the_numbers_behind_them_travel_with_them(self, regression):
        levels = pd.concat(regression.levels)
        assert set(levels["question"]) == {"RQ2", "RQ3"}
        assert set(levels[SCOPE_COLUMN]) == {"spline only", "unseen-family mean"}
        tests = pd.concat(regression.tests)
        # The known family's plain pairs, and the unseen range's nine cells.
        assert (tests["question"] == "RQ2").sum() == 3 * len(GROUPS)
        assert (tests["question"] == "RQ3").sum() == 9


class TestTheSelectionHalf:
    def test_rq2_reads_every_pair_and_the_highest(self, selection):
        read = values(selection)
        assert read[("RQ2", SELECTION, "rank vs value")] == "rank>value"
        assert read[("RQ2", SELECTION, "concat vs value")] == NOT_SIGNIFICANT
        assert read[("RQ2", SELECTION, "highest")] == "rank"

    def test_rq3_reads_the_three_cells(self, selection):
        read = {
            key[2]: value for key, value in values(selection).items() if key[0] == "RQ3"
        }
        assert read == {
            "value": "not above",
            "rank": "not above",
            "concat": "augmented above",
        }


class TestTheFirstQuestion:
    @staticmethod
    def levels(all_post: float) -> pd.DataFrame:
        pre, post = (LEVEL_COLUMNS[TEST_METRICS[view]] for view in ("pre", "post"))
        return pd.DataFrame(
            [
                {"predictors": ALL_PREDICTORS, pre: 0.9, post: all_post},
                {"predictors": INVARIANT_PREDICTORS, pre: 0.5, post: 0.45},
            ]
        )

    @staticmethod
    def comparison(significant: bool) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "left": INVARIANT_PREDICTORS,
                    "right": ALL_PREDICTORS,
                    "higher": INVARIANT_PREDICTORS,
                    "significant": significant,
                }
            ]
        )

    def test_both_conditions_and_the_test_are_read(self):
        read = values(read_predictors(self.levels(0.1), self.comparison(True)))
        assert read == {
            ("RQ1", REGRESSION, "both shortcut conditions"): "True",
            ("RQ1", REGRESSION, "invariant vs all"): "invariant higher, significant",
        }

    def test_a_shortcut_missing_one_condition_is_read_as_none(self):
        """`all` still above after the warp: the second condition fails."""
        read = values(read_predictors(self.levels(0.6), self.comparison(False)))
        assert read[("RQ1", REGRESSION, "both shortcut conditions")] == "False"
        assert read[("RQ1", REGRESSION, "invariant vs all")] == NOT_SIGNIFICANT


def measured(rq2_highest: str, extra: tuple[Reading, ...] = ()) -> dict:
    """One learner's readings in both sets: RQ1 in ELA only."""
    ela = Measured(
        [
            Reading("RQ1", REGRESSION, "both shortcut conditions", "True"),
            Reading("RQ2", REGRESSION, "G1 highest", rq2_highest),
            Reading("RQ2", SELECTION, "highest", "rank"),
            Reading("RQ3", REGRESSION, "G1 x value", "augmented above"),
            *extra,
        ]
    )
    deep = Measured([Reading("RQ2", SELECTION, "highest", "rank")])
    return {"ela": ela, "deep-ela": deep}


class TestTheTable:
    def test_the_main_learner_is_the_reference_and_comes_first(self):
        summary, _ = agreement(
            {"xgboost": measured("rank"), "realmlp": measured("rank")}, "realmlp"
        )
        assert list(summary["learner"]) == ["realmlp", "realmlp", "xgboost", "xgboost"]
        first = summary.iloc[0]
        assert first["RQ1"] == f"{REGRESSION}: {REFERENCE}"
        assert first["RQ2"] == f"{REGRESSION}: {REFERENCE} / {SELECTION}: {REFERENCE}"

    def test_reading_every_item_alike_is_agreement(self):
        summary, detail = agreement(
            {"realmlp": measured("rank"), "xgboost": measured("rank")}, "realmlp"
        )
        row = summary[
            (summary["learner"] == "xgboost") & (summary["feature_set"] == "ela")
        ]
        assert row.iloc[0]["RQ2"] == f"{REGRESSION}: {AGREE} / {SELECTION}: {AGREE}"
        assert detail["agrees"].all()

    def test_a_disagreement_names_the_item_and_both_readings(self):
        summary, detail = agreement(
            {"realmlp": measured("rank"), "xgboost": measured("value")}, "realmlp"
        )
        row = summary[
            (summary["learner"] == "xgboost") & (summary["feature_set"] == "ela")
        ]
        assert row.iloc[0]["RQ2"] == (
            f"{REGRESSION}: {DISAGREE} (G1 highest: value (RealMLP: rank)) / "
            f"{SELECTION}: {AGREE}"
        )
        assert row.iloc[0]["RQ1"] == f"{REGRESSION}: {AGREE}"
        differs = detail[~detail["agrees"]]
        assert list(differs["item"]) == ["G1 highest"]
        assert list(differs["realmlp"]) == ["rank"]

    def test_an_item_only_one_learner_reads_is_a_difference(self):
        extra = (Reading("RQ3", REGRESSION, "G2 x rank", "not above"),)
        summary, _ = agreement(
            {"realmlp": measured("rank"), "xgboost": measured("rank", extra)},
            "realmlp",
        )
        row = summary[
            (summary["learner"] == "xgboost") & (summary["feature_set"] == "ela")
        ]
        assert "G2 x rank: not above (RealMLP: absent)" in row.iloc[0]["RQ3"]

    def test_deep_ela_has_no_first_question(self):
        """Experiment 1 is read in ELA only."""
        summary, _ = agreement(
            {"realmlp": measured("rank"), "xgboost": measured("rank")}, "realmlp"
        )
        deep = summary[summary["feature_set"] == "deep-ela"]
        assert set(deep["RQ1"]) == {NOT_MEASURED}
        assert set(deep["RQ3"]) == {NOT_MEASURED}


class TestTheLongTables:
    def test_every_table_is_named_by_its_learner_and_set(self):
        one = Measured(levels=[pd.DataFrame({"value": [0.1]})])
        two = Measured(levels=[pd.DataFrame({"value": [0.2]})])
        table = long_table({"realmlp": {"ela": one}, "xgboost": {"ela": two}}, "levels")
        assert list(table["learner"]) == ["realmlp", "xgboost"]
        assert list(table["feature_set"]) == ["ela", "ela"]
        assert list(table["value"]) == [0.1, 0.2]
