"""The rows of the appendix's bootstrap table."""

from __future__ import annotations

from aas_shortcut.define_bootstrap_rows import MIXED, SELECTION, comparisons
from aas_shortcut.define_reporting_scopes import SCOPE_CONTROL, SCOPE_UNSEEN


def test_rq1_is_measured_on_ela_alone():
    rq1 = [row for row in comparisons() if row.question == "RQ1"]
    assert [row.label for row in rq1] == [
        "invariant - all",
        "pre - post",
        "pre - post",
        "all - invariant",
    ]
    assert [row.left.predictors for row in rq1[1:3]] == ["all", "invariant"]
    assert not [row for row in comparisons(("deep-ela",)) if row.question == "RQ1"]


def test_representations_share_seeds_and_trainings_draw_them_apart():
    for row in comparisons():
        same = row.left.training == row.right.training
        assert row.paired is same
        assert (row.label == "augmented - plain") is not same


def test_rq2_reads_the_known_family_and_rq3_the_unseen_mean():
    for row in comparisons():
        expected = SCOPE_CONTROL if row.question in ("RQ1", "RQ2") else SCOPE_UNSEEN
        assert row.left.scope == row.right.scope == expected


def test_the_selection_rows_are_the_mixed_portfolio_on_the_transformed_block():
    rows = [row for row in comparisons() if row.left.part == SELECTION]
    assert {row.left.group for row in rows} == {MIXED}
    assert {row.left.view for row in rows} == {"post"}
    # Per feature set: two in RQ2, three augmentation rows and three after it.
    assert len(rows) == 2 * (2 + 3 + 3)


def test_the_row_count_per_feature_set():
    rows = comparisons(("deep-ela",))
    # RQ2 regression 3 x 3, selection 2; RQ3 regression 2 x 3 x 3, selection 3;
    # after augmentation 3 x 3 + 3; untransformed minus transformed, G1 and G2 x 3.
    assert len(rows) == 9 + 2 + 18 + 3 + 12 + 6


def test_a_difference_between_views_shares_the_seeds():
    rows = [row for row in comparisons() if row.label == "pre - post"]
    assert rows and all(row.paired for row in rows)
    assert {(row.left.view, row.right.view) for row in rows} == {("pre", "post")}
