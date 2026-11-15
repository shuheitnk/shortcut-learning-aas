"""The two ranges of families RQ3 is reported over.

The four unseen families, and the known one alone; never all five together.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.define_reporting_scopes import (
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
    SCOPES,
    UNSEEN_FAMILIES,
    measurable_scopes,
    scope_families,
    scope_mean,
    scope_rows,
)
from aas_shortcut.define_warp_audits import AUDITS, CONTROL

FAMILIES = tuple(audit.label for audit in AUDITS)

# One value per family, far enough apart that any mix of them is a different
# number from every other mix, and the known family's not a sum of the others'.
VALUES: dict[str, float] = {
    label: 0.1 * (index + 1) + 0.0123 for index, label in enumerate(FAMILIES)
}


def per_family(families=FAMILIES) -> pd.DataFrame:
    """One cell's value per family, under two cells."""
    return pd.DataFrame(
        [
            {"cell": cell, "family": label, "value": VALUES[label] + offset}
            for cell, offset in (("a", 0.0), ("b", 1.0))
            for label in families
        ]
    )


class TestTheRanges:
    def test_the_unseen_families_are_every_audit_but_the_known_one(self):
        assert set(UNSEEN_FAMILIES) == set(FAMILIES) - {CONTROL.label}
        assert len(UNSEEN_FAMILIES) == 4

    def test_the_known_range_is_the_control_family_alone(self):
        assert scope_families(SCOPE_CONTROL) == (CONTROL.label,)
        assert CONTROL.name in SCOPE_CONTROL

    def test_no_range_reads_all_five_families(self):
        """The five-family mean is not reported."""
        assert all(len(families) < len(FAMILIES) for families in SCOPES.values())

    def test_the_main_range_comes_first(self):
        assert list(SCOPES) == [SCOPE_UNSEEN, SCOPE_CONTROL]

    def test_an_unknown_range_is_rejected(self):
        with pytest.raises(ValueError, match="unknown range"):
            scope_families("family mean")


class TestWhichRangesARunCovers:
    def test_every_family_covers_both(self):
        covered, lacking = measurable_scopes(FAMILIES)
        assert covered == list(SCOPES)
        assert lacking == {}

    def test_the_known_family_alone_covers_its_range_only(self):
        covered, lacking = measurable_scopes([CONTROL.label])
        assert covered == [SCOPE_CONTROL]
        assert lacking == {SCOPE_UNSEEN: list(UNSEEN_FAMILIES)}

    def test_three_of_the_four_unseen_families_do_not_cover_their_range(self):
        covered, lacking = measurable_scopes(FAMILIES[:-1])
        assert covered == [SCOPE_CONTROL]
        assert lacking == {SCOPE_UNSEEN: [FAMILIES[-1]]}


class TestTheReduction:
    def test_the_unseen_range_averages_its_four_families(self):
        reduced = scope_mean(per_family(), ["cell"], SCOPE_UNSEEN, "family")
        mean = sum(VALUES[label] for label in UNSEEN_FAMILIES) / len(UNSEEN_FAMILIES)
        assert reduced.set_index("cell")["value"].to_dict() == pytest.approx(
            {"a": mean, "b": mean + 1.0}
        )

    def test_the_known_range_returns_the_known_family_s_value_unchanged(self):
        reduced = scope_mean(per_family(), ["cell"], SCOPE_CONTROL, "family")
        held = reduced.set_index("cell")["value"]
        assert held["a"] == VALUES[CONTROL.label]
        assert held["b"] == VALUES[CONTROL.label] + 1.0

    def test_a_range_missing_a_family_is_rejected(self):
        with pytest.raises(ValueError, match="needs the families"):
            scope_mean(per_family(FAMILIES[:-1]), ["cell"], SCOPE_UNSEEN, "family")

    def test_the_rows_are_the_range_s_families_alone(self):
        rows = scope_rows(per_family(), SCOPE_UNSEEN, "family")
        assert set(rows["family"]) == set(UNSEEN_FAMILIES)
