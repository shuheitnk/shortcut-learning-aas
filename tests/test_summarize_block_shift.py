"""The preanalysis's shift of the target, measured on any block."""

from __future__ import annotations

from aas_shortcut.summarize_block_shift import superiority_shift
from tests.test_measure_representation_transfer import G1, G3, make_view_superiorities


def test_only_the_moved_pair_shifts():
    rows = make_view_superiorities(n_instances=10).assign(dimension=10)
    shift = superiority_shift(rows, "spline/warpheld").set_index("quantity")
    assert shift.loc[G1.slug, "value_raw"] == 0.0
    assert shift.loc[G1.slug, "p_moved"] == 0.0
    assert shift.loc[G3.slug, "value_raw"] > 0.0
    assert set(shift["audit"]) == {"spline/warpheld"}
