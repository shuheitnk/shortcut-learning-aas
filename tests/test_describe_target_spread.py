"""The spread of the regression's target on a block."""

from __future__ import annotations

import numpy as np
import pytest

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.describe_target_spread import member_means, target_spread
from tests.test_measure_representation_transfer import G3, make_view_superiorities


def test_the_variance_is_the_r2_denominator_over_n():
    rows = make_view_superiorities(n_instances=10)
    spread = target_spread(rows, "spline/warpheld")
    assert len(spread) == len(FOCUS_PAIRS) * 2
    g3 = rows[(rows["pair"] == G3.slug) & (rows["transform"] == "post")]["superiority"]
    got = spread[(spread["group"] == "G3") & (spread["transform"] == "post")]
    assert got["variance"].iloc[0] == pytest.approx(np.var(g3.to_numpy()))
    assert got["n"].iloc[0] == 10


def test_the_second_member_s_mean_is_the_complement():
    rows = make_view_superiorities(n_instances=10)
    means = member_means(rows, "spline/warpheld", G3.slug)
    pre = means[means["transform"] == "pre"].set_index("algorithm")["mean"]
    assert pre[G3.first] + pre[G3.second] == pytest.approx(1.0)
