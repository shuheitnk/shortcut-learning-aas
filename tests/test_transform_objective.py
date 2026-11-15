"""Tests for an instance read through its own monotone map, as the ``post`` view does.

The map moves the objective but not its ranking, and the wrapped problem still
answers for the search space and the optimum.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from aas_shortcut.create_problem import create_problem
from aas_shortcut.sample_monotone_transform import transform_for_instance
from aas_shortcut.transform_objective import FuncTransform

DIM = 5
COORDINATES = (8, 1, DIM)


def warped(function_id: int, instance_id: int, dim: int) -> FuncTransform:
    """One instance read through its map, as the ``post`` view builds it."""
    return FuncTransform(
        create_problem(function_id, instance_id, dim),
        transform_for_instance(42, "bbob", function_id, instance_id, dim),
    )


@pytest.fixture(scope="module")
def sample():
    """Draw points to compare the two views of an instance on."""
    rng = np.random.default_rng(0)
    plain = create_problem(*COORDINATES)
    lb = np.asarray(plain.bounds.lb, dtype=np.float64)
    ub = np.asarray(plain.bounds.ub, dtype=np.float64)
    return rng.uniform(lb, ub, size=(200, DIM))


class TestReparametrisation:
    def test_the_ranking_of_the_objective_is_preserved_exactly(self, sample):
        """The whole claim: a comparison-based optimizer cannot see the map."""
        plain = create_problem(*COORDINATES)
        view = warped(*COORDINATES)
        assert np.array_equal(
            rankdata([plain(x) for x in sample], method="average"),
            rankdata([view(x) for x in sample], method="average"),
        )

    def test_the_objective_is_actually_moved(self, sample):
        """Otherwise the view would be the instance under another name."""
        plain = create_problem(*COORDINATES)
        view = warped(*COORDINATES)
        assert not np.allclose([plain(x) for x in sample], [view(x) for x in sample])

    def test_the_search_space_is_untouched(self):
        plain = create_problem(*COORDINATES)
        view = warped(*COORDINATES)
        assert np.array_equal(view.bounds.lb, plain.bounds.lb)
        assert np.array_equal(view.bounds.ub, plain.bounds.ub)

    def test_the_optimum_is_reported_as_the_instance_reports_it(self):
        """The optimizer's target is the same point of the same problem."""
        plain = create_problem(*COORDINATES)
        view = warped(*COORDINATES)
        assert np.array_equal(view.optimum.x, plain.optimum.x)
