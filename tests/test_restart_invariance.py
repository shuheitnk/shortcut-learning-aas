r"""Strict-monotone invariance of the restart criteria, at two strengths.

**Criterion invariance (every member).** Each :math:`C_A` reads the search state's
geometry or the evaluation counter, never an objective value, so at the same state it
agrees under :math:`f` and :math:`g \circ f` for any strictly increasing :math:`g`.

**Behavioural invariance (G1 only).** Rank-based searchers' states also coincide, so
the whole restart sequence is invariant. G2 and G3 trajectories move by design.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from aas_shortcut.create_problem import create_problem
from aas_shortcut.define_algorithm_portfolio import GROUP_RANK_SEARCH, PORTFOLIO
from aas_shortcut.optimizers.define_optimizer_base import BudgetedObjective
from aas_shortcut.sample_monotone_transform import transform_for_instance
from tests.define_restart_mechanism import MECHANISMS

DIM = 5
BUDGET = 1500
LB, UB = np.full(DIM, -5.0), np.full(DIM, 5.0)

# Each map must be a function of ``y`` alone; one whose parameters are read off the
# batch would silently test nothing.
MONOTONE_MAPS = {
    "affine": lambda y: 3.0 * y + 17.0,
    "cube_root": np.cbrt,
    "arcsinh": np.arcsinh,
    "arctan": np.arctan,
    "steep_affine": lambda y: 1e6 * y - 4.2e5,
}


class TestCriterionReadsNoObjectiveValue:
    """The structural guarantee: the criteria are functions of geometry alone."""

    def test_nelder_mead_criterion_is_the_simplex_alone(self):
        from aas_shortcut.optimizers import build_nelder_mead as nm

        simplex = np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]])
        lb, ub = np.full(2, -5.0), np.full(2, 5.0)
        # There is no argument through which an objective value could enter.
        assert "func" not in nm.capability_lost.__code__.co_varnames
        assert nm.capability_lost(simplex, lb, ub, 1.0)
        assert not nm.capability_lost(simplex, lb, ub, 1e-6)

    def test_shade_criterion_is_population_and_archive_alone(self):
        from aas_shortcut.optimizers import build_differential_evolution as de

        lb, ub = np.full(2, -5.0), np.full(2, 5.0)
        population = SimpleNamespace(x=np.array([[0.0, 1.0], [0.0, 0.0]]))
        state = cast(Any, SimpleNamespace(population=population, archive=None))
        assert de.capability_lost(state, lb, ub, 0.5)
        assert not de.capability_lost(state, lb, ub, 0.05)

    def test_cmaes_criterion_is_sigma_and_the_covariance_alone(self):
        from aas_shortcut.optimizers import build_cmaes as cmaes

        state = cast(
            Any,
            SimpleNamespace(
                p=SimpleNamespace(
                    mutation=SimpleNamespace(sigma=0.5),
                    adaptation=SimpleNamespace(d=np.array([0.1, 0.4, 0.2])),
                )
            ),
        )
        # sigma * max(d) / width = 0.5 * 0.4 / 10
        assert cmaes.search_capability(state, 10.0) == pytest.approx(0.02)


@pytest.mark.slow
class TestBehaviouralInvariance:
    """G1: the whole restart sequence survives a strictly increasing transform."""

    @staticmethod
    def _run(spec, function_id: int, seed: int, transform):
        problem = create_problem(function_id, 1, DIM)
        evaluated: list[float] = []

        def objective(x):
            raw = float(problem(x))
            evaluated.append(raw)
            return raw if transform is None else float(transform(np.array([raw]))[0])

        wrapped = BudgetedObjective(objective, BUDGET, bounds=(LB, UB))
        report = spec.runner(wrapped, DIM, seed, BUDGET, LB, UB)
        return report, np.asarray(evaluated)

    @pytest.mark.parametrize("function_id", [1, 8, 21])
    @pytest.mark.parametrize("name", sorted(MONOTONE_MAPS))
    def test_rank_based_search_is_untouched(self, function_id: int, name: str):
        for spec in PORTFOLIO:
            if spec.group != GROUP_RANK_SEARCH:
                continue
            plain, raw_plain = self._run(spec, function_id, 7, None)
            moved, raw_moved = self._run(spec, function_id, 7, MONOTONE_MAPS[name])

            assert plain.restarts == moved.restarts, (
                f"{spec.name} restarted differently under {name} on f{function_id}"
            )
            assert plain.multistarts == moved.multistarts
            # The same points, in the same order: the transform changed nothing.
            assert np.array_equal(raw_plain, raw_moved)

    @pytest.mark.parametrize("function_id", [1, 8, 21])
    def test_the_studys_own_transform_is_covered(self, function_id: int):
        """`pre` vs `post`: the per-instance map the experiments apply."""
        warp = transform_for_instance(42, "bbob", function_id, 1, DIM)
        for spec in PORTFOLIO:
            if spec.group != GROUP_RANK_SEARCH:
                continue
            plain, raw_plain = self._run(spec, function_id, 11, None)
            moved, raw_moved = self._run(spec, function_id, 11, warp)
            assert plain.restarts == moved.restarts
            assert np.array_equal(raw_plain, raw_moved)


class TestSpecificationAndCodeAgree:
    """The settings document describes the restart rules the code runs.

    Cheap structural checks that catch a criterion described one way and implemented
    another.
    """

    @staticmethod
    def _spec() -> str:
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / "docs" / "SETTINGS.md"
        return path.read_text(encoding="utf-8")

    def test_every_member_appears_in_the_restart_table(self):
        spec = self._spec()
        table = spec[spec.index("### 5.3 Restart rule") : spec.index("### 5.4")]
        for name in MECHANISMS:
            assert name in table, f"{name} is not in docs/SETTINGS.md's restart table"

    def test_the_document_does_not_claim_one_rule_for_every_member(self):
        """Each member restarts on its own search scale: one threshold, not one rule."""
        spec = self._spec().lower()
        for forbidden in (
            "the same restart rule",
            "one restart rule for every member",
            "a single restart rule",
            "an identical restart rule",
        ):
            assert forbidden not in spec
