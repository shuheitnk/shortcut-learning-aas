"""Benchmark problem construction and the sweep grid."""

from __future__ import annotations

import pytest

from aas_shortcut.create_problem import (
    MAX_BBOB_FUNCTION_ID,
    create_problem,
    resolve_problem_grid,
)


class TestCreateProblem:
    @pytest.mark.parametrize("function_id", [1, 12, MAX_BBOB_FUNCTION_ID])
    def test_bbob_ids_build_a_problem(self, function_id: int) -> None:
        problem = create_problem(function_id, instance_id=1, dim=2)
        assert problem.meta_data.n_variables == 2

    def test_instances_of_one_function_differ(self) -> None:
        first = create_problem(1, instance_id=1, dim=2)
        second = create_problem(1, instance_id=2, dim=2)
        assert first.optimum.y != second.optimum.y

    def test_problem_is_evaluable_and_bounded(self) -> None:
        problem = create_problem(1, instance_id=1, dim=2)
        assert isinstance(float(problem([0.0, 0.0])), float)
        assert len(problem.bounds.lb) == 2

    @pytest.mark.parametrize("function_id", [0, -1, 25, 100])
    def test_unknown_function_id_raises(self, function_id: int) -> None:
        with pytest.raises(ValueError, match="Unknown function id"):
            create_problem(function_id, instance_id=1, dim=2)


class TestResolveProblemGrid:
    def test_bbob_grid_is_one_based_and_inclusive(self) -> None:
        functions, instances = resolve_problem_grid("bbob", 24, 10)
        assert list(functions) == list(range(1, 25))
        assert list(instances) == list(range(1, 11))

    def test_bbob_grid_respects_smaller_limits(self) -> None:
        functions, instances = resolve_problem_grid("bbob", 3, 2)
        assert list(functions) == [1, 2, 3]
        assert list(instances) == [1, 2]

    def test_unknown_problem_class_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown problem class"):
            resolve_problem_grid("cec", 24, 10)

    @pytest.mark.parametrize("max_functions", [0, 25, -1])
    def test_out_of_range_function_limit_raises(self, max_functions: int) -> None:
        with pytest.raises(ValueError, match="max_functions"):
            resolve_problem_grid("bbob", max_functions, 10)

    def test_non_positive_instance_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="max_instances"):
            resolve_problem_grid("bbob", 24, 0)
