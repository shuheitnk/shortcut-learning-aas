"""Benchmark problem construction."""

from __future__ import annotations

import ioh

from aas_shortcut.define_paths import validate_problem_class

# Highest function id of the BBOB suite.
MAX_BBOB_FUNCTION_ID = 24


def create_problem(function_id: int, instance_id: int, dim: int):
    """Instantiate one BBOB problem, ``function_id`` in 1-24."""
    if not 1 <= function_id <= MAX_BBOB_FUNCTION_ID:
        raise ValueError(f"Unknown function id: {function_id}")
    return ioh.get_problem(
        function_id,
        instance=instance_id,
        dimension=dim,
        problem_class=ioh.ProblemClass.BBOB,
    )


def resolve_problem_grid(
    problem_class: str,
    max_functions: int,
    max_instances: int,
) -> tuple[range, range]:
    """Return the ``(functions, instances)`` ranges to sweep for a suite."""
    validate_problem_class(problem_class)
    if not 1 <= max_functions <= MAX_BBOB_FUNCTION_ID:
        raise ValueError(
            f"max_functions must be within 1..{MAX_BBOB_FUNCTION_ID}, "
            f"got {max_functions}"
        )
    if max_instances < 1:
        raise ValueError(f"max_instances must be >= 1, got {max_instances}")
    return range(1, max_functions + 1), range(1, max_instances + 1)
