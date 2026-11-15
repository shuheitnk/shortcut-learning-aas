"""The shared sample of points every feature condition is computed from."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from aas_shortcut.create_problem import create_problem
from aas_shortcut.define_paths import (
    validate_encoding,
    validate_problem_class,
    validate_transform,
)
from aas_shortcut.derive_seeds import derive_seed
from aas_shortcut.sample_monotone_transform import (
    DEFAULT_WARP,
    WarpConfig,
    transform_for_instance,
)


def sample_seed(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    rep: int,
) -> int:
    """Derive the sample seed of one ``(instance, repetition)``."""
    validate_problem_class(problem_class)
    # `"doe"` is the seed's domain token; changing it changes every sample.
    return derive_seed(
        "doe", base_seed, problem_class, function_id, instance_id, dim, rep
    )


def latin_hypercube(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Draw an ``n x dim`` Latin-hypercube sample in the unit cube."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")

    edges = np.linspace(0.0, 1.0, n + 1)
    lower = edges[:n, None]
    upper = edges[1:, None]
    points = lower + rng.random((n, dim)) * (upper - lower)
    for axis in range(dim):
        points[:, axis] = points[rng.permutation(n), axis]
    return points


def normalize(y: np.ndarray) -> np.ndarray:
    """Scale values to ``[0, 1]``, mapping a constant input to all zeros."""
    y = np.asarray(y, dtype=np.float64)
    y_range = y.max() - y.min()
    if y_range == 0:
        return np.zeros_like(y)
    return (y - y.min()) / y_range


@dataclass(frozen=True)
class SampledPoints:
    """One sample of one instance, evaluated under both objective views."""

    problem_class: str
    function_id: int
    instance_id: int
    dimension: int
    rep: int
    sample_seed: int
    # Sample points in the unit cube, handed to the feature calculators.
    X_unit: np.ndarray
    # Raw objective values.
    y_pre: np.ndarray
    # Transformed objective values.
    y_post: np.ndarray
    optimum_y: float

    @property
    def sample_size(self) -> int:
        """Number of sampled points."""
        return int(self.X_unit.shape[0])

    def objective(self, transform: str) -> np.ndarray:
        """Return the objective values under ``pre`` or ``post``."""
        return self.y_post if validate_transform(transform) == "post" else self.y_pre

    def ranks_agree(self) -> bool:
        """Whether ``pre`` and ``post`` induce exactly the same ranking."""
        return bool(
            np.array_equal(
                rankdata(self.y_pre, method="average"),
                rankdata(self.y_post, method="average"),
            )
        )


def create_sampled_points(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    rep: int,
    sample_size: int,
    warp: WarpConfig = DEFAULT_WARP,
) -> SampledPoints:
    """Draw and evaluate the shared sample of one ``(instance, repetition)``."""
    problem = create_problem(function_id, instance_id, dim)
    lb = np.asarray(problem.bounds.lb, dtype=np.float64)
    ub = np.asarray(problem.bounds.ub, dtype=np.float64)

    seed = sample_seed(base_seed, problem_class, function_id, instance_id, dim, rep)
    rng = np.random.default_rng(seed)
    X_unit = latin_hypercube(sample_size, dim, rng)
    X_scaled = X_unit * (ub - lb) + lb

    y_pre = np.array([problem(x) for x in X_scaled], dtype=np.float64)
    optimum_y = float(problem.optimum.y)
    # One warp per instance, shared by all its repetitions.
    transform = transform_for_instance(
        base_seed, problem_class, function_id, instance_id, dim, warp
    )
    y_post = transform(y_pre)

    return SampledPoints(
        problem_class=problem_class,
        function_id=int(function_id),
        instance_id=int(instance_id),
        dimension=int(dim),
        rep=int(rep),
        sample_seed=int(seed),
        X_unit=X_unit,
        y_pre=y_pre,
        y_post=y_post,
        optimum_y=optimum_y,
    )


def ela_inputs(
    sample: SampledPoints, transform: str, encoding: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the ``(X, y)`` pair the feature calculators consume."""
    validate_encoding(encoding)
    y = sample.objective(transform)
    if encoding == "rank":
        y = rankdata(y, method="average")

    X_df = pd.DataFrame(
        sample.X_unit, columns=[f"x{i + 1}" for i in range(sample.dimension)]
    )
    return X_df, pd.Series(normalize(y), name="y")
