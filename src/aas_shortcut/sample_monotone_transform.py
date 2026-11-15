"""The per-instance random strictly increasing objective transformation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, PchipInterpolator

from aas_shortcut.create_problem import create_problem
from aas_shortcut.derive_seeds import derive_seed

# Identifier of the transformation, recorded in every stage fingerprint so that
# changing how the map is drawn invalidates the artifacts drawn the old way.
WARP_ID = "spline-warp(minmax(f; f_min, f_max_hat)); identity above 1"

# Uniform draws the maximum is estimated from.
MAX_ESTIMATE_SAMPLE_SIZE = 10_000

# Half-open range of the spline's control-point count. Few knots keep the warp a
# plausible reparametrisation rather than noise.
SPLINE_KNOTS = (5, 9)

# Fewest control points a spline draw is defined for: `CubicSpline` needs three,
# and a half-open range needs its ends apart.
MINIMUM_KNOTS = 3
# Default amplitude ``alpha``: every family reads it as the spread of its own
# log-scale parameter and collapses to the identity at ``alpha = 0``.
KNOT_SCALE = 1.0
# Points the warp is built on; fine enough that the interpolant reproduces the
# integral well below float32 resolution.
GRID_SIZE = 2001
# Interior breakpoints of a piecewise-linear draw, in the range of SPLINE_KNOTS so
# the two families differ in smoothness, not in degrees of freedom.
PIECEWISE_BREAKS = (4, 8)

# PCHIP through strictly increasing knots is strictly increasing; a family whose
# point is a corner is read linearly instead.
PCHIP_INTERPOLATION = "pchip"
LINEAR_INTERPOLATION = "linear"
WARP_INTERPOLATIONS = (PCHIP_INTERPOLATION, LINEAR_INTERPOLATION)

# The normalised integral of ``exp`` of a random cubic spline: smooth, with a
# bounded derivative.
SPLINE_FAMILY = "spline"
# ``z ** p``: for ``p < 1`` the derivative is unbounded at the optimum, which no
# spline draw can reproduce.
POWER_FAMILY = "power"
# Random slopes on random intervals: not differentiable at the breakpoints.
PIECEWISE_FAMILY = "piecewise-linear"
WARP_FAMILIES = (SPLINE_FAMILY, POWER_FAMILY, PIECEWISE_FAMILY)

# Seed domain of the training augmentation; changing the token would move every
# instance's map.
AUGMENTATION_DOMAIN = "warp"
# Seed domain of the held-out audit, independent of the augmentation's draws.
HELD_OUT_DOMAIN = "warpheld"
WARP_DOMAINS = (AUGMENTATION_DOMAIN, HELD_OUT_DOMAIN)


@dataclass(frozen=True)
class WarpConfig:
    """Which transformation family is drawn, how strongly, and from which seeds."""

    family: str = SPLINE_FAMILY
    # ``0`` is the identity in every family, a control: the remaining min-max
    # normalisation is invisible to features computed on per-sample scaled values.
    alpha: float = KNOT_SCALE
    domain: str = AUGMENTATION_DOMAIN
    # Only the spline family reads it; another family rejects a non-default value.
    knots: tuple[int, int] = SPLINE_KNOTS
    # Repetition index; it enters the curve's seed only, so repetitions share the
    # maximum estimate that normalises the objective.
    draw: int = 0

    def __post_init__(self) -> None:
        """Reject a setting no draw is defined for."""
        if self.family not in WARP_FAMILIES:
            raise ValueError(
                f"unknown warp family {self.family!r}, expected one of {WARP_FAMILIES}"
            )
        if self.domain not in WARP_DOMAINS:
            raise ValueError(
                f"unknown seed domain {self.domain!r}, expected one of {WARP_DOMAINS}"
            )
        if not self.alpha >= 0.0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.draw < 0:
            raise ValueError(f"the repetition index must be >= 0, got {self.draw}")
        if len(self.knots) != 2:
            raise ValueError(
                f"knots is a (low, high) range, got {len(self.knots)} value(s)"
            )
        low, high = self.knots
        if not MINIMUM_KNOTS <= low < high:
            raise ValueError(
                f"knots must satisfy {MINIMUM_KNOTS} <= low < high, got {self.knots}"
            )
        if self.knots != SPLINE_KNOTS and self.family != SPLINE_FAMILY:
            raise ValueError(
                f"only the {SPLINE_FAMILY!r} family draws control points; "
                f"{self.family!r} would ignore knots={self.knots}"
            )

    @property
    def identifier(self) -> str:
        """Return the string the fingerprint of a stage records this draw under."""
        if self == DEFAULT_WARP:
            return WARP_ID
        text = (
            f"{self.family}-warp(minmax(f; f_min, f_max_hat)); identity above 1; "
            f"alpha={self.alpha:g}; domain={self.domain}"
        )
        # Each part is appended only off its default, so default draws keep their ID.
        if self.knots != SPLINE_KNOTS:
            text += f"; knots={self.knots[0]}-{self.knots[1]}"
        if self.draw:
            text += f"; draw={self.draw}"
        return text


# One shared instance as the argument default, so the cache of
# :func:`transform_for_instance` hits.
DEFAULT_WARP = WarpConfig()


def warp_seed(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    domain: str = AUGMENTATION_DOMAIN,
    draw: int = 0,
) -> int:
    """Derive the seed of one instance's warp."""
    if domain not in WARP_DOMAINS:
        raise ValueError(
            f"unknown seed domain {domain!r}, expected one of {WARP_DOMAINS}"
        )
    # The domain token is the seed's namespace, not a name.
    parts: tuple[object, ...] = (
        domain,
        base_seed,
        problem_class,
        function_id,
        instance_id,
        dim,
    )
    # Draw 0 adds no part, so the first repetition keeps the default seed.
    if draw:
        parts = (*parts, draw)
    return derive_seed(*parts)


def maximum_seed(
    base_seed: int, problem_class: str, function_id: int, instance_id: int, dim: int
) -> int:
    """Derive the seed of one instance's maximum-estimation sample."""
    # `"warpmax"` is the seed's domain token, not a name.
    return derive_seed(
        "warpmax", base_seed, problem_class, function_id, instance_id, dim
    )


def estimate_maximum(problem: Any, seed: int) -> float:
    """Return the largest objective value in a uniform sample of the box."""
    lb = np.asarray(problem.bounds.lb, dtype=np.float64)
    ub = np.asarray(problem.bounds.ub, dtype=np.float64)
    rng = np.random.default_rng(seed)
    points = rng.uniform(lb, ub, size=(MAX_ESTIMATE_SAMPLE_SIZE, lb.size))
    return float(max(problem(point) for point in points))


def _spline_warp(
    rng: np.random.Generator, config: WarpConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return one draw from the smooth family."""
    grid = np.linspace(0.0, 1.0, GRID_SIZE)
    knots = np.linspace(0.0, 1.0, int(rng.integers(*config.knots)))
    density = np.exp(
        CubicSpline(knots, rng.normal(0.0, config.alpha, knots.size))(grid)
    )
    cumulative = cumulative_trapezoid(density, grid, initial=0.0)
    values = cumulative / cumulative[-1]
    values[0], values[-1] = 0.0, 1.0  # the division leaves float dust on the ends
    return grid, values


def _power_warp(
    rng: np.random.Generator, config: WarpConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return one draw from the power family, ``z ** exp(N(0, alpha))``."""
    exponent = float(np.exp(rng.normal(0.0, config.alpha)))
    grid = np.linspace(0.0, 1.0, GRID_SIZE) ** 2
    return grid, grid**exponent


def _piecewise_linear_warp(
    rng: np.random.Generator, config: WarpConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return one draw from the piecewise-linear family."""
    breaks = np.sort(rng.uniform(size=int(rng.integers(*PIECEWISE_BREAKS))))
    grid = np.concatenate(([0.0], breaks, [1.0]))
    widths = np.diff(grid)
    slopes = np.exp(rng.normal(0.0, config.alpha, widths.size))
    values = np.concatenate(([0.0], np.cumsum(slopes * widths)))
    values /= values[-1]
    values[-1] = 1.0
    return grid, values


# How each family is drawn, and how the result is read back between its knots.
# The piecewise-linear one is read linearly because interpolating it smoothly
# would remove the property it exists to test.
WARP_FAMILY_SAMPLERS = {
    SPLINE_FAMILY: (_spline_warp, PCHIP_INTERPOLATION),
    POWER_FAMILY: (_power_warp, PCHIP_INTERPOLATION),
    PIECEWISE_FAMILY: (_piecewise_linear_warp, LINEAR_INTERPOLATION),
}


def _sample_warp(
    rng: np.random.Generator, config: WarpConfig
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the grid, values and interpolation of one random monotone map."""
    sampler, interpolation = WARP_FAMILY_SAMPLERS[config.family]
    grid, values = sampler(rng, config)
    return grid, values, interpolation


def _linear_interpolator(
    grid: np.ndarray, values: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable reading ``values`` between the knots by straight lines."""
    return lambda z: np.interp(z, grid, values)


class MonotoneTransform:
    """One instance's strictly increasing objective transformation."""

    def __init__(
        self,
        optimum_y: float,
        maximum_y: float,
        grid: np.ndarray,
        values: np.ndarray,
        seed: int,
        interpolation: str = PCHIP_INTERPOLATION,
    ) -> None:
        if not maximum_y > optimum_y:
            raise ValueError(
                f"maximum_y must exceed optimum_y, got {maximum_y} <= {optimum_y}"
            )
        if interpolation not in WARP_INTERPOLATIONS:
            raise ValueError(
                f"unknown interpolation {interpolation!r}, "
                f"expected one of {WARP_INTERPOLATIONS}"
            )
        self.optimum_y = float(optimum_y)
        self.maximum_y = float(maximum_y)
        self.grid = grid
        self.values = values
        self.seed = int(seed)
        self.interpolation = interpolation
        # Both interpolants keep strict monotonicity; linear keeps the corners.
        self._warp = (
            PchipInterpolator(grid, values)
            if interpolation == PCHIP_INTERPOLATION
            else _linear_interpolator(grid, values)
        )

    def __call__(self, y: Any) -> np.ndarray:
        """Apply the transformation elementwise, returning ``float64``."""
        z = (np.asarray(y, dtype=np.float64) - self.optimum_y) / (
            self.maximum_y - self.optimum_y
        )
        # Noise can put a near-optimal value just below the tabulated optimum;
        # clamping keeps the warp inside its grid.
        z = np.maximum(z, 0.0)
        return np.where(z <= 1.0, self._warp(np.minimum(z, 1.0)), z)

    @property
    def strength(self) -> float:
        """Return ``max |g(z) - z|``, how far the warp departs from the identity."""
        return float(np.max(np.abs(self.values - self.grid)))

    def __repr__(self) -> str:
        """Return a debug representation naming the anchors and the strength."""
        return (
            f"MonotoneTransform(optimum_y={self.optimum_y!r}, "
            f"maximum_y={self.maximum_y!r}, seed={self.seed!r}, "
            f"interpolation={self.interpolation!r}, "
            f"strength={self.strength:.3f})"
        )


def sample_monotone_transform(
    problem: Any,
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    config: WarpConfig = DEFAULT_WARP,
) -> MonotoneTransform:
    """Draw the transformation of one instance, a pure function of its arguments."""
    seed = warp_seed(
        base_seed,
        problem_class,
        function_id,
        instance_id,
        dim,
        config.domain,
        config.draw,
    )
    grid, values, interpolation = _sample_warp(np.random.default_rng(seed), config)
    return MonotoneTransform(
        optimum_y=float(problem.optimum.y),
        maximum_y=estimate_maximum(
            problem,
            maximum_seed(base_seed, problem_class, function_id, instance_id, dim),
        ),
        grid=grid,
        values=values,
        seed=seed,
        interpolation=interpolation,
    )


@lru_cache(maxsize=512)
def transform_for_instance(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    config: WarpConfig = DEFAULT_WARP,
) -> MonotoneTransform:
    """Return the transformation of one instance, by coordinates alone."""
    return sample_monotone_transform(
        create_problem(function_id, instance_id, dim),
        base_seed,
        problem_class,
        function_id,
        instance_id,
        dim,
        config,
    )


# How far each map of a block departs from the identity, written to a report root.
WARP_STRENGTH_TABLE = Path("warp_strength.csv")


def warp_strength(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    config: WarpConfig = DEFAULT_WARP,
) -> float:
    """Return ``max |g(z) - z|`` of one instance's map.

    The curve alone decides it, so no problem is built and no maximum is estimated.
    """
    seed = warp_seed(
        base_seed,
        problem_class,
        function_id,
        instance_id,
        dim,
        config.domain,
        config.draw,
    )
    grid, values, _ = _sample_warp(np.random.default_rng(seed), config)
    return float(np.max(np.abs(values - grid)))


def warp_strengths(
    problems: pd.DataFrame, base_seed: int, dim: int, config: WarpConfig
) -> pd.DataFrame:
    """Return the strength of the map every problem of a block is drawn under."""
    keys = sorted(
        {
            (str(problem_class), int(function_id), int(instance_id))
            for problem_class, function_id, instance_id in zip(
                problems["problem_class"],
                problems["function_id"],
                problems["instance_id"],
                strict=True,
            )
        }
    )
    return pd.DataFrame(
        {
            "problem_class": [problem_class for problem_class, _, _ in keys],
            "function_id": [function_id for _, function_id, _ in keys],
            "instance_id": [instance_id for _, _, instance_id in keys],
            "dimension": dim,
            "family": config.family,
            "domain": config.domain,
            "draw": config.draw,
            "strength": [
                warp_strength(
                    base_seed, problem_class, function_id, instance_id, dim, config
                )
                for problem_class, function_id, instance_id in keys
            ],
        }
    )
