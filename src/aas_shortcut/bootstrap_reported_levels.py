"""Appendix: percentile bootstrap intervals of the reported levels.

Four units are resampled with replacement, each independently: the model seeds,
the deployments, the functions, and within each drawn function its test instances.
Functions are a unit of their own because the instances of one function are
strongly correlated. Every sample recomputes a level as the paper does: a
deployment's R² over the drawn problems (or its mean Θ over them), the median over
the drawn deployments, the mean over a range's families, the median over the drawn
seeds. A drawn problem enters as a weight, the number of times it was drawn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.derive_seeds import derive_seed

SEED_DOMAIN = "bootstrap"

# The two ways of drawing the problems: functions and their instances (the main
# version), or only the instances within every function, each function kept once.
FUNCTIONS_AND_INSTANCES = "functions and instances"
INSTANCES_WITHIN_FUNCTIONS = "instances within functions"
SEED_DOMAINS: dict[str, str] = {
    FUNCTIONS_AND_INSTANCES: SEED_DOMAIN,
    INSTANCES_WITHIN_FUNCTIONS: "bootstrap-within-functions",
}
N_SAMPLES = 10_000
CONFIDENCE = 0.95

# `predict_superiority.r2_score`'s rule for a target of no spread.
ZERO_SPREAD = 1e-12

# Samples per block of the weighted products, to bound memory.
CHUNK = 2_000


@dataclass(frozen=True)
class BootstrapDraws:
    """Every sample's draws: seed and deployment indices, and problem weights.

    ``other_seeds`` is an independent draw for the second side of an unpaired
    comparison, which shares the deployments and problems but not the seeds.
    """

    seeds: np.ndarray
    other_seeds: np.ndarray
    reps: np.ndarray
    weights: np.ndarray

    @property
    def n_samples(self) -> int:
        """How many samples were drawn."""
        return int(self.weights.shape[0])


def problem_order(problems: pd.DataFrame) -> pd.DataFrame:
    """Return the problems, one row each, in the order every array is laid out."""
    keys = cast(pd.DataFrame, problems[["function_id", "instance_id"]])
    return (
        keys.drop_duplicates()
        .sort_values(["function_id", "instance_id"])
        .reset_index(drop=True)
    )


def draw_samples(
    problems: pd.DataFrame,
    n_seeds: int,
    n_reps: int,
    n_samples: int = N_SAMPLES,
    base_seed: int = 42,
    resampling: str = FUNCTIONS_AND_INSTANCES,
) -> BootstrapDraws:
    """Draw every sample's seeds, deployments, functions and instances."""
    if resampling not in SEED_DOMAINS:
        raise ValueError(
            f"unknown resampling {resampling!r}; expected {list(SEED_DOMAINS)}"
        )
    order = problem_order(problems)
    sizes = order.groupby("function_id").size()
    if (sizes != sizes.iloc[0]).any():
        raise ValueError(
            f"every function needs as many instances; got {sizes.to_dict()}"
        )
    n_functions, n_instances = len(sizes), int(sizes.iloc[0])
    rng = np.random.default_rng(derive_seed(base_seed, SEED_DOMAINS[resampling]))
    seeds = rng.integers(0, n_seeds, size=(n_samples, n_seeds))
    other_seeds = rng.integers(0, n_seeds, size=(n_samples, n_seeds))
    reps = rng.integers(0, n_reps, size=(n_samples, n_reps))
    if resampling == FUNCTIONS_AND_INSTANCES:
        functions = rng.integers(0, n_functions, size=(n_samples, n_functions))
    else:
        functions = np.tile(np.arange(n_functions), (n_samples, 1))
    instances = rng.integers(0, n_instances, size=(n_samples, n_functions, n_instances))
    drawn = functions[:, :, None] * n_instances + instances
    weights = np.zeros((n_samples, n_functions * n_instances))
    rows = np.repeat(np.arange(n_samples), n_functions * n_instances)
    np.add.at(weights, (rows, drawn.reshape(-1)), 1.0)
    return BootstrapDraws(seeds, other_seeds, reps, weights)


def identity_draws(n_seeds: int, n_reps: int, n_problems: int) -> BootstrapDraws:
    """The one sample that is the data itself: the point estimate."""
    seeds = np.arange(n_seeds)[None, :]
    return BootstrapDraws(
        seeds,
        seeds,
        np.arange(n_reps)[None, :],
        np.ones((1, n_problems)),
    )


def weighted_r2(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """R² of every deployment on every sample's weighted problems.

    ``observed`` is (problems,), ``predicted`` (seeds, reps, problems), ``weights``
    (samples, problems); the result is (samples, seeds, reps). With integer weights
    it is `r2_score` on the problems repeated as drawn.
    """
    n_seeds, n_reps, n_problems = predicted.shape
    errors = ((observed[None, None, :] - predicted) ** 2).reshape(-1, n_problems)
    counts = weights.sum(axis=1)
    mean = weights @ observed / counts
    total = weights @ (observed**2) - counts * mean**2
    present = weights > 0
    high = np.where(present, observed[None, :], -np.inf).max(axis=1)
    low = np.where(present, observed[None, :], np.inf).min(axis=1)
    largest = np.where(present, np.abs(observed)[None, :], 0.0).max(axis=1)
    flat = (high - low) <= ZERO_SPREAD * np.where(largest > 0, largest, 1.0)
    total = np.where(flat, 1.0, total)
    out = np.empty((weights.shape[0], n_seeds * n_reps))
    for start in range(0, weights.shape[0], CHUNK):
        part = slice(start, start + CHUNK)
        residual = weights[part] @ errors.T
        out[part] = 1.0 - residual / total[part, None]
    out[flat] = 0.0
    return out.reshape(-1, n_seeds, n_reps)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Mean of every deployment's values over every sample's weighted problems."""
    n_seeds, n_reps, n_problems = values.shape
    flat = values.reshape(-1, n_problems)
    counts = weights.sum(axis=1)
    out = np.empty((weights.shape[0], n_seeds * n_reps))
    for start in range(0, weights.shape[0], CHUNK):
        part = slice(start, start + CHUNK)
        out[part] = (weights[part] @ flat.T) / counts[part, None]
    return out.reshape(-1, n_seeds, n_reps)


def seed_values(families: Sequence[np.ndarray], draws: BootstrapDraws) -> np.ndarray:
    """Every seed's value on every sample, as the paper reduces a cell.

    The median over the drawn deployments, then the mean over the families;
    ``families`` holds (samples, seeds, reps) arrays.
    """
    per_family = []
    for values in families:
        picked = np.take_along_axis(values, draws.reps[:, None, :], axis=2)
        per_family.append(np.median(picked, axis=2))
    return np.mean(per_family, axis=0)


def level(seeds: np.ndarray, picked: np.ndarray) -> np.ndarray:
    """Every sample's level: the median over its drawn seeds."""
    return np.median(np.take_along_axis(seeds, picked, axis=1), axis=1)


def percentile_interval(
    samples: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """The percentile interval of the bootstrap distribution."""
    tail = 100.0 * (1.0 - confidence) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return float(low), float(high)


def _cube(
    rows: pd.DataFrame,
    column: str,
    order: pd.DataFrame,
    seeds: Sequence[int],
    reps: Sequence[int],
) -> np.ndarray:
    """Lay one cell's per-problem column out as (seeds, reps, problems)."""
    expected = len(seeds) * len(reps) * len(order)
    if len(rows) != expected:
        raise ValueError(f"the cell holds {len(rows)} row(s), expected {expected}")
    problem = pd.MultiIndex.from_frame(order).get_indexer(
        pd.MultiIndex.from_frame(
            cast(pd.DataFrame, rows[["function_id", "instance_id"]].astype(int))
        )
    )
    seed = pd.Index(list(seeds)).get_indexer(rows["seed"].astype(int))
    rep = pd.Index(list(reps)).get_indexer(rows["rep"].astype(int))
    if (problem < 0).any() or (seed < 0).any() or (rep < 0).any():
        raise ValueError(
            "the cell holds a problem, seed or deployment outside the layout"
        )
    cube = np.full((len(seeds), len(reps), len(order)), np.nan)
    cube[seed, rep, problem] = rows[column].to_numpy(dtype=np.float64)
    if np.isnan(cube).any():
        raise ValueError("the cell holds a problem twice and lacks another")
    return cube


def regression_cube(
    rows: pd.DataFrame,
    order: pd.DataFrame,
    seeds: Sequence[int],
    reps: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """One cell's targets (problems,) and predictions (seeds, reps, problems)."""
    observed = _cube(rows, "observed", order, seeds, reps)
    if not (observed == observed[:1, :1]).all():
        raise ValueError("a problem's target differs between seeds or deployments")
    return observed[0, 0], _cube(rows, "predicted", order, seeds, reps)


def selection_cube(
    rows: pd.DataFrame,
    order: pd.DataFrame,
    seeds: Sequence[int],
    reps: Sequence[int],
) -> np.ndarray:
    """One cell's per-problem Θ as (seeds, reps, problems)."""
    return _cube(rows, "superiority_vs_sbs", order, seeds, reps)
