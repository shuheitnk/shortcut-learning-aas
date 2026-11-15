"""Whether the pair tuning's calibration survives the whole training block.

`tune_pair` minimises the KS distance of each pair's superiorities to
``Uniform(0, 1)`` on 24 problems (the tuning instance); the regression is fitted on
the 2400 of the training block. This module measures the calibration again on both
blocks, from the benchmark's superiority shard.

The deciles show the shape the summary cannot: a U-shaped block and a flat one can
reach the same D_KS.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS, AlgorithmPair
from aas_shortcut.define_paths import TRANSFORMS, RunPaths
from aas_shortcut.define_table_schema import (
    SUPERIORITY_CALIBRATION_DECILE_SCHEMA,
    SUPERIORITY_CALIBRATION_SCHEMA,
)
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.split_bbob_instances import TRAIN_INSTANCES, TUNING_INSTANCES
from aas_shortcut.tune_algorithms import (
    ESTIMATOR,
    comparisons_per_rate,
    floor_for,
    ks_distance,
)

LOGGER = logging.getLogger(__name__)


CALIBRATION_SHARD = "superiority_calibration"
DECILE_SHARD = "superiority_calibration_deciles"

# Finer bins would show sampling noise rather than the shape.
DECILES = 10

# Superiorities inside this band are informative rather than one member dominating;
# the same band as `tune_algorithms.summarise`.
INFORMATIVE_BAND = (0.2, 0.8)


@dataclass(frozen=True)
class CalibrationConfig:
    """Which blocks of which shard the calibration is read on."""

    problem_class: str = "bbob"
    dim: int = 10
    # The binding was chosen on `pre`, so the calibration is read there.
    transform: str = TRANSFORMS[0]
    pairs: tuple[AlgorithmPair, ...] = FOCUS_PAIRS
    # ``{block name: instance ids}``; the tuning block is a subset of the training one.
    blocks: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: {
            "tuning": TUNING_INSTANCES,
            "training": TRAIN_INSTANCES,
        }
    )

    def __post_init__(self) -> None:
        """Reject an unknown view or an empty axis."""
        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"unknown view {self.transform!r}; expected one of {list(TRANSFORMS)}"
            )
        if not self.pairs:
            raise ValueError("the calibration needs at least one pair")
        if not self.blocks:
            raise ValueError("the calibration needs at least one block")
        for name, instances in self.blocks.items():
            if not instances:
                raise ValueError(f"block {name!r} names no instance")

    def as_dict(self) -> dict:
        """Fingerprintable description of the configuration."""
        return {
            "stage": "superiority_calibration",
            # The KS floor depends on the estimator's granularity.
            "estimator": ESTIMATOR,
            "problem_class": self.problem_class,
            "dim": self.dim,
            "transform": self.transform,
            "pairs": [pair.slug for pair in self.pairs],
            "blocks": {
                name: list(instances) for name, instances in sorted(self.blocks.items())
            },
            "deciles": DECILES,
            "informative_band": list(INFORMATIVE_BAND),
        }


def _rates(
    superiorities: pd.DataFrame,
    pair: AlgorithmPair,
    instances: Sequence[int],
    config: CalibrationConfig,
) -> tuple[np.ndarray, int]:
    """Return one block's superiorities for one pair, and their single trial count.

    The D_KS floor is defined for one granularity, so mixed trial counts are rejected.
    """
    rows = superiorities[
        (superiorities["pair"] == pair.slug)
        & (superiorities["transform"] == config.transform)
        & (superiorities["instance_id"].isin(list(instances)))
    ]
    if rows.empty:
        raise ValueError(
            f"the superiority shard holds no {config.transform} row for {pair.slug} "
            f"on instances {list(instances)[:3]}..."
        )
    trials = sorted(int(value) for value in rows["n_trials"].unique())
    if len(trials) != 1:
        raise ValueError(
            f"{pair.slug} was measured at {trials} trials on this block; the "
            "granularity floor is defined against a single trial count"
        )
    return rows["superiority"].to_numpy(dtype=np.float64), trials[0]


def calibration_summary(
    superiorities: pd.DataFrame, config: CalibrationConfig | None = None
) -> pd.DataFrame:
    """Return how uniform each pair's target is on each block."""
    config = config or CalibrationConfig()
    rows: list[dict] = []
    for block, instances in config.blocks.items():
        for pair in config.pairs:
            values, trials = _rates(superiorities, pair, instances, config)
            saturated = (values == 0.0) | (values == 1.0)
            informative = (values > INFORMATIVE_BAND[0]) & (
                values < INFORMATIVE_BAND[1]
            )
            rows.append(
                {
                    "block": block,
                    "group": group_of(pair),
                    "pair": pair.slug,
                    "transform": config.transform,
                    "n_problems": int(values.size),
                    "n_trials": trials,
                    "ks_distance": ks_distance(values),
                    # A12 compares every run with every run, so a rate is a
                    # multiple of 1 / (2 * trials**2), as in the tuning's floor.
                    "ks_floor": floor_for(values.size, comparisons_per_rate(trials)),
                    "mean": float(values.mean()),
                    # Population variance: the block is the population the
                    # regression is fitted on, not a sample from a larger one.
                    "variance": float(values.var(ddof=0)),
                    "n_saturated": int(saturated.sum()),
                    "saturated_share": float(saturated.mean()),
                    "n_informative": int(informative.sum()),
                    "informative_share": float(informative.mean()),
                }
            )
    return pd.DataFrame(rows)


def calibration_deciles(
    superiorities: pd.DataFrame, config: CalibrationConfig | None = None
) -> pd.DataFrame:
    """Return each block's counts in ten equal bins of ``[0, 1]``, against uniform."""
    config = config or CalibrationConfig()
    edges = np.linspace(0.0, 1.0, DECILES + 1)
    rows: list[dict] = []
    for block, instances in config.blocks.items():
        for pair in config.pairs:
            values, _ = _rates(superiorities, pair, instances, config)
            counts, _ = np.histogram(values, bins=DECILES, range=(0.0, 1.0))
            for index, count in enumerate(counts):
                rows.append(
                    {
                        "block": block,
                        "group": group_of(pair),
                        "pair": pair.slug,
                        "transform": config.transform,
                        "decile": index,
                        "lower": float(edges[index]),
                        "upper": float(edges[index + 1]),
                        "n_problems": int(count),
                        "expected_uniform": float(values.size) / DECILES,
                    }
                )
    return pd.DataFrame(rows)


def run_calibration_stage(
    superiorities: pd.DataFrame,
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
    config: CalibrationConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure both tables and persist them; return ``(summary, deciles)``."""
    config = config or CalibrationConfig()
    summary = calibration_summary(superiorities, config).assign(
        scale=scale, config_hash=config_hash
    )
    deciles = calibration_deciles(superiorities, config).assign(
        scale=scale, config_hash=config_hash
    )

    written_summary = write_table(
        summary,
        run_paths.metric(CALIBRATION_SHARD).data,
        SUPERIORITY_CALIBRATION_SCHEMA,
    )
    written_deciles = write_table(
        deciles,
        run_paths.metric(DECILE_SHARD).data,
        SUPERIORITY_CALIBRATION_DECILE_SCHEMA,
    )
    LOGGER.info(
        "superiority calibration: wrote %s (%d rows) and %s (%d rows)",
        run_paths.metric(CALIBRATION_SHARD).data,
        len(written_summary),
        run_paths.metric(DECILE_SHARD).data,
        len(written_deciles),
    )
    return written_summary, written_deciles
