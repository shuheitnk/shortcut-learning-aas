"""The repetitions a reported number is the median over.

A repetition re-seeds the regressor; every cell is fitted under each of
:data:`MODEL_SEEDS` and the tables report the median over them.
"""

from __future__ import annotations

from dataclasses import replace

from aas_shortcut.build_regression_model import ModelConfig

# Odd, so a median is one of the repetitions' own values, not an interpolation.
REPETITIONS = 11

# The regressor seed of each repetition; the first is the default seed.
MODEL_SEEDS: tuple[int, ...] = tuple(
    ModelConfig.random_state + index for index in range(REPETITIONS)
)


def repetitions() -> tuple[int, ...]:
    """Return the repetition indices, in the order they are run and reported."""
    return tuple(range(REPETITIONS))


def model_for(config: ModelConfig, repetition: int) -> ModelConfig:
    """Return the regressor setting of one repetition."""
    _check(repetition)
    return replace(config, random_state=MODEL_SEEDS[repetition])


def _check(repetition: int) -> None:
    """Reject a repetition the study does not run."""
    if repetition not in repetitions():
        raise ValueError(
            f"repetition {repetition} is not one of the {REPETITIONS} the study "
            f"runs; a number reported from another index is not the median"
        )
