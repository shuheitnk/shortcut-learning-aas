"""Matching the number of parameter updates across the two training conditions.

The augmented condition fits on every instance twice, so at a fixed epoch count it
takes twice the optimiser steps, and the number of updates alone moves
generalisation (Hoffer et al., NeurIPS 2017). The epoch budget is therefore divided
by the number of views; the updates then match up to one step per epoch, since
RealMLP-TD drops each epoch's last partial batch. RealMLP-TD's schedules follow the
fraction of training done, so they need no change; the best epoch is chosen from
half as many candidates.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aas_shortcut.build_regression_model import (
    EARLY_STOPPED_KINDS,
    NEURAL_KINDS,
    REALMLP_TD_BATCH_SIZE,
    ModelConfig,
    validation_split_sizes,
)

# Views the unaugmented condition spends, which the budget is matched against.
BASELINE_VIEWS = 1


def steps_per_epoch(rows: int, config: ModelConfig) -> int:
    """Optimiser steps one epoch takes on a matrix of ``rows`` rows.

    The validation rows come off first, and pytabkit drops the last partial batch.
    """
    fitted, _ = validation_split_sizes(rows, config.validation_fraction)
    batch = max(1, min(REALMLP_TD_BATCH_SIZE, fitted))
    return fitted // batch


def total_updates(rows: int, config: ModelConfig) -> int:
    """Optimiser steps a whole fit takes, at ``config``'s epoch budget."""
    return steps_per_epoch(rows, config) * config.epochs


def matched_config(config: ModelConfig, views: int) -> ModelConfig:
    """Return ``config`` with its epoch budget divided by ``views``.

    ``views`` must divide the epochs: rounding would move the two conditions'
    updates apart by more than the dropped partial batches do.
    """
    if views < 1:
        raise ValueError(
            f"the training side spends {views} view(s); expected at least 1"
        )
    if views == BASELINE_VIEWS:
        return config
    if config.epochs % views:
        raise ValueError(
            f"epochs={config.epochs} is not divisible by the {views} view(s) the "
            "training side spends; the two conditions would take different "
            "numbers of parameter updates"
        )
    return replace(config, epochs=config.epochs // views)


def describe_budget(rows: int, config: ModelConfig, views: int) -> str:
    """One line naming the updates a condition's fit will take."""
    matched = matched_config(config, views)
    return (
        f"{views} view(s), {rows} row(s): {steps_per_epoch(rows, matched)} step(s) "
        f"x {matched.epochs} epoch(s) = {total_updates(rows, matched)} update(s)"
    )


def learning_settings_of(config: ModelConfig, views: int) -> dict[str, Any]:
    """What a fit on ``views`` views depends on that ``config.as_dict()`` omits.

    A cell's unit records them, so a change of either is a different unit. Each is
    recorded only for the backends it acts on.
    """
    settings: dict[str, Any] = {}
    if config.kind in NEURAL_KINDS:
        settings["epoch_divisor"] = views
    if config.kind in EARLY_STOPPED_KINDS:
        # Whole problems are held out for early stopping, so a problem's augmented
        # copy stays on its side.
        settings["early_stopping_split"] = {
            "unit": "problem",
            "fraction": config.validation_fraction,
        }
    return settings
