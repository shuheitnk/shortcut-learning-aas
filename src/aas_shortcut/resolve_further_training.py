"""Appendix: a plain training condition as large as the augmented one.

``plain-4800`` fits on instances 11-210, all untransformed: the paper's training
block and a further block of 100 instances per function, benchmarked under the
same bindings. It spends the augmented condition's update budget, so the two
differ only in whether the second half of the rows is transformed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.bind_portfolio_parameters import (
    CONDITION_SELECTION,
    PortfolioBinding,
)
from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.match_update_budget import matched_config
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
    TRAINING_VIEWS,
)
from aas_shortcut.resolve_benchmark_units import (
    PRE,
    PairUnit,
    assemble,
    resolve_pair_unit,
)
from aas_shortcut.resolve_feature_units import resolve_training_features
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    training_superiority_role,
)
from aas_shortcut.resolve_portfolio_blocks import (
    PortfolioBindings,
    PortfolioUnit,
    SelectionHeldOut,
    SelectionTraining,
    resolve_portfolio_unit,
    training_mean_superiority_role,
)
from aas_shortcut.resolve_regression_cells import RegressionData, regression_data
from aas_shortcut.resolve_selection_cells import SelectionData, selection_data
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore
from aas_shortcut.split_bbob_instances import TRAIN_INSTANCES

TRAINING_PLAIN_4800 = "plain-4800"

# One hundred more instances per function, after the paper's training block.
FURTHER_INSTANCES: tuple[int, ...] = tuple(
    range(TRAIN_INSTANCES[-1] + 1, TRAIN_INSTANCES[-1] + 1 + len(TRAIN_INSTANCES))
)
PLAIN_4800_INSTANCES: tuple[int, ...] = (*TRAIN_INSTANCES, *FURTHER_INSTANCES)

FURTHER = "further"


def further_role(role: str) -> str:
    """The input role of the further block's counterpart of a training unit."""
    return f"{role}:{FURTHER}"


def plain_4800_model(model: ModelConfig) -> ModelConfig:
    """The model at the augmented condition's update budget."""
    return matched_config(model, len(TRAINING_VIEWS[TRAINING_AUGMENTED]))


@dataclass(frozen=True)
class FurtherFeatures:
    """The feature unit covering instances 1-210, both views."""

    features: StageArtifact
    feature_table: pd.DataFrame


def resolve_further_features(
    store: StageStore, context: ExperimentContext
) -> FurtherFeatures:
    """Resolve the training features over the further block too."""
    wider = replace(context, max_instances=PLAIN_4800_INSTANCES[-1])
    features, table = resolve_training_features(store, wider)
    return FurtherFeatures(features, table)


def check_shared_features(wider: pd.DataFrame, paper: pd.DataFrame) -> int:
    """Stop unless the wider unit holds the paper's training features bit for bit."""
    keys = [
        column
        for column in wider.columns
        if column not in ("value", "config_hash", "elapsed_seconds")
    ]
    instances = paper["instance_id"].unique().tolist()
    shared = cast(pd.DataFrame, wider[wider["instance_id"].isin(instances)])
    if len(shared) != len(paper):
        raise ValueError(
            f"the wider feature unit holds {len(shared)} row(s) on the paper's "
            f"instances, the paper's unit {len(paper)}"
        )
    joined = paper[[*keys, "value"]].merge(
        shared[[*keys, "value"]], on=keys, how="left", suffixes=("", "_wider")
    )
    held = joined["value"].to_numpy(dtype=np.float64)
    again = joined["value_wider"].to_numpy(dtype=np.float64)
    same = (held == again) | (np.isnan(held) & np.isnan(again))
    if not same.all():
        raise ValueError(
            f"{int((~same).sum())} of {len(joined)} training feature value(s) differ "
            "between the wider unit and the paper's"
        )
    return len(joined)


def check_same_matrix(wider: pd.DataFrame, paper: pd.DataFrame) -> int:
    """Stop unless the wider data's matrix on the paper's rows is the paper's."""
    pd.testing.assert_frame_equal(
        wider.reset_index(drop=True), paper.reset_index(drop=True), check_exact=True
    )
    return len(paper)


def resolve_further_pairs(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    step: str,
) -> PairUnit:
    """The focus pairs' members on the further block, untransformed."""
    return resolve_pair_unit(
        store, context, binding, tuning, view=PRE, block=FURTHER_INSTANCES, step=step
    )


def resolve_further_portfolio(
    store: StageStore, context: ExperimentContext, bindings: PortfolioBindings
) -> PortfolioUnit:
    """The portfolio on the further block, untransformed, under the selection role.

    The SBS is not run there: it stays the one named on the paper's training block.
    """
    return resolve_portfolio_unit(
        store,
        context,
        bindings,
        CONDITION_SELECTION,
        view=PRE,
        block=FURTHER_INSTANCES,
    )


def _features_on(
    table: pd.DataFrame, instances: Sequence[int], features: Sequence[str]
) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        table[
            table["instance_id"].isin(list(instances))
            & table["feature"].isin(list(features))
        ],
    )


def plain_4800_regression_data(
    further: FurtherFeatures,
    training: TrainingBlock,
    pairs: PairUnit,
    blocks: Sequence[HeldOutBlock],
    features: Sequence[str],
) -> RegressionData:
    """The plain cells' data with the further block's rows added."""
    if training.post is not None:
        raise ValueError("plain-4800 fits on the untransformed view alone")
    paper = regression_data(training, blocks, features)
    inputs = dict(paper.inputs)
    inputs["train_features"] = further.features.fingerprint
    inputs[further_role(training_superiority_role(PRE))] = pairs.superiority.fingerprint
    return replace(
        paper,
        train_features=_features_on(
            further.feature_table, PLAIN_4800_INSTANCES, features
        ),
        train_superiorities=assemble(
            [training.pre.superiorities, pairs.superiorities], "view_superiority"
        ),
        inputs=inputs,
    )


def plain_4800_selection_data(
    further: FurtherFeatures,
    training: SelectionTraining,
    portfolio: PortfolioUnit,
    blocks: Sequence[SelectionHeldOut],
    features: Sequence[str],
) -> SelectionData:
    """The plain cells' data with the further block's rows; the SBS's left as it was."""
    if training.post:
        raise ValueError("plain-4800 fits on the untransformed view alone")
    paper = selection_data(training, blocks, features)
    inputs = dict(paper.inputs)
    inputs["train_features"] = further.features.fingerprint
    inputs[further_role(training_mean_superiority_role(PRE, CONDITION_SELECTION))] = (
        portfolio.rank.fingerprint
    )
    return replace(
        paper,
        features=_features_on(further.feature_table, PLAIN_4800_INSTANCES, features),
        mean_superiorities=assemble(
            [
                training.pre[CONDITION_SELECTION].mean_superiorities,
                portfolio.mean_superiorities,
            ],
            "view_mean_superiority",
        ),
        inputs=inputs,
    )


def relabel(table: pd.DataFrame) -> pd.DataFrame:
    """Name the plain-4800 cells' rows by their condition."""
    if set(table["training"].astype(str)) != {TRAINING_PLAIN}:
        raise ValueError("plain-4800's cells are fitted as plain ones")
    return table.assign(training=TRAINING_PLAIN_4800)
