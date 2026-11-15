"""The training and held-out blocks of the pair benchmark.

The transformed training unit is resolved only when an augmented cell is fitted.
The untransformed held-out unit is shared by every family. Each block names the
fingerprints of its units, which the regression cells record as inputs.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import pandas as pd

from aas_shortcut.bind_portfolio_parameters import PortfolioBinding
from aas_shortcut.calculate_features import OrderPreservationError
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_warp_audits import AUDITS, Audit
from aas_shortcut.measure_selection_transfer import TRAINING_CONDITIONS, training_views
from aas_shortcut.resolve_benchmark_units import (
    POST,
    PRE,
    PairUnit,
    assemble,
    resolve_pair_unit,
)
from aas_shortcut.resolve_feature_units import (
    resolve_deployment_features,
    resolve_training_features,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP
from aas_shortcut.split_bbob_instances import TEST_INSTANCES, TRAIN_INSTANCES

LOGGER = logging.getLogger(__name__)


def training_superiority_role(view: str) -> str:
    """Return the input role of the training block's superiorities for one view."""
    return f"train_superiority_{view}"


@dataclass(frozen=True)
class TrainingBlock:
    """Instances 11-110: their features, and the pair benchmark under the views read.

    ``post`` is absent when only plain cells are fitted on the block.
    """

    features: StageArtifact
    feature_table: pd.DataFrame
    pre: PairUnit
    post: PairUnit | None = None

    @property
    def units(self) -> dict[str, PairUnit]:
        """The benchmark units the block was resolved with, by view."""
        units = {PRE: self.pre}
        if self.post is not None:
            units[POST] = self.post
        return units

    @property
    def superiorities(self) -> pd.DataFrame:
        """The resolved views' superiorities, in the order one table of them has."""
        return assemble(
            [unit.superiorities for unit in self.units.values()], "view_superiority"
        )

    @property
    def inputs(self) -> dict[str, str]:
        """The fingerprints a cell fitted on this block may record."""
        inputs = {"train_features": self.features.fingerprint}
        for view, unit in self.units.items():
            inputs[training_superiority_role(view)] = unit.superiority.fingerprint
        return inputs


@dataclass(frozen=True)
class HeldOutBlock:
    """Instances 1-10 under one family: eleven deployments and both views' runs."""

    label: str
    features: StageArtifact
    feature_table: pd.DataFrame
    pre: PairUnit
    post: PairUnit

    @property
    def superiorities(self) -> pd.DataFrame:
        """Both views' superiorities on the block, in the order one table has."""
        return assemble(
            (self.pre.superiorities, self.post.superiorities), "view_superiority"
        )

    @property
    def inputs(self) -> dict[str, str]:
        """The fingerprints a cell scored on this block records."""
        return {
            f"test_features:{self.label}": self.features.fingerprint,
            "test_superiority_pre": self.pre.superiority.fingerprint,
            f"test_superiority_post:{self.label}": self.post.superiority.fingerprint,
        }


def resolve_training_block(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    step: str,
    trainings: Sequence[str] = TRAINING_CONDITIONS,
) -> TrainingBlock:
    """Resolve the training block, computing what the store lacks.

    Only the views ``trainings`` need are benchmarked. The warp must be the default
    one, since the training features were computed under it.
    """
    if context.warp != DEFAULT_WARP:
        raise ValueError(
            f"the training block's warp is {context.warp.identifier}, but its "
            f"features are computed under {DEFAULT_WARP.identifier}"
        )
    views = training_views(trainings)
    features, table = resolve_training_features(store, context)
    pre = resolve_pair_unit(
        store, context, binding, tuning, view=PRE, block=TRAIN_INSTANCES, step=step
    )
    post = None
    if POST in views:
        post = resolve_pair_unit(
            store,
            context,
            binding,
            tuning,
            view=POST,
            block=TRAIN_INSTANCES,
            warp=context.warp,
            step=step,
        )
    return TrainingBlock(features, table, pre, post)


def resolve_held_out_pre(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    step: str,
) -> PairUnit:
    """Resolve the untransformed held-out unit every family shares."""
    return resolve_pair_unit(
        store, context, binding, tuning, view=PRE, block=TEST_INSTANCES, step=step
    )


def resolve_held_out_block(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    audit: Audit,
    pre: PairUnit,
    step: str,
) -> HeldOutBlock:
    """Resolve one family's held-out block, computing what the store lacks."""
    family = replace(context, warp=audit.warp)
    features, table = resolve_deployment_features(store, family)
    post = resolve_pair_unit(
        store,
        family,
        binding,
        tuning,
        view=POST,
        block=TEST_INSTANCES,
        warp=audit.warp,
        step=step,
    )
    return HeldOutBlock(audit.label, features, table, pre, post)


def resolve_held_out_blocks(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    step: str,
    audits: Sequence[Audit] = AUDITS,
) -> dict[str, HeldOutBlock]:
    """Resolve every family's held-out block, by label, in the order given."""
    pre = resolve_held_out_pre(store, context, binding, tuning, step)
    return {
        audit.label: resolve_held_out_block(
            store, context, binding, tuning, audit, pre, step
        )
        for audit in audits
    }


def resolve_held_out_in_range(
    store: StageStore,
    context: ExperimentContext,
    binding: PortfolioBinding,
    tuning: StageArtifact,
    step: str,
    audits: Sequence[Audit] = AUDITS,
) -> list[HeldOutBlock]:
    """Resolve each family's held-out block, skipping families out of their range.

    A family that cannot preserve the sample's order is skipped with a warning.
    Experiments 2 and 4 both use this function, so their cells name the same blocks.
    """
    pre = resolve_held_out_pre(store, context, binding, tuning, step)
    blocks: list[HeldOutBlock] = []
    for audit in audits:
        try:
            blocks.append(
                resolve_held_out_block(
                    store, context, binding, tuning, audit, pre, step
                )
            )
        except OrderPreservationError as failure:
            LOGGER.warning(
                "%s: the family does not preserve the sample's order here, so "
                "this audit is out of its range: %s",
                audit.label,
                failure,
            )
    return blocks


def block_inputs(
    training: TrainingBlock, blocks: Sequence[HeldOutBlock]
) -> dict[str, str]:
    """Return the input fingerprints of a cell fitted and scored on these blocks.

    Blocks naming the shared ``pre`` unit differently were resolved under different
    bindings and are rejected.
    """
    inputs = dict(training.inputs)
    for block in blocks:
        for role, fingerprint in block.inputs.items():
            if inputs.setdefault(role, fingerprint) != fingerprint:
                raise ValueError(
                    f"{block.label}: {role} is {fingerprint}, but another block "
                    f"names {inputs[role]}"
                )
    return inputs
