"""The blocks of the portfolio benchmark that the selection experiments read.

The six-member portfolio is benchmarked under two bindings (the selection's at the
search budget, the single best solver's at the total budget), split into units by
view and block. Each block records the fingerprints of the units it came from.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import pandas as pd

from aas_shortcut.bind_portfolio_parameters import (
    CONDITION_SBS,
    CONDITION_SELECTION,
    PortfolioBinding,
)
from aas_shortcut.calculate_features import OrderPreservationError
from aas_shortcut.calculate_mean_superiority import run_view_mean_superiority_stage
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import TABLE_SUFFIX, shard_stem
from aas_shortcut.define_warp_audits import AUDITS, Audit
from aas_shortcut.measure_selection_transfer import TRAINING_CONDITIONS, training_views
from aas_shortcut.resolve_benchmark_units import (
    POST,
    PRE,
    assemble,
    benchmark_unit,
    resolve_benchmark,
)
from aas_shortcut.resolve_feature_units import (
    resolve_deployment_features,
    resolve_training_features,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit
from aas_shortcut.resolve_tuning_stages import resolve_role_bindings
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP, WarpConfig
from aas_shortcut.split_bbob_instances import (
    PROBLEM_CLASS,
    TEST_INSTANCES,
    TRAIN_INSTANCES,
)

LOGGER = logging.getLogger(__name__)

# The selector's binding, then the baseline's.
CONDITIONS = (CONDITION_SELECTION, CONDITION_SBS)


def training_mean_superiority_role(view: str, condition: str) -> str:
    """Return the input role of the training block's mean superiorities of one run."""
    return f"train_mean_superiority_{view}:{condition}"


def condition_budget(context: ExperimentContext, condition: str) -> int:
    """Return the budget a condition's runs are given.

    The selection spends the sample on features and searches with the rest; the
    single best solver gets the whole, so both have the same total.
    """
    if condition == CONDITION_SELECTION:
        return context.search_budget
    if condition == CONDITION_SBS:
        return context.total_budget
    raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


@dataclass(frozen=True)
class PortfolioBindings:
    """The two bindings, and the role tuning they were read from."""

    bindings: Mapping[str, PortfolioBinding]
    roles: StageArtifact

    def provenance(self, condition: str) -> dict[str, Any]:
        """Return where a condition's binding came from, for a unit's record."""
        return {
            "binding": {
                "stage": self.roles.unit.stage,
                "fingerprint": self.roles.fingerprint,
                "condition": condition,
            }
        }


def resolve_portfolio_bindings(
    store: StageStore, context: ExperimentContext
) -> PortfolioBindings:
    """Resolve the role tuning and the two bindings it yields."""
    bindings, roles = resolve_role_bindings(store, context)
    return PortfolioBindings(bindings, roles)


def mean_superiority_settings(
    context: ExperimentContext, condition: str
) -> dict[str, Any]:
    """Return the settings of a mean-superiority unit, besides its benchmark."""
    return {
        "stage": "view_mean_superiority",
        "condition": condition,
        "dim": context.dim,
        "algorithms": list(ALGORITHMS),
    }


def mean_superiority_table(dim: int, condition: str) -> str:
    """Return the path of a mean-superiority unit's table inside the unit."""
    stem = shard_stem(f"view_mean_superiority_{condition}", PROBLEM_CLASS, dim)
    return f"mean_superiority/{stem}{TABLE_SUFFIX}"


def resolve_mean_superiorities(
    store: StageStore,
    context: ExperimentContext,
    condition: str,
    performance: tuple[StageArtifact, pd.DataFrame],
) -> tuple[StageArtifact, pd.DataFrame]:
    """Return the mean superiorities of one benchmark unit, computing them if absent."""
    source, rows = performance
    unit = StageUnit(
        "view_mean_superiority",
        mean_superiority_settings(context, condition),
        inputs={"performance": source.fingerprint},
    )

    def compute(paths: Any, config_hash: str) -> None:
        run_view_mean_superiority_stage(
            rows,
            PROBLEM_CLASS,
            context.dim,
            paths,
            context.scale,
            config_hash,
            condition=condition,
        )

    artifact = store.resolve(unit, compute)
    return artifact, artifact.read_table(mean_superiority_table(context.dim, condition))


@dataclass(frozen=True)
class PortfolioUnit:
    """One unit of the portfolio benchmark, resolved, with its mean superiorities."""

    benchmark: StageArtifact
    performance: pd.DataFrame
    rank: StageArtifact
    mean_superiorities: pd.DataFrame


def resolve_portfolio_unit(
    store: StageStore,
    context: ExperimentContext,
    bindings: PortfolioBindings,
    condition: str,
    *,
    view: str,
    block: Iterable[int],
    warp: WarpConfig = DEFAULT_WARP,
) -> PortfolioUnit:
    """Resolve one unit of the portfolio benchmark under one condition."""
    binding = bindings.bindings[condition]
    config = benchmark_unit(
        context,
        algorithms=ALGORITHMS,
        binding=binding,
        view=view,
        block=block,
        budget=condition_budget(context, condition),
        condition=condition,
        warp=warp,
    )
    benchmark, performance = resolve_benchmark(
        store, config, binding, bindings.provenance(condition)
    )
    rank, mean_superiorities = resolve_mean_superiorities(
        store, context, condition, (benchmark, performance)
    )
    return PortfolioUnit(benchmark, performance, rank, mean_superiorities)


@dataclass(frozen=True)
class SelectionTraining:
    """Instances 11-110: their features, and the portfolio under the views read.

    ``post`` holds the selection condition's transformed runs when augmented cells
    are fitted, and is empty otherwise.
    """

    features: StageArtifact
    feature_table: pd.DataFrame
    # By condition.
    pre: Mapping[str, PortfolioUnit]
    post: Mapping[str, PortfolioUnit] = field(default_factory=dict)

    def mean_superiorities(self, condition: str) -> pd.DataFrame:
        """Return the resolved views' mean superiorities of one condition."""
        units = [self.pre[condition]]
        if condition in self.post:
            units.append(self.post[condition])
        return assemble(
            [unit.mean_superiorities for unit in units], "view_mean_superiority"
        )

    @property
    def inputs(self) -> dict[str, str]:
        """Return the fingerprints a cell fitted on this block may record."""
        inputs = {"train_features": self.features.fingerprint}
        for condition in CONDITIONS:
            inputs[training_mean_superiority_role(PRE, condition)] = self.pre[
                condition
            ].rank.fingerprint
            if condition in self.post:
                inputs[training_mean_superiority_role(POST, condition)] = self.post[
                    condition
                ].rank.fingerprint
        return inputs


@dataclass(frozen=True)
class SelectionHeldOut:
    """Instances 1-10 under one family: eleven deployments and both views' runs."""

    label: str
    features: StageArtifact
    feature_table: pd.DataFrame
    # By condition; ``pre`` is the same for every family.
    pre: Mapping[str, PortfolioUnit]
    post: Mapping[str, PortfolioUnit]

    def performance(self, condition: str) -> pd.DataFrame:
        """Return both views' runs of one condition on the block."""
        return assemble(
            (self.pre[condition].performance, self.post[condition].performance),
            "view_performance",
        )

    def mean_superiorities(self, condition: str) -> pd.DataFrame:
        """Return both views' mean superiorities of one condition on the block."""
        return assemble(
            (
                self.pre[condition].mean_superiorities,
                self.post[condition].mean_superiorities,
            ),
            "view_mean_superiority",
        )

    @property
    def inputs(self) -> dict[str, str]:
        """Return the fingerprints a cell scored on this block records."""
        inputs = {f"test_features:{self.label}": self.features.fingerprint}
        for condition in CONDITIONS:
            pre, post = self.pre[condition], self.post[condition]
            inputs[f"test_performance_pre:{condition}"] = pre.benchmark.fingerprint
            inputs[f"test_mean_superiority_pre:{condition}"] = pre.rank.fingerprint
            inputs[f"test_performance_post:{condition}:{self.label}"] = (
                post.benchmark.fingerprint
            )
            inputs[f"test_mean_superiority_post:{condition}:{self.label}"] = (
                post.rank.fingerprint
            )
        return inputs


def resolve_selection_training(
    store: StageStore,
    context: ExperimentContext,
    bindings: PortfolioBindings,
    trainings: Sequence[str] = TRAINING_CONDITIONS,
) -> SelectionTraining:
    """Resolve the training block, computing what the store lacks.

    The single best solver's transformed training run is not resolved: no cell reads
    it, since the baseline is named from the untransformed block.
    """
    if context.warp != DEFAULT_WARP:
        raise ValueError(
            f"the training block's warp is {context.warp.identifier}, but its "
            f"features are computed under {DEFAULT_WARP.identifier}"
        )
    views = training_views(trainings)
    features, table = resolve_training_features(store, context)
    pre = {
        condition: resolve_portfolio_unit(
            store, context, bindings, condition, view=PRE, block=TRAIN_INSTANCES
        )
        for condition in CONDITIONS
    }
    post: dict[str, PortfolioUnit] = {}
    if POST in views:
        post[CONDITION_SELECTION] = resolve_portfolio_unit(
            store,
            context,
            bindings,
            CONDITION_SELECTION,
            view=POST,
            block=TRAIN_INSTANCES,
            warp=context.warp,
        )
    return SelectionTraining(features, table, pre, post)


def resolve_selection_held_out_pre(
    store: StageStore, context: ExperimentContext, bindings: PortfolioBindings
) -> dict[str, PortfolioUnit]:
    """Resolve the untransformed held-out units every family shares, by condition."""
    return {
        condition: resolve_portfolio_unit(
            store, context, bindings, condition, view=PRE, block=TEST_INSTANCES
        )
        for condition in CONDITIONS
    }


def resolve_selection_held_out(
    store: StageStore,
    context: ExperimentContext,
    bindings: PortfolioBindings,
    audit: Audit,
    pre: Mapping[str, PortfolioUnit],
) -> SelectionHeldOut:
    """Resolve one family's held-out block, computing what the store lacks."""
    family = replace(context, warp=audit.warp)
    features, table = resolve_deployment_features(store, family)
    post = {
        condition: resolve_portfolio_unit(
            store,
            family,
            bindings,
            condition,
            view=POST,
            block=TEST_INSTANCES,
            warp=audit.warp,
        )
        for condition in CONDITIONS
    }
    return SelectionHeldOut(audit.label, features, table, dict(pre), post)


def resolve_selection_held_out_in_range(
    store: StageStore,
    context: ExperimentContext,
    bindings: PortfolioBindings,
    audits: Sequence[Audit] = AUDITS,
) -> list[SelectionHeldOut]:
    """Resolve each family's held-out block, skipping a family out of its range.

    A family that cannot preserve the sample's order is skipped with a warning.
    Experiments 3 and 5 both resolve through here, so their cells name the same blocks.
    """
    pre = resolve_selection_held_out_pre(store, context, bindings)
    blocks: list[SelectionHeldOut] = []
    for audit in audits:
        try:
            blocks.append(
                resolve_selection_held_out(store, context, bindings, audit, pre)
            )
        except OrderPreservationError as failure:
            LOGGER.warning(
                "%s: the family does not preserve the sample's order here, so "
                "this audit is out of its range: %s",
                audit.label,
                failure,
            )
    return blocks


def selection_inputs(
    training: SelectionTraining, blocks: Sequence[SelectionHeldOut]
) -> dict[str, str]:
    """Return the fingerprints of a cell fitted on training and scored on the blocks.

    Two blocks naming a shared unit differently raise (not one binding).
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
