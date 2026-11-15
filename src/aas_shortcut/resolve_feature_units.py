"""Resolving feature units, and checking every one against the paper's columns.

A unit's fingerprint depends only on the feature stage's config, so every experiment
resolves the same units. Each resolution stops if the stage dropped a column the
paper reads.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from aas_shortcut.calculate_features import FeatureStageConfig, run_feature_stage
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_feature_columns import check_feature_unit
from aas_shortcut.define_paths import MANIFEST_SUFFIX, TABLE_SUFFIX, shard_stem
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS, TEST_INSTANCES


def feature_table(config: FeatureStageConfig, *, deployment: bool) -> str:
    """Where the feature stage writes the shard inside a unit."""
    kind = "deployment_features" if deployment else "features"
    stem = shard_stem(kind, config.problem_class, config.dim)
    return f"features/{stem}{TABLE_SUFFIX}"


def feature_unit(config: FeatureStageConfig) -> StageUnit:
    """The unit of one feature configuration."""
    return StageUnit("features", config.as_dict())


def resolve_feature_unit(
    store: StageStore, config: FeatureStageConfig, *, deployment: bool
) -> tuple[StageArtifact, pd.DataFrame]:
    """Return a feature unit and its table, computing it if the store lacks it."""
    relative = feature_table(config, deployment=deployment)

    def compute(paths: Any, config_hash: str) -> None:
        shard = (
            paths.deployment_features(config.problem_class, config.dim)
            if deployment
            else paths.features(config.problem_class, config.dim)
        )
        run_feature_stage(config, paths, config_hash, shard=shard)

    artifact = store.resolve(feature_unit(config), compute)
    table = artifact.read_table(relative)
    with artifact.file(f"{relative}{MANIFEST_SUFFIX}").open(encoding="utf-8") as handle:
        diagnostics = json.load(handle)["diagnostics"]
    check_feature_unit(table, diagnostics, str(artifact.path), config.scale)
    return artifact, table


def resolve_training_features(
    store: StageStore, context: ExperimentContext
) -> tuple[StageArtifact, pd.DataFrame]:
    """The training block's single sample, both views."""
    return resolve_feature_unit(
        store, context.feature_config(PROBLEM_CLASS), deployment=False
    )


def resolve_deployment_features(
    store: StageStore,
    context: ExperimentContext,
    instances: tuple[int, ...] = TEST_INSTANCES,
) -> tuple[StageArtifact, pd.DataFrame]:
    """The held-out block's eleven deployments under the context's transformation."""
    config = context.deployment_feature_config(PROBLEM_CLASS, instances)
    return resolve_feature_unit(store, config, deployment=True)
