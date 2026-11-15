"""Regression cells: one unit per fit of one pair's superiority.

A cell is a representation, an encoding, a target pair, a training condition and a
backend; its unit holds every training run's score on every deployment. The missing
cells of one call are fitted in one pass to keep one pool of workers busy; a cell's
result depends only on its own matrices and seeds, not on which cells share the pass.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aas_shortcut.define_algorithm_portfolio import AlgorithmPair
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import TRANSFORMS, RunPaths
from aas_shortcut.define_table_schema import SCHEMAS
from aas_shortcut.define_warp_audits import AUDITS, Audit
from aas_shortcut.match_update_budget import learning_settings_of
from aas_shortcut.measure_augmented_regression import (
    DEPLOYMENT_SHARD,
    AugmentedRegressionConfig,
    run_augmented_regression_stage,
    seed_levels,
)
from aas_shortcut.measure_selection_transfer import TRAINING_VIEWS
from aas_shortcut.read_write_tables import read_table, write_table
from aas_shortcut.resolve_benchmark_units import assemble
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    block_inputs,
    training_superiority_role,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit
from aas_shortcut.split_bbob_instances import TRAIN_INSTANCES

_SCHEMAS = {DEPLOYMENT_SHARD: "deployment_score"}


@dataclass(frozen=True)
class RegressionCell:
    """One fit: an encoding of a pair's superiority, a training condition, a backend."""

    pair: AlgorithmPair
    variant: str
    training: str
    kind: str


@dataclass(frozen=True)
class RegressionData:
    """What every cell of one call is fitted on and scored against."""

    train_features: pd.DataFrame
    train_superiorities: pd.DataFrame
    test_features: Mapping[str, pd.DataFrame]
    test_superiorities: Mapping[str, pd.DataFrame]
    # Fingerprint of every unit those tables came from, by role.
    inputs: Mapping[str, str]


def family_regression_config(
    context: ExperimentContext,
    feature_set: str,
    candidates: Sequence[str],
    audits: Sequence[Audit] = AUDITS,
) -> AugmentedRegressionConfig:
    """Experiment 4's regression of one representation, every family scored."""
    return AugmentedRegressionConfig(
        feature_set=feature_set,
        audits=tuple(audit.label for audit in audits),
        features=paper_columns(context.feature_sets, context.scale),
        candidates=tuple(candidates),
        model=context.model,
        n_jobs=context.n_jobs,
    )


def regression_data(
    training: TrainingBlock,
    blocks: Sequence[HeldOutBlock],
    features: Sequence[str],
) -> RegressionData:
    """What a configuration's cells are fitted on and scored against."""
    kept = list(features)
    source = training.feature_table
    train_features = cast(
        pd.DataFrame,
        source[
            source["instance_id"].isin(TRAIN_INSTANCES) & source["feature"].isin(kept)
        ],
    )
    test_features = {
        block.label: cast(
            pd.DataFrame, block.feature_table[block.feature_table["feature"].isin(kept)]
        )
        for block in blocks
    }
    return RegressionData(
        train_features=train_features,
        train_superiorities=training.superiorities,
        test_features=test_features,
        test_superiorities={block.label: block.superiorities for block in blocks},
        inputs=block_inputs(training, blocks),
    )


def cells_of(config: AugmentedRegressionConfig) -> list[RegressionCell]:
    """Every cell of a configuration, in the order the stage fits them."""
    return [
        RegressionCell(pair, variant, training, kind)
        for pair in config.pairs
        for variant in config.variants
        for training in config.trainings
        for kind in config.candidates
    ]


def learning_settings(
    config: AugmentedRegressionConfig, cell: RegressionCell
) -> dict[str, Any]:
    """What a cell's fit depends on that the stage's configuration does not carry."""
    return learning_settings_of(
        config.config_for(cell.kind), len(TRAINING_VIEWS[cell.training])
    )


def _narrowed(
    config: AugmentedRegressionConfig, cell: RegressionCell
) -> AugmentedRegressionConfig:
    return replace(
        config,
        pairs=(cell.pair,),
        variants=(cell.variant,),
        trainings=(cell.training,),
        candidates=(cell.kind,),
    )


def cell_inputs(inputs: Mapping[str, str], training: str) -> dict[str, str]:
    """The fingerprints a cell of one training condition reads.

    A plain cell does not name the transformed training superiorities, so it is not
    refitted when only they change. The training features stay one unit for both
    views, since splitting it would change every downstream fingerprint.
    """
    read = TRAINING_VIEWS[training]
    missing = [view for view in read if training_superiority_role(view) not in inputs]
    if missing:
        raise ValueError(
            f"a {training!r} cell fits on the training block's {missing} view(s), "
            "which the block was not resolved under"
        )
    unread = {
        training_superiority_role(view) for view in TRANSFORMS if view not in read
    }
    return {role: value for role, value in inputs.items() if role not in unread}


def regression_cell_unit(
    config: AugmentedRegressionConfig,
    cell: RegressionCell,
    inputs: Mapping[str, str],
) -> StageUnit:
    """The unit of one cell of a regression configuration."""
    settings = {
        **_narrowed(config, cell).as_dict(),
        "learning": learning_settings(config, cell),
    }
    return StageUnit(
        "augmented_regression", settings, inputs=cell_inputs(inputs, cell.training)
    )


def cell_table(shard: str) -> str:
    """Where a cell's unit holds one of the stage's tables."""
    return f"metrics/{shard}.parquet"


def rows_on_family(
    cells: Mapping[RegressionCell, StageArtifact], label: str
) -> pd.DataFrame:
    """The cells' rows on one family, laid out as one representation's table.

    This is how Experiment 2 reads Experiment 4's plain cells; a row carries the
    median of the eleven training runs.
    """
    trainings = {cell.training for cell in cells}
    if len(trainings) > 1:
        raise ValueError(
            f"the cells are of the training conditions {sorted(trainings)}; one "
            "representation's table holds one"
        )
    frames: list[pd.DataFrame] = []
    for cell, artifact in cells.items():
        table = seed_levels(artifact.read_table(cell_table(DEPLOYMENT_SHARD)))
        audit = table["audit"].astype(str)
        if not (audit == label).any():
            raise ValueError(f"the cell {cell} holds no row on {label}")
        frames.append(
            cast(
                pd.DataFrame, table[audit == label].drop(columns=["training", "audit"])
            )
        )
    return assemble(frames, "deployment_score")


def _cell_rows(
    table: pd.DataFrame, config: AugmentedRegressionConfig, cell: RegressionCell
) -> pd.Series:
    return (
        (table["feature_set"].astype(str) == config.feature_set)
        & (table["feature_variant"].astype(str) == cell.variant)
        & (table["training"].astype(str) == cell.training)
        & (table["pair"].astype(str) == cell.pair.slug)
        & (table["model"].astype(str) == cell.kind)
    )


def resolve_regression_cells(
    store: StageStore,
    config: AugmentedRegressionConfig,
    cells: Sequence[RegressionCell],
    data: RegressionData,
    scale: str,
) -> dict[RegressionCell, StageArtifact]:
    """Return every cell, fitting the missing ones -- and only those -- together."""
    units = {cell: regression_cell_unit(config, cell, data.inputs) for cell in cells}
    shards = [DEPLOYMENT_SHARD]

    def compute(
        missing: Sequence[StageUnit], scratches: Mapping[str, RunPaths]
    ) -> None:
        wanted = [cell for cell, unit in units.items() if unit in missing]
        with tempfile.TemporaryDirectory(dir=store.root) as scratch:
            paths = RunPaths(Path(scratch))
            run_augmented_regression_stage(
                data.train_features,
                data.train_superiorities,
                data.test_features,
                data.test_superiorities,
                paths,
                scale,
                "batch",
                config,
                deployment_name=DEPLOYMENT_SHARD,
                cells=[
                    (cell.pair, cell.variant, cell.training, cell.kind)
                    for cell in wanted
                ],
            )
            written = {shard: read_table(paths.metric(shard).data) for shard in shards}
        for cell in wanted:
            unit = units[cell]
            target = scratches[unit.fingerprint]
            for shard, table in written.items():
                rows = table[_cell_rows(table, config, cell)]
                if rows.empty:
                    raise ValueError(
                        f"the regression wrote no {shard} row for {cell}; the cell "
                        "cannot be stored"
                    )
                write_table(
                    rows.assign(config_hash=unit.fingerprint),
                    target.metric(shard).data,
                    SCHEMAS[_SCHEMAS[shard]],
                )

    artifacts = store.resolve_together(list(units.values()), compute)
    return {cell: artifacts[unit.fingerprint] for cell, unit in units.items()}


def resolve_family_deployments(
    store: StageStore,
    context: ExperimentContext,
    feature_set: str,
    candidates: Sequence[str],
    training: TrainingBlock,
    blocks: Sequence[HeldOutBlock],
    audits: Sequence[Audit],
) -> pd.DataFrame:
    """Every training run of Experiment 4's regression of one representation.

    Experiment 4 and the appendix both read the regression through here, so they
    cannot name different units.
    """
    config = family_regression_config(context, feature_set, candidates, audits)
    cells = cells_of(config)
    artifacts = resolve_regression_cells(
        store,
        config,
        cells,
        regression_data(training, blocks, config.features),
        context.scale,
    )
    return assemble(
        [artifacts[cell].read_table(cell_table(DEPLOYMENT_SHARD)) for cell in cells],
        "deployment_score",
    )
