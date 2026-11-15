"""Appendix: a regression cell's prediction for every test problem and deployment.

A regression unit keeps one R² per deployment. The bootstrap over problems and the
appendix's per-problem checks need each prediction, so the cell is fitted again on
the same matrix, configuration and seed, and each deployment is predicted as the
rows its score was taken on. Rescoring the predictions must give the stored R² bit
for bit (:func:`check_reproduced`).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
from joblib import delayed

from aas_shortcut.augment_training_samples import REP_COLUMN
from aas_shortcut.build_regression_model import fit_model
from aas_shortcut.collect_parallel_results import as_rows
from aas_shortcut.define_table_schema import TableSchema
from aas_shortcut.evaluate_two_view_cell import (
    GROUP_COLUMNS,
    SINGLE_SAMPLE,
    TEST_METRICS,
)
from aas_shortcut.execute_parallel_jobs import parallel_for
from aas_shortcut.match_update_budget import matched_config
from aas_shortcut.measure_augmented_regression import (
    AugmentedRegressionConfig,
    held_out_blocks,
    predictors_of,
    training_matrix,
)
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.measure_selection_transfer import TRAINING_VIEWS
from aas_shortcut.predict_superiority import TARGET, observed_and_predicted, r2_score
from aas_shortcut.resolve_regression_cells import RegressionCell, RegressionData

LOGGER = logging.getLogger(__name__)

STAGE = "regression_predictions"
PREDICTION_TABLE = "metrics/predictions.parquet"

PREDICTION_SCHEMA = TableSchema(
    name="regression_prediction",
    columns={
        "feature_set": "string",
        "feature_variant": "string",
        "training": "string",
        "pair": "string",
        "group": "string",
        "model": "string",
        "seed": "int32",
        "audit": "string",
        "transform": "string",
        "rep": "int32",
        "function_id": "int32",
        "instance_id": "int32",
        "observed": "float64",
        "predicted": "float64",
        "scale": "string",
        "config_hash": "string",
    },
    key_columns=("seed", "audit", "transform", "rep", "function_id", "instance_id"),
    finite_columns=("observed", "predicted"),
)

PROBLEM_KEYS: tuple[str, ...] = ("function_id", "instance_id")


def deployment_rows(block: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    """Split a block as `evaluate_two_view_cell.deployment_scores` scores it."""
    if REP_COLUMN not in block.columns or block[REP_COLUMN].nunique() < 2:
        return [(SINGLE_SAMPLE, block)]
    return [
        (int(cast(int, rep)), rows)
        for rep, rows in block.groupby(REP_COLUMN, sort=True)
    ]


def predict_seed(
    train: pd.DataFrame,
    blocks: Mapping[str, Mapping[str, pd.DataFrame]],
    columns: list[str],
    cell: RegressionCell,
    config: AugmentedRegressionConfig,
    seed: int,
) -> list[pd.DataFrame]:
    """Fit one cell under one seed, as `score_seed` does, and keep its predictions."""
    views = len(TRAINING_VIEWS[cell.training])
    model_config = replace(
        matched_config(config.config_for(cell.kind), views), random_state=seed
    )
    model = fit_model(train, columns, model_config, TARGET, group_columns=GROUP_COLUMNS)
    frames: list[pd.DataFrame] = []
    for label, views_of in blocks.items():
        for transform, block in views_of.items():
            for rep, rows in deployment_rows(block):
                observed, predicted = observed_and_predicted(model, rows, columns)
                frames.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "audit": label,
                            "transform": transform,
                            "rep": rep,
                            "function_id": rows["function_id"].to_numpy(),
                            "instance_id": rows["instance_id"].to_numpy(),
                            "observed": observed,
                            "predicted": predicted,
                        }
                    )
                )
    return frames


def measure_predictions(
    config: AugmentedRegressionConfig,
    cells: Sequence[RegressionCell],
    data: RegressionData,
) -> dict[RegressionCell, pd.DataFrame]:
    """Refit every cell's seeds and return each cell's predictions."""
    trains = {
        cell: training_matrix(
            data.train_features,
            data.train_superiorities,
            cell.variant,
            cell.training,
            cell.pair.slug,
            config,
        )
        for cell in cells
    }
    blocks = {
        cell: held_out_blocks(
            data.test_features,
            data.test_superiorities,
            cell.variant,
            cell.pair.slug,
            config,
        )
        for cell in cells
    }
    jobs = [(cell, seed) for cell in cells for seed in config.model_seeds]
    LOGGER.info(
        "regression predictions: %d cell(s), %d fit(s) on %d worker(s)",
        len(cells),
        len(jobs),
        config.n_jobs,
    )
    results = as_rows(
        parallel_for(config.n_jobs, len(jobs))(
            delayed(predict_seed)(
                trains[cell],
                blocks[cell],
                predictors_of(trains[cell]),
                cell,
                config,
                seed,
            )
            for cell, seed in jobs
        )
    )
    collected: dict[RegressionCell, list[pd.DataFrame]] = {cell: [] for cell in cells}
    for (cell, _), frames in zip(jobs, results, strict=True):
        collected[cell].extend(frames)
    return {
        cell: pd.concat(frames, ignore_index=True).assign(
            feature_set=config.feature_set,
            feature_variant=cell.variant,
            training=cell.training,
            pair=cell.pair.slug,
            group=group_of(cell.pair),
            model=cell.kind,
        )
        for cell, frames in collected.items()
    }


def rescore(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return R² per seed, audit, view and deployment, from the predictions."""
    rows = [
        {
            "seed": int(seed),
            "audit": str(audit),
            "metric": TEST_METRICS[str(transform)],
            "rep": int(rep),
            "value": r2_score(
                group["observed"].to_numpy(), group["predicted"].to_numpy()
            ),
        }
        for (seed, audit, transform, rep), group in predictions.groupby(
            ["seed", "audit", "transform", "rep"], sort=True, observed=True
        )
    ]
    return pd.DataFrame(rows)


def check_reproduced(predictions: pd.DataFrame, deployments: pd.DataFrame) -> int:
    """Stop unless the predictions give each stored R² bit for bit; count them."""
    keys = ["seed", "audit", "metric", "rep"]
    stored = deployments[deployments["metric"].isin(list(TEST_METRICS.values()))]
    stored = stored.assign(
        seed=stored["seed"].astype(int),
        audit=stored["audit"].astype(str),
        metric=stored["metric"].astype(str),
        rep=stored["rep"].astype(int),
    )[[*keys, "value"]]
    joined = rescore(predictions).merge(
        stored, on=keys, how="outer", suffixes=("_refit", "_stored"), indicator=True
    )
    unmatched = joined[joined["_merge"] != "both"]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} score(s) are on one side only, e.g. "
            f"{unmatched.iloc[0][keys].to_dict()}"
        )
    refit = joined["value_refit"].to_numpy(dtype=np.float64)
    held = joined["value_stored"].to_numpy(dtype=np.float64)
    differ = refit != held
    if differ.any():
        worst = float(np.max(np.abs(refit - held)))
        raise ValueError(
            f"{int(differ.sum())} of {len(joined)} refitted R² differ from the "
            f"stored ones (largest gap {worst:.3g}); the refit is not the fit"
        )
    return len(joined)
