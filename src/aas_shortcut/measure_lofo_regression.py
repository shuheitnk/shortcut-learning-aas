"""Appendix: the superiority regression under leave-one-function-out (LOFO).

Fold f fits on the training instances of the other 23 BBOB functions -- with the
augmented condition, their transformed copies too, built as Experiment 4 builds
them -- and predicts the test instances of function f, untransformed, on every
deployment. R² is taken once per deployment over the 240 pooled out-of-fold
predictions, never per fold.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from aas_shortcut.augment_training_samples import REP_COLUMN
from aas_shortcut.build_regression_model import ModelConfig, fit_model
from aas_shortcut.define_algorithm_portfolio import AlgorithmPair
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.define_repetitions import MODEL_SEEDS, REPETITIONS
from aas_shortcut.define_table_schema import TableSchema
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.evaluate_two_view_cell import GROUP_COLUMNS
from aas_shortcut.execute_parallel_jobs import run_jobs
from aas_shortcut.match_update_budget import learning_settings_of, matched_config
from aas_shortcut.measure_augmented_regression import (
    AugmentedRegressionConfig,
    held_out_blocks,
    predictors_of,
    training_matrix,
)
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.measure_selection_transfer import TRAINING_VIEWS
from aas_shortcut.predict_superiority import TARGET, observed_and_predicted, r2_score
from aas_shortcut.split_bbob_instances import TEST_INSTANCES, TRAIN_INSTANCES

LOGGER = logging.getLogger(__name__)

STAGE = "lofo_regression"
PROTOCOL = "leave-one-function-out"
METRIC = "r2_lofo"
FUNCTIONS: tuple[int, ...] = tuple(range(1, 25))

PREDICTION_TABLE = "metrics/lofo_predictions.parquet"
SCORE_TABLE = "metrics/lofo_scores.parquet"

_CELL = {
    "feature_set": "string",
    "feature_variant": "string",
    "training": "string",
    "pair": "string",
    "group": "string",
    "model": "string",
}
_PROVENANCE = {"scale": "string", "config_hash": "string"}

PREDICTION_SCHEMA = TableSchema(
    name="lofo_prediction",
    columns={
        **_CELL,
        "seed": "int32",
        "function_id": "int32",
        "instance_id": "int32",
        "rep": "int32",
        "observed": "float64",
        "predicted": "float64",
        **_PROVENANCE,
    },
    key_columns=("seed", "rep", "function_id", "instance_id"),
    finite_columns=("observed", "predicted"),
)

SCORE_SCHEMA = TableSchema(
    name="lofo_score",
    columns={
        **_CELL,
        "n_features": "int32",
        "seed": "int32",
        "rep": "int32",
        "metric": "string",
        "value": "float64",
        "n_test": "int32",
        **_PROVENANCE,
    },
    key_columns=("seed", "rep"),
    finite_columns=("value",),
)


@dataclass(frozen=True)
class LofoRegressionCell:
    """One cell: a pair's superiority in one encoding, one training condition."""

    feature_set: str
    variant: str
    pair: AlgorithmPair
    training: str
    features: tuple[str, ...]
    model: ModelConfig = field(default_factory=ModelConfig)
    functions: tuple[int, ...] = FUNCTIONS
    train_instances: tuple[int, ...] = TRAIN_INSTANCES
    test_instances: tuple[int, ...] = TEST_INSTANCES
    n_seeds: int = REPETITIONS

    @property
    def model_seeds(self) -> tuple[int, ...]:
        """The training runs, the study's first ``n_seeds`` model seeds."""
        return MODEL_SEEDS[: self.n_seeds]

    @property
    def views(self) -> int:
        """How many views of each training problem the matrix carries."""
        return len(TRAINING_VIEWS[self.training])

    @property
    def fitted_model(self) -> ModelConfig:
        """The model configuration with the matched epoch budget."""
        return matched_config(self.model, self.views)

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description of the cell."""
        return {
            "stage": STAGE,
            "protocol": PROTOCOL,
            "feature_set": self.feature_set,
            "variant": self.variant,
            "pair": self.pair.slug,
            "training": self.training,
            "features": sorted(self.features),
            "functions": list(self.functions),
            "train_instances": list(self.train_instances),
            "test_instances": list(self.test_instances),
            "model_seeds": list(self.model_seeds),
            "model": self.model.as_dict(),
            "learning": learning_settings_of(self.model, self.views),
        }

    def _matrix_config(self) -> AugmentedRegressionConfig:
        return AugmentedRegressionConfig(
            feature_set=self.feature_set,
            variants=(self.variant,),
            pairs=(self.pair,),
            trainings=(self.training,),
            audits=(CONTROL.label,),
            features=self.features,
            train_instances=self.train_instances,
            test_instances=self.test_instances,
            candidates=(self.model.kind,),
            model=self.model,
            n_seeds=self.n_seeds,
        )

    def matrices(
        self,
        train_features: pd.DataFrame,
        train_superiorities: pd.DataFrame,
        test_features: pd.DataFrame,
        test_superiorities: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return the training matrix and the untransformed test matrix.

        Both are Experiment 4's own matrices, narrowed to the cell's columns; the
        test block is its ``pre`` view with one row per problem and deployment.
        """
        config = self._matrix_config()
        kept = list(self.features)
        train_features = cast(
            pd.DataFrame, train_features[train_features["feature"].isin(kept)]
        )
        test_features = cast(
            pd.DataFrame, test_features[test_features["feature"].isin(kept)]
        )
        train = training_matrix(
            train_features,
            train_superiorities,
            self.variant,
            self.training,
            self.pair.slug,
            config,
        )
        test = held_out_blocks(
            {CONTROL.label: test_features},
            {CONTROL.label: test_superiorities},
            self.variant,
            self.pair.slug,
            config,
        )[CONTROL.label][TRANSFORMS[0]]
        return train, test


def predict_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    model: ModelConfig,
    seed: int,
) -> pd.DataFrame:
    """Fit on one fold's training rows and predict its held-out rows."""
    fitted = fit_model(
        train,
        columns,
        replace(model, random_state=seed),
        TARGET,
        group_columns=GROUP_COLUMNS,
    )
    observed, predicted = observed_and_predicted(fitted, test, columns)
    return pd.DataFrame(
        {
            "function_id": test["function_id"].to_numpy(dtype=np.int32),
            "instance_id": test["instance_id"].to_numpy(dtype=np.int32),
            "rep": test[REP_COLUMN].to_numpy(dtype=np.int32),
            "observed": observed,
            "predicted": predicted,
        }
    )


def pooled_scores(predictions: pd.DataFrame, n_test: int) -> pd.DataFrame:
    """R² per seed and deployment over the pooled out-of-fold predictions."""
    rows = []
    for (seed, rep), group in predictions.groupby(["seed", "rep"], sort=True):
        if len(group) != n_test:
            raise ValueError(
                f"seed {seed} deployment {rep} holds {len(group)} predictions, "
                f"expected {n_test}"
            )
        rows.append(
            {
                "seed": seed,
                "rep": rep,
                "metric": METRIC,
                "value": r2_score(
                    group["observed"].to_numpy(), group["predicted"].to_numpy()
                ),
                "n_test": len(group),
            }
        )
    return pd.DataFrame(rows)


def measure_cells(
    cells: Sequence[LofoRegressionCell],
    train_features: pd.DataFrame,
    train_superiorities: pd.DataFrame,
    test_features: pd.DataFrame,
    test_superiorities: pd.DataFrame,
    n_jobs: int,
) -> dict[LofoRegressionCell, tuple[pd.DataFrame, pd.DataFrame]]:
    """Run every fold of every seed of the cells; return predictions and scores."""
    tasks: list[tuple[Any, ...]] = []
    keys: list[tuple[LofoRegressionCell, int]] = []
    widths: dict[LofoRegressionCell, int] = {}
    for cell in cells:
        train, test = cell.matrices(
            train_features, train_superiorities, test_features, test_superiorities
        )
        columns = predictors_of(train)
        widths[cell] = len(columns)
        for function_id in cell.functions:
            fold_train = cast(pd.DataFrame, train[train["function_id"] != function_id])
            fold_test = cast(pd.DataFrame, test[test["function_id"] == function_id])
            if fold_train.empty or fold_test.empty:
                raise ValueError(f"fold f{function_id} of {cell.pair.slug} is empty")
            for seed in cell.model_seeds:
                tasks.append((fold_train, fold_test, columns, cell.fitted_model, seed))
                keys.append((cell, seed))
    LOGGER.info(
        "LOFO regression: %d cell(s), %d fit(s) on %d worker(s)",
        len(cells),
        len(tasks),
        n_jobs,
    )
    collected: dict[LofoRegressionCell, list[pd.DataFrame]] = {c: [] for c in cells}
    for index, frame in run_jobs("LOFO regression", tasks, predict_fold, n_jobs=n_jobs):
        cell, seed = keys[index]
        collected[cell].append(frame.assign(seed=seed))

    results = {}
    for cell, frames in collected.items():
        identity = {
            "feature_set": cell.feature_set,
            "feature_variant": cell.variant,
            "training": cell.training,
            "pair": cell.pair.slug,
            "group": group_of(cell.pair),
            "model": cell.model.kind,
        }
        predictions = pd.concat(frames, ignore_index=True)
        scores = pooled_scores(
            predictions, len(cell.functions) * len(cell.test_instances)
        )
        results[cell] = (
            predictions.assign(**identity),
            scores.assign(**identity, n_features=widths[cell]),
        )
    return results
