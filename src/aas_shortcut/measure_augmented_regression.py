"""Experiment 4: the superiority regression, with and without transform augmentation.

``plain`` fits on the ``pre`` view; ``augmented`` adds the same instances under their
own transformation, each row with its own view's superiority. Each cell is fitted
once and scored on the held-out block under all five transformation families
(audit points), since the training block does not depend on the family.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast

import pandas as pd
from joblib import delayed

from aas_shortcut.augment_training_samples import (
    REDUCTION_MEAN,
    REDUCTION_ROWS,
    REP_COLUMN,
    build_matrix,
)
from aas_shortcut.build_feature_matrix import (
    INSTANCE_COLUMNS,
    feature_columns,
    join_targets,
)
from aas_shortcut.build_regression_model import (
    DEFAULT_CANDIDATES,
    MODEL_KINDS,
    ModelConfig,
    fit_model,
)
from aas_shortcut.collect_parallel_results import as_rows
from aas_shortcut.compare_training_runs import (
    SEED_COLUMN,
    median_over_seeds,
    per_seed,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS, GROUPS, AlgorithmPair
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    TRANSFORMS,
    RunPaths,
    fingerprint_variants,
)
from aas_shortcut.define_repetitions import MODEL_SEEDS, REPETITIONS
from aas_shortcut.define_reporting_scopes import scope_rows
from aas_shortcut.define_table_schema import DEPLOYMENT_SCORE_SCHEMA
from aas_shortcut.evaluate_two_view_cell import (
    GROUP_COLUMNS,
    TEST_METRICS,
    deployment_scores,
)
from aas_shortcut.execute_parallel_jobs import parallel_for
from aas_shortcut.match_update_budget import describe_budget, matched_config
from aas_shortcut.measure_representation_transfer import VARIANTS, group_of
from aas_shortcut.measure_selection_transfer import (
    TRAINING_CONDITIONS,
    TRAINING_VIEWS,
)
from aas_shortcut.predict_superiority import TARGET
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.split_bbob_instances import TEST_INSTANCES, TRAIN_INSTANCES

LOGGER = logging.getLogger(__name__)


# Every training run's score on every deployment of every held-out block.
DEPLOYMENT_SHARD = "deployment_scores"


@dataclass(frozen=True)
class AugmentedRegressionConfig:
    """Settings of Experiment 4's regression."""

    feature_set: str = ELA_FEATURE_SET
    variants: tuple[str, ...] = VARIANTS
    pairs: tuple[AlgorithmPair, ...] = FOCUS_PAIRS
    trainings: tuple[str, ...] = TRAINING_CONDITIONS
    # Audit points, by label, in table order.
    audits: tuple[str, ...] = ()
    # The features every audit point and the training block all carry.
    features: tuple[str, ...] = ()
    train_instances: tuple[int, ...] = TRAIN_INSTANCES
    test_instances: tuple[int, ...] = TEST_INSTANCES
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES
    model: ModelConfig = field(default_factory=ModelConfig)
    # Training runs per cell (the first model seeds); tests use fewer to stay fast.
    n_seeds: int = REPETITIONS
    n_jobs: int = 1

    def __post_init__(self) -> None:
        """Reject unknown or empty axes and overlapping training and test blocks."""
        unknown = [name for name in self.variants if name not in VARIANTS]
        if unknown:
            raise ValueError(
                f"unknown feature variant(s) {unknown}; expected a subset of "
                f"{list(VARIANTS)}"
            )
        unknown = [name for name in self.trainings if name not in TRAINING_VIEWS]
        if unknown:
            raise ValueError(
                f"unknown training condition(s) {unknown}; expected a subset of "
                f"{list(TRAINING_CONDITIONS)}"
            )
        unknown = [kind for kind in self.candidates if kind not in MODEL_KINDS]
        if unknown:
            raise ValueError(
                f"unknown candidate backend(s) {unknown}; expected a subset of "
                f"{list(MODEL_KINDS)}"
            )
        for axis, values in (
            ("representation", self.variants),
            ("target pair", self.pairs),
            ("training condition", self.trainings),
            ("audit point", self.audits),
            ("candidate backend", self.candidates),
        ):
            if not values:
                raise ValueError(f"the comparison needs at least one {axis}")
        shared = sorted(set(self.train_instances) & set(self.test_instances))
        if shared:
            raise ValueError(
                f"instances {shared} are on both sides of the split; the test "
                "block must be disjoint from the training block"
            )
        if not 1 <= self.n_seeds <= len(MODEL_SEEDS):
            raise ValueError(
                f"n_seeds is {self.n_seeds}; a cell fits between 1 and "
                f"{len(MODEL_SEEDS)} of the study's model seeds"
            )

    @property
    def model_seeds(self) -> tuple[int, ...]:
        """The training runs every cell fits, in the study's order."""
        return MODEL_SEEDS[: self.n_seeds]

    def config_for(self, kind: str) -> ModelConfig:
        """Return the shared hyper-parameters at one candidate's backend."""
        return replace(self.model, kind=kind)

    def as_dict(self) -> dict:
        """Fingerprintable description of the configuration."""
        return {
            "stage": "augmented_regression",
            "feature_set": self.feature_set,
            "variants": fingerprint_variants(self.variants),
            "pairs": [pair.slug for pair in self.pairs],
            "trainings": list(self.trainings),
            "audits": list(self.audits),
            "features": sorted(self.features),
            "train_instances": list(self.train_instances),
            "test_instances": list(self.test_instances),
            "candidates": list(self.candidates),
            "model_seeds": list(self.model_seeds),
            "models": {
                kind: self.config_for(kind).as_dict() for kind in self.candidates
            },
        }


# ---------------------------------------------------------------------------
# Matrices
# ---------------------------------------------------------------------------


def _labelled(
    features: pd.DataFrame,
    superiorities: pd.DataFrame,
    transform: str,
    variant: str,
    instances: Sequence[int],
    pair: str,
    feature_set: str,
    *,
    per_rep: bool,
) -> pd.DataFrame:
    """Build one block's feature matrix, in one encoding, with its target attached.

    ``per_rep`` keeps each deployment sample as its own row: averaging the samples
    would score a practitioner who had sampled the problem eleven times.
    """
    matrix = build_matrix(
        features,
        transform,
        variant,
        feature_set=feature_set,
        instances=list(instances),
        reduction=REDUCTION_ROWS if per_rep else REDUCTION_MEAN,
    )
    targets = cast(pd.DataFrame, superiorities[superiorities["transform"] == transform])
    if targets.empty:
        held = sorted(str(name) for name in superiorities["transform"].unique())
        raise ValueError(
            f"the superiority table holds no {transform} row; it covers {held}"
        )
    joined = join_targets(
        matrix,
        targets,
        pair,
        validate="m:1" if per_rep else "1:1",
    )
    if joined.empty:
        raise ValueError(
            f"no {transform} row of {pair} survives the join of features and win "
            f"rates on instances {list(instances)}"
        )
    return joined.assign(transform=transform).reset_index(drop=True)


def training_matrix(
    features: pd.DataFrame,
    superiorities: pd.DataFrame,
    variant: str,
    training: str,
    pair: str,
    config: AugmentedRegressionConfig,
) -> pd.DataFrame:
    """Build one cell's training matrix under one training condition.

    The ``plain`` matrix is, bit for bit, the first half of the ``augmented`` one.
    """
    return pd.concat(
        [
            _labelled(
                features,
                superiorities,
                transform,
                variant,
                config.train_instances,
                pair,
                config.feature_set,
                per_rep=False,
            )
            for transform in TRAINING_VIEWS[training]
        ],
        ignore_index=True,
    )


def held_out_blocks(
    features_by_audit: Mapping[str, pd.DataFrame],
    superiorities_by_audit: Mapping[str, pd.DataFrame],
    variant: str,
    pair: str,
    config: AugmentedRegressionConfig,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Build ``{audit label: {view: matrix}}`` for one cell.

    The ``pre`` block does not depend on the family, so its five copies must score
    identically.
    """
    return {
        label: {
            transform: _labelled(
                features_by_audit[label],
                superiorities_by_audit[label],
                transform,
                variant,
                config.test_instances,
                pair,
                config.feature_set,
                per_rep=True,
            )
            for transform in TRANSFORMS
        }
        for label in config.audits
    }


def predictors_of(matrix: pd.DataFrame) -> list[str]:
    """Return the matrix's predictor columns, target and identifiers excluded."""
    identifiers = {TARGET, REP_COLUMN, "transform", *INSTANCE_COLUMNS}
    return [column for column in feature_columns(matrix) if column not in identifiers]


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _deployment_sizes(block: pd.DataFrame) -> dict[int, int]:
    """Rows each deployment of a held-out block was scored on."""
    if REP_COLUMN not in block.columns:
        return {}
    counted = block.groupby(REP_COLUMN, observed=True).size()
    return {int(cast(int, rep)): int(rows) for rep, rows in counted.items()}


def score_seed(
    train: pd.DataFrame,
    blocks: Mapping[str, Mapping[str, pd.DataFrame]],
    columns: list[str],
    pair: AlgorithmPair,
    variant: str,
    training: str,
    kind: str,
    config: AugmentedRegressionConfig,
    seed: int,
) -> list[dict]:
    """Fit one cell under one model seed and score every held-out block.

    Each training run is its own task so that a one-cell stage still uses every
    worker.
    """
    if not columns:
        raise ValueError(
            f"the {variant!r} matrix of {config.feature_set} has no predictor "
            "column; there is nothing to fit"
        )
    for label, views in blocks.items():
        for transform, block in views.items():
            missing = [column for column in columns if column not in block.columns]
            if missing:
                raise ValueError(
                    f"the {label} {transform} block is missing {len(missing)} "
                    f"predictor(s) the training matrix has, e.g. {missing[:3]}"
                )

    # Divide the epoch budget by the number of training views, so the augmented
    # matrix gets as many updates as the one-view fit.
    views = len(TRAINING_VIEWS[training])
    model_config = replace(
        matched_config(config.config_for(kind), views), random_state=seed
    )
    identity = {
        "feature_set": config.feature_set,
        "feature_variant": variant,
        "pair": pair.slug,
        "group": group_of(pair),
        "training": training,
        "model": kind,
        "n_features": len(columns),
        "seed": seed,
    }
    model = fit_model(train, columns, model_config, TARGET, group_columns=GROUP_COLUMNS)
    rows: list[dict] = []
    for label, views_of in blocks.items():
        for transform, block in views_of.items():
            sizes = _deployment_sizes(block)
            rows.extend(
                {
                    **identity,
                    "audit": label,
                    "rep": rep,
                    "metric": TEST_METRICS[transform],
                    "value": value,
                    "n_test": sizes.get(rep, len(block)),
                }
                for rep, value in deployment_scores(model, block, columns).items()
            )
    LOGGER.info(
        "%s/%s/%s/%s seed %d (%d features, %d training row(s)): done",
        group_of(pair),
        variant,
        training,
        kind,
        seed,
        len(columns),
        len(train),
    )
    return rows


def run_augmented_regression_stage(
    train_features: pd.DataFrame,
    train_superiorities: pd.DataFrame,
    test_features: Mapping[str, pd.DataFrame],
    test_superiorities: Mapping[str, pd.DataFrame],
    run_paths: RunPaths,
    scale: str,
    config_hash: str,
    config: AugmentedRegressionConfig | None = None,
    *,
    deployment_name: str = DEPLOYMENT_SHARD,
    cells: Sequence[tuple[AlgorithmPair, str, str, str]] | None = None,
) -> pd.DataFrame:
    """Fit every cell's training runs and persist what each deployment scored.

    ``cells`` (``(pair, encoding, training, backend)`` tuples) selects a subset;
    ``None`` fits all. A cell's rows depend only on its own matrices and seeds.
    """
    config = config or AugmentedRegressionConfig()
    if cells is None:
        cells = [
            (pair, variant, training, kind)
            for pair in config.pairs
            for variant in config.variants
            for training in config.trainings
            for kind in config.candidates
        ]
    else:
        cells = list(cells)
        outside = [
            cell
            for cell in cells
            if cell[0] not in config.pairs
            or cell[1] not in config.variants
            or cell[2] not in config.trainings
            or cell[3] not in config.candidates
        ]
        if outside or not cells:
            raise ValueError(
                "the cells to fit must be some of the configuration's; got "
                f"{len(cells)} cell(s), {len(outside)} outside it"
            )
    trains = {
        (pair.slug, variant, training): training_matrix(
            train_features, train_superiorities, variant, training, pair.slug, config
        )
        for pair, variant, training in dict.fromkeys(
            (pair, variant, training) for pair, variant, training, _ in cells
        )
    }
    blocks = {
        (pair.slug, variant): held_out_blocks(
            test_features, test_superiorities, variant, pair.slug, config
        )
        for pair, variant in dict.fromkeys(
            (pair, variant) for pair, variant, *_ in cells
        )
    }
    training_rows: dict[str, int] = {}
    for pair, variant, training, _ in cells:
        training_rows.setdefault(training, len(trains[(pair.slug, variant, training)]))
    LOGGER.info(
        "augmented regression: %d cell(s) x %d training run(s) on %d worker(s); "
        "%d audit point(s); training rows %s",
        len(cells),
        len(config.model_seeds),
        config.n_jobs,
        len(config.audits),
        training_rows,
    )
    for training, rows in training_rows.items():
        LOGGER.info(
            "augmented regression: %s: %s",
            training,
            describe_budget(rows, config.model, len(TRAINING_VIEWS[training])),
        )

    jobs = [
        (pair, variant, training, kind, seed)
        for pair, variant, training, kind in cells
        for seed in config.model_seeds
    ]
    results = as_rows(
        parallel_for(config.n_jobs, len(jobs))(
            delayed(score_seed)(
                trains[(pair.slug, variant, training)],
                blocks[(pair.slug, variant)],
                predictors_of(trains[(pair.slug, variant, training)]),
                pair,
                variant,
                training,
                kind,
                config,
                seed,
            )
            for pair, variant, training, kind, seed in jobs
        )
    )
    deployments = pd.DataFrame([row for rows in results for row in rows]).assign(
        scale=scale, config_hash=config_hash
    )
    shard = run_paths.metric(deployment_name)
    written = write_table(deployments, shard.data, DEPLOYMENT_SCORE_SCHEMA)
    LOGGER.info("augmented regression: wrote %s (%d rows)", shard.data, len(written))
    return written


# ---------------------------------------------------------------------------
# Reading the table
# ---------------------------------------------------------------------------


# The columns naming a cell and family, without the training run and sample.
LEVEL_KEYS: tuple[str, ...] = (
    "feature_set",
    "feature_variant",
    "training",
    "audit",
    "pair",
    "group",
    "model",
    "n_features",
    "metric",
    "n_test",
    "scale",
    "config_hash",
)


def seed_levels(
    deployments: pd.DataFrame, seeds: Sequence[int] = MODEL_SEEDS
) -> pd.DataFrame:
    """Return each cell, family and view's median over runs of the median deployment."""
    rows = deployments[deployments["metric"].isin(list(TEST_METRICS.values()))]
    keys = [name for name in LEVEL_KEYS if name in rows.columns]
    runs = (
        rows.groupby([*keys, SEED_COLUMN], observed=True)["value"]
        .median()
        .reset_index()
    )
    held = runs.groupby(keys, observed=True)[SEED_COLUMN].agg(set)
    short = held[held.map(lambda found: not set(seeds) <= found)]
    if not short.empty:
        raise ValueError(
            f"{len(short)} cell(s) do not carry every training run, e.g. "
            f"{short.index[0]} with {sorted(short.iloc[0])}"
        )
    return median_over_seeds(runs, keys)


def board_columns(
    variants: Sequence[str] = VARIANTS,
    trainings: Sequence[str] = TRAINING_CONDITIONS,
) -> list[tuple[str, str]]:
    """Return the table's ``(representation, training)`` columns, training innermost.

    Training varies fastest so the plain/augmented pair a reader compares sits side
    by side.
    """
    return [(variant, training) for variant in variants for training in trainings]


def _scored_rows(
    table: pd.DataFrame, transform: str, audits: Sequence[str]
) -> pd.DataFrame:
    """Return the rows of one held-out block, restricted to ``audits``."""
    metric = TEST_METRICS[transform]
    rows = table[table["metric"] == metric]
    if audits:
        rows = rows[rows["audit"].isin(list(audits))]
    if rows.empty:
        held = sorted(set(table["audit"]))
        raise ValueError(
            f"the table holds no {metric} row for audit point(s) "
            f"{list(audits) or 'any'}; it covers {held}"
        )
    return rows


def _spread(values: pd.Series) -> float:
    """How far a cell moved between the families it was scored on."""
    return float(values.max() - values.min())


def _group_cells(cells: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    """Lay one value per cell out as three groups by six cells."""
    wide = cells.pivot_table(
        index="group", columns=["feature_variant", "training"], values="value"
    )
    columns = [name for name in board_columns() if name in wide.columns]
    return wide.reindex(
        index=[name for name in groups if name in wide.index], columns=columns
    )


def group_board(
    deployments: pd.DataFrame,
    scope: str,
    transform: str = TRANSFORMS[1],
    groups: Sequence[str] = GROUPS,
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Return the main table over one scope: groups by ``(representation, training)``.

    A cell is the median over training runs of each run's mean, over the scope's
    families, of its median deployment.
    """
    runs = per_seed(deployments, scope, TEST_METRICS[transform], seeds)
    return _group_cells(median_over_seeds(runs), groups)


def group_spread(
    table: pd.DataFrame,
    scope: str,
    transform: str = TRANSFORMS[1],
    groups: Sequence[str] = GROUPS,
) -> pd.DataFrame:
    """Return each :func:`group_board` cell's range (max - min) over its families."""
    rows = scope_rows(_scored_rows(table, transform, ()), scope, "audit")
    grouped = rows.groupby(["group", "feature_variant", "training"], observed=True)
    spread = (
        cast(pd.Series, grouped["value"].agg(_spread)).rename("value").reset_index()
    )
    return _group_cells(spread, groups)


# The main table's numbers, one row per cell, at full precision.
BOARD_LEVEL_COLUMNS: tuple[str, ...] = (
    "scope",
    "transform",
    "group",
    "feature_variant",
    "training",
    "level",
    "spread",
)


def board_levels(
    mean: pd.DataFrame, spread: pd.DataFrame | None, scope: str, transform: str
) -> pd.DataFrame:
    """Return one main table's cells, level and spread, one row per cell.

    ``spread`` is None for a scope of one family.
    """
    levels = mean.to_numpy(dtype=float)
    widths = None if spread is None else spread.to_numpy(dtype=float)
    if widths is not None and widths.shape != levels.shape:
        raise ValueError(
            f"the mean and the spread must be the same table; got {levels.shape} "
            f"against {widths.shape}"
        )
    variants = [str(name) for name in mean.columns.get_level_values(0)]
    trainings = [str(name) for name in mean.columns.get_level_values(1)]
    rows = [
        {
            "scope": scope,
            "transform": transform,
            "group": str(group),
            "feature_variant": variant,
            "training": training,
            "level": float(levels[row, column]),
            "spread": float("nan") if widths is None else float(widths[row, column]),
        }
        for row, group in enumerate(mean.index)
        for column, (variant, training) in enumerate(
            zip(variants, trainings, strict=True)
        )
    ]
    return pd.DataFrame(rows, columns=list(BOARD_LEVEL_COLUMNS))


def best_variant(
    deployments: pd.DataFrame,
    training: str,
    scope: str,
    transform: str = TRANSFORMS[1],
    seeds: Sequence[int] = MODEL_SEEDS,
) -> dict[str, str]:
    """Return the encoding with the highest :func:`group_board` score, per group."""
    metric = TEST_METRICS[transform]
    rows = cast(pd.DataFrame, deployments[deployments["training"] == training])
    levels = median_over_seeds(per_seed(rows, scope, metric, seeds)).set_index(
        ["group", "feature_variant"]
    )["value"]
    winners: dict[str, str] = {}
    for group in GROUPS:
        if group not in levels.index.get_level_values("group"):
            continue
        cell = levels.xs(group, level="group")
        best = max(cell)
        tied = [str(name) for name, value in cell.items() if float(value) == best]
        winners[group] = min(tied, key=lambda name: VARIANTS.index(name))
    return winners
