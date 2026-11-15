"""Experiment 1: the invariant predictors against every predictor, on one cell.

The two cells differ only in their predictors: ``all`` is Experiment 4's ``plain``
x ``value`` x G1 cell under the control family (the 59 ELA columns), ``invariant``
is the 36 of those columns that are comparison-based by definition. Neither is
fitted here; this module reads both and compares them.

Column suffixes name the unit: ``*_seed`` is one training run (the median of its
eleven deployments), ``*_median`` is the reported level (the median over runs).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.classify_feature_invariance import is_structurally_invariant
from aas_shortcut.compare_training_runs import (
    ALPHA,
    EFFECT_COLUMN,
    PAIRED,
    PAIRED_EFFECT,
    SIZE_COLUMN,
    UNIT,
    compare_paired_runs,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import ELA_FEATURE_SET, RunPaths
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_augmented_regression import (
    DEPLOYMENT_SHARD,
    AugmentedRegressionConfig,
)
from aas_shortcut.measure_representation_transfer import group_of
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.read_write_tables import read_table

# The two predictor sets, in the order the figure draws them.
ALL_PREDICTORS = "all"
INVARIANT_PREDICTORS = "invariant"
PREDICTOR_SETS: tuple[str, str] = (ALL_PREDICTORS, INVARIANT_PREDICTORS)

# The only pair whose superiority a strictly increasing transformation cannot move,
# so the only one whose fall reads as a shortcut.
REPORTED_PAIR = next(pair for pair in FOCUS_PAIRS if pair.pair_type == "G1_G1")

# A rank-encoded matrix is invariant on both sides by construction, so only the
# value encoding gives a comparison.
REPORTED_VARIANT = "value"

# The reported levels: the median over training runs of each run's score.
LEVEL_COLUMNS: dict[str, str] = {
    TEST_METRICS["pre"]: "r2_pre_median",
    TEST_METRICS["post"]: "r2_post_median",
}

# One training run's observation in the test: its score on the transformed block.
SEED_METRIC = TEST_METRICS["post"]


def invariant_columns(scale: str) -> tuple[str, ...]:
    """Return the ``invariant`` cell's predictors: the invariant paper ELA columns.

    A subset of the ``all`` cell's columns, so the two cells differ in predictors
    and not in feature tables.
    """
    columns = paper_columns([ELA_FEATURE_SET], scale)
    invariant = tuple(name for name in columns if is_structurally_invariant(name))
    if not invariant:
        raise ValueError(
            f"none of the {len(columns)} ELA column(s) is comparison-based by "
            "definition; there is nothing to fit"
        )
    return invariant


def invariant_cell_config(
    model: ModelConfig,
    candidates: Sequence[str],
    columns: Sequence[str],
    n_jobs: int = 1,
) -> AugmentedRegressionConfig:
    """Return Experiment 4's regression narrowed to the ``all`` cell, on ``columns``."""
    return AugmentedRegressionConfig(
        feature_set=ELA_FEATURE_SET,
        variants=(REPORTED_VARIANT,),
        pairs=(REPORTED_PAIR,),
        trainings=(TRAINING_PLAIN,),
        audits=(CONTROL.label,),
        features=tuple(columns),
        candidates=tuple(candidates),
        model=model,
        n_jobs=n_jobs,
    )


@dataclass(frozen=True)
class Cell:
    """Where one predictor set's numbers are read from."""

    predictors: str
    # Run root, relative to the output root.
    run: str
    # The per-deployment table inside it.
    deployments: str = DEPLOYMENT_SHARD

    def paths(self, output_root: Path | str) -> RunPaths:
        """Return the run root this cell is read from."""
        return RunPaths.from_root(Path(output_root) / self.run)


@dataclass(frozen=True)
class SourceCells:
    """The two artifacts the comparison stands on, and what narrows them to a cell."""

    all_cell: Cell
    invariant_cell: Cell
    feature_set: str = ELA_FEATURE_SET
    feature_variant: str = "value"
    training: str = TRAINING_PLAIN
    audit: str = CONTROL.label
    pair: str = REPORTED_PAIR.slug
    seeds: tuple[int, ...] = MODEL_SEEDS

    @property
    def cells(self) -> tuple[Cell, Cell]:
        """The two cells, in the order the figure and the tables read them."""
        return (self.all_cell, self.invariant_cell)

    def describe(self) -> str:
        """One line naming the condition both cells are read under."""
        return (
            f"{group_of(REPORTED_PAIR)} x {self.feature_variant} x {self.training} "
            f"x {self.audit}, {self.feature_set}"
        )


def _narrow(table: pd.DataFrame, source: SourceCells, audit: str) -> pd.DataFrame:
    """Restrict a metric or deployment table to the one cell being compared."""
    wanted = {
        "feature_set": source.feature_set,
        "feature_variant": source.feature_variant,
        "training": source.training,
        "pair": source.pair,
        "audit": audit,
    }
    kept = table
    for column, value in wanted.items():
        if column in kept.columns:
            kept = kept[kept[column] == value]
    return kept


def _seed_scores(
    output_root: Path | str, source: SourceCells, cell: Cell, metric: str
) -> pd.Series:
    """Return every training run's score on one block: the median of its deployments.

    Every training run must be present: a level or a test over a subset is not the
    one that was fixed.
    """
    shard = cell.paths(output_root).metric(cell.deployments)
    scored = _narrow(read_table(shard.data), source, source.audit)
    scored = scored[scored["metric"] == metric]
    if scored.empty:
        raise ValueError(
            f"{shard.data} holds no {metric!r} row for {source.describe()}; "
            f"the {cell.predictors!r} half of the comparison is not there"
        )
    reduced = scored.groupby("seed", observed=True)["value"].median()
    missing = sorted(set(source.seeds) - {int(seed) for seed in reduced.index})
    if missing:
        raise ValueError(
            f"{shard.data} is missing training run(s) {missing} for "
            f"{source.describe()}; the comparison is over all "
            f"{len(source.seeds)} of them"
        )
    return reduced


def median_levels(output_root: Path | str, source: SourceCells) -> pd.DataFrame:
    """Return each predictor set's column count and median ``R^2`` on both blocks."""
    rows = []
    for cell in source.cells:
        shard = cell.paths(output_root).metric(cell.deployments)
        scored = _narrow(read_table(shard.data), source, source.audit)
        rows.append(
            {
                "predictors": cell.predictors,
                "n_features": int(cast(int, scored["n_features"].max())),
                **{
                    name: float(
                        _seed_scores(output_root, source, cell, metric).median()
                    )
                    for metric, name in LEVEL_COLUMNS.items()
                },
            }
        )
    return pd.DataFrame(rows)


def shortcut_detected(levels: pd.DataFrame) -> bool:
    """Whether ``all`` beats ``invariant`` before the transform and loses after it.

    Read off the levels, not the test: the study's definition does not ask for
    significance.
    """
    scored = levels.set_index("predictors")
    missing = [name for name in PREDICTOR_SETS if name not in scored.index]
    if missing:
        raise ValueError(f"the levels hold no {missing} row; nothing to compare")
    pre, post = (LEVEL_COLUMNS[TEST_METRICS[view]] for view in ("pre", "post"))
    above_before = (
        scored.loc[ALL_PREDICTORS, pre] > scored.loc[INVARIANT_PREDICTORS, pre]
    )
    above_after = (
        scored.loc[INVARIANT_PREDICTORS, post] > scored.loc[ALL_PREDICTORS, post]
    )
    return bool(above_before and above_after)


def per_seed_runs(output_root: Path | str, source: SourceCells) -> pd.DataFrame:
    """Return every training run's score on the transformed block, per predictor set."""
    frames = []
    for cell in source.cells:
        reduced = _seed_scores(output_root, source, cell, SEED_METRIC).rename(
            "r2_post_seed"
        )
        frames.append(
            reduced.reset_index().assign(predictors=cell.predictors)[
                ["predictors", "seed", "r2_post_seed"]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def compare(runs: pd.DataFrame, seeds: Sequence[int] = MODEL_SEEDS) -> pd.DataFrame:
    """Test the two predictor sets against each other, paired by seed, two-sided."""
    wide = runs.pivot_table(index="seed", columns="predictors", values="r2_post_seed")
    wide = wide.reindex(index=list(seeds))
    missing = [name for name in PREDICTOR_SETS if name not in wide.columns]
    if missing:
        raise ValueError(f"the runs hold no {missing} column; there is nothing to pair")

    higher, p_value, size = compare_paired_runs(
        wide[INVARIANT_PREDICTORS].to_numpy(),
        wide[ALL_PREDICTORS].to_numpy(),
        (INVARIANT_PREDICTORS, ALL_PREDICTORS),
    )
    return pd.DataFrame(
        [
            {
                "left": INVARIANT_PREDICTORS,
                "right": ALL_PREDICTORS,
                "unit": UNIT,
                "metric": "r2_post_seed",
                "test": PAIRED,
                "n_pairs": len(wide),
                "higher": higher,
                "p": p_value,
                EFFECT_COLUMN: PAIRED_EFFECT,
                SIZE_COLUMN: size,
                "significant": p_value < ALPHA,
            }
        ]
    )


def hero_frame(levels: pd.DataFrame, pair: str = REPORTED_PAIR.slug) -> pd.DataFrame:
    """Return the levels in the long shape `figure_shortcut_hero` reads."""
    rows = [
        {
            "pair": pair,
            "predictors": row["predictors"],
            "metric": metric,
            "value": row[name],
        }
        for _, row in levels.iterrows()
        for metric, name in LEVEL_COLUMNS.items()
    ]
    return pd.DataFrame(rows)
