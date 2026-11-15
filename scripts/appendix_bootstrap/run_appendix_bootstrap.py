"""Appendix - bootstrap intervals of the levels RQ1 to RQ3 compare.

Resamples the model seeds, the deployments, the functions and each drawn
function's test instances, and recomputes every level as the paper does. The
regression's per-problem predictions come from refitting Experiment 4's cells,
which must reproduce every stored R² bit for bit; the selection's per-problem Θ
is read from Experiment 5's units.

uv run python scripts/appendix_bootstrap/run_appendix_bootstrap.py \
    paths.output_root=outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.bootstrap_reported_levels import (
    CONFIDENCE,
    N_SAMPLES,
    SEED_DOMAINS,
    BootstrapDraws,
    draw_samples,
    identity_draws,
    level,
    percentile_interval,
    problem_order,
    regression_cube,
    seed_values,
    selection_cube,
    weighted_mean,
    weighted_r2,
)
from aas_shortcut.compare_selection_runs import FAMILY_COLUMN
from aas_shortcut.compare_selection_runs import seed_values as selection_seed_values
from aas_shortcut.compare_training_runs import median_over_seeds, per_seed
from aas_shortcut.define_bootstrap_rows import (
    REGRESSION,
    Comparison,
    Side,
    comparisons,
)
from aas_shortcut.define_paths import representation_path
from aas_shortcut.define_repetitions import MODEL_SEEDS, REPETITIONS
from aas_shortcut.define_reporting_scopes import scope_families, scope_mean
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.resolve_appendix_inputs import (
    PREDICTORS,
    resolve_regression_tables,
    resolve_selection_tables,
)
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_overrides,
    add_store_options,
    build_context,
    build_parser,
    chosen_feature_sets,
    compose_config,
    record_run,
    resolve_run_paths,
    store_policy,
)

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

INTERVAL_TABLE = Path("bootstrap_intervals.csv")

# A point estimate further than this from the reported level stops the run: the
# weighted sums add in another order than the report's, nothing else may differ.
POINT_TOLERANCE = 1e-12


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run the bootstrap appendix: percentile intervals of the levels RQ1 to "
            "RQ3 compare, resampling seeds, deployments, functions and instances. "
            "Trailing arguments are passed to Hydra as overrides."
        ),
        epilog="example: run_appendix_bootstrap.py task.n_jobs=26",
    )
    add_store_options(parser)
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help=f"bootstrap samples (default {N_SAMPLES}; fewer only to try the run)",
    )
    return add_overrides(parser).parse_args(argv)


REGRESSION_KEYS: tuple[str, ...] = (
    PREDICTORS,
    "group",
    "feature_set",
    "feature_variant",
    "training",
    "transform",
    "audit",
)
SELECTION_KEYS: tuple[str, ...] = (
    "feature_set",
    "feature_variant",
    "training",
    "transform",
    FAMILY_COLUMN,
)
CUBE_COLUMNS: tuple[str, ...] = (
    "seed",
    "rep",
    "function_id",
    "instance_id",
    "observed",
    "predicted",
    "superiority_vs_sbs",
)


def _split(table: pd.DataFrame, keys: Sequence[str]) -> dict[tuple, pd.DataFrame]:
    """The table's rows by key, keeping only what a cube reads."""
    kept = [column for column in CUBE_COLUMNS if column in table.columns]
    return {
        tuple(str(value) for value in key): cast(pd.DataFrame, rows[kept])
        for key, rows in table.groupby(list(keys), observed=True, sort=False)
    }


class Tables:
    """The per-problem tables every side is read from, and the stored levels."""

    def __init__(
        self,
        predictions: pd.DataFrame,
        deployments: pd.DataFrame,
        problems: pd.DataFrame,
        records: pd.DataFrame,
    ) -> None:
        self.deployments = deployments
        self.records = records
        # Split once: the tables run to millions of rows and every side reads one
        # family of one cell.
        self.regression = _split(predictions, REGRESSION_KEYS)
        self.selection = _split(problems, SELECTION_KEYS)

    def rows(self, side: Side, family: str) -> pd.DataFrame:
        """One side's per-problem rows on one family."""
        cell = (side.feature_set, side.variant, side.training, side.view, family)
        if side.part == REGRESSION:
            key = (side.predictors, side.group, *cell)
            found = self.regression.get(key)
        else:
            found = self.selection.get(cell)
        if found is None:
            raise ValueError(f"no per-problem rows for {side} on {family}")
        return found

    def reported(self, side: Side) -> float:
        """The level the experiment's own report functions give for one side."""
        if side.part == REGRESSION:
            rows = cast(
                pd.DataFrame,
                self.deployments[self.deployments[PREDICTORS] == side.predictors],
            )
            levels = median_over_seeds(
                per_seed(rows, side.scope, TEST_METRICS[side.view])
            )
            levels = levels[levels["group"].astype(str) == side.group]
        else:
            per_family = selection_seed_values(
                self.records, by=(FAMILY_COLUMN,), view=side.view
            )
            levels = median_over_seeds(
                scope_mean(
                    per_family,
                    ["feature_set", "feature_variant", "training", "seed"],
                    side.scope,
                    FAMILY_COLUMN,
                ),
                ("feature_set", "feature_variant", "training"),
            )
        hit = levels[
            (levels["feature_set"].astype(str) == side.feature_set)
            & (levels["feature_variant"].astype(str) == side.variant)
            & (levels["training"].astype(str) == side.training)
        ]
        if len(hit) != 1:
            raise ValueError(f"{len(hit)} reported level(s) for {side}")
        return float(hit["value"].iloc[0])


class SideValues:
    """Every side's per-seed values on the data and on every sample, computed once."""

    def __init__(self, tables: Tables, order: pd.DataFrame, draws: BootstrapDraws):
        self.tables = tables
        self.order = order
        self.draws = draws
        self.point = identity_draws(len(MODEL_SEEDS), REPETITIONS, len(order))
        self.cache: dict[Side, tuple[np.ndarray, np.ndarray]] = {}

    def __call__(self, side: Side) -> tuple[np.ndarray, np.ndarray]:
        """Return (point, samples) of one side's per-seed values."""
        if side not in self.cache:
            reps = list(range(1, REPETITIONS + 1))
            on_point, on_samples = [], []
            for family in scope_families(side.scope):
                rows = self.tables.rows(side, family)
                if side.part == REGRESSION:
                    observed, predicted = regression_cube(
                        rows, self.order, MODEL_SEEDS, reps
                    )
                    on_point.append(
                        weighted_r2(observed, predicted, self.point.weights)
                    )
                    on_samples.append(
                        weighted_r2(observed, predicted, self.draws.weights)
                    )
                else:
                    values = selection_cube(rows, self.order, MODEL_SEEDS, reps)
                    on_point.append(weighted_mean(values, self.point.weights))
                    on_samples.append(weighted_mean(values, self.draws.weights))
            self.cache[side] = (
                seed_values(on_point, self.point),
                seed_values(on_samples, self.draws),
            )
        return self.cache[side]


def interval_row(comparison: Comparison, values: SideValues, resampling: str) -> dict:
    """One row of the table: both levels and their difference, with intervals."""
    draws = values.draws
    left_point, left_samples = values(comparison.left)
    right_point, right_samples = values(comparison.right)
    point_seeds = values.point.seeds
    left_level = float(level(left_point, point_seeds)[0])
    right_level = float(level(right_point, point_seeds)[0])
    reported = {
        "left": values.tables.reported(comparison.left),
        "right": values.tables.reported(comparison.right),
    }
    for name, point in (("left", left_level), ("right", right_level)):
        if abs(point - reported[name]) > POINT_TOLERANCE:
            raise ValueError(
                f"{comparison.question} {comparison.label}: the {name} point estimate "
                f"{point!r} is not the reported level {reported[name]!r}"
            )
    left = level(left_samples, draws.seeds)
    right_seeds = draws.seeds if comparison.paired else draws.other_seeds
    difference = left - level(right_samples, right_seeds)
    side = comparison.left
    return {
        "question": comparison.question,
        "part": side.part,
        "feature_set": side.feature_set,
        "group": side.group,
        "scope": side.scope,
        "view": side.view,
        "comparison": comparison.label,
        "left_predictors": side.predictors if side.part == REGRESSION else "",
        "right_predictors": (
            comparison.right.predictors if side.part == REGRESSION else ""
        ),
        "left_variant": side.variant,
        "left_training": side.training,
        "right_variant": comparison.right.variant,
        "right_training": comparison.right.training,
        "seeds": "shared" if comparison.paired else "drawn apart",
        "left_level": left_level,
        "left_lower": percentile_interval(left)[0],
        "left_upper": percentile_interval(left)[1],
        "right_level": right_level,
        "right_lower": percentile_interval(level(right_samples, draws.seeds))[0],
        "right_upper": percentile_interval(level(right_samples, draws.seeds))[1],
        "difference": left_level - right_level,
        "difference_lower": percentile_interval(difference)[0],
        "difference_upper": percentile_interval(difference)[1],
        "point_gap": max(
            abs(left_level - reported["left"]), abs(right_level - reported["right"])
        ),
        "n_samples": draws.n_samples,
        "confidence": CONFIDENCE,
        "resampling": resampling,
        "seed_domain": SEED_DOMAINS[resampling],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the appendix and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    spec = ExperimentSpec.load(SPEC_PATH)
    cfg = compose_config(spec.hydra_overrides(args.preset, args.overrides))
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)
    feature_sets = chosen_feature_sets(context.feature_sets)
    candidates = list(cfg.task.model_selection.candidates)
    if len(candidates) != 1:
        print(f"stopped: one learner expected, got {candidates}", file=sys.stderr)
        return 1

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        regression = resolve_regression_tables(store, context, candidates, feature_sets)
        selection = resolve_selection_tables(store, context)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    tables = Tables(
        regression.predictions,
        regression.deployments,
        selection.units.problems(list(selection.units.labels)),
        selection.units.records(),
    )
    order = problem_order(regression.predictions)
    frames = []
    for resampling in SEED_DOMAINS:
        draws = draw_samples(
            order,
            len(MODEL_SEEDS),
            REPETITIONS,
            args.samples,
            base_seed=context.seed,
            resampling=resampling,
        )
        values = SideValues(tables, order, draws)
        frames.append(
            pd.DataFrame(
                [
                    interval_row(comparison, values, resampling)
                    for comparison in comparisons(tuple(feature_sets))
                ]
            )
        )
    rows = pd.concat(frames, ignore_index=True)

    run_paths.mkdirs()
    for feature_set in feature_sets:
        target = run_paths.root / representation_path(INTERVAL_TABLE, feature_set)
        rows[rows["feature_set"] == feature_set].to_csv(target, index=False)
        print(f"written: {target}")
    store.write_list(run_paths.root)
    record_run(
        spec,
        cfg,
        run_paths,
        extra={"feature_sets": feature_sets, "samples": args.samples},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
