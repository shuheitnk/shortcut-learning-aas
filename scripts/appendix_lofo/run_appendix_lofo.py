"""Appendix - does a condition that holds up under the RRT also hold up under LOFO?

The RRT value is read from Experiment 4's units (mean over the four unseen
families). The LOFO value comes from 24 folds, one per BBOB function: fit on the
other 23 functions' training instances, score on the held-out function's
untransformed test instances.

uv run python scripts/appendix_lofo/run_appendix_lofo.py \
    paths.output_root=outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from aas_shortcut.create_figures.draw_lofo_figure import (
    figure_lofo_against_rrt,
    shared_limits,
)
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import representation_path
from aas_shortcut.define_warp_audits import AUDITS, CONTROL
from aas_shortcut.measure_lofo_regression import (
    PREDICTION_TABLE,
    SCORE_TABLE,
    LofoRegressionCell,
)
from aas_shortcut.measure_representation_transfer import VARIANTS
from aas_shortcut.measure_selection_transfer import TRAINING_CONDITIONS
from aas_shortcut.resolve_lofo_cells import resolve_lofo_regression
from aas_shortcut.resolve_pair_blocks import (
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_regression_cells import resolve_family_deployments
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.resolve_tuning_stages import resolve_pair_binding
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
from aas_shortcut.summarize_lofo_levels import (
    function_rho_by_group,
    level_table,
    rrt_correlations,
    spread_correlations,
    summary_table,
)

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# Experiment 4's pair tuning, which every cell binds from.
TUNING_STEP = "E2-1"

REGRESSION_LEVELS = Path("lofo_regression_levels.csv")
REGRESSION_SUMMARY = Path("lofo_regression_summary.csv")
SPREAD_CORRELATIONS = Path("lofo_spread_correlations.csv")
RRT_CORRELATIONS = Path("lofo_rrt_correlations.csv")
FUNCTION_RHO = Path("lofo_function_rho.csv")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run the LOFO appendix: the superiority regression under "
            "leave-one-function-out, against Experiment 4's RRT levels. Trailing "
            "arguments are passed to Hydra as overrides."
        ),
        epilog="example: run_appendix_lofo.py task.n_jobs=26",
    )
    add_store_options(parser)
    return add_overrides(parser).parse_args(argv)


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
    model = replace(context.model, kind=candidates[0])
    features = paper_columns(context.feature_sets, context.scale)

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
        training = resolve_training_block(store, context, binding, tuning, TUNING_STEP)
        blocks = resolve_held_out_in_range(
            store, context, binding, tuning, TUNING_STEP, AUDITS
        )
        test = next(block for block in blocks if block.label == CONTROL.label)
        rrt = {
            feature_set: resolve_family_deployments(
                store, context, feature_set, candidates, training, blocks, AUDITS
            )
            for feature_set in feature_sets
        }
        scores: dict[str, list[pd.DataFrame]] = {name: [] for name in feature_sets}
        predictions: dict[str, list[pd.DataFrame]] = {n: [] for n in feature_sets}
        for feature_set in feature_sets:
            for condition in TRAINING_CONDITIONS:
                for variant in VARIANTS:
                    cells = [
                        LofoRegressionCell(
                            feature_set=feature_set,
                            variant=variant,
                            pair=pair,
                            training=condition,
                            features=features,
                            model=model,
                        )
                        for pair in FOCUS_PAIRS
                    ]
                    artifacts = resolve_lofo_regression(
                        store, cells, training, test, context.scale, context.n_jobs
                    )
                    scores[feature_set] += [
                        artifacts[cell].read_table(SCORE_TABLE) for cell in cells
                    ]
                    predictions[feature_set] += [
                        artifacts[cell].read_table(PREDICTION_TABLE) for cell in cells
                    ]
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    run_paths.mkdirs()
    levels: dict[str, pd.DataFrame] = {}
    correlations: dict[str, pd.DataFrame] = {}
    for feature_set in feature_sets:
        scored = pd.concat(scores[feature_set], ignore_index=True)
        summary = summary_table(
            scored, pd.concat(predictions[feature_set], ignore_index=True)
        )
        levels[feature_set] = level_table(scored, rrt[feature_set])
        correlations[feature_set] = rrt_correlations(levels[feature_set])
        for name, table in (
            (REGRESSION_LEVELS, levels[feature_set]),
            (REGRESSION_SUMMARY, summary),
            (SPREAD_CORRELATIONS, spread_correlations(summary)),
            (RRT_CORRELATIONS, correlations[feature_set]),
            (FUNCTION_RHO, function_rho_by_group(summary)),
        ):
            target = run_paths.root / representation_path(name, feature_set)
            table.to_csv(target, index=False)
            print(f"written: {target}")

    # Both feature sets' figures share axis ranges so they can be compared.
    limits = shared_limits(list(levels.values()))
    for feature_set in feature_sets:
        for path in figure_lofo_against_rrt(
            levels[feature_set],
            correlations[feature_set],
            feature_set,
            limits,
            run_paths,
        ):
            print(f"written: {path}")

    store.write_list(run_paths.root)
    record_run(spec, cfg, run_paths, extra={"feature_sets": feature_sets})
    return 0


if __name__ == "__main__":
    sys.exit(main())
