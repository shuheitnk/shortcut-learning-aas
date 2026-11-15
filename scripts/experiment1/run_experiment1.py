"""Experiment 1 - Shortcut Learning under an Invariant Target.

One cell: the G1 superiority regressed on the structurally invariant ELA columns,
fitted on the untransformed training block and scored on the control family's
held-out block. The comparison with Experiment 4's `all` cell is drawn afterwards
by `report_predictor_comparison.py`.

uv run python scripts/experiment1/run_experiment1.py task.n_jobs=26
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pandas as pd

from aas_shortcut.assemble_paired_views import check_source_agreement
from aas_shortcut.augment_training_samples import DEPLOYED_REP
from aas_shortcut.compare_predictor_sets import (
    INVARIANT_PREDICTORS,
)
from aas_shortcut.define_paths import ELA_FEATURE_SET, TRANSFORMS
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_representation_transfer import (
    PREMISE_TABLE,
    describe_target_invariance,
    invariance_holds,
    premise_table,
)
from aas_shortcut.resolve_predictor_cells import (
    ela_values,
    resolve_predictor_blocks,
    resolve_predictor_cells,
)
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_learner_option,
    add_overrides,
    add_store_options,
    build_context,
    build_parser,
    compose_config,
    record_run,
    resolve_run_paths,
    store_policy,
)
from aas_shortcut.split_bbob_instances import TEST_INSTANCES

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# The step whose pair tuning every cell binds from.
TUNING_STEP = "E2-1"

# Relative tolerance for the untransformed anchor, as in Experiment 4.
ANCHOR_TOLERANCE = 1e-9


def _check_premises(superiorities: pd.DataFrame, block: str) -> bool:
    """Check that the block's G1 superiority did not move under the warp."""
    if invariance_holds(superiorities):
        return True
    LOGGER.error(
        "the G1 superiority moved on the %s; its members search by comparison "
        "alone, so any difference is a broken measurement",
        block,
    )
    return False


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run Experiment 1: the G1 superiority regression on the structurally "
            "invariant predictors, under Experiment 4's conditions. Trailing "
            "arguments are passed to Hydra as overrides."
        ),
        epilog="example: run_experiment1.py task.n_jobs=26",
    )
    add_store_options(parser)
    add_learner_option(parser)
    return add_overrides(parser).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    spec = ExperimentSpec.load(SPEC_PATH)
    cfg = compose_config(
        spec.hydra_overrides(args.preset, args.overrides, learner=args.learner)
    )
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        blocks = resolve_predictor_blocks(store, context, TUNING_STEP)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    source_features = ela_values(blocks.training.feature_table)

    # The untransformed half of the held-out block does not depend on the family,
    # so its features must match the training block's; otherwise the run would
    # score a different problem, not a different warp.
    held_out_features = ela_values(blocks.control.feature_table)
    shared = check_source_agreement(
        cast(pd.DataFrame, held_out_features[held_out_features["rep"] == DEPLOYED_REP]),
        source_features,
        TEST_INSTANCES,
        ANCHOR_TOLERANCE,
        "the training block",
    )

    # Only the held-out block is read under both views; the training block's
    # invariance is checked by Experiment 4, which benchmarks it transformed.
    if not _check_premises(blocks.control.superiorities, CONTROL.label):
        return 1
    run_paths.mkdirs()
    premises = premise_table(
        {CONTROL.label: describe_target_invariance(blocks.control.superiorities)},
        {CONTROL.label: shared},
        ANCHOR_TOLERANCE,
    )
    premises.to_csv(run_paths.root / PREMISE_TABLE, index=False)

    try:
        cells = resolve_predictor_cells(
            store, tuple(cfg.task.model_selection.candidates), context, blocks
        )
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1

    store.write_list(run_paths.root)
    record_run(
        spec,
        cfg,
        run_paths,
        extra={
            "feature_set": ELA_FEATURE_SET,
            "predictors": INVARIANT_PREDICTORS,
            "n_predictors": len(cells.columns),
            "transforms": list(TRANSFORMS),
            "invariant_cell": str(cells.invariant.path),
            "all_cell": str(cells.all.path),
            "argv": list(argv or sys.argv[1:]),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
