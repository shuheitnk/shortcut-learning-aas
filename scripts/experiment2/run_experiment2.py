"""Experiment 2 - Which Representation Survives a Monotone Transformation.

Reads Experiment 4's non-augmented cells on the control family (spline/warpheld)
and fits nothing of its own. Run alone, it still resolves all five families'
held-out blocks, because those cells are named by every family they are scored on.
Also writes the pair-tuning histogram and its tables.

uv run python scripts/experiment2/run_experiment2.py task.n_jobs=28
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from aas_shortcut.create_figures.draw_hero_figures import (
    HERO_FORMATS,
    figure_representation_hero,
)
from aas_shortcut.create_figures.draw_tuning_figures import figure_tuning_histogram
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_warp_audits import AUDITS, CONTROL
from aas_shortcut.measure_representation_transfer import (
    invariance_holds,
)
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    TrainingBlock,
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_regression_cells import (
    cells_of,
    family_regression_config,
    regression_data,
    resolve_regression_cells,
    rows_on_family,
)
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageArtifact,
    StageStore,
)
from aas_shortcut.resolve_tuning_stages import (
    CALIBRATION_TABLES,
    PAIR_TABLE,
    resolve_calibration,
    resolve_pair_binding,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_learner_option,
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
from aas_shortcut.tune_algorithms import summarize_pair_tuning

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")

# The step whose pair tuning the benchmark binds from.
TUNING_STEP = "E2-1"

PAIR_TUNING_TABLE = Path("pair_tuning.csv")


@dataclass(frozen=True)
class Blocks:
    """The training block, the families' held-out blocks and the pair tuning."""

    training: TrainingBlock
    held_out: list[HeldOutBlock]
    tuning: StageArtifact
    calibration: StageArtifact

    @property
    def control(self) -> HeldOutBlock:
        """The control family's held-out block, the one the experiment reports."""
        for block in self.held_out:
            if block.label == CONTROL.label:
                return block
        raise ValueError(f"{CONTROL.label} is out of its range on this block")


def resolve_blocks(store: StageStore, context: Any) -> Blocks:
    """Resolve the data of Experiment 4's non-augmented cells, and the calibration.

    The blocks are resolved by Experiment 4's functions so the cells are shared.
    The training block is untransformed only, as those cells fit on it alone.
    """
    binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
    training = resolve_training_block(
        store, context, binding, tuning, TUNING_STEP, trainings=(TRAINING_PLAIN,)
    )
    held_out = resolve_held_out_in_range(store, context, binding, tuning, TUNING_STEP)
    calibration = resolve_calibration(
        store, context, training.pre.superiority, training.pre.superiorities
    )
    return Blocks(training, held_out, tuning, calibration)


def draw_tuning(blocks: Blocks, run_paths: RunPaths) -> list[Path]:
    """Draw each pair's superiorities before and after the tuning, as histograms."""
    tuning = blocks.tuning.read_table(PAIR_TABLE)
    return figure_tuning_histogram(tuning, run_paths, HERO_FORMATS)


def write_calibration(blocks: Blocks, run_paths: RunPaths) -> list[Path]:
    """Copy the calibration's two tables from its unit into the run root.

    They measure the superiorities' spread on the tuning's 24 problems and on the
    2400 training problems, beside the histogram drawn on the former.
    """
    written: list[Path] = []
    for name in CALIBRATION_TABLES:
        target = run_paths.root / Path(name).with_suffix(".csv").name
        blocks.calibration.read_table(name).to_csv(target, index=False)
        written.append(target)
    return written


def write_pair_tuning(blocks: Blocks, run_paths: RunPaths) -> Path:
    """Write each pair's default and selected configuration from the tuning unit.

    The histogram's panels as numbers: the Kolmogorov-Smirnov distance to
    ``U(0, 1)``, the saturated problems and the grid searched.
    """
    target = run_paths.root / PAIR_TUNING_TABLE
    summarize_pair_tuning(blocks.tuning.read_table(PAIR_TABLE)).to_csv(
        target, index=False
    )
    return target


def _report_missing(store: StageStore, missing: MissingStageArtifact) -> int:
    """Say which unit the run may not compute, and return the exit status."""
    print(f"stopped: {missing}", file=sys.stderr)
    return 0 if store.policy == PLAN else 1


def _check_premises(superiorities: pd.DataFrame) -> bool:
    """Check that G1's superiority did not move on the control family's block.

    G1's members search by comparison alone, so any difference is a broken
    measurement. The training block's invariance is checked by Experiment 4.
    """
    return invariance_holds(superiorities)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run Experiment 2: value-based, rank-based and concat superiority models "
            "for all three algorithm groups, scored on the original and on the "
            "transformed test block. Trailing arguments are passed to Hydra as "
            "overrides."
        ),
        epilog="example: run_experiment2.py task.n_jobs=28",
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
        spec.hydra_overrides(
            args.preset,
            args.overrides,
            learner=args.learner,
        )
    )
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)
    feature_sets = chosen_feature_sets(context.feature_sets)
    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        blocks = resolve_blocks(store, context)
    except MissingStageArtifact as missing:
        return _report_missing(store, missing)
    run_paths.mkdirs()
    store.write_list(run_paths.root)

    if not _check_premises(blocks.control.superiorities):
        LOGGER.error("experiment 2: a premise failed; the result is not readable")
        return 1

    # A cell is reported as the median of its eleven training runs, each run's
    # score the median of its eleven deployments.
    candidates = tuple(cfg.task.model_selection.candidates)
    scored = [
        audit for audit in AUDITS if audit.label in {b.label for b in blocks.held_out}
    ]
    tables = []
    for feature_set in feature_sets:
        config = family_regression_config(context, feature_set, candidates, scored)
        cells = [cell for cell in cells_of(config) if cell.training == TRAINING_PLAIN]
        try:
            resolved = resolve_regression_cells(
                store,
                config,
                cells,
                regression_data(blocks.training, blocks.held_out, config.features),
                context.scale,
            )
        except MissingStageArtifact as missing:
            return _report_missing(store, missing)
        table = rows_on_family({cell: resolved[cell] for cell in cells}, CONTROL.label)
        tables.append(table)

    reported = pd.concat(tables, ignore_index=True)
    # Drawn in the same design as Experiment 1's figure.
    written = figure_representation_hero(reported, run_paths, HERO_FORMATS)
    written += draw_tuning(blocks, run_paths)
    written.append(write_pair_tuning(blocks, run_paths))
    written += write_calibration(blocks, run_paths)
    for path in written:
        print(f"written: {path}")
    store.write_list(run_paths.root)
    record_run(
        spec,
        cfg,
        run_paths,
        extra={
            "feature_sets": feature_sets,
            "argv": list(argv or sys.argv[1:]),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
