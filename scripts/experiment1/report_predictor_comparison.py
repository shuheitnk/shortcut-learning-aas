"""Experiment 1's comparison: the invariant predictors against every predictor.

Reads Experiment 4's `all` cell and Experiment 1's `invariant` cell from the stage
store (never computed here) and writes the levels, the test and the figure. A
level is the median over eleven training runs of each run's median over its eleven
deployments; the test pairs the same eleven runs by seed.

uv run python scripts/experiment1/report_predictor_comparison.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from run_experiment1 import SPEC_PATH, TUNING_STEP

from aas_shortcut.compare_predictor_sets import (
    compare,
    hero_frame,
    median_levels,
    per_seed_runs,
)
from aas_shortcut.create_figures.draw_hero_figures import (
    HERO_FORMATS,
    figure_shortcut_hero,
)
from aas_shortcut.resolve_predictor_cells import (
    comparison_sources,
    resolve_predictor_blocks,
    resolve_predictor_cells,
)
from aas_shortcut.resolve_stage_artifacts import (
    REQUIRE_CACHED,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_learner_option,
    add_overrides,
    build_context,
    compose_config,
    resolve_run_paths,
)

LEVEL_TABLE = "predictor_levels.csv"
TEST_TABLE = "predictor_tests.csv"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare Experiment 4's `all` cell with Experiment 1's `invariant` "
            "cell on the control family's held-out block, and draw Fig. 14. "
            "Trailing arguments are passed to Hydra as overrides, as to "
            "run_experiment1.py."
        )
    )
    parser.add_argument(
        "--preset", default=None, help="scale to read at; full by default"
    )
    add_learner_option(parser)
    return add_overrides(parser).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Read both cells, write the comparison and draw its figure."""
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
    root = Path(cfg.paths.output_root)
    store = StageStore.under(root, REQUIRE_CACHED)
    try:
        cells = resolve_predictor_cells(
            store,
            tuple(cfg.task.model_selection.candidates),
            context,
            resolve_predictor_blocks(store, context, TUNING_STEP),
        )
    except MissingStageArtifact as missing:
        print(
            f"stopped: {missing}\nrun scripts/experiment1/run_experiment1.py first",
            file=sys.stderr,
        )
        return 1
    source = comparison_sources(cells, root)
    levels = median_levels(root, source)
    runs = per_seed_runs(root, source)
    tested = compare(runs, source.seeds)

    run_paths.mkdirs()
    written = []
    for table, name in ((levels, LEVEL_TABLE), (tested, TEST_TABLE)):
        target = run_paths.root / name
        table.to_csv(target, index=False)
        written.append(target)

    # The figure is drawn from the same levels the table carries.
    written += figure_shortcut_hero(hero_frame(levels), run_paths, HERO_FORMATS)
    for path in written:
        print(f"written: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
