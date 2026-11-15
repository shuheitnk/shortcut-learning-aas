"""Whether RQ1 to RQ3 hold when the learner is changed: the appendix's table.

For RealMLP, the random forest and XGBoost, reads the cells of Experiments 1, 4
and 5 from the stage store (computing nothing) and sets each learner's verdicts
beside RealMLP's. Run every experiment with ``--learner`` first (`just reproduce`).

uv run python scripts/report_learner_agreement.py
uv run python scripts/report_learner_agreement.py --preset debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

from aas_shortcut.compare_predictor_sets import compare, median_levels, per_seed_runs
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import ELA_FEATURE_SET, run_root_name
from aas_shortcut.define_warp_audits import AUDITS
from aas_shortcut.read_learner_verdicts import (
    Measured,
    agreement,
    long_table,
    read_predictors,
    read_regression,
    read_selection,
)
from aas_shortcut.resolve_pair_blocks import (
    resolve_held_out_in_range,
    resolve_training_block,
)
from aas_shortcut.resolve_portfolio_blocks import (
    resolve_portfolio_bindings,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
)
from aas_shortcut.resolve_predictor_cells import (
    comparison_sources,
    resolve_predictor_blocks,
    resolve_predictor_cells,
)
from aas_shortcut.resolve_regression_cells import resolve_family_deployments
from aas_shortcut.resolve_selection_cells import resolve_family_records
from aas_shortcut.resolve_stage_artifacts import (
    REQUIRE_CACHED,
    MissingStageArtifact,
    StageStore,
)
from aas_shortcut.resolve_tuning_stages import resolve_pair_binding
from aas_shortcut.run_experiment_spec import (
    LEARNERS,
    MAIN_LEARNER,
    ExperimentSpec,
    add_overrides,
    build_context,
    chosen_feature_sets,
    compose_config,
    resolve_run_paths,
)

# Each experiment is resolved under its own entry point's configuration.
SPECS = {
    number: Path(__file__).resolve().parent / f"experiment{number}" / "config.yaml"
    for number in (1, 4, 5)
}

# The step whose pair tuning every regression cell binds from.
TUNING_STEP = "E2-1"

# This script's root, scoped to the preset as an experiment's is, and its tables.
ROOT_NAME = "learner_agreement"
SUMMARY_TABLE = "learner_agreement.csv"
DETAIL_TABLE = "learner_agreement_detail.csv"
LEVEL_TABLE = "learner_levels.csv"
TEST_TABLE = "learner_tests.csv"


def _context(
    number: int, preset: str | None, overrides: Sequence[str], learner: str
) -> tuple[DictConfig, ExperimentContext]:
    """One experiment's configuration and context, as its entry point builds them."""
    spec = ExperimentSpec.load(SPECS[number])
    cfg = compose_config(spec.hydra_overrides(preset, overrides, learner=learner))
    return cfg, build_context(cfg, resolve_run_paths(cfg))


def measure(
    store: StageStore, preset: str | None, overrides: Sequence[str], learner: str
) -> dict[str, Measured]:
    """Read one learner's verdicts in every representation set, from the store."""
    cfg, context = _context(4, preset, overrides, learner)
    binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
    training = resolve_training_block(store, context, binding, tuning, TUNING_STEP)
    blocks = resolve_held_out_in_range(
        store, context, binding, tuning, TUNING_STEP, AUDITS
    )
    labels = [block.label for block in blocks]
    scored = [audit for audit in AUDITS if audit.label in labels]
    measured = {
        feature_set: read_regression(
            resolve_family_deployments(
                store,
                context,
                feature_set,
                tuple(cfg.task.model_selection.candidates),
                training,
                blocks,
                scored,
            ),
            labels,
        )
        for feature_set in chosen_feature_sets(context.feature_sets)
    }

    _, context = _context(5, preset, overrides, learner)
    bindings = resolve_portfolio_bindings(store, context)
    records, _ = resolve_family_records(
        store,
        context,
        resolve_selection_training(store, context, bindings),
        resolve_selection_held_out_in_range(store, context, bindings, AUDITS),
    )
    for feature_set, rows in records.groupby("feature_set", observed=True):
        measured[str(feature_set)].extend(read_selection(rows))

    cfg, context = _context(1, preset, overrides, learner)
    cells = resolve_predictor_cells(
        store,
        tuple(cfg.task.model_selection.candidates),
        context,
        resolve_predictor_blocks(store, context, TUNING_STEP),
    )
    root = Path(cfg.paths.output_root)
    source = comparison_sources(cells, root)
    predictors = read_predictors(
        median_levels(root, source), compare(per_seed_runs(root, source), source.seeds)
    )
    # RQ1 first, as the table's columns read.
    predictors.extend(measured[ELA_FEATURE_SET])
    measured[ELA_FEATURE_SET] = predictors
    return measured


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this script's command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Read RQ1-RQ3's verdicts for RealMLP, the random forest and XGBoost "
            "from the stage store, computing nothing, and set each beside "
            "RealMLP's. Trailing arguments are passed "
            "to Hydra as overrides, as to the entry points."
        )
    )
    parser.add_argument(
        "--preset", default=None, help="scale to read at; full by default"
    )
    return add_overrides(parser).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the tables and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    cfg, _ = _context(4, args.preset, args.overrides, MAIN_LEARNER)
    store = StageStore.under(cfg.paths.output_root, REQUIRE_CACHED)
    measured: dict[str, dict[str, Measured]] = {}
    try:
        for learner in LEARNERS:
            print(f"reading {learner}", flush=True)
            measured[learner] = measure(store, args.preset, args.overrides, learner)
    except MissingStageArtifact as missing:
        print(
            f"stopped: {missing}\nrun every experiment with --learner first",
            file=sys.stderr,
        )
        return 1

    reported = ExperimentSpec.load(SPECS[4]).preset
    root = Path(cfg.paths.output_root).resolve() / run_root_name(
        ROOT_NAME, args.preset or reported, reported
    )
    root.mkdir(parents=True, exist_ok=True)
    summary, detail = agreement(measured, MAIN_LEARNER)
    tables = {
        SUMMARY_TABLE: summary,
        DETAIL_TABLE: detail,
        LEVEL_TABLE: long_table(measured, "levels"),
        TEST_TABLE: long_table(measured, "tests"),
    }
    for name, table in tables.items():
        table.to_csv(root / name, index=False)
        print(f"written: {root / name}")
    store.write_list(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
