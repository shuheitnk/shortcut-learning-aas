"""Experiment 3's entry point: Experiment 5's plain cells, read on the control family.

Experiment 3 shares its cells, oracles and data with Experiment 5 and fits nothing
of its own; run alone, it computes all five families' test blocks, because the
plain cells are named by every family they are scored on.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from aas_shortcut.bind_portfolio_parameters import (
    CONDITION_SBS,
    CONDITION_SELECTION,
    summarize_role_tuning,
)
from aas_shortcut.compare_selection_runs import (
    PROBLEM_COLUMNS,
    compare_selection_representations,
    problem_levels,
    selection_runs,
    view_levels,
)
from aas_shortcut.create_figures.render_experiment_figures import render_selection_split
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import (
    FEATURE_SETS,
    ORACLE_VARIANT,
    TRANSFORMS,
    RunPaths,
    representation_path,
)
from aas_shortcut.define_reporting_scopes import SCOPE_COLUMN, SCOPE_CONTROL
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN, VARIANTS
from aas_shortcut.resolve_benchmark_units import assemble
from aas_shortcut.resolve_portfolio_blocks import (
    PortfolioBindings,
    PortfolioUnit,
    SelectionHeldOut,
    SelectionTraining,
    resolve_portfolio_bindings,
    resolve_selection_held_out_in_range,
    resolve_selection_training,
)
from aas_shortcut.resolve_selection_cells import (
    SelectionCell,
    SelectionSetting,
    assemble_oracles,
    assemble_problems,
    assemble_records,
    cells_of,
    resolve_oracles,
    resolve_selection_cells,
    selection_data,
)
from aas_shortcut.resolve_stage_artifacts import (
    PLAN,
    MissingStageArtifact,
    StageArtifact,
    StageStore,
)
from aas_shortcut.resolve_tuning_stages import ROLE_TABLE
from aas_shortcut.run_experiment_spec import (
    ExperimentSpec,
    add_learner_option,
    add_overrides,
    add_store_options,
    build_context,
    build_parser,
    compose_config,
    experiment_label,
    record_run,
    resolve_run_paths,
    store_policy,
)
from aas_shortcut.select_algorithms import (
    SelectionConfig,
    rank_single_best_solver,
    view_of,
)
from aas_shortcut.split_bbob_instances import (
    MAX_INSTANCE,
    PROBLEM_CLASS,
    TRAIN_INSTANCES,
)

# The selectors are fitted on `pre` alone.
TRAININGS: tuple[str, ...] = (TRAINING_PLAIN,)

# Columns of Fig. 16's table: the method, the block and the problem.
FIGURE_COLUMNS: tuple[str, ...] = (
    "feature_set",
    "feature_variant",
    "training",
    "transform",
    *PROBLEM_COLUMNS,
    "superiority_vs_sbs",
)

# The levels the report prints and Fig. 16 marks, one table per representation set.
LEVEL_TABLE = Path("selection_levels.csv")

# The role tuning's configurations and the SBS ranking, written to the run root.
ROLE_TUNING_TABLE = Path("role_tuning.csv")
SBS_TABLE = Path("single_best_solver.csv")

# As Experiment 5's table, with the block in place of the training condition. The
# oracle repeats per row: it is the ceiling, not a fourth representation.
LEVEL_COLUMNS: tuple[str, ...] = (
    "feature_set",
    SCOPE_COLUMN,
    "representation",
    "transform",
    "superiority_vs_sbs_mean",
    "oracle",
)


@dataclass(frozen=True)
class SelectionUnits:
    """What Experiment 3 reads: Experiment 5's plain cells, oracles and data."""

    training: SelectionTraining
    blocks: list[SelectionHeldOut]
    config: SelectionConfig
    cells: dict[SelectionCell, StageArtifact]
    oracles: dict[str, StageArtifact]
    bindings: PortfolioBindings

    @property
    def control(self) -> SelectionHeldOut:
        """Return the control family's block, the one the experiment reports."""
        for block in self.blocks:
            if block.label == CONTROL.label:
                return block
        raise ValueError(f"{CONTROL.label} is out of its range on this block")

    def oracle(self) -> pd.DataFrame:
        """Return the oracle's problems on the control family."""
        return assemble_oracles(self.oracles, [CONTROL.label])

    def records(self) -> pd.DataFrame:
        """Return every training run's deployments on the control family."""
        return assemble_records(self.cells, [CONTROL.label])

    def problems(self) -> pd.DataFrame:
        """Return every training run's problems on the control family."""
        return assemble_problems(self.cells, [CONTROL.label])

    def _units(self, condition: str) -> list[PortfolioUnit]:
        # `pre` on the training and test blocks, `post` on the test block.
        return [
            self.training.pre[condition],
            self.control.pre[condition],
            self.control.post[condition],
        ]

    def performance(self, condition: str) -> pd.DataFrame:
        """Return one condition's runs: training block, and test block in both views."""
        return assemble(
            [unit.performance for unit in self._units(condition)], "view_performance"
        )

    def mean_superiorities(self, condition: str) -> pd.DataFrame:
        """Return one condition's mean superiorities over the same runs."""
        return assemble(
            [unit.mean_superiorities for unit in self._units(condition)],
            "view_mean_superiority",
        )


def resolve_selection_blocks(
    store: StageStore, context: Any
) -> tuple[PortfolioBindings, SelectionTraining, list[SelectionHeldOut]]:
    """Resolve Experiment 5's data for its plain cells, as Experiment 5 does.

    A plain cell is named by every family it is scored on, so the cells are shared
    only if the blocks are.
    """
    bindings = resolve_portfolio_bindings(store, context)
    training = resolve_selection_training(store, context, bindings, trainings=TRAININGS)
    blocks = resolve_selection_held_out_in_range(store, context, bindings)
    return bindings, training, blocks


def resolve_selection_units(store: StageStore, context: Any) -> SelectionUnits:
    """Resolve everything Experiment 3 reads, computing what the store's policy lets."""
    bindings, training, blocks = resolve_selection_blocks(store, context)
    config = context.selection_config()
    features = paper_columns(context.feature_sets, context.scale)
    setting = SelectionSetting(
        features=tuple(features),
        dim=context.dim,
        base_seed=context.seed,
        sample_size=context.feature_config(PROBLEM_CLASS).sample_size,
    )
    data = selection_data(training, blocks, features)
    cells = resolve_selection_cells(
        store, config, cells_of(config, TRAININGS), data, setting, context.scale
    )
    oracles = resolve_oracles(store, config, data, setting, context.scale)
    return SelectionUnits(training, blocks, config, cells, oracles, bindings)


def level_board(
    levels: pd.DataFrame, oracle: pd.DataFrame, feature_set: str
) -> pd.DataFrame:
    """Return one representation set's levels by block, with the oracle's.

    The oracle's level is its mean over the problems.
    """
    rows = levels[levels["feature_set"] == feature_set]
    ceiling = oracle[
        (oracle["feature_set"] == feature_set) & (oracle["training"] == TRAINING_PLAIN)
    ]
    board = rows.pivot_table(
        index="feature_variant", columns="transform", values="value"
    ).reindex(index=list(VARIANTS), columns=list(TRANSFORMS))
    board.loc[ORACLE_VARIANT] = [
        float(ceiling[ceiling["transform"] == view]["superiority_vs_sbs"].mean())
        for view in TRANSFORMS
    ]
    return board


def level_table(board: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Lay one representation set's :func:`level_board` out as the written table.

    Full precision, so quoted numbers are rounded from the full value.
    """
    return pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                SCOPE_COLUMN: SCOPE_CONTROL,
                "representation": variant,
                "transform": view,
                "superiority_vs_sbs_mean": float(board.loc[variant, view]),
                "oracle": float(board.loc[ORACLE_VARIANT, view]),
            }
            for variant in VARIANTS
            for view in TRANSFORMS
        ],
        columns=list(LEVEL_COLUMNS),
    )


def write_levels(
    levels: pd.DataFrame, oracle: pd.DataFrame, run_paths: RunPaths
) -> list[Path]:
    """Write the levels the report printed, one table per representation set."""
    written: list[Path] = []
    for feature_set in FEATURE_SETS:
        if (levels["feature_set"] != feature_set).all():
            continue
        table = level_table(level_board(levels, oracle, feature_set), feature_set)
        target = run_paths.root / representation_path(LEVEL_TABLE, feature_set)
        table.to_csv(target, index=False)
        written.append(target)
    return written


# The seed-level test of the representations.
REPRESENTATION_TEST_TABLE = Path("representation_tests.csv")


def report_seed_tests(records: pd.DataFrame, run_paths: RunPaths) -> list[Path]:
    """Write the representations' seed-level comparison, one table per set.

    Fitting on `pre` alone, only the representations' comparison applies.
    """
    written: list[Path] = []
    for name, rows in records.groupby("feature_set", observed=True):
        table = compare_selection_representations(
            selection_runs(rows, SCOPE_CONTROL), SCOPE_CONTROL
        )
        target = run_paths.root / representation_path(
            REPRESENTATION_TEST_TABLE, str(name)
        )
        table.to_csv(target, index=False)
        written.append(target)
    return written


def write_portfolio_tables(
    units: SelectionUnits, context: Any, run_paths: RunPaths
) -> list[Path]:
    """Write the role tuning's configurations and the single best solver's ranking.

    The ranking is checked against the SBS the cells were scored against.
    """
    roles = summarize_role_tuning(
        units.bindings.roles.read_table(ROLE_TABLE),
        {
            CONDITION_SELECTION: context.search_budget,
            CONDITION_SBS: context.total_budget,
        },
    )
    training = units.training.mean_superiorities(CONDITION_SBS)
    training = view_of(
        cast(pd.DataFrame, training[training["instance_id"].isin(TRAIN_INSTANCES)]),
        "pre",
    )
    ranking = rank_single_best_solver(training, units.config.algorithms)
    flagged = set(ranking.loc[ranking["is_sbs"], "algorithm"].astype(str))
    recorded = set(units.oracle()["sbs_algorithm"].astype(str))
    if flagged != recorded:
        raise ValueError(
            f"the ranking flags {sorted(flagged)} but the cells were scored "
            f"against {sorted(recorded)}"
        )
    written = []
    for table, name in ((roles, ROLE_TUNING_TABLE), (ranking, SBS_TABLE)):
        target = run_paths.root / name
        table.to_csv(target, index=False)
        written.append(target)
    return written


def parse_args(argv: Sequence[str] | None, spec_id: str) -> argparse.Namespace:
    """Parse the entry point's command line, with the stage store's flags."""
    parser = build_parser(
        description=(
            f"Run {experiment_label(spec_id)}: value-based, rank-based and concat "
            f"selectors over the tuned heterogeneous portfolio, scored against the "
            f"single best solver on the original and on the transformed test block. "
            f"Trailing arguments are passed to Hydra as overrides."
        ),
        epilog=f"example: run_{spec_id}.py task.n_jobs=28",
    )
    add_store_options(parser)
    add_learner_option(parser)
    return add_overrides(parser).parse_args(argv)


def run_selection_experiment(argv: Sequence[str] | None, spec_path: Path) -> int:
    """Drive the selection experiment."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    spec = ExperimentSpec.load(spec_path)
    args = parse_args(argv, spec.id)
    cfg = compose_config(
        spec.hydra_overrides(
            args.preset,
            args.overrides,
            learner=args.learner,
        )
    )
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)
    if context.max_instances < MAX_INSTANCE:
        raise ValueError(
            f"the split needs instances 1-{MAX_INSTANCE}, but the preset asks for "
            f"{context.max_instances}; run with preset.max_instances={MAX_INSTANCE}"
        )

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        units = resolve_selection_units(store, context)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1
    run_paths.mkdirs()

    records = units.records()
    levels = view_levels(records)
    oracle = units.oracle()
    for path in write_levels(levels, oracle, run_paths):
        print(f"written: {path}")
    for path in report_seed_tests(records, run_paths):
        print(f"written: {path}")
    for path in write_portfolio_tables(units, context, run_paths):
        print(f"written: {path}")

    # Fig. 16: per-problem medians beside the oracle's problems; rules at the
    # table's levels.
    problems = problem_levels(units.problems())
    figure_table = cast(
        pd.DataFrame,
        pd.concat(
            [oracle[list(FIGURE_COLUMNS)], problems[list(FIGURE_COLUMNS)]],
            ignore_index=True,
        ),
    )
    for path in render_selection_split(
        figure_table, run_paths, training=TRAINING_PLAIN, levels=levels
    ):
        print(f"written: {path}")
    store.write_list(run_paths.root)
    record_run(
        spec,
        cfg,
        run_paths,
        extra={"trainings": list(TRAININGS), "argv": list(argv or sys.argv[1:])},
    )
    return 0
