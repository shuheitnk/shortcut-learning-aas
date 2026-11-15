"""The preanalysis - Standardized Shift of Features and Superiorities.

Measured on the superiority regression's own held-out block, warp draw, eleven
deployed samples and target; the block is shared with Experiments 1, 2 and 4.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from aas_shortcut.calculate_delta_z import PRIMARY_STANDARDIZATION, STATISTIC
from aas_shortcut.create_figures.draw_shift_figures import SHIFT_FORMATS
from aas_shortcut.create_figures.render_experiment_figures import (
    render_distribution_shift,
)
from aas_shortcut.define_algorithm_portfolio import GROUPS
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import (
    SUPERIORITY_QUANTITY,
    TRANSFORMS,
    RunPaths,
)
from aas_shortcut.define_repetitions import REPETITIONS
from aas_shortcut.define_warp_audits import CONTROL
from aas_shortcut.measure_distribution_shift import (
    feature_observations,
    measure_shift,
    run_shift_summary_stage,
    view_superiority_observations,
)
from aas_shortcut.record_manifests import stage_fingerprint
from aas_shortcut.resolve_pair_blocks import (
    HeldOutBlock,
    resolve_held_out_block,
    resolve_held_out_pre,
)
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
    compose_config,
    record_run,
    resolve_run_paths,
    store_policy,
)
from aas_shortcut.sample_monotone_transform import (
    WARP_STRENGTH_TABLE,
    warp_strengths,
)
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS, TEST_INSTANCES

LOGGER = logging.getLogger(__name__)

SPEC_PATH = Path(__file__).with_name("config.yaml")


# Only `value` features can move under a transformation; `rank` ones are invariant
# by construction and checked by the feature stage.
REPORTED_VARIANT = "value"


# The eleven deployments, as the feature shard numbers them.
DEPLOYMENTS: tuple[int, ...] = tuple(range(1, REPETITIONS + 1))

# The step whose pair tuning the benchmark binds from.
TUNING_STEP = "E2-1"


def _resolve(store: StageStore, context) -> HeldOutBlock:
    """Resolve the control family's held-out block, computing what the store lacks."""
    binding, tuning = resolve_pair_binding(store, context, TUNING_STEP)
    pre = resolve_held_out_pre(store, context, binding, tuning, TUNING_STEP)
    return resolve_held_out_block(
        store, context, binding, tuning, CONTROL, pre, TUNING_STEP
    )


def _held_out(table: pd.DataFrame, *, shard: str) -> pd.DataFrame:
    """Keep the held-out problems and check that both views cover every one.

    The shift is between two views of one problem, so a problem only one view
    reaches is not a small shift but no measurement at all.
    """
    block = table[table["instance_id"].isin(TEST_INSTANCES)]
    covered = block.groupby("transform", observed=True)[
        ["function_id", "instance_id"]
    ].apply(lambda rows: set(map(tuple, rows.drop_duplicates().to_numpy())))
    expected = {
        (function, instance)
        for function in sorted(block["function_id"].unique())
        for instance in TEST_INSTANCES
    }
    # Loop over the views needed, not those present: a view missing entirely fails.
    for transform in TRANSFORMS:
        missing = sorted(expected - covered.get(transform, set()))
        if missing:
            raise ValueError(
                f"{shard}: the {transform!r} view is missing {len(missing)} "
                f"problem(s) of the held-out block, e.g. f{missing[0][0]} "
                f"i{missing[0][1]}"
            )
    return block


def _fitted_columns(context) -> set[str]:
    """Return the paper's fixed feature columns, the ones the regression fits on."""
    return set(paper_columns(context.feature_sets, context.scale))


def _deployed(features: pd.DataFrame, columns: set[str] | None = None) -> pd.DataFrame:
    """Keep the deployments, the encoding and the columns the regression reports."""
    kept = features[
        features["rep"].isin(DEPLOYMENTS)
        & (features["feature_variant"] == REPORTED_VARIANT)
    ]
    if columns is None:
        return kept
    return kept[kept["feature"].isin(sorted(columns))]


def _check_deployments(features: pd.DataFrame) -> None:
    """Check that every problem, view and column holds every deployment.

    Each deployment's scale is taken over its problems, so a missing problem would
    change what was measured rather than the transformation's effect.
    """
    held = features.groupby(
        ["function_id", "instance_id", "transform", "feature"], observed=True
    )["rep"].agg(lambda reps: tuple(sorted(set(reps))))
    short = held[held.map(lambda reps: reps != DEPLOYMENTS)]
    if not short.empty:
        function, instance, transform, feature = short.index[0]
        raise ValueError(
            f"{len(short)} problem-view-column(s) do not hold deployments "
            f"{DEPLOYMENTS[0]}-{DEPLOYMENTS[-1]}, e.g. f{function} i{instance} "
            f"{transform} {feature} with {list(short.iloc[0])}"
        )


def _observations(
    features: pd.DataFrame, superiorities: pd.DataFrame, columns: set[str] | None = None
) -> pd.DataFrame:
    """Put the two quantities into the one shape the shift is measured in."""
    return pd.concat(
        [
            feature_observations(_deployed(features, columns)),
            view_superiority_observations(superiorities, DEPLOYMENTS),
        ],
        ignore_index=True,
    )


def _check_control(summary: pd.DataFrame) -> bool:
    """Check that the G1 (rank-based) pair's superiority did not move.

    G1 reads only the objective's ordering and both views share trial seeds, so a
    shift is a measurement bug. A failure still writes the outputs for inspection.
    """
    g1 = _reported(summary)
    g1 = g1[
        (g1["quantity_kind"] == SUPERIORITY_QUANTITY)
        & (g1["quantity_class"] == GROUPS[0])
    ]
    moved = g1[(g1["value"] > 0.0) | (g1["value_raw"] > 0.0)]
    if not moved.empty:
        LOGGER.error(
            "preanalysis: the G1 control failed -- the rank-based pair's superiority "
            "moved on %.0f%% of the problems, so the two views did not produce "
            "identical trajectories",
            100 * float(moved["p_moved"].iloc[0]),
        )
        return False
    return True


def _write_strengths(superiorities: pd.DataFrame, context, run_paths: RunPaths) -> Path:
    """Write how far each problem's map departs from the identity.

    The curve alone decides the strength, so no stage unit is needed.
    """
    strengths = warp_strengths(
        superiorities, int(context.seed), context.dim, CONTROL.warp
    )
    target = run_paths.root / WARP_STRENGTH_TABLE
    strengths.to_csv(target, index=False)
    return target


def _reported(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep the summary rows the figures are made from."""
    return summary[
        (summary["statistic"] == STATISTIC)
        & (summary["standardization"] == PRIMARY_STANDARDIZATION)
    ]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse this entry point's command line."""
    parser = build_parser(
        description=(
            "Run the preanalysis: how far a strictly increasing transformation moves "
            "a landscape feature, and how far it moves a superiority, both measured "
            "on the superiority regression's held-out block and in units of how far "
            "the quantity varies over it. Trailing arguments are passed to Hydra "
            "as overrides."
        ),
        epilog="example: run_preanalysis.py task.n_jobs=28",
    )
    add_store_options(parser)
    return add_overrides(parser).parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the preanalysis and return a process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    spec = ExperimentSpec.load(SPEC_PATH)
    cfg = compose_config(spec.hydra_overrides(args.preset, args.overrides))
    run_paths = resolve_run_paths(cfg)
    context = build_context(cfg, run_paths)

    store = StageStore.under(cfg.paths.output_root, store_policy(args))
    try:
        block = _resolve(store, context)
    except MissingStageArtifact as missing:
        print(f"stopped: {missing}", file=sys.stderr)
        return 0 if store.policy == PLAN else 1
    features = _held_out(block.feature_table, shard="the deployment features")
    superiorities = _held_out(block.superiorities, shard="the superiorities")
    columns = _fitted_columns(context)
    _check_deployments(_deployed(features, columns))

    # Fingerprint the input units, not where they live: a run root's name is not
    # part of what was measured.
    config_hash = stage_fingerprint(
        {
            "stage": "shift",
            "dim": context.dim,
            "reps": list(DEPLOYMENTS),
            "feature_variant": REPORTED_VARIANT,
            "n_features": len(columns),
            "inputs": dict(sorted(block.inputs.items())),
        }
    )
    shift = measure_shift(_observations(features, superiorities, columns))
    summary = run_shift_summary_stage(
        shift, PROBLEM_CLASS, context.dim, run_paths, context.scale, config_hash
    )

    readable = _check_control(summary)
    written = [_write_strengths(superiorities, context, run_paths)]
    written += render_distribution_shift(
        _reported(summary), run_paths, SHIFT_FORMATS, REPORTED_VARIANT
    )
    for path in written:
        print(f"written: {path}")
    store.write_list(run_paths.root)
    record_run(
        spec,
        cfg,
        run_paths,
        status="completed" if readable else "control-failed",
        extra={"argv": list(argv or sys.argv[1:])},
    )
    return 0 if readable else 1


if __name__ == "__main__":
    sys.exit(main())
