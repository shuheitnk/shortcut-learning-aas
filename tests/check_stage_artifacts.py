"""Checks over the stored units of a stage store, for the tests that read a real one.

They re-check on disk what each stage already refuses as it writes a unit: an
over-budget run, a moved rank feature, missing rows, out-of-range values, and files
that do not match the unit's record.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from aas_shortcut.benchmark_objective_views import ViewBenchmarkConfig
from aas_shortcut.calculate_features import FeatureStageConfig
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_paths import FEATURE_ENCODINGS, TRANSFORMS
from aas_shortcut.define_table_schema import (
    SchemaViolation,
    validate_table,
)
from aas_shortcut.record_manifests import sha256_file
from aas_shortcut.resolve_stage_artifacts import STAGE_RECORD_NAME
from tests.conftest import MEAN_SUPERIORITY_SCHEMA, deep_ela_feature_names

STAGE_SCHEMAS = {
    "features": "features",
    "view_performance": "view_performance",
    "view_superiority": "view_superiority",
    "tune_pair": "tuning_pair",
    "augmented_regression": "deployment_score",
}

VIEW_RUN = ["function_id", "instance_id", "trial", "algorithm", "transform"]


def stored_units(store: Path) -> list[tuple[str, Path, dict]]:
    """Every complete unit under a store, as ``(stage, directory, record)``."""
    units = []
    for path in sorted(store.glob(f"*/*/{STAGE_RECORD_NAME}")):
        record = json.loads(path.read_text(encoding="utf-8"))
        units.append((record["stage"], path.parent, record))
    return units


def check_unit_files(directory: Path, record: dict) -> list[str]:
    """Return what differs between a unit's files and the record it keeps of them."""
    listed = record["files"]
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != STAGE_RECORD_NAME
    }
    problems = [
        f"{name} is not in the record" for name in sorted(present - set(listed))
    ]
    for name, described in sorted(listed.items()):
        path = directory / name
        if not path.exists():
            problems.append(f"{name} is missing")
        elif sha256_file(path) != described["sha256"]:
            problems.append(f"{name} is not the file the record names")
    return problems


def view_benchmark_config(settings: dict) -> ViewBenchmarkConfig:
    """The view benchmark a unit's settings describe, for its grid of runs."""
    return ViewBenchmarkConfig(
        problem_class=settings["problem_class"],
        dim=settings["dim"],
        max_functions=settings["max_functions"],
        max_instances=settings["max_instances"],
        num_trials=settings["num_trials"],
        budget=settings["budget"],
        base_seed=settings["base_seed"],
        algorithms=tuple(settings["algorithms"]),
        transforms=tuple(settings["transforms"]),
        instances_by_transform=settings.get("instances_by_transform"),
    )


def feature_stage_config(settings: dict) -> FeatureStageConfig:
    """The feature stage a unit's settings describe, for its grid of samples."""
    return FeatureStageConfig(
        problem_class=settings["problem_class"],
        dim=settings["dim"],
        sampling_factor=settings["sampling_factor"],
        num_reps=settings["num_reps"],
        max_functions=settings["max_functions"],
        max_instances=settings["max_instances"],
        base_seed=settings["base_seed"],
        scale=settings["scale"],
        feature_sets=tuple(settings["feature_sets"]),
    )


def check_budget(
    performance: pd.DataFrame, *, require_multistart: bool = True
) -> list[SchemaViolation]:
    """Test 6: no run exceeded its budget, and restarts are accounted for."""
    violations: list[SchemaViolation] = []

    over = performance[performance["evaluations_used"] > performance["budget"]]
    if not over.empty:
        worst = int((over["evaluations_used"] - over["budget"]).max())
        violations.append(
            SchemaViolation(
                "performance",
                "budget",
                f"{len(over)} run(s) exceeded the budget, by up to {worst} "
                "evaluation(s)",
            )
        )

    empty = performance[performance["evaluations_used"] <= 0]
    if not empty.empty:
        violations.append(
            SchemaViolation(
                "performance",
                "budget",
                f"{len(empty)} run(s) consumed no evaluations at all",
            )
        )

    multistart = performance[performance["restarts"] > 0]["algorithm"].unique()
    if require_multistart and len(multistart) == 0:
        violations.append(
            SchemaViolation(
                "performance",
                "multistart",
                "no algorithm ever restarted; the shared multi-start wrapper "
                "never fired",
            )
        )
    return violations


def check_rank_invariance(features: pd.DataFrame) -> list[SchemaViolation]:
    """Test 1, re-checked on the stored artifact."""
    ranks = features[features["feature_variant"] == "rank"]
    if ranks.empty:
        return [
            SchemaViolation("features", "rank_invariance", "no rank features stored")
        ]

    index = ["function_id", "instance_id", "rep", "feature"]
    wide = ranks.pivot_table(
        index=index, columns="transform", values="value", aggfunc="mean"
    )
    if "pre" not in wide.columns or "post" not in wide.columns:
        return [
            SchemaViolation(
                "features", "rank_invariance", "the rank features lack a transform"
            )
        ]

    pre = wide["pre"].to_numpy(dtype=np.float64)
    post = wide["post"].to_numpy(dtype=np.float64)
    if np.allclose(pre, post, rtol=1e-7, atol=1e-9, equal_nan=True):
        return []

    gap = np.abs(pre - post)
    return [
        SchemaViolation(
            "features",
            "rank_invariance",
            f"{int(np.count_nonzero(gap > 0))} rank feature value(s) differ "
            f"between pre and post; worst |diff| = {np.nanmax(gap):.3e}",
        )
    ]


def check_feature_completeness(
    features: pd.DataFrame, config: FeatureStageConfig
) -> list[SchemaViolation]:
    """Test 7: every instance, repetition, transform, set and encoding is present."""
    functions, instances = config.grid()
    required = {
        "function_id": list(functions),
        "instance_id": list(instances),
        "transform": list(TRANSFORMS),
        "feature_set": list(config.feature_sets),
        "feature_variant": list(FEATURE_ENCODINGS),
        "rep": list(range(1, config.num_reps + 1)),
    }
    deep_names = (
        deep_ela_feature_names(config.deep_ela) if config.wants_deep_ela else []
    )
    if deep_names:
        required["feature"] = deep_names
    return validate_table(
        features,
        "features",
        expected_rows=config.expected_feature_rows(
            cast(int, features["feature"].nunique())
        ),
        required_values=required,
    )


def check_view_completeness(
    performance: pd.DataFrame, config: ViewBenchmarkConfig
) -> list[SchemaViolation]:
    """Test 7: every run the benchmark declares is there once, and no other."""
    expected = {
        (function_id, instance_id, trial, algorithm, transform)
        for function_id, instance_id, trial, algorithm, transform in config.jobs()
    }
    held = list(performance[VIEW_RUN].itertuples(index=False, name=None))
    violations = validate_table(
        performance, "view_performance", expected_rows=len(expected)
    )
    missing, extra = expected - set(held), set(held) - expected
    if missing:
        violations.append(
            SchemaViolation(
                "view_performance",
                "incomplete",
                f"{len(missing)} declared run(s) are absent, e.g. {sorted(missing)[0]}",
            )
        )
    if extra:
        violations.append(
            SchemaViolation(
                "view_performance",
                "incomplete",
                f"{len(extra)} run(s) are not declared, e.g. {sorted(extra)[0]}",
            )
        )
    return violations


def check_mean_superiority(mean_superiorities: pd.DataFrame) -> list[SchemaViolation]:
    """Tests 7 and 8 for the mean-superiority table."""
    violations = validate_table(
        mean_superiorities,
        MEAN_SUPERIORITY_SCHEMA,
        required_values={"algorithm": list(ALGORITHMS)},
    )

    # Every pair of members hands out exactly one win between them, so a complete
    # ranking of k members scores k(k-1)/2 wins over k-1 opponents: k/2.
    expected_total = len(ALGORITHMS) / 2
    totals = mean_superiorities.groupby(
        ["problem_class", "function_id", "instance_id", "dimension"],
        observed=True,
    )["mean_superiority"].sum()
    off = totals[~np.isclose(totals.to_numpy(dtype=np.float64), expected_total)]
    if not off.empty:
        violations.append(
            SchemaViolation(
                "mean_superiority",
                "incomplete",
                f"{len(off)} instance(s) have mean superiorities summing to something "
                f"other than {expected_total}; first {off.index[0]} sums to "
                f"{float(off.iloc[0]):.3f}",
            )
        )
    return violations
