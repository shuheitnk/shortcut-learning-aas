"""Shared fixtures.

The synthetic tables mirror the real artifacts column for column and dtype for dtype,
so a test against a fixture tests the same contract the pipeline writes.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_module

from aas_shortcut.bind_portfolio_parameters import CONDITION_SELECTION, PortfolioBinding
from aas_shortcut.calculate_deep_ela_features import (
    FEATURE_PREFIX,
    MODEL_FEATURE_COUNT,
    DeepElaConfig,
)
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_config_schema import register_configs
from aas_shortcut.define_paths import (
    DEEP_ELA_FEATURE_SET,
    ELA_FEATURE_SET,
    FEATURE_ENCODINGS,
    PROBLEM_CLASSES,
    SCALES,
    TRANSFORMS,
    RunPaths,
)
from aas_shortcut.define_table_schema import (
    _INSTANCE,
    _PROVENANCE,
    FEATURES_SCHEMA,
    TableSchema,
    conform,
)
from aas_shortcut.define_tunable_portfolio import TUNABLE
from aas_shortcut.run_experiment_spec import CONFIG_MODULE

# Single-view tables for the synthetic fixtures; the stages write the two-view forms,
# and these are the per-view slices the superiority functions read.
PERFORMANCE_SCHEMA = TableSchema(
    name="performance",
    columns={
        **_INSTANCE,
        "algorithm": "string",
        "trial": "int32",
        "seed": "int64",
        "budget": "int32",
        "evaluations_used": "int32",
        # Optional: they describe a run rather than decide one, and no analysis
        # stage reads them (see `RestartReport`).
        "restarts": "int32",
        "multistarts": "Int32",
        "native_reanneals": "Int32",
        "best_y": "float64",
        "optimum_y": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "algorithm",
        "trial",
        # One trial is observed at two budgets, so the budget identifies an
        # observation.
        "budget",
    ),
    finite_columns=("best_y", "optimum_y"),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "scale": SCALES,
    },
    nullable_columns=("multistarts", "native_reanneals"),
    optional_columns=("multistarts", "native_reanneals"),
)


SUPERIORITY_SCHEMA = TableSchema(
    name="superiority",
    columns={
        **_INSTANCE,
        "pair": "string",
        "algorithm_a": "string",
        "algorithm_b": "string",
        "pair_type": "string",
        "n_trials": "int32",
        "superiority": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "pair",
    ),
    finite_columns=("superiority",),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "scale": SCALES,
    },
    bounds={"superiority": (0.0, 1.0)},
)


# The selector's target: each member's mean superiority ``J`` on each instance.
# 1 is the best member, 0 the worst.
MEAN_SUPERIORITY_SCHEMA = TableSchema(
    name="mean_superiority",
    columns={
        **_INSTANCE,
        "algorithm": "string",
        "group": "string",
        "n_trials": "int32",
        "mean_superiority": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "algorithm",
    ),
    finite_columns=("mean_superiority",),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "scale": SCALES,
    },
    bounds={"mean_superiority": (0.0, 1.0)},
)

FEATURE_NAMES = (
    "ela_meta.lin_simple.adj_r2",
    "ela_distr.skewness",
    "fitness_distance.fitness_mean",
)

# Feature names of the synthetic learned representation.
DEEP_FEATURE_NAMES = (
    "deep_ela.large_50d_v1.X0",
    "deep_ela.large_50d_v1.X1",
)

FEATURE_NAMES_BY_SET = {
    ELA_FEATURE_SET: FEATURE_NAMES,
    DEEP_ELA_FEATURE_SET: DEEP_FEATURE_NAMES,
}

CONFIG_HASH = "0123456789abcdef"


def identical_tables(
    a: pd.DataFrame,
    b: pd.DataFrame,
    ignore: Sequence[str] = ("config_hash",),
) -> tuple[bool, str]:
    """Compare two tables value for value, floats at the bit level.

    Rows are matched by sorting on every non-float column, so row order does not
    matter. Returns whether they agree and the first difference found.
    """
    a = a.drop(columns=[column for column in ignore if column in a.columns])
    b = b.drop(columns=[column for column in ignore if column in b.columns])
    if set(a.columns) != set(b.columns):
        return False, f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return False, f"rows differ: {len(a)} vs {len(b)}"
    columns = list(a.columns)
    b = b.reindex(columns=columns)
    for column in columns:
        if a[column].dtype != b[column].dtype:
            return False, (
                f"column {column}: dtype {a[column].dtype} vs {b[column].dtype}"
            )
    keys = [column for column in columns if not pd.api.types.is_float_dtype(a[column])]
    if keys:
        a = a.assign(**{f"__{c}": a[c].astype(str) for c in keys})
        b = b.assign(**{f"__{c}": b[c].astype(str) for c in keys})
        order = [f"__{c}" for c in keys]
        a = a.sort_values(order, kind="mergesort").reset_index(drop=True)
        b = b.sort_values(order, kind="mergesort").reset_index(drop=True)
    for column in columns:
        if column in keys:
            if not np.array_equal(
                a[f"__{column}"].to_numpy(), b[f"__{column}"].to_numpy()
            ):
                return False, f"column {column} differs"
            continue
        x = a[column].to_numpy(np.float64)
        y = b[column].to_numpy(np.float64)
        if not np.array_equal(x.view(np.uint64), y.view(np.uint64)):
            n = int((x.view(np.uint64) != y.view(np.uint64)).sum())
            worst = float(np.nanmax(np.abs(x - y))) if n else 0.0
            return False, (
                f"column {column}: {n} value(s) differ, max |diff| {worst:.3e}"
            )
    return True, f"{len(a)} rows x {len(columns)} columns bit-identical"


def compose_config(*overrides: str):
    """Compose the root configuration exactly as an entry point does."""
    register_configs()
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


@pytest.fixture
def run_paths(tmp_path) -> RunPaths:
    """An empty run layout rooted in a temporary directory."""
    paths = RunPaths.from_root(tmp_path / "run")
    paths.mkdirs()
    return paths


def deep_ela_feature_names(config: DeepElaConfig | None = None) -> list[str]:
    """Return the names Deep-ELA produces, without running the model."""
    config = config or DeepElaConfig()
    width = MODEL_FEATURE_COUNT[config.model]
    return [f"{FEATURE_PREFIX}{config.model}.X{index}" for index in range(width)]


def make_features(
    problem_class: str = "bbob",
    function_id: int = 1,
    n_instances: int = 8,
    n_reps: int = 3,
    dim: int = 2,
    scale: str = "debug",
    *,
    rank_invariant: bool = True,
    value_shift: float = 1.5,
    seed: int = 0,
    feature_sets: tuple[str, ...] = (ELA_FEATURE_SET,),
) -> pd.DataFrame:
    """Build a long-format feature table with the real schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for instance_id in range(1, n_instances + 1):
        for rep in range(1, n_reps + 1):
            for feature_set in feature_sets:
                for feature in FEATURE_NAMES_BY_SET[feature_set]:
                    base = float(rng.normal(instance_id, 0.2))
                    shift = value_shift * float(rng.uniform(0.7, 1.3))
                    for encoding in FEATURE_ENCODINGS:
                        for transform in TRANSFORMS:
                            if encoding == "rank":
                                value = base
                                if not rank_invariant and transform == "post":
                                    value = base + 0.5
                            else:
                                value = base + (shift if transform == "post" else 0.0)
                            rows.append(
                                {
                                    "problem_class": problem_class,
                                    "function_id": function_id,
                                    "instance_id": instance_id,
                                    "dimension": dim,
                                    "transform": transform,
                                    "feature_set": feature_set,
                                    "feature_variant": encoding,
                                    "rep": rep,
                                    "sample_seed": 1000 + instance_id * 10 + rep,
                                    "sample_size": 100,
                                    "feature": feature,
                                    "value": value,
                                    "scale": scale,
                                    "config_hash": CONFIG_HASH,
                                }
                            )
    return conform(pd.DataFrame(rows), FEATURES_SCHEMA)


def _restart_counters(algorithm: str, trial: int) -> dict[str, int]:
    """Restart counters consistent with the member's declared mechanism."""
    from tests.define_restart_mechanism import MECHANISMS

    mechanism = MECHANISMS[algorithm]
    restarts = 1 + (trial % 3)
    if mechanism.keeps_native_mechanism:
        return {
            "restarts": restarts,
            "multistarts": 0,
            "native_reanneals": restarts,
        }
    multistarted = "multistart" in mechanism.native_handling
    return {
        "restarts": restarts,
        "multistarts": restarts + 1 if multistarted else 0,
        "native_reanneals": 0,
    }


def make_performance(
    problem_class: str = "bbob",
    function_id: int = 1,
    n_instances: int = 8,
    n_trials: int = 5,
    dim: int = 2,
    budget: int = 200,
    scale: str = "debug",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a long-format performance table with the real schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for instance_id in range(1, n_instances + 1):
        for trial in range(1, n_trials + 1):
            for algorithm in ALGORITHMS:
                rows.append(
                    {
                        "problem_class": problem_class,
                        "function_id": function_id,
                        "instance_id": instance_id,
                        "dimension": dim,
                        "algorithm": algorithm,
                        "trial": trial,
                        "seed": 5000 + trial,
                        "budget": budget,
                        "evaluations_used": budget - trial,
                        # Counters agree with the member's declared mechanism.
                        **_restart_counters(algorithm, trial),
                        "best_y": float(rng.normal(instance_id % 3, 0.5)),
                        "optimum_y": 0.0,
                        "scale": scale,
                        "config_hash": CONFIG_HASH,
                    }
                )
    return conform(pd.DataFrame(rows), PERFORMANCE_SCHEMA)


def make_sbs_performance(
    performance: pd.DataFrame, total_budget: int, penalty: float = -0.25
) -> pd.DataFrame:
    """Build the ``sbs`` condition's shard from the ``selection`` condition's."""
    sbs = performance.copy()
    sbs["budget"] = total_budget
    sbs["evaluations_used"] = total_budget - sbs["trial"]
    sbs["best_y"] = sbs["best_y"] + penalty
    return conform(sbs, PERFORMANCE_SCHEMA)


def make_superiority(performance: pd.DataFrame) -> pd.DataFrame:
    """Derive a superiority table from a performance table."""
    from aas_shortcut.calculate_superiority import compute_superiorities

    table = compute_superiorities(performance)
    table = table.assign(scale=performance["scale"].iloc[0], config_hash=CONFIG_HASH)
    return conform(table, SUPERIORITY_SCHEMA)


def make_mean_superiorities(performance: pd.DataFrame) -> pd.DataFrame:
    """Derive a mean-superiority table from a performance table."""
    from aas_shortcut.calculate_mean_superiority import compute_mean_superiorities

    table = compute_mean_superiorities(performance)
    table = table.assign(scale=performance["scale"].iloc[0], config_hash=CONFIG_HASH)
    return conform(table, MEAN_SUPERIORITY_SCHEMA)


@pytest.fixture
def features() -> pd.DataFrame:
    """A synthetic long-format feature table for MA-BBOB."""
    return make_features()


@pytest.fixture
def performance() -> pd.DataFrame:
    """A synthetic long-format performance table for MA-BBOB."""
    return make_performance()


@pytest.fixture
def superiorities(performance: pd.DataFrame) -> pd.DataFrame:
    """A synthetic long-format superiority table for MA-BBOB."""
    return make_superiority(performance)


@pytest.fixture
def mean_superiorities(performance: pd.DataFrame) -> pd.DataFrame:
    """A synthetic long-format mean-superiority table for MA-BBOB."""
    return make_mean_superiorities(performance)


def published_binding(
    condition: str = CONDITION_SELECTION, budget: int | None = None
) -> PortfolioBinding:
    """Every member at its published configuration, for the tests that run them."""
    return PortfolioBinding(
        condition=condition,
        objective="default",
        budget=budget,
        assignments=tuple(
            (name, tuple(sorted((str(k), v) for k, v in entry.resolve().items())))
            for name, entry in sorted(TUNABLE.items())
        ),
    )
