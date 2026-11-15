"""Long-format table schemas and their validator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aas_shortcut.calculate_delta_z import PRIMARY_STANDARDIZATION, STATISTIC
from aas_shortcut.define_paths import (
    FEATURE_ENCODINGS,
    FEATURE_SETS,
    FEATURE_VARIANTS,
    NOT_A_FEATURE,
    PROBLEM_CLASSES,
    QUANTITY_KINDS,
    SCALES,
    SELECTION_VARIANTS,
    TRANSFORMS,
)


@dataclass(frozen=True)
class SchemaViolation:
    """One reason a table failed validation."""

    table: str
    kind: str
    detail: str

    def __str__(self) -> str:
        """Render the violation as a single log-friendly line."""
        return f"[{self.table}] {self.kind}: {self.detail}"


@dataclass(frozen=True)
class TableSchema:
    """The contract one long-format table must satisfy."""

    name: str
    columns: Mapping[str, str]
    key_columns: Sequence[str]
    finite_columns: Sequence[str] = ()
    domains: Mapping[str, Sequence[str]] = field(default_factory=dict)
    bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    nullable_columns: Sequence[str] = ()
    optional_columns: Sequence[str] = ()

    @property
    def column_order(self) -> list[str]:
        """Canonical column order for the written table."""
        return list(self.columns)


# Columns tracing a row back to the run that produced it.
_PROVENANCE: dict[str, str] = {
    "scale": "string",
    "config_hash": "string",
}

_INSTANCE: dict[str, str] = {
    "problem_class": "string",
    "function_id": "int32",
    "instance_id": "int32",
    "dimension": "int32",
}

FEATURES_SCHEMA = TableSchema(
    name="features",
    columns={
        **_INSTANCE,
        "transform": "string",
        "feature_set": "string",
        "feature_variant": "string",
        "rep": "int32",
        "sample_seed": "int64",
        "sample_size": "int32",
        "feature": "string",
        "value": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "transform",
        "feature_set",
        "feature_variant",
        "rep",
        "feature",
    ),
    finite_columns=("value",),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "transform": TRANSFORMS,
        "feature_set": FEATURE_SETS,
        "feature_variant": FEATURE_ENCODINGS,
        "scale": SCALES,
    },
)


SELECTION_SCHEMA = TableSchema(
    name="selection",
    columns={
        **_INSTANCE,
        "feature_set": "string",
        "feature_variant": "string",
        # The algorithm chosen most often over the trials, and how often.
        "modal_algorithm": "string",
        "modal_share": "float64",
        "n_selected_algorithms": "int32",
        "sbs_algorithm": "string",
        "n_trials": "int32",
        # Evaluation budget of each side.
        "selection_budget": "int32",
        "sbs_budget": "int32",
        "superiority_vs_sbs": "float64",
        "median_best_y_selection": "float64",
        "median_best_y_sbs": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "feature_set",
        "feature_variant",
    ),
    finite_columns=(
        "superiority_vs_sbs",
        "median_best_y_selection",
        "median_best_y_sbs",
    ),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "feature_set": FEATURE_SETS,
        # Includes the oracle, which is scored on this axis but is not a feature set.
        "feature_variant": SELECTION_VARIANTS,
        "scale": SCALES,
    },
    bounds={"superiority_vs_sbs": (0.0, 1.0), "modal_share": (0.0, 1.0)},
)


TUNING_PAIR_SCHEMA = TableSchema(
    name="tuning_pair",
    columns={
        "group": "string",
        "pair": "string",
        "tuned": "string",
        "fixed": "string",
        "config_id": "string",
        # JSON.
        "params": "string",
        "is_default": "bool",
        "is_selected": "bool",
        "function_id": "int32",
        "instance_id": "int32",
        "dimension": "int32",
        "budget": "int32",
        "superiority": "float64",
        # D_KS of the configuration's whole sample (repeated per row) and the
        # granularity floor to read it against.
        "score": "float64",
        "floor": "float64",
        "n_saturated": "int32",
        **_PROVENANCE,
    },
    key_columns=("pair", "config_id", "function_id", "instance_id"),
    finite_columns=("superiority", "score", "floor"),
    domains={"scale": SCALES},
    bounds={"superiority": (0.0, 1.0), "score": (0.0, 1.0)},
)


# Each instance is benchmarked on ``f`` and on its warped counterpart, so
# ``transform`` is part of the key.
VIEW_PERFORMANCE_SCHEMA = TableSchema(
    name="view_performance",
    columns={
        **_INSTANCE,
        "transform": "string",
        "algorithm": "string",
        "trial": "int32",
        "seed": "int64",
        "budget": "int32",
        "evaluations_used": "int32",
        "restarts": "int32",
        "multistarts": "Int32",
        "native_reanneals": "Int32",
        # Always on the raw objective, so the two views share one scale.
        "best_y": "float64",
        "optimum_y": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "transform",
        "algorithm",
        "trial",
    ),
    finite_columns=("best_y", "optimum_y"),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "transform": TRANSFORMS,
        "scale": SCALES,
    },
    nullable_columns=("multistarts", "native_reanneals"),
    optional_columns=("multistarts", "native_reanneals"),
)

# The regression's target: the Vargha-Delaney superiority under each view, kept apart.
VIEW_SUPERIORITY_SCHEMA = TableSchema(
    name="view_superiority",
    columns={
        **_INSTANCE,
        "transform": "string",
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
        "transform",
        "pair",
    ),
    finite_columns=("superiority",),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "transform": TRANSFORMS,
        "scale": SCALES,
    },
    bounds={"superiority": (0.0, 1.0)},
)

# Whether the pair tuning's calibration on 24 problems holds on the 2400 the
# regression is fitted on, measured on the target itself.
SUPERIORITY_CALIBRATION_SCHEMA = TableSchema(
    name="superiority_calibration",
    columns={
        "block": "string",
        "group": "string",
        "pair": "string",
        "transform": "string",
        "n_problems": "int32",
        "n_trials": "int32",
        # D_KS against Uniform(0, 1) and the best a block of this size and
        # granularity could reach, which it is read against.
        "ks_distance": "float64",
        "ks_floor": "float64",
        "mean": "float64",
        "variance": "float64",
        "n_saturated": "int32",
        "saturated_share": "float64",
        "n_informative": "int32",
        "informative_share": "float64",
        **_PROVENANCE,
    },
    key_columns=("block", "pair", "transform"),
    finite_columns=("ks_distance", "ks_floor", "mean", "variance"),
    domains={"transform": TRANSFORMS, "scale": SCALES},
    bounds={
        "ks_distance": (0.0, 1.0),
        "ks_floor": (0.0, 1.0),
        "mean": (0.0, 1.0),
        "saturated_share": (0.0, 1.0),
        "informative_share": (0.0, 1.0),
    },
)

# The decile shape behind each row of the table above.
SUPERIORITY_CALIBRATION_DECILE_SCHEMA = TableSchema(
    name="superiority_calibration_decile",
    columns={
        "block": "string",
        "group": "string",
        "pair": "string",
        "transform": "string",
        "decile": "int32",
        "lower": "float64",
        "upper": "float64",
        "n_problems": "int32",
        # Count a uniform block of this size would put in the bin.
        "expected_uniform": "float64",
        **_PROVENANCE,
    },
    key_columns=("block", "pair", "transform", "decile"),
    finite_columns=("lower", "upper", "expected_uniform"),
    domains={"transform": TRANSFORMS, "scale": SCALES},
    bounds={"lower": (0.0, 1.0), "upper": (0.0, 1.0)},
)


# Every training run of a regression cell scored on every deployment of its test
# blocks. Seeds and deployments are shared across the cells compared, so the
# comparisons are paired.
DEPLOYMENT_SCORE_SCHEMA = TableSchema(
    name="deployment_score",
    columns={
        "feature_set": "string",
        "feature_variant": "string",
        "training": "string",
        # Warp of the test block, as ``<family>/<seed domain>``.
        "audit": "string",
        "pair": "string",
        "group": "string",
        "model": "string",
        "n_features": "int32",
        # One of MODEL_SEEDS.
        "seed": "int32",
        # The deployment, as the feature shard numbers them.
        "rep": "int32",
        "metric": "string",
        "value": "float64",
        "n_test": "int32",
        **_PROVENANCE,
    },
    key_columns=(
        "feature_set",
        "feature_variant",
        "training",
        "audit",
        "pair",
        "model",
        "metric",
        "seed",
        "rep",
    ),
    finite_columns=("value",),
    domains={
        "feature_set": FEATURE_SETS,
        "feature_variant": FEATURE_VARIANTS,
        "scale": SCALES,
    },
)

# The preanalysis's shifts: reduced over instances within each deployment, then the
# median over deployments.
SHIFT_SUMMARY_SCHEMA = TableSchema(
    name="shift_summary",
    columns={
        "problem_class": "string",
        "dimension": "int32",
        "quantity_kind": "string",
        "feature_set": "string",
        "feature_variant": "string",
        "quantity": "string",
        "quantity_class": "string",
        "standardization": "string",
        "statistic": "string",
        # Standardized shift. Null where a deployment had no defined instance: a
        # quantity constant across instances has no unit, and zero would claim no move.
        "value": "float64",
        # Unstandardized shift, in the quantity's own units; always defined.
        "value_raw": "float64",
        # Fraction of instances on which the quantity moved at all.
        "p_moved": "float64",
        "n_instances": "int32",
        # Instances excluded from `value` because their shift was undefined.
        "n_undefined": "int32",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "dimension",
        "quantity_kind",
        "feature_set",
        "feature_variant",
        "quantity",
        "standardization",
        "statistic",
    ),
    finite_columns=("value_raw", "p_moved"),
    nullable_columns=("value",),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "quantity_kind": QUANTITY_KINDS,
        "feature_set": (*FEATURE_SETS, NOT_A_FEATURE),
        "feature_variant": (*FEATURE_VARIANTS, NOT_A_FEATURE),
        "standardization": (PRIMARY_STANDARDIZATION,),
        "statistic": (STATISTIC,),
        "scale": SCALES,
    },
    bounds={"p_moved": (0.0, 1.0)},
)


# The selector's target: each member's mean superiority per instance and view.
# 1 is the best member, 0 the worst.
VIEW_MEAN_SUPERIORITY_SCHEMA = TableSchema(
    name="view_mean_superiority",
    columns={
        **_INSTANCE,
        "transform": "string",
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
        "transform",
        "algorithm",
    ),
    finite_columns=("mean_superiority",),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "transform": TRANSFORMS,
        "scale": SCALES,
    },
    bounds={"mean_superiority": (0.0, 1.0)},
)

# A selection cell's mean superiority over the SBS per training run and deployment,
# as :data:`DEPLOYMENT_SCORE_SCHEMA` is for a regression cell.
SELECTION_DEPLOYMENT_SCORE_SCHEMA = TableSchema(
    name="selection_deployment_score",
    columns={
        # The view the test block was measured under.
        "transform": "string",
        # The views the selector was fitted on.
        "training": "string",
        "feature_set": "string",
        "feature_variant": "string",
        "dimension": "int32",
        # One of MODEL_SEEDS.
        "seed": "int32",
        # The deployment, as the feature shard numbers them.
        "rep": "int32",
        # Mean over the test problems.
        "superiority_vs_sbs": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "transform",
        "training",
        "feature_set",
        "feature_variant",
        "dimension",
        "seed",
        "rep",
    ),
    finite_columns=("superiority_vs_sbs",),
    domains={
        "transform": TRANSFORMS,
        "feature_set": FEATURE_SETS,
        "feature_variant": SELECTION_VARIANTS,
        "scale": SCALES,
    },
    bounds={"superiority_vs_sbs": (0.0, 1.0)},
)

# Per-problem rows behind :data:`SELECTION_DEPLOYMENT_SCORE_SCHEMA`'s means.
SELECTION_PROBLEM_SCORE_SCHEMA = TableSchema(
    name="selection_problem_score",
    columns={
        # The view the test block was measured under.
        "transform": "string",
        # The views the selector was fitted on.
        "training": "string",
        "feature_set": "string",
        "feature_variant": "string",
        # One of MODEL_SEEDS.
        "seed": "int32",
        # The deployment, as the feature shard numbers them.
        "rep": "int32",
        **_INSTANCE,
        # Probability, over the trials, that the selection beats the SBS.
        "superiority_vs_sbs": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "transform",
        "training",
        "feature_set",
        "feature_variant",
        "seed",
        "rep",
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
    ),
    finite_columns=("superiority_vs_sbs",),
    domains={
        "transform": TRANSFORMS,
        "problem_class": PROBLEM_CLASSES,
        "feature_set": FEATURE_SETS,
        "feature_variant": SELECTION_VARIANTS,
        "scale": SCALES,
    },
    bounds={"superiority_vs_sbs": (0.0, 1.0)},
)

# The oracle's superiority over the SBS per test problem, view and training
# condition: :data:`SELECTION_SCHEMA` plus those two columns.
VIEW_SELECTION_SCHEMA = TableSchema(
    name="view_selection",
    columns={
        **_INSTANCE,
        # The view the test block was measured under.
        "transform": "string",
        # The views the selector was fitted on.
        "training": "string",
        "feature_set": "string",
        "feature_variant": "string",
        "modal_algorithm": "string",
        "modal_share": "float64",
        "n_selected_algorithms": "int32",
        "sbs_algorithm": "string",
        "n_trials": "int32",
        "selection_budget": "int32",
        "sbs_budget": "int32",
        "superiority_vs_sbs": "float64",
        "median_best_y_selection": "float64",
        "median_best_y_sbs": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "problem_class",
        "function_id",
        "instance_id",
        "dimension",
        "transform",
        "training",
        "feature_set",
        "feature_variant",
    ),
    finite_columns=(
        "superiority_vs_sbs",
        "median_best_y_selection",
        "median_best_y_sbs",
    ),
    domains={
        "problem_class": PROBLEM_CLASSES,
        "transform": TRANSFORMS,
        "feature_set": FEATURE_SETS,
        "feature_variant": SELECTION_VARIANTS,
        "scale": SCALES,
    },
    bounds={"superiority_vs_sbs": (0.0, 1.0), "modal_share": (0.0, 1.0)},
)

GRID_ERROR_SCHEMA = TableSchema(
    name="grid_error",
    columns={
        # The two budgets are independent measurements and are never pooled.
        "budget": "int32",
        "algorithm": "string",
        "config_id": "string",
        "function_id": "int32",
        "instance_id": "int32",
        "trial": "int32",
        # Final distance to the optimum. Not reduced to a rank, which depends on
        # every other configuration and could not be recomputed later.
        "error": "float64",
        **_PROVENANCE,
    },
    key_columns=(
        "budget",
        "algorithm",
        "config_id",
        "function_id",
        "instance_id",
        "trial",
    ),
    finite_columns=("error",),
    domains={"scale": SCALES},
)


TUNING_ROLE_SCHEMA = TableSchema(
    name="tuning_role",
    columns={
        # The role, by its budget: 1000d for selection, 1050d for no selection.
        "budget": "int32",
        "algorithm": "string",
        "config_id": "string",
        "params": "string",
        "is_default": "bool",
        "is_selected": "bool",
        "function_id": "int32",
        "instance_id": "int32",
        # Against every configuration in the grid, so members share one scale and
        # the oracle's `max` over them is meaningful.
        "mean_superiority": "float64",
        # The same over each half of the trials, for the split-half check.
        "mean_superiority_odd": "float64",
        "mean_superiority_even": "float64",
        "dimension": "int32",
        "n_trials": "int32",
        **_PROVENANCE,
    },
    key_columns=("budget", "algorithm", "config_id", "function_id", "instance_id"),
    finite_columns=(
        "mean_superiority",
        "mean_superiority_odd",
        "mean_superiority_even",
    ),
    domains={"scale": SCALES},
    bounds={
        "mean_superiority": (0.0, 1.0),
        "mean_superiority_odd": (0.0, 1.0),
        "mean_superiority_even": (0.0, 1.0),
    },
)


SCHEMAS: dict[str, TableSchema] = {
    schema.name: schema
    for schema in (
        FEATURES_SCHEMA,
        SELECTION_SCHEMA,
        TUNING_PAIR_SCHEMA,
        SUPERIORITY_CALIBRATION_SCHEMA,
        SUPERIORITY_CALIBRATION_DECILE_SCHEMA,
        VIEW_PERFORMANCE_SCHEMA,
        VIEW_SUPERIORITY_SCHEMA,
        DEPLOYMENT_SCORE_SCHEMA,
        SHIFT_SUMMARY_SCHEMA,
        VIEW_SELECTION_SCHEMA,
        SELECTION_DEPLOYMENT_SCORE_SCHEMA,
        SELECTION_PROBLEM_SCORE_SCHEMA,
        VIEW_MEAN_SUPERIORITY_SCHEMA,
        GRID_ERROR_SCHEMA,
        TUNING_ROLE_SCHEMA,
    )
}


def conform(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Return ``df`` with the schema's columns, order and dtypes."""
    missing = [column for column in schema.columns if column not in df.columns]
    absent_optional = [c for c in missing if c in set(schema.optional_columns)]
    required = [c for c in missing if c not in set(schema.optional_columns)]
    if required:
        raise ValueError(f"{schema.name}: missing column(s) {required}")
    if absent_optional:
        # Older shards may lack an optional column.
        df = df.assign(**dict.fromkeys(absent_optional, pd.NA))
    extra = [column for column in df.columns if column not in schema.columns]
    if extra:
        raise ValueError(f"{schema.name}: unexpected column(s) {extra}")

    out = df[schema.column_order].copy()
    for column, dtype in schema.columns.items():
        out[column] = out[column].astype(dtype)
    return out


def _check_dtypes(df: pd.DataFrame, schema: TableSchema) -> list[SchemaViolation]:
    violations = []
    for column, dtype in schema.columns.items():
        actual = str(df[column].dtype)
        if actual != dtype:
            violations.append(
                SchemaViolation(
                    schema.name,
                    "dtype",
                    f"column {column!r} is {actual}, expected {dtype}",
                )
            )
    return violations


def _check_missing(df: pd.DataFrame, schema: TableSchema) -> list[SchemaViolation]:
    violations = []
    nullable = set(schema.nullable_columns)
    for column in schema.columns:
        if column in nullable:
            continue
        n_missing = int(df[column].isna().sum())
        if n_missing:
            violations.append(
                SchemaViolation(
                    schema.name,
                    "missing_value",
                    f"column {column!r} has {n_missing} missing value(s)",
                )
            )
    return violations


def _check_finite(df: pd.DataFrame, schema: TableSchema) -> list[SchemaViolation]:
    violations = []
    for column in schema.finite_columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
        n_nan = int(np.isnan(values).sum())
        n_inf = int(np.isinf(values).sum())
        if n_nan:
            violations.append(
                SchemaViolation(
                    schema.name, "nan", f"column {column!r} has {n_nan} NaN value(s)"
                )
            )
        if n_inf:
            violations.append(
                SchemaViolation(
                    schema.name, "inf", f"column {column!r} has {n_inf} +/-Inf value(s)"
                )
            )
    return violations


def _check_domains(df: pd.DataFrame, schema: TableSchema) -> list[SchemaViolation]:
    violations = []
    for column, allowed in schema.domains.items():
        seen = set(df[column].dropna().unique())
        unknown = sorted(str(value) for value in seen - set(allowed))
        if unknown:
            violations.append(
                SchemaViolation(
                    schema.name,
                    "domain",
                    f"column {column!r} has value(s) {unknown} outside {list(allowed)}",
                )
            )
    return violations


def _check_bounds(df: pd.DataFrame, schema: TableSchema) -> list[SchemaViolation]:
    violations = []
    for column, (low, high) in schema.bounds.items():
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
        outside = np.isfinite(values) & ((values < low) | (values > high))
        n_outside = int(outside.sum())
        if n_outside:
            violations.append(
                SchemaViolation(
                    schema.name,
                    "bounds",
                    f"column {column!r} has {n_outside} value(s) outside "
                    f"[{low}, {high}]",
                )
            )
    return violations


def _check_duplicate_keys(
    df: pd.DataFrame, schema: TableSchema
) -> list[SchemaViolation]:
    key = list(schema.key_columns)
    duplicated = df.duplicated(subset=key, keep=False)
    n_duplicated = int(duplicated.sum())
    if not n_duplicated:
        return []
    example = df.loc[duplicated, key].head(1).to_dict("records")
    return [
        SchemaViolation(
            schema.name,
            "duplicate_key",
            f"{n_duplicated} row(s) share a key on {key}; first example {example}",
        )
    ]


def _check_config_hash(df: pd.DataFrame, schema: TableSchema) -> list[SchemaViolation]:
    if "config_hash" not in schema.columns:
        return []
    hashes = sorted(str(value) for value in df["config_hash"].dropna().unique())
    if len(hashes) > 1:
        return [
            SchemaViolation(
                schema.name,
                "config_hash",
                f"table mixes {len(hashes)} configurations: {hashes}",
            )
        ]
    return []


def validate_table(
    df: pd.DataFrame,
    schema: TableSchema | str,
    *,
    expected_rows: int | None = None,
    required_values: Mapping[str, Iterable[object]] | None = None,
) -> list[SchemaViolation]:
    """Return every violation of one table against its schema (empty if valid)."""
    if isinstance(schema, str):
        schema = SCHEMAS[schema]

    missing = [column for column in schema.columns if column not in df.columns]
    if missing:
        return [
            SchemaViolation(
                schema.name, "missing_column", f"column(s) {missing} are absent"
            )
        ]
    extra = [column for column in df.columns if column not in schema.columns]

    violations: list[SchemaViolation] = []
    if extra:
        violations.append(
            SchemaViolation(
                schema.name, "unexpected_column", f"column(s) {extra} are not declared"
            )
        )
    violations.extend(_check_dtypes(df, schema))
    violations.extend(_check_missing(df, schema))
    violations.extend(_check_finite(df, schema))
    violations.extend(_check_domains(df, schema))
    violations.extend(_check_bounds(df, schema))
    violations.extend(_check_duplicate_keys(df, schema))
    violations.extend(_check_config_hash(df, schema))

    if expected_rows is not None and len(df) != expected_rows:
        violations.append(
            SchemaViolation(
                schema.name,
                "row_count",
                f"table has {len(df)} row(s), expected {expected_rows}",
            )
        )

    for column, values in (required_values or {}).items():
        present = set(df[column].unique())
        absent = sorted(str(value) for value in set(values) - present)
        if absent:
            violations.append(
                SchemaViolation(
                    schema.name,
                    "incomplete",
                    f"column {column!r} is missing {len(absent)} required "
                    f"value(s), e.g. {absent[:5]}",
                )
            )

    return violations


def describe_violations(violations: Sequence[SchemaViolation]) -> str:
    """Render violations as a multi-line message for an exception or a log."""
    return "\n".join(f"  - {violation}" for violation in violations)
