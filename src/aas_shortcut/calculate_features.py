"""Stage ``features``: ELA features for every instance and repetition."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from pflacco import classical_ela_features, misc_features

from aas_shortcut.calculate_deep_ela_features import (
    DeepElaConfig,
    availability,
    compute_deep_ela_features,
)
from aas_shortcut.create_problem import resolve_problem_grid
from aas_shortcut.create_sampled_points import (
    SampledPoints,
    create_sampled_points,
    ela_inputs,
)
from aas_shortcut.define_paths import (
    DEEP_ELA_FEATURE_SET,
    ELA_FEATURE_SET,
    FEATURE_ENCODINGS,
    TRANSFORMS,
    RunPaths,
    Shard,
    manifest_path_for,
    validate_feature_set,
)
from aas_shortcut.define_table_schema import FEATURES_SCHEMA
from aas_shortcut.execute_parallel_jobs import run_jobs
from aas_shortcut.read_write_tables import PartWriter, plan_flush_size
from aas_shortcut.record_manifests import write_manifest
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP, WarpConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Calculator:
    """A pflacco feature calculator plus the arguments pinned for determinism."""

    func: Callable[..., dict]
    # Prefix of every feature this calculator returns, e.g. ``ela_distr``.
    group: str
    kwargs: dict[str, object] = field(default_factory=dict)
    # Whether ``func`` takes a ``seed`` that must be pinned per repetition.
    needs_seed: bool = False

    @property
    def name(self) -> str:
        """The wrapped calculator's function name."""
        return self.func.__name__

    def __call__(self, X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
        """Run the calculator with its determinism arguments applied."""
        kwargs = dict(self.kwargs)
        if self.needs_seed:
            kwargs["seed"] = seed
        return self.func(X, y, **kwargs)


# The study's feature groups, in a fixed order so that the resulting feature
# names appear in a deterministic sequence.
ELA_CALCULATORS: tuple[Calculator, ...] = (
    Calculator(classical_ela_features.calculate_ela_distribution, "ela_distr"),
    Calculator(classical_ela_features.calculate_ela_level, "ela_level"),
    Calculator(classical_ela_features.calculate_ela_meta, "ela_meta"),
    Calculator(classical_ela_features.calculate_pca, "pca"),
    # The default tie breaker "sample" calls np.random.choice on equidistant points.
    Calculator(
        classical_ela_features.calculate_nbc,
        "nbc",
        kwargs={"dist_tie_breaker": "first"},
    ),
    Calculator(classical_ela_features.calculate_dispersion, "disp"),
    # Draws its nearest-neighbour walk start from the global RNG unless seeded.
    Calculator(
        classical_ela_features.calculate_information_content, "ic", needs_seed=True
    ),
    Calculator(
        misc_features.calculate_fitness_distance_correlation, "fitness_distance"
    ),
)

# pflacco reports its own timings alongside the features; they are wall-clock
# measurements and would make otherwise identical runs differ.
RUNTIME_SUFFIX = "costs_runtime"

# Smallest sample ``calculate_ela_level`` accepts.
MIN_ELA_SAMPLE_SIZE = 100

# Tolerances for the pre/post agreement of the rank features.
RANK_INVARIANCE_ATOL = 1e-9
RANK_INVARIANCE_RTOL = 1e-7


class RankInvarianceError(RuntimeError):
    """Raised when the rank features are not invariant under the transform."""


class OrderPreservationError(RuntimeError):
    """Raised when the transformation did not preserve the sample's ranking."""


class IncompleteShardError(RuntimeError):
    """Raised when a finished shard holds fewer rows than its grid requires."""


def compute_ela_features(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    seed: int,
    rejections: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Run every calculator in :data:`ELA_CALCULATORS` and drop timing entries."""
    features: dict[str, float] = {}
    for calculator in ELA_CALCULATORS:
        try:
            features.update(calculator(X_df, y_series, seed))
        except Exception as error:
            LOGGER.warning(
                "calculator %s refused this sample (%d points): %s",
                calculator.name,
                len(y_series),
                error,
            )
            if rejections is not None:
                rejections.append(
                    {
                        "calculator": calculator.name,
                        "group": calculator.group,
                        "n_points": len(y_series),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    return {
        key: float(value)
        for key, value in features.items()
        if not key.endswith(RUNTIME_SUFFIX)
    }


def compute_feature_set(
    feature_set: str,
    X_df: pd.DataFrame,
    y_series: pd.Series,
    seed: int,
    deep_ela_config: DeepElaConfig,
    rejections: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    """Compute one feature set for one ``(X, y)`` pair."""
    if validate_feature_set(feature_set) == ELA_FEATURE_SET:
        return compute_ela_features(X_df, y_series, seed, rejections)
    return compute_deep_ela_features(X_df, y_series, seed, deep_ela_config)


def _check_rank_invariance(
    conditions: dict[tuple[str, str, str], dict[str, float]],
    sample: SampledPoints,
    feature_set: str,
) -> None:
    """Assert that one set's rank features agree between ``pre`` and ``post``."""
    pre = conditions[(feature_set, "pre", "rank")]
    post = conditions[(feature_set, "post", "rank")]

    names = sorted(set(pre) | set(post))
    pre_values = np.array([pre.get(name, np.nan) for name in names])
    post_values = np.array([post.get(name, np.nan) for name in names])

    agree = np.allclose(
        pre_values,
        post_values,
        rtol=RANK_INVARIANCE_RTOL,
        atol=RANK_INVARIANCE_ATOL,
        equal_nan=True,
    )
    if agree:
        return

    gap = np.abs(pre_values - post_values)
    worst = int(np.nanargmax(gap))
    raise RankInvarianceError(
        f"{feature_set} rank features differ between pre and post for "
        f"{sample.problem_class} f{sample.function_id} i{sample.instance_id} "
        f"rep {sample.rep} (sample_seed={sample.sample_seed}): worst feature "
        f"{names[worst]!r} pre={pre_values[worst]!r} post={post_values[worst]!r} "
        f"|diff|={gap[worst]:.3e}"
    )


def features_for_sample(
    sample: SampledPoints,
    *,
    check_invariance: bool = True,
    feature_sets: tuple[str, ...] = (ELA_FEATURE_SET,),
    deep_ela_config: DeepElaConfig | None = None,
    rejections: list[dict[str, object]] | None = None,
) -> list[dict]:
    """Compute every requested feature set under all four conditions, as long rows."""
    if check_invariance and not sample.ranks_agree():
        raise OrderPreservationError(
            f"the transformation changed the ranking of the sample for "
            f"{sample.problem_class} f{sample.function_id} "
            f"i{sample.instance_id} rep {sample.rep} "
            f"(sample_seed={sample.sample_seed})"
        )

    deep_ela_config = deep_ela_config or DeepElaConfig()
    conditions: dict[tuple[str, str, str], dict[str, float]] = {}
    for feature_set in feature_sets:
        for transform in TRANSFORMS:
            for encoding in FEATURE_ENCODINGS:
                X_df, y_series = ela_inputs(sample, transform, encoding)
                conditions[(feature_set, transform, encoding)] = compute_feature_set(
                    feature_set,
                    X_df,
                    y_series,
                    sample.sample_seed,
                    deep_ela_config,
                    rejections,
                )

    if check_invariance:
        for feature_set in feature_sets:
            _check_rank_invariance(conditions, sample, feature_set)

    rows = []
    for (feature_set, transform, encoding), features in conditions.items():
        for name, value in features.items():
            rows.append(
                {
                    "problem_class": sample.problem_class,
                    "function_id": sample.function_id,
                    "instance_id": sample.instance_id,
                    "dimension": sample.dimension,
                    "transform": transform,
                    "feature_set": feature_set,
                    "feature_variant": encoding,
                    "rep": sample.rep,
                    "sample_seed": sample.sample_seed,
                    "sample_size": sample.sample_size,
                    "feature": name,
                    "value": value,
                }
            )
    return rows


def _feature_job(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    rep: int,
    sample_size: int,
    check_invariance: bool,
    feature_sets: tuple[str, ...],
    deep_ela_config: DeepElaConfig,
    warp: WarpConfig,
) -> tuple[list[dict], list[dict[str, object]]]:
    """Draw one sample and turn it into long-format feature rows."""
    rejections: list[dict[str, object]] = []
    started = time.perf_counter()
    sample = create_sampled_points(
        base_seed=base_seed,
        problem_class=problem_class,
        function_id=function_id,
        instance_id=instance_id,
        dim=dim,
        rep=rep,
        sample_size=sample_size,
        warp=warp,
    )
    rows = features_for_sample(
        sample,
        check_invariance=check_invariance,
        feature_sets=feature_sets,
        deep_ela_config=deep_ela_config,
        rejections=rejections,
    )
    for record in rejections:
        record.update(
            {
                "function_id": int(function_id),
                "instance_id": int(instance_id),
                "rep": int(rep),
            }
        )
    LOGGER.debug(
        "features f%02d i%03d rep %02d | %d row(s) | %.2fs",
        function_id,
        instance_id,
        rep,
        len(rows),
        time.perf_counter() - started,
    )
    return rows, rejections


@dataclass(frozen=True)
class FeatureStageConfig:
    """Everything that determines the content of one feature shard."""

    problem_class: str
    dim: int
    sampling_factor: int
    num_reps: int
    max_functions: int
    max_instances: int
    base_seed: int
    scale: str
    check_invariance: bool = True
    n_jobs: int = 1
    # Repetitions completed between flushes to disk.
    save_every: int = 25
    feature_sets: tuple[str, ...] = (ELA_FEATURE_SET,)
    deep_ela: DeepElaConfig = field(default_factory=DeepElaConfig)
    # The transformation behind the ``post`` view; its identifier is fingerprinted.
    warp: WarpConfig = DEFAULT_WARP

    @property
    def sample_size(self) -> int:
        """Number of sampled points, ``dim * sampling_factor``."""
        return self.dim * self.sampling_factor

    @property
    def wants_deep_ela(self) -> bool:
        """Whether the learned representation is among the requested sets."""
        return DEEP_ELA_FEATURE_SET in self.feature_sets

    def grid(self) -> tuple[range, range]:
        """Return the ``(functions, instances)`` ranges to sweep."""
        return resolve_problem_grid(
            self.problem_class, self.max_functions, self.max_instances
        )

    def jobs(self) -> list[tuple[int, int, int]]:
        """Return every ``(function_id, instance_id, rep)`` triple, in order."""
        functions, instances = self.grid()
        return [
            (function_id, instance_id, rep)
            for function_id in functions
            for instance_id in instances
            for rep in range(1, self.num_reps + 1)
        ]

    def expected_feature_rows(self, n_features: int) -> int:
        """Rows a complete shard must contain, for the completeness check."""
        conditions = len(TRANSFORMS) * len(FEATURE_ENCODINGS)
        return len(self.jobs()) * conditions * n_features

    def as_dict(self) -> dict:
        """Fingerprintable description of the configuration."""
        config = {
            "stage": "features",
            "problem_class": self.problem_class,
            "dim": self.dim,
            "sampling_factor": self.sampling_factor,
            "num_reps": self.num_reps,
            "max_functions": self.max_functions,
            "max_instances": self.max_instances,
            # Unused, but part of every existing shard's fingerprint.
            "num_mabbob": None,
            "base_seed": self.base_seed,
            "scale": self.scale,
            "feature_sets": list(self.feature_sets),
            "calculators": [calculator.name for calculator in ELA_CALCULATORS],
            "transform": self.warp.identifier,
        }
        # Only when computed, so a Deep-ELA setting never invalidates an ELA-only shard.
        if self.wants_deep_ela:
            config["deep_ela"] = self.deep_ela.as_dict()
        return config

    def validate(self) -> None:
        """Reject configurations the feature calculators cannot serve.

        Deep-ELA's availability is checked here, so missing weights are found before
        the tuning and benchmark that run ahead of this stage.
        """
        if self.sample_size < MIN_ELA_SAMPLE_SIZE:
            needed = -(-MIN_ELA_SAMPLE_SIZE // self.dim)
            raise ValueError(
                f"dim * sampling_factor = {self.dim} * {self.sampling_factor} "
                f"= {self.sample_size} is below the {MIN_ELA_SAMPLE_SIZE} points "
                f"calculate_ela_level needs; raise sampling_factor to at least "
                f"{needed}"
            )
        if not self.feature_sets:
            raise ValueError("at least one feature set must be requested")
        for feature_set in self.feature_sets:
            validate_feature_set(feature_set)
        if self.wants_deep_ela:
            self.deep_ela.validate(self.dim)
            availability().require()


@dataclass
class FeatureDiagnostics:
    """What a feature stage dropped, and why."""

    rejections: list[dict[str, object]] = field(default_factory=list)
    # Features removed because they were non-finite somewhere.
    nonfinite: list[str] = field(default_factory=list)
    # Features removed because some condition did not have them.
    unbalanced: list[str] = field(default_factory=list)
    # Where each non-finite value occurred, and whether it was NaN or an infinity.
    nonfinite_observations: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Render for the manifest, collapsing the refusals to their counts."""
        by_cause: dict[tuple[str, str], int] = {}
        for record in self.rejections:
            key = (str(record["calculator"]), str(record["error"]))
            by_cause[key] = by_cause.get(key, 0) + 1
        by_feature: dict[str, list[dict[str, object]]] = {}
        for record in self.nonfinite_observations:
            by_feature.setdefault(str(record["feature"]), []).append(
                {key: value for key, value in record.items() if key != "feature"}
            )
        return {
            "dropped_features": {
                "nonfinite": sorted(self.nonfinite),
                "unbalanced": sorted(self.unbalanced),
                "n_dropped": len(set(self.nonfinite) | set(self.unbalanced)),
            },
            "nonfinite_observations": {
                feature: {"n_observations": len(records), "observations": records}
                for feature, records in sorted(by_feature.items())
            },
            "calculator_rejections": [
                {"calculator": calculator, "error": error, "count": count}
                for (calculator, error), count in sorted(by_cause.items())
            ],
            "n_rejections": len(self.rejections),
        }


# The columns that name one observation of a feature, as far as a table has them.
OBSERVATION_COLUMNS = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
    "transform",
    "feature_set",
    "feature_variant",
    "rep",
)


def _nonfinite_kind(value: float) -> str:
    """Name a non-finite value the way a manifest records it."""
    if np.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _plain(value: object) -> object:
    """Return a NumPy scalar as the Python value a manifest can hold."""
    return value.item() if isinstance(value, np.generic) else value


def drop_nonfinite_features(
    table: pd.DataFrame,
    dropped: list[str] | None = None,
    observations: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Remove any feature that is not finite somewhere in the table."""
    if table.empty:
        return table

    values = table["value"].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    finite_by_feature = (
        table.assign(finite=finite).groupby("feature", observed=True)["finite"].all()
    )
    bad = cast(list[str], sorted(finite_by_feature[~finite_by_feature].index))
    if not bad:
        return table

    affected = table[table["feature"].isin(bad)]
    n_bad = int((~np.isfinite(affected["value"].to_numpy(dtype=np.float64))).sum())
    LOGGER.warning(
        "dropping %d feature(s) that are non-finite for %d of %d observation(s): %s",
        len(bad),
        n_bad,
        len(affected),
        ", ".join(bad),
    )
    if dropped is not None:
        dropped.extend(bad)
    if observations is not None:
        identity = [column for column in OBSERVATION_COLUMNS if column in table.columns]
        failed = table.loc[~finite, [*identity, "feature", "value"]]
        for row in failed.itertuples(index=False, name=None):
            record = {
                column: _plain(value)
                for column, value in zip(identity, row[: len(identity)], strict=True)
            }
            record["feature"] = str(row[len(identity)])
            record["value"] = _nonfinite_kind(float(row[len(identity) + 1]))
            observations.append(record)
    return table[~table["feature"].isin(bad)].reset_index(drop=True)


def drop_unbalanced_features(
    table: pd.DataFrame, dropped: list[str] | None = None
) -> pd.DataFrame:
    """Remove features that are missing from some condition."""
    if table.empty:
        return table

    counts = table.groupby("feature", observed=True).size()
    expected = int(cast(int, counts.max()))
    incomplete = cast(list[str], sorted(counts[counts < expected].index))
    if not incomplete:
        return table

    LOGGER.warning(
        "dropping %d feature(s) that are absent from some condition (present "
        "for fewer than %d observations): %s",
        len(incomplete),
        expected,
        ", ".join(incomplete),
    )
    if dropped is not None:
        dropped.extend(incomplete)
    return table[~table["feature"].isin(incomplete)].reset_index(drop=True)


def sanitize_features(
    table: pd.DataFrame, diagnostics: FeatureDiagnostics | None = None
) -> pd.DataFrame:
    """Make a feature shard rectangular and numerically valid."""
    diagnostics = diagnostics or FeatureDiagnostics()
    balanced = drop_unbalanced_features(table, diagnostics.unbalanced)
    return drop_nonfinite_features(
        balanced, diagnostics.nonfinite, diagnostics.nonfinite_observations
    )


# Enough to decide what a resumed run still has to do, without reading the whole
# shard.
JOB_IDENTITY = ("function_id", "instance_id", "rep")


def _completed_jobs(existing: pd.DataFrame) -> set[tuple[int, int, int]]:
    """Return the ``(function, instance, rep)`` triples already on disk."""
    if existing.empty:
        return set()
    key = existing[["function_id", "instance_id", "rep"]].drop_duplicates()
    return {
        (int(function_id), int(instance_id), int(rep))
        for function_id, instance_id, rep in key.itertuples(index=False, name=None)
    }


def run_feature_stage(
    config: FeatureStageConfig,
    run_paths: RunPaths,
    config_hash: str,
    *,
    batch_size: int | None = None,
    shard: Shard | None = None,
    diagnostics: FeatureDiagnostics | None = None,
) -> pd.DataFrame:
    """Compute and persist the feature shard for one suite and dimension."""
    diagnostics = diagnostics if diagnostics is not None else FeatureDiagnostics()
    config.validate()
    shard = shard or run_paths.features(config.problem_class, config.dim)
    writer = PartWriter(shard, FEATURES_SCHEMA, fingerprint=config_hash)

    cached = writer.reuse_or_reset()
    if cached is not None:
        n_features = int(cached["feature"].nunique()) if not cached.empty else 0
        if n_features and len(cached) == config.expected_feature_rows(n_features):
            LOGGER.info("features: %s already complete", shard.data)
            return assert_complete(cached, config, shard)

    # Not `cached`: a run interrupted before `consolidate` leaves finished samples in
    # the part directory, which would otherwise be appended twice.
    done = _completed_jobs(writer.existing(columns=JOB_IDENTITY))
    pending = [job for job in config.jobs() if job not in done]
    LOGGER.info(
        "features: %s %dD | %d job(s) pending, %d already done | n=%d | sets=%s",
        config.problem_class,
        config.dim,
        len(pending),
        len(done),
        config.sample_size,
        ", ".join(config.feature_sets),
    )

    tasks = [
        (
            config.base_seed,
            config.problem_class,
            function_id,
            instance_id,
            config.dim,
            rep,
            config.sample_size,
            config.check_invariance,
            tuple(config.feature_sets),
            config.deep_ela,
            config.warp,
        )
        for function_id, instance_id, rep in pending
    ]
    batch_size = plan_flush_size(len(pending), batch_size or config.save_every)

    buffer: list[dict] = []
    for finished, (_, (job_rows, job_rejections)) in enumerate(
        run_jobs(
            "features",
            tasks,
            _feature_job,
            n_jobs=config.n_jobs,
        ),
        start=1,
    ):
        buffer.extend(
            {**row, "scale": config.scale, "config_hash": config_hash}
            for row in job_rows
        )
        diagnostics.rejections.extend(job_rejections)
        # Checkpoint on completion count, not on batch boundaries, so that a slow
        # sample does not idle the other workers.
        if finished % batch_size == 0:
            writer.append(buffer)
            buffer.clear()

    if buffer:
        writer.append(buffer)

    table = assert_complete(
        writer.consolidate(prepare=lambda table: sanitize_features(table, diagnostics)),
        config,
        shard,
    )
    # Not written on the "already complete" path, where nothing was sanitized and an
    # empty `dropped_features` would be false.
    write_manifest(
        manifest_path_for(shard.data),
        shard.data.stem,
        config.as_dict(),
        outputs=[shard.data],
        status="finished",
        diagnostics=diagnostics.as_dict(),
    )
    return table


def assert_complete(
    table: pd.DataFrame, config: FeatureStageConfig, shard: Shard
) -> pd.DataFrame:
    """Refuse a shard that does not hold every row its grid declares.

    The count uses the features present, so a dropped feature does not trip it.
    """
    n_features = int(table["feature"].nunique()) if not table.empty else 0
    expected = config.expected_feature_rows(n_features)
    if n_features and len(table) == expected:
        return table
    raise IncompleteShardError(
        f"{shard.data} holds {len(table)} row(s) over {n_features} feature(s), "
        f"but {len(config.jobs())} job(s) x {len(TRANSFORMS) * len(FEATURE_ENCODINGS)} "
        f"condition(s) x {n_features} feature(s) = {expected}; the shard is "
        "incomplete and must not be used. Rerun the stage, or delete it and "
        "start over."
    )
