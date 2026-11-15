"""Experiment vocabulary and artifact layout."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# ``pre`` is ``f(x)``; ``post`` is ``f(x)`` under the instance's ranking-preserving
# transformation (:mod:`aas_shortcut.sample_monotone_transform`).
TRANSFORMS = ("pre", "post")

# Encodings the feature stage stores. ``concat`` is built from these two at read
# time rather than stored.
FEATURE_ENCODINGS = ("value", "rank")

# The two encodings side by side (never averaged).
CONCAT_VARIANT = "concat"

FEATURE_VARIANTS = ("value", "rank", CONCAT_VARIANT)

# Fingerprints spell ``concat`` as ``combined``, so that the variant's name does not
# change the identity of the stored measurements.
FINGERPRINT_VARIANT_NAMES = {CONCAT_VARIANT: "combined"}

# What the preanalysis measures a distribution shift of: a landscape feature or the
# pairwise superiority of two members. One table carries both, on one scale.
QUANTITY_KINDS = ("feature", "superiority")

FEATURE_QUANTITY = "feature"
SUPERIORITY_QUANTITY = "superiority"

# ``feature_set`` and ``feature_variant`` on a superiority row: "does not apply",
# where a null would read as "not known".
NOT_A_FEATURE = "none"

# The perfect instance-adaptive selector.
ORACLE_VARIANT = "oracle"

SELECTION_VARIANTS = (*FEATURE_VARIANTS, ORACLE_VARIANT)

# Feature representations: handcrafted ELA and the learned Deep-ELA embedding.
FEATURE_SETS = ("ela", "deep-ela")

ELA_FEATURE_SET = "ela"
DEEP_ELA_FEATURE_SET = "deep-ela"

# File-name suffix of one representation's table or figure; the two representations
# are never pooled into one file.
REPRESENTATION_SUFFIXES = {ELA_FEATURE_SET: "", DEEP_ELA_FEATURE_SET: "_deep-ela"}

PROBLEM_CLASSES = ("bbob",)


# ``full`` is the scale the paper reports; ``debug`` exercises every code path in
# minutes. Feature columns are declared only for these scales.
SCALES = ("debug", "full")

MANIFEST_SUFFIX = ".manifest.json"

# Written by an entry point when it finishes: resolved configuration, code version
# and a digest of every artifact in the root. Not ``*.manifest.json``, which names
# one shard's manifest.
RUN_MANIFEST_NAME = "run_manifest.json"

TABLE_SUFFIX = ".parquet"

# Shared stages' artifacts, one directory per stage and fingerprint.
STAGE_STORE_NAME = "_stages"


def run_root_name(base: str, preset: str, reported_preset: str) -> str:
    """Return the run root one experiment writes to at one scale.

    Non-reported scales get a suffix (``experiment3_debug``), so a rehearsal never
    overwrites a reported artifact whose file name carries no scale.
    """
    return base if preset == reported_preset else f"{base}_{preset}"


def validate_choice(value: str, allowed: tuple[str, ...], what: str) -> str:
    """Return ``value`` unchanged after checking it is one of ``allowed``."""
    if value not in allowed:
        raise ValueError(f"Unknown {what} {value!r}; expected one of {list(allowed)}")
    return value


def validate_transform(transform: str) -> str:
    """Validate a ``pre``/``post`` transform name."""
    return validate_choice(transform, TRANSFORMS, "transform")


def validate_encoding(encoding: str) -> str:
    """Validate a stored feature encoding name."""
    return validate_choice(encoding, FEATURE_ENCODINGS, "feature encoding")


def validate_feature_variant(variant: str) -> str:
    """Validate a feature variant name, including ``concat``."""
    return validate_choice(variant, FEATURE_VARIANTS, "feature variant")


def fingerprint_variant(variant: str) -> str:
    """Spell one variant the way the shards on disk were fingerprinted."""
    return FINGERPRINT_VARIANT_NAMES.get(variant, variant)


def fingerprint_variants(variants: Iterable[str]) -> list[str]:
    """Spell a sequence of variants for a fingerprint, in the order given."""
    return [fingerprint_variant(variant) for variant in variants]


def validate_feature_set(feature_set: str) -> str:
    """Validate a feature-set name (``ela`` or ``deep-ela``)."""
    return validate_choice(feature_set, FEATURE_SETS, "feature set")


def validate_problem_class(problem_class: str) -> str:
    """Validate a benchmark suite name."""
    return validate_choice(problem_class, PROBLEM_CLASSES, "problem class")


def problem_slug(problem_class: str) -> str:
    """Return a filename-safe slug for a problem class name."""
    return validate_problem_class(problem_class).replace("-", "")


def representation_path(path: Path | str, feature_set: str) -> Path:
    """Return where one representation's copy of a table is written."""
    path = Path(path)
    suffix = REPRESENTATION_SUFFIXES.get(
        validate_feature_set(feature_set), f"_{feature_set}"
    )
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def manifest_path_for(data_path: Path) -> Path:
    """Return the manifest path that accompanies a data file."""
    data_path = Path(data_path)
    return data_path.with_suffix(data_path.suffix + MANIFEST_SUFFIX)


def shard_stem(kind: str, problem_class: str, dim: int) -> str:
    """Build the filename stem of one ``(kind, suite, dimension)`` shard."""
    validate_problem_class(problem_class)
    return f"{kind}_{problem_slug(problem_class)}_{dim}d"


@dataclass(frozen=True)
class Shard:
    """A long-format table; its manifest sits beside it (:func:`manifest_path_for`)."""

    data: Path

    def exists(self) -> bool:
        """Whether the data file has been written."""
        return self.data.exists()


@dataclass(frozen=True)
class RunPaths:
    """Resolved directory layout for one experiment run root."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | str) -> RunPaths:
        """Build a layout from a run root, resolving it to an absolute path."""
        return cls(Path(root).expanduser().resolve())

    @property
    def features_dir(self) -> Path:
        """Directory holding extracted ELA features."""
        return self.root / "features"

    @property
    def performance_dir(self) -> Path:
        """Directory holding raw optimizer performance logs."""
        return self.root / "performance"

    @property
    def superiority_dir(self) -> Path:
        """Directory holding pairwise superiorities."""
        return self.root / "superiority"

    @property
    def mean_superiority_dir(self) -> Path:
        """Directory holding instance-wise algorithm mean superiorities."""
        return self.root / "mean_superiority"

    @property
    def metrics_dir(self) -> Path:
        """Directory holding |dz| and prediction metric tables."""
        return self.root / "metrics"

    @property
    def figures_dir(self) -> Path:
        """Directory holding the paper figures."""
        return self.root / "figures"

    def features(self, problem_class: str, dim: int) -> Shard:
        """Return the feature shard for one suite and dimension."""
        stem = shard_stem("features", problem_class, dim)
        return Shard(self.features_dir / f"{stem}{TABLE_SUFFIX}")

    def deployment_features(self, problem_class: str, dim: int) -> Shard:
        """Return the held-out block's shard of one sample per deployment run.

        Separate from :meth:`features` because it holds each instance once per
        deployment, while training reads the single-sample shard.
        """
        stem = shard_stem("deployment_features", problem_class, dim)
        return Shard(self.features_dir / f"{stem}{TABLE_SUFFIX}")

    def view_performance(
        self, problem_class: str, dim: int, condition: str | None = None
    ) -> Shard:
        """Return a two-view performance shard."""
        name = (
            "performance_views"
            if condition is None
            else f"performance_views_{condition}"
        )
        stem = shard_stem(name, problem_class, dim)
        return Shard(self.performance_dir / f"{stem}{TABLE_SUFFIX}")

    def view_superiority(self, problem_class: str, dim: int) -> Shard:
        """Return the superiority shard, which carries both views."""
        stem = shard_stem("view_superiority", problem_class, dim)
        return Shard(self.superiority_dir / f"{stem}{TABLE_SUFFIX}")

    def view_mean_superiority(
        self, problem_class: str, dim: int, condition: str | None = None
    ) -> Shard:
        """Return the mean-superiority shard, which carries both views."""
        name = (
            "view_mean_superiority"
            if condition is None
            else f"view_mean_superiority_{condition}"
        )
        stem = shard_stem(name, problem_class, dim)
        return Shard(self.mean_superiority_dir / f"{stem}{TABLE_SUFFIX}")

    def metric(self, name: str) -> Shard:
        """Return a named metric table, e.g. ``superiority_calibration``."""
        return Shard(self.metrics_dir / f"{name}{TABLE_SUFFIX}")

    def figure(self, name: str, extension: str = "pdf") -> Path:
        """Return the path of a named figure."""
        return self.figures_dir / f"{name}.{extension}"

    def mkdirs(self) -> None:
        """Create every output directory in the layout."""
        for directory in (
            self.features_dir,
            self.performance_dir,
            self.superiority_dir,
            self.mean_superiority_dir,
            self.metrics_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
