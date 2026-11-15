"""Typed Hydra configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from aas_shortcut.build_regression_model import (
    DEFAULT_CANDIDATES,
    REALMLP,
    ModelConfig,
)
from aas_shortcut.calculate_deep_ela_features import DEFAULT_MODEL, DEFAULT_REPETITIONS
from aas_shortcut.define_paths import ELA_FEATURE_SET, FEATURE_VARIANTS, PROBLEM_CLASSES
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP
from aas_shortcut.split_bbob_instances import TUNING_INSTANCES


@dataclass
class PresetConfig:
    """Scale parameters that distinguish a debug run from the paper's."""

    # Also the default run sub-directory and the ``scale`` column of every artifact.
    name: str = MISSING
    dim: int = MISSING
    # ELA sample size is ``dim * sampling_factor``.
    sampling_factor: int = MISSING
    # Optimizer evaluation budget is ``dim * fe_budget_factor``.
    fe_budget_factor: int = MISSING
    # Independent sampling repetitions per instance.
    num_feature_reps: int = MISSING
    # Independent algorithm trials per instance.
    num_trials: int = MISSING
    # Highest BBOB function id to include (1-based, inclusive).
    max_functions: int = MISSING
    # Highest BBOB instance id to include (1-based, inclusive).
    max_instances: int = MISSING


@dataclass
class PathsConfig:
    """Where artifacts are written."""

    output_root: str = "outputs"
    run_name: str = "${preset.name}"


@dataclass
class ModelSchema:
    """The regression backend and the hyper-parameters every candidate is given."""

    # Backend for the stages that fit a single one rather than comparing several.
    kind: str = REALMLP
    # XGBoost's patience; not divided by the views the augmented condition spends.
    early_stopping_rounds: int = ModelConfig.early_stopping_rounds
    # RealMLP-TD's epochs; the augmented condition divides them by its views.
    epochs: int = ModelConfig.epochs
    # Share held back, as whole problems, to choose an epoch or round on.
    validation_fraction: float = ModelConfig.validation_fraction


@dataclass
class ModelSelectionSchema:
    """Which backends the regression cells are fitted with."""

    candidates: list[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATES))


@dataclass
class TaskConfig:
    """Base class for the ``task`` config group."""

    name: str = MISSING


@dataclass
class PipelineTaskConfig(TaskConfig):
    """The settings every experiment's entry point composes its run from."""

    name: str = "pipeline"
    problem_classes: list[str] = field(default_factory=lambda: list(PROBLEM_CLASSES))
    # ``-1`` means every core. Results do not depend on it: each unit of work is
    # seeded from its own coordinates.
    n_jobs: int = -1
    # Jobs completed between flushes to disk.
    save_every: int = 25
    check_invariance: bool = True
    feature_variants: list[str] = field(default_factory=lambda: list(FEATURE_VARIANTS))
    # ``deep-ela`` needs the vendored weights.
    feature_sets: list[str] = field(default_factory=lambda: [ELA_FEATURE_SET])
    # Family the `post` view's transformation is drawn from.
    warp_family: str = DEFAULT_WARP.family
    # ``warp`` for the training maps, ``warpheld`` for the evaluation blocks.
    warp_domain: str = DEFAULT_WARP.domain
    # Half-open range of the spline family's control-point count; moving it makes
    # an unseen family from the same construction.
    warp_knots: list[int] = field(default_factory=lambda: list(DEFAULT_WARP.knots))
    deep_ela_model: str = DEFAULT_MODEL
    # Dimension permutations the Deep-ELA model is averaged over.
    deep_ela_repetitions: int = DEFAULT_REPETITIONS
    # Credit the selector with the best point its own sample evaluated.
    include_best_sampled_point: bool = True
    # Instance ids the portfolio is tuned on. Taken from the training block: a test
    # problem must not help choose the parameters its own target is measured under.
    tuning_instance_ids: list[int] = field(
        default_factory=lambda: list(TUNING_INSTANCES)
    )
    model: ModelSchema = field(default_factory=ModelSchema)
    model_selection: ModelSelectionSchema = field(default_factory=ModelSelectionSchema)


@dataclass
class RootConfig:
    """Top-level configuration composed by Hydra."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            {"preset": "debug"},
            {"task": "pipeline"},
            "_self_",
        ]
    )
    preset: PresetConfig = MISSING
    task: Any = MISSING
    paths: PathsConfig = field(default_factory=PathsConfig)
    seed: int = 42
    num_threads: int = 1


def register_configs() -> ConfigStore:
    """Register every schema with Hydra's :class:`ConfigStore` and return the store."""
    store = ConfigStore.instance()
    store.store(name="root_schema", node=RootConfig)
    store.store(group="preset", name="preset_schema", node=PresetConfig)
    store.store(group="task", name="pipeline_schema", node=PipelineTaskConfig)
    return store
