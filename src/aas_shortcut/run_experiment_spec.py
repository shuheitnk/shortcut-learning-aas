"""One experiment's specification, and the configuration its entry point composes."""

from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_module
from omegaconf import DictConfig, OmegaConf

from aas_shortcut.build_regression_model import (
    RANDOM_FOREST,
    REALMLP,
    XGBOOST,
    ModelConfig,
)
from aas_shortcut.calculate_deep_ela_features import DeepElaConfig
from aas_shortcut.define_config_schema import register_configs
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    FEATURE_SETS,
    RunPaths,
    run_root_name,
    validate_feature_set,
)
from aas_shortcut.record_manifests import write_run_manifest
from aas_shortcut.resolve_stage_artifacts import COMPUTE, PLAN, REQUIRE_CACHED
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP, WarpConfig

LOGGER = logging.getLogger(__name__)

# Importable location of the Hydra config tree.
CONFIG_MODULE = "aas_shortcut.configs"

# An id is the preanalysis, ``experiment<N>`` or ``appendix_<name>``; it names the
# experiment's directory, entry point and report root alike.
PREANALYSIS = "preanalysis"
EXPERIMENT_ID = re.compile(r"preanalysis|experiment[1-9][0-9]*|appendix_[a-z]+")

REQUIRED_KEYS = ("id", "preset", "problem_classes")
OPTIONAL_KEYS = ("overrides",)

# The override that names a run root; rewritten per preset and learner.
RUN_NAME_KEY = "paths.run_name"

# The paper's main learner and the two the appendix repeats the experiments with.
LEARNERS: tuple[str, ...] = (REALMLP, RANDOM_FOREST, XGBOOST)

# The main learner's runs keep plain root names and override nothing, since the
# configuration already fits it.
MAIN_LEARNER = REALMLP


def learner_run_name(base: str, learner: str = MAIN_LEARNER) -> str:
    """Return the run root ``base`` names for one learner's runs."""
    _check_learner(learner)
    return base if learner == MAIN_LEARNER else f"{base}_{learner}"


def learner_overrides(learner: str = MAIN_LEARNER) -> list[str]:
    """Return the overrides that fit every regression and selection with a learner.

    Both change together, so a run never mixes two learners.
    """
    _check_learner(learner)
    if learner == MAIN_LEARNER:
        return []
    return [
        f"task.model_selection.candidates=[{learner}]",
        f"task.model.kind={learner}",
    ]


def _check_learner(learner: str) -> None:
    if learner not in LEARNERS:
        raise ValueError(f"unknown learner {learner!r}; expected one of {LEARNERS}")


def experiment_label(spec_id: str) -> str:
    """Return the paper's name for an experiment: Preanalysis, or Experiment N."""
    if spec_id == PREANALYSIS:
        return "Preanalysis"
    return f"Experiment {spec_id.removeprefix('experiment')}"


def _strings(value: Any) -> tuple[str, ...]:
    """Return a YAML sequence as a tuple of strings."""
    if value is None:
        return ()
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment: its scale, its suite and the overrides its run needs."""

    id: str
    preset: str
    problem_classes: tuple[str, ...]
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a spec whose id is not an experiment's."""
        if not EXPERIMENT_ID.fullmatch(self.id):
            raise ValueError(
                f"experiment {self.id!r}: an id is {PREANALYSIS!r}, 'experiment<N>' "
                "or 'appendix_<name>'"
            )

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec:
        """Read a spec file, rejecting unknown keys.

        An ignored key would let an experiment claim a condition nothing acts on.
        """
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

        missing = [key for key in REQUIRED_KEYS if key not in raw]
        unknown = [key for key in raw if key not in (*REQUIRED_KEYS, *OPTIONAL_KEYS)]
        if missing or unknown:
            raise ValueError(
                f"{path}: missing key(s) {missing}, unknown key(s) {unknown}"
            )

        return cls(
            id=str(raw["id"]),
            preset=str(raw["preset"]),
            problem_classes=_strings(raw["problem_classes"]),
            overrides=_strings(raw.get("overrides")),
        )

    def run_name(self, base: str, preset: str | None = None) -> str:
        """Return the run root ``base`` resolves to at a scale.

        The spec's own scale keeps ``base``; any other is suffixed with its preset.
        """
        return run_root_name(base, preset or self.preset, self.preset)

    def scoped_overrides(
        self, preset: str | None = None, learner: str = MAIN_LEARNER
    ) -> list[str]:
        """Return this spec's overrides scoped to the preset and the learner."""
        scoped = []
        for override in self.overrides:
            key, _, value = override.partition("=")
            if key == RUN_NAME_KEY:
                named = learner_run_name(value, learner)
                override = f"{RUN_NAME_KEY}={self.run_name(named, preset)}"
            scoped.append(override)
        return [*scoped, *learner_overrides(learner)]

    def hydra_overrides(
        self,
        preset: str | None = None,
        extra: Sequence[str] = (),
        *,
        learner: str = MAIN_LEARNER,
    ) -> list[str]:
        """Build the override list this experiment's run is composed from.

        ``extra`` comes last, so an explicit ``paths.run_name`` there wins.
        """
        return [
            f"preset={preset or self.preset}",
            "task=pipeline",
            f"task.problem_classes=[{','.join(self.problem_classes)}]",
            *self.scoped_overrides(preset, learner),
            *extra,
        ]


def compose_config(overrides: Sequence[str]) -> DictConfig:
    """Compose the project's configuration from the overrides."""
    register_configs()
    with initialize_config_module(config_module=CONFIG_MODULE, version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def resolve_run_paths(cfg: DictConfig) -> RunPaths:
    """Return the output layout for this run."""
    return RunPaths.from_root(Path(cfg.paths.output_root) / cfg.paths.run_name)


def build_model_config(cfg: DictConfig, num_threads: int) -> ModelConfig:
    """Build the regression backend's configuration from the ``task.model`` node."""
    node = cfg.get("model")
    settings = OmegaConf.to_container(node, resolve=True) if node is not None else {}
    if not isinstance(settings, dict):
        raise ValueError(f"task.model must be a mapping, got {type(node).__name__}")
    return ModelConfig(thread_count=num_threads, **settings)


def build_context(cfg: DictConfig, run_paths: RunPaths) -> ExperimentContext:
    """Assemble the experiment context from the resolved Hydra config."""
    task = cfg.task
    return ExperimentContext(
        run_paths=run_paths,
        scale=cfg.preset.name,
        dim=cfg.preset.dim,
        sampling_factor=cfg.preset.sampling_factor,
        fe_budget_factor=cfg.preset.fe_budget_factor,
        num_feature_reps=cfg.preset.num_feature_reps,
        num_trials=cfg.preset.num_trials,
        max_functions=cfg.preset.max_functions,
        max_instances=cfg.preset.max_instances,
        seed=cfg.seed,
        problem_classes=tuple(task.problem_classes),
        n_jobs=task.get("n_jobs", -1),
        save_every=task.get("save_every", 25),
        check_invariance=task.get("check_invariance", True),
        feature_variants=tuple(
            task.get("feature_variants", ("value", "rank", "concat"))
        ),
        feature_sets=tuple(task.get("feature_sets", (ELA_FEATURE_SET,))),
        deep_ela=DeepElaConfig(
            model=task.get("deep_ela_model", DeepElaConfig.model),
            repetitions=task.get("deep_ela_repetitions", DeepElaConfig.repetitions),
            num_threads=cfg.num_threads,
        ),
        warp=WarpConfig(
            family=task.get("warp_family", DEFAULT_WARP.family),
            alpha=DEFAULT_WARP.alpha,
            domain=task.get("warp_domain", DEFAULT_WARP.domain),
            knots=cast(
                tuple[int, int],
                tuple(
                    int(value) for value in task.get("warp_knots", DEFAULT_WARP.knots)
                ),
            ),
            draw=DEFAULT_WARP.draw,
        ),
        include_best_sampled_point=task.get("include_best_sampled_point", True),
        tuning_instance_ids=tuple(
            task.get("tuning_instance_ids", ExperimentContext.tuning_instance_ids)
        ),
        model=build_model_config(task, cfg.num_threads),
    )


def record_run(
    spec: ExperimentSpec,
    cfg: DictConfig,
    run_paths: RunPaths,
    *,
    inputs: Sequence[Path] = (),
    status: str = "completed",
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write the run manifest of one invocation of an experiment's entry point."""
    config = {
        "experiment": spec.id,
        "hydra": OmegaConf.to_container(cfg, resolve=True),
        **(dict(extra) if extra else {}),
    }
    path = write_run_manifest(
        run_paths.root,
        spec.id,
        config,
        inputs=inputs,
        status=status,
    )
    LOGGER.info("run manifest: %s", path)
    return path


def build_parser(description: str, epilog: str) -> argparse.ArgumentParser:
    """Return the parser every experiment entry point starts from."""
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "--preset",
        default=None,
        help="scale to run at; full by default",
    )
    return parser


def add_store_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the two ways of running without computing a shared stage.

    ``--require-cached`` keeps a missing artifact from starting hours of computation
    unnoticed.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plan",
        action="store_true",
        help="read every shared stage that exists, and stop at the first to compute",
    )
    group.add_argument(
        "--require-cached",
        action="store_true",
        help="fail instead of computing any shared stage",
    )
    return parser


def add_learner_option(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--learner``, which fits every regression and selection with one learner."""
    parser.add_argument(
        "--learner",
        choices=LEARNERS,
        default=MAIN_LEARNER,
        help=(
            "learner of every regression and selection; a learner other than "
            f"{MAIN_LEARNER} reports to experiment<N>_<learner>"
        ),
    )
    return parser


def store_policy(args: argparse.Namespace) -> str:
    """Return the store policy :func:`add_store_options`' flags ask for."""
    if getattr(args, "plan", False):
        return PLAN
    if getattr(args, "require_cached", False):
        return REQUIRE_CACHED
    return COMPUTE


def chosen_feature_sets(task_feature_sets: Sequence[str]) -> list[str]:
    """Return the representations a run measures, in :data:`FEATURE_SETS` order."""
    for name in task_feature_sets:
        validate_feature_set(name)
    chosen = [name for name in FEATURE_SETS if name in task_feature_sets]
    if not chosen:
        raise ValueError("task.feature_sets names no feature set to measure")
    return chosen


def add_overrides(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the trailing Hydra overrides, which must come after every option."""
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. task.n_jobs=8 'task.feature_sets=[ela]'",
    )
    return parser
