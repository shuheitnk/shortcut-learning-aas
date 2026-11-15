"""The settings an experiment's stages are configured from."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.calculate_deep_ela_features import DeepElaConfig
from aas_shortcut.calculate_features import FeatureStageConfig
from aas_shortcut.define_paths import ELA_FEATURE_SET, RunPaths
from aas_shortcut.define_repetitions import REPETITIONS
from aas_shortcut.sample_monotone_transform import DEFAULT_WARP, WarpConfig
from aas_shortcut.select_algorithms import SelectionConfig
from aas_shortcut.split_bbob_instances import TUNING_INSTANCES, check_tuning_instances
from aas_shortcut.tune_algorithms import PairConditions


@dataclass(frozen=True)
class ExperimentContext:
    """Everything an experiment configures its stages from, resolved from Hydra."""

    run_paths: RunPaths
    scale: str
    dim: int
    sampling_factor: int
    fe_budget_factor: int
    num_feature_reps: int
    num_trials: int
    max_functions: int
    max_instances: int
    seed: int
    problem_classes: tuple[str, ...]
    n_jobs: int = 1
    save_every: int = 25
    check_invariance: bool = True
    feature_variants: tuple[str, ...] = ("value", "rank", "concat")
    feature_sets: tuple[str, ...] = (ELA_FEATURE_SET,)
    deep_ela: DeepElaConfig = field(default_factory=DeepElaConfig)
    include_best_sampled_point: bool = True
    tuning_instance_ids: tuple[int, ...] = TUNING_INSTANCES
    model: ModelConfig = field(default_factory=ModelConfig)
    # One setting per run, so features and performance see the same ``post`` map.
    warp: WarpConfig = DEFAULT_WARP

    def feature_config(self, problem_class: str) -> FeatureStageConfig:
        """Build the feature stage's configuration for one suite."""
        return FeatureStageConfig(
            problem_class=problem_class,
            dim=self.dim,
            sampling_factor=self.sampling_factor,
            num_reps=self.num_feature_reps,
            max_functions=self.max_functions,
            max_instances=self.max_instances,
            base_seed=self.seed,
            scale=self.scale,
            check_invariance=self.check_invariance,
            n_jobs=self.n_jobs,
            save_every=self.save_every,
            feature_sets=tuple(self.feature_sets),
            deep_ela=self.deep_ela,
            warp=self.warp,
        )

    def deployment_feature_config(
        self, problem_class: str, instances: tuple[int, ...]
    ) -> FeatureStageConfig:
        """Build the held-out block's one-sample-per-deployment configuration.

        Only the number of samples and the instances differ from the feature
        stage, so a deployment sees the features the study measures.
        """
        return replace(
            self.feature_config(problem_class),
            num_reps=REPETITIONS,
            max_instances=max(instances),
        )

    @property
    def search_budget(self) -> int:
        """Evaluations a selector's chosen algorithm gets, after its sample."""
        return self.fe_budget_factor * self.dim

    @property
    def total_budget(self) -> int:
        """Evaluations the single best solver gets — sample included."""
        return (self.fe_budget_factor + self.sampling_factor) * self.dim

    def selection_config(self) -> SelectionConfig:
        """Build the selection experiment's configuration."""
        return SelectionConfig(
            feature_variants=tuple(self.feature_variants),
            feature_sets=tuple(self.feature_sets),
            include_best_sampled_point=self.include_best_sampled_point,
            model=self.model,
            n_jobs=self.n_jobs,
            n_seeds=REPETITIONS,
        )

    def tuning_conditions(self, budget_factor: int | None = None) -> PairConditions:
        """Build the tuning's benchmark conditions for one budget."""
        return PairConditions(
            dim=self.dim,
            budget_factor=budget_factor or self.fe_budget_factor,
            function_ids=tuple(range(1, self.max_functions + 1)),
            instance_ids=check_tuning_instances(self.tuning_instance_ids),
            n_trials=self.num_trials,
            base_seed=self.seed,
        )
