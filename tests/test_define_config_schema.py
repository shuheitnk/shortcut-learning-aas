"""Tests for Hydra config composition and the context an entry point builds from it.

They compose the real config tree, so a YAML that stops composing fails here rather
than hours into a sweep.
"""

from __future__ import annotations

from typing import cast

import pytest
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from aas_shortcut.build_regression_model import DEFAULT_CANDIDATES, MODEL_KINDS, REALMLP
from aas_shortcut.calculate_features import MIN_ELA_SAMPLE_SIZE
from aas_shortcut.define_algorithm_portfolio import minimum_fe_budget
from aas_shortcut.define_config_schema import register_configs
from aas_shortcut.define_paths import FEATURE_VARIANTS, PROBLEM_CLASSES, SCALES
from aas_shortcut.run_experiment_spec import build_context, resolve_run_paths
from tests.conftest import compose_config

# The only scales a column list is declared for: the paper's and the CI one.
ALL_PRESETS = ("debug", "full")


class TestComposition:
    def test_defaults_compose(self):
        cfg = compose_config()
        assert cfg.preset.name == "debug"
        assert cfg.task.name == "pipeline"

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_every_preset_composes(self, preset):
        cfg = compose_config(f"preset={preset}")
        assert cfg.preset.name == preset

    def test_the_task_composes(self):
        cfg = compose_config("task=pipeline")
        assert cfg.task.name == "pipeline"

    def test_a_typo_in_an_override_fails(self):
        with pytest.raises(
            (ConfigCompositionException, ConfigAttributeError, Exception)
        ):
            compose_config("preset.dimm=5")

    def test_run_root_follows_the_preset(self):
        cfg = compose_config("preset=full")
        assert resolve_run_paths(cfg).root.name == "full"

    def test_output_root_is_overridable(self):
        cfg = compose_config("paths.output_root=elsewhere")
        assert "elsewhere" in str(resolve_run_paths(cfg).root)


class TestPresetsMatchTheDesign:
    """The design fixes these numbers; drift must be visible."""

    @pytest.mark.parametrize(
        ("preset", "expected"),
        [
            (
                "full",
                {
                    "dim": 10,
                    "sampling_factor": 50,
                    "fe_budget_factor": 1000,
                    "num_feature_reps": 1,
                    "num_trials": 31,
                    "max_functions": 24,
                    "max_instances": 10,
                },
            ),
        ],
    )
    def test_scale_parameters(self, preset, expected):
        cfg = compose_config(f"preset={preset}")
        for key, value in expected.items():
            assert getattr(cfg.preset, key) == value, key

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_sample_is_large_enough_for_the_calculators(self, preset):
        cfg = compose_config(f"preset={preset}")
        assert cfg.preset.dim * cfg.preset.sampling_factor >= MIN_ELA_SAMPLE_SIZE

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_budget_is_large_enough_for_the_portfolio(self, preset):
        cfg = compose_config(f"preset={preset}")
        budget = cfg.preset.fe_budget_factor * cfg.preset.dim
        assert budget >= minimum_fe_budget(cfg.preset.dim)

    @pytest.mark.parametrize("preset", ALL_PRESETS)
    def test_preset_name_is_a_known_scale(self, preset):
        assert compose_config(f"preset={preset}").preset.name in SCALES

    def test_the_papers_scale_uses_the_design_budget_rule(self):
        """Search budget = 1000 * dim."""
        assert compose_config("preset=full").preset.fe_budget_factor == 1000

    def test_a_removed_scale_does_not_compose(self):
        """A scale without a column list does not compose."""
        for preset in ("smoke", "pilot"):
            with pytest.raises(MissingConfigException, match=f"preset/{preset}"):
                compose_config(f"preset={preset}")


class TestPipelineTask:
    def test_defaults_cover_the_whole_grid(self):
        cfg = compose_config()
        assert list(cfg.task.problem_classes) == list(PROBLEM_CLASSES)
        assert list(cfg.task.feature_variants) == list(FEATURE_VARIANTS)

    def test_n_jobs_defaults_to_every_available_core(self, run_paths):
        """-1 is joblib's convention; results do not depend on it."""
        cfg = compose_config()
        assert cfg.task.n_jobs == -1
        context = build_context(cfg, run_paths)
        assert context.n_jobs == -1

    def test_context_carries_the_preset_through(self, run_paths):
        cfg = compose_config("preset=full", "task.n_jobs=4", "seed=7")
        context = build_context(cfg, run_paths)
        assert context.dim == 10
        assert context.num_feature_reps == 1
        assert context.num_trials == 31
        assert context.n_jobs == 4
        assert context.seed == 7
        assert context.scale == "full"


class TestRootConfig:
    def test_seed_defaults_to_the_documented_value(self):
        assert compose_config().seed == 42

    def test_threads_default_to_one(self):
        """Anything else forfeits bit-identical results."""
        assert compose_config().num_threads == 1

    def test_the_config_is_fully_resolvable(self):
        """An unresolved interpolation would surface as a crash mid-run."""
        cfg = compose_config("preset=full")
        resolved = cast(dict, OmegaConf.to_container(cfg, resolve=True))
        assert resolved["paths"]["run_name"] == "full"

    def test_register_configs_is_idempotent(self):
        first = register_configs()
        second = register_configs()
        assert first is second


class TestModelWiring:
    """``task.model`` and ``task.model_selection`` reach the experiments."""

    def test_the_main_learner_is_fitted_by_default(self):
        assert tuple(compose_config().task.model_selection.candidates) == (
            DEFAULT_CANDIDATES
        )
        assert DEFAULT_CANDIDATES == (REALMLP,)

    @pytest.mark.parametrize("kind", MODEL_KINDS)
    def test_every_backend_composes_as_a_candidate(self, kind):
        cfg = compose_config(f"task.model_selection.candidates=[{kind}]")
        assert tuple(cfg.task.model_selection.candidates) == (kind,)

    def test_the_shared_hyper_parameters_reach_the_context(self):
        cfg = compose_config("task.model.epochs=9")
        assert build_context(cfg, resolve_run_paths(cfg)).model.epochs == 9

    def test_xgboosts_stopping_reaches_the_context(self):
        cfg = compose_config("task.model.early_stopping_rounds=50")
        assert build_context(
            cfg, resolve_run_paths(cfg)
        ).model.early_stopping_rounds == (50)

    @pytest.mark.parametrize("item", ["subsample", "colsample_bytree", "reg_lambda"])
    def test_xgboosts_old_items_are_gone(self, item):
        """XGB-TD fixes them; an override would set nothing."""
        with pytest.raises(ConfigCompositionException):
            compose_config(f"task.model.{item}=0.5")

    def test_the_backend_is_named_in_configuration(self):
        """Experiments 3 and 5 fit one model and read it straight from here."""
        cfg = compose_config("task.model.kind=xgboost")
        assert build_context(cfg, resolve_run_paths(cfg)).model.kind == "xgboost"

    def test_the_default_backend_is_the_one_the_study_measured_best(self):
        """The default backend is RealMLP-TD."""
        cfg = compose_config()
        assert build_context(cfg, resolve_run_paths(cfg)).model.kind == "realmlp"

    def test_an_unknown_backend_is_rejected(self):
        cfg = compose_config("task.model.kind=lightgbm")
        with pytest.raises(ValueError, match="unknown model kind"):
            build_context(cfg, resolve_run_paths(cfg))
