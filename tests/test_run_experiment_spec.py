"""Tests for the experiment entry points (``scripts/experimentN/``).

They check each numbered step of a spec, the configuration it composes, and the
shared units it resolves. Entry points are discovered from the directory, so a
directory outside the paper's experiments, or a mismatched spec ``id``, is caught.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from aas_shortcut.define_paths import (
    FEATURE_SETS,
    PROBLEM_CLASSES,
    SCALES,
)
from aas_shortcut.resolve_feature_units import feature_unit
from aas_shortcut.resolve_stage_artifacts import COMPUTE, PLAN, REQUIRE_CACHED
from aas_shortcut.resolve_tuning_stages import grid_unit, pair_tuning_unit, role_unit
from aas_shortcut.run_experiment_spec import (
    LEARNERS,
    MAIN_LEARNER,
    PREANALYSIS,
    ExperimentSpec,
    add_learner_option,
    add_store_options,
    build_context,
    chosen_feature_sets,
    compose_config,
    learner_overrides,
    learner_run_name,
    resolve_run_paths,
    store_policy,
)

# The experiment entry points: the preanalysis, the appendix's, then Experiments
# 1 to 5.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
EXPERIMENTS = tuple(
    sorted(
        (path.parent.name for path in SCRIPTS.glob("*/config.yaml")),
        key=lambda name: (name != PREANALYSIS, name),
    )
)
SPEC_PATHS = tuple(SCRIPTS / name / "config.yaml" for name in EXPERIMENTS)

# The appendix's entry points beside the six.
APPENDICES = (
    "appendix_blocks",
    "appendix_bootstrap",
    "appendix_checks",
    "appendix_doubled",
    "appendix_lofo",
)

TEST_PRESET = "debug"


@pytest.fixture(params=SPEC_PATHS, ids=lambda path: path.parent.name)
def spec(request) -> ExperimentSpec:
    """Each experiment's spec in turn."""
    return ExperimentSpec.load(request.param)


class TestSpecFiles:
    def test_every_experiment_has_an_entry_point(self):
        for path in SPEC_PATHS:
            assert path.exists()
            assert path.with_name(f"run_{path.parent.name}.py").exists()

    def test_the_entry_points_are_the_paper_s_six_experiments(self):
        """The six and the appendix, each reporting to a root of its own name."""
        assert list(EXPERIMENTS) == [
            PREANALYSIS,
            *APPENDICES,
            *(f"experiment{n}" for n in range(1, 6)),
        ]
        for path in SPEC_PATHS:
            assert _spec_run_name(ExperimentSpec.load(path)) == path.parent.name

    def test_every_spec_id_matches_its_directory(self):
        assert [ExperimentSpec.load(path).id for path in SPEC_PATHS] == list(
            EXPERIMENTS
        )

    def test_every_spec_names_a_known_preset_and_suite(self, spec):
        assert spec.preset in SCALES
        assert set(spec.problem_classes) <= set(PROBLEM_CLASSES)


class TestValidation:
    def test_an_unknown_key_in_a_spec_file_is_rejected(self, tmp_path):
        """A silently ignored key would let a spec claim a condition nothing acts on."""
        path = tmp_path / "config.yaml"
        path.write_text("id: 9\nfe_budget: 12345\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown key"):
            ExperimentSpec.load(path)


class TestComposition:
    def test_the_overrides_compose_this_experiment_s_run(self, spec):
        cfg = compose_config(spec.hydra_overrides(TEST_PRESET))
        assert cfg.preset.name == TEST_PRESET
        assert list(cfg.task.problem_classes) == list(spec.problem_classes)


class TestPresetScopedRunRoots:
    """A rehearsal of an experiment never writes into the root it reports from."""

    def test_the_reported_scale_keeps_the_spec_run_root(self, spec):
        base = _spec_run_name(spec)
        if base is None:
            pytest.skip("the spec names no run root of its own")
        cfg = compose_config(spec.hydra_overrides(spec.preset))
        assert cfg.paths.run_name == base

    def test_a_rehearsal_is_suffixed_with_its_preset(self, spec):
        base = _spec_run_name(spec)
        if base is None:
            pytest.skip("the spec names no run root of its own")
        cfg = compose_config(spec.hydra_overrides(TEST_PRESET))
        assert cfg.paths.run_name == f"{base}_{TEST_PRESET}"

    def test_an_explicit_run_root_still_wins(self, spec):
        cfg = compose_config(
            spec.hydra_overrides(TEST_PRESET, ["paths.run_name=somewhere_else"])
        )
        assert cfg.paths.run_name == "somewhere_else"


# The ``config_hash`` of the shared units without inputs, at the reported scale. A
# change that moves one changes what is measured and needs a rerun and a new value.
REPORTED_UNITS = {
    # The preanalysis and Experiment 1 are absent: their units are the ones below, read
    # under other settings, or have inputs.
    "experiment2": {
        "tune_pair": "ccdef07d6e569b5b",
        "features:bbob": "0ba6e9e9fc4ccb64",
    },
    "experiment3": {
        "grid_errors": "f5d081b19e4b7a71",
        "tuning_roles": "a7f885a0d0827723",
        "features:bbob": "0ba6e9e9fc4ccb64",
    },
}


class TestReportedConfigurations:
    """The specs compose, at the reported scale, to what the artifacts record."""

    @pytest.mark.parametrize("name", sorted(REPORTED_UNITS))
    def test_the_shared_units_fingerprint_as_reported(self, name, tmp_path):
        spec = ExperimentSpec.load(SCRIPTS / name / "config.yaml")
        cfg = compose_config(
            [*spec.hydra_overrides(spec.preset), f"paths.output_root={tmp_path}"]
        )
        context = build_context(cfg, resolve_run_paths(cfg))
        units = {
            "tune_pair": pair_tuning_unit(context),
            "grid_errors": grid_unit(context),
            "tuning_roles": role_unit(context),
            "features:bbob": feature_unit(context.feature_config("bbob")),
        }
        expected = REPORTED_UNITS[name]
        assert {name: units[name].fingerprint for name in expected} == expected

    def test_experiment_1_narrows_experiment_4_and_changes_nothing_else(self):
        """Experiment 1's cell is Experiment 4's, narrowed on the declared axes only.

        Every other setting (blocks, backend, criterion, hyper-parameters, number of
        training runs) must match, or the comparison is not under one condition.
        """
        from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
        from aas_shortcut.define_warp_audits import AUDITS, CONTROL
        from aas_shortcut.measure_augmented_regression import (
            AugmentedRegressionConfig,
        )
        from aas_shortcut.measure_selection_transfer import TRAINING_PLAIN

        # The predictor set only the feature tables can give, so both sides use the
        # same placeholder: under test is everything the configurations do not narrow.
        columns = ("a", "b")
        narrowed = {"variants", "pairs", "trainings", "audits", "features"}

        def config_of(number: int, **narrowing) -> AugmentedRegressionConfig:
            spec = ExperimentSpec.load(SCRIPTS / f"experiment{number}" / "config.yaml")
            cfg = compose_config(spec.hydra_overrides(spec.preset))
            context = build_context(cfg, resolve_run_paths(cfg))
            return AugmentedRegressionConfig(
                feature_set="ela",
                features=columns,
                candidates=tuple(cfg.task.model_selection.candidates),
                model=context.model,
                **narrowing,
            )

        narrowing = config_of(
            1,
            variants=("value",),
            pairs=(next(p for p in FOCUS_PAIRS if p.pair_type == "G1_G1"),),
            trainings=(TRAINING_PLAIN,),
            audits=(CONTROL.label,),
        )
        full = config_of(4, audits=tuple(audit.label for audit in AUDITS))

        shared = {k: v for k, v in narrowing.as_dict().items() if k not in narrowed}
        assert shared == {k: v for k, v in full.as_dict().items() if k not in narrowed}

        # Each narrowed axis restricts Experiment 4's: a pair or family it does not
        # measure would compare two experiments rather than two predictor sets.
        assert set(narrowing.variants) <= set(full.variants)
        assert set(narrowing.pairs) <= set(full.pairs)
        assert set(narrowing.trainings) <= set(full.trainings)
        assert set(narrowing.audits) <= set(full.audits)
        assert narrowing.audits == (CONTROL.label,)

    def test_experiment_1_fingerprints_the_columns_it_narrowed_to(self):
        """Experiment 1's fingerprint carries the invariant columns it narrowed to.

        `classify_feature_invariance` can change the columns while every field of the
        configuration stays the same; `AugmentedRegressionConfig.features` puts them
        in the fingerprint.
        """
        from aas_shortcut.classify_feature_invariance import is_structurally_invariant
        from aas_shortcut.measure_augmented_regression import (
            AugmentedRegressionConfig,
        )
        from aas_shortcut.record_manifests import stage_fingerprint

        # The column that moved, and two others that did not, so the difference
        # under test is the membership rather than the size of the set.
        assert is_structurally_invariant("ic.m0"), "the scenario is not the real one"
        columns = ("ela_meta.lin_simple.adj_r2", "ic.m0", "nbc.nn_nb.sd_ratio")

        def fingerprint(features: tuple[str, ...]) -> str:
            return stage_fingerprint(
                AugmentedRegressionConfig(
                    audits=("spline/warpheld",), features=features
                ).as_dict()
            )

        without = tuple(name for name in columns if name != "ic.m0")
        assert fingerprint(columns) != fingerprint(without)
        # And it is the membership, not the order they arrive in.
        assert fingerprint(columns) == fingerprint(tuple(reversed(columns)))


class TestStoreOptions:
    """The two flags that let a paper experiment run without computing a stage."""

    @staticmethod
    def _parse(argv):
        return add_store_options(argparse.ArgumentParser()).parse_args(argv)

    def test_by_default_a_missing_stage_is_computed(self):
        assert store_policy(self._parse([])) == COMPUTE

    def test_each_flag_selects_its_policy(self):
        assert store_policy(self._parse(["--plan"])) == PLAN
        assert store_policy(self._parse(["--require-cached"])) == REQUIRE_CACHED

    def test_the_two_flags_exclude_each_other(self):
        with pytest.raises(SystemExit):
            self._parse(["--plan", "--require-cached"])


# The entry points that fit a regression or a selection, and so take `--learner`.
# The preanalysis fits neither.
LEARNER_ENTRY_POINTS = (
    "experiment1/run_experiment1.py",
    "experiment1/report_predictor_comparison.py",
    "experiment2/run_experiment2.py",
    "experiment3/run_experiment3.py",
    "experiment5/run_experiment5.py",
    "experiment4/run_experiment4.py",
)


def _spec_run_name(spec: ExperimentSpec) -> str | None:
    """The run root a spec names, before any scoping."""
    for override in spec.overrides:
        key, _, value = override.partition("=")
        if key == "paths.run_name":
            return value
    return None


class TestLearnerSwitch:
    """``--learner`` refits every regression and selection, into its own roots."""

    @staticmethod
    def _parse(argv):
        return add_learner_option(argparse.ArgumentParser()).parse_args(argv)

    def test_the_paper_s_learner_is_the_default(self):
        assert self._parse([]).learner == MAIN_LEARNER == "realmlp"
        assert LEARNERS == ("realmlp", "random-forest", "xgboost")

    def test_an_unknown_learner_is_refused(self):
        with pytest.raises(SystemExit):
            self._parse(["--learner", "lightgbm"])
        with pytest.raises(ValueError, match="unknown learner"):
            learner_overrides("lightgbm")
        with pytest.raises(ValueError, match="unknown learner"):
            learner_run_name("experiment4", "lightgbm")

    def test_the_paper_s_learner_changes_no_override(self, spec):
        """The main learner adds no override, so its units and reports do not move."""
        assert learner_overrides(MAIN_LEARNER) == []
        assert spec.hydra_overrides(
            TEST_PRESET, learner=MAIN_LEARNER
        ) == spec.hydra_overrides(TEST_PRESET)

    @pytest.mark.parametrize("learner", LEARNERS[1:])
    def test_another_learner_fits_the_regression_and_the_selection(self, spec, learner):
        cfg = compose_config(spec.hydra_overrides(TEST_PRESET, learner=learner))
        assert tuple(cfg.task.model_selection.candidates) == (learner,)
        context = build_context(cfg, resolve_run_paths(cfg))
        assert context.model.kind == learner
        assert context.selection_config().model.kind == learner

    @pytest.mark.parametrize("learner", LEARNERS[1:])
    def test_another_learner_reports_to_its_own_root(self, spec, learner):
        base = _spec_run_name(spec)
        if base is None:
            pytest.skip("the spec names no run root of its own")
        reported = compose_config(spec.hydra_overrides(spec.preset, learner=learner))
        assert reported.paths.run_name == f"{base}_{learner}"
        rehearsal = compose_config(spec.hydra_overrides(TEST_PRESET, learner=learner))
        assert rehearsal.paths.run_name == f"{base}_{learner}_{TEST_PRESET}"

    def test_an_explicit_run_root_still_wins(self, spec):
        cfg = compose_config(
            spec.hydra_overrides(
                TEST_PRESET, ["paths.run_name=somewhere_else"], learner=LEARNERS[1]
            )
        )
        assert cfg.paths.run_name == "somewhere_else"

    @pytest.mark.parametrize("relative", LEARNER_ENTRY_POINTS)
    def test_every_entry_point_that_fits_takes_the_switch(
        self, relative, capsys, monkeypatch
    ):
        path = SCRIPTS / relative
        name = f"entry_point_{path.parent.name}_{path.stem}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        # A dataclass looks its module up by name while the module is executed,
        # and an entry point may import its sibling, as run from its directory.
        monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.syspath_prepend(str(path.parent))
        module_spec.loader.exec_module(module)
        with pytest.raises(SystemExit) as exited:
            module.main(["--help"])
        assert exited.value.code == 0
        assert "--learner" in capsys.readouterr().out


class TestChosenFeatureSets:
    """The representations Experiments 2 and 4 measure, each its own regression."""

    def test_the_task_s_feature_sets_are_measured(self):
        assert chosen_feature_sets(["ela"]) == ["ela"]
        assert chosen_feature_sets(("deep-ela",)) == ["deep-ela"]

    def test_they_come_in_feature_set_order_whatever_the_task_s_order(self):
        assert chosen_feature_sets(["deep-ela", "ela"]) == list(FEATURE_SETS)

    @pytest.mark.parametrize("preset", SCALES)
    @pytest.mark.parametrize("name", ["experiment2", "experiment4"])
    def test_by_default_every_representation_is_measured(self, name, preset):
        spec = ExperimentSpec.load(SCRIPTS / name / "config.yaml")
        cfg = compose_config(spec.hydra_overrides(preset))
        assert chosen_feature_sets(cfg.task.feature_sets) == list(FEATURE_SETS)

    def test_a_task_naming_an_unknown_feature_set_is_refused(self):
        with pytest.raises(ValueError, match="Unknown feature set"):
            chosen_feature_sets(["ela", "elastic"])

    def test_a_task_naming_no_feature_set_is_refused(self):
        with pytest.raises(ValueError, match="no feature set"):
            chosen_feature_sets([])
