"""The eight required tests of the experimental design.

Each guards a condition under which the results would be uninterpretable. Where a
condition can be read off stored units, it is also checked on a debug run of the
preanalysis from an empty output root (``debug_root``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from aas_shortcut.benchmark_algorithms import run_job
from aas_shortcut.benchmark_objective_views import ViewBenchmarkConfig
from aas_shortcut.bind_portfolio_parameters import (
    CONDITION_SELECTION,
)
from aas_shortcut.build_regression_model import ModelConfig, fit_model
from aas_shortcut.calculate_features import (
    FeatureStageConfig,
    OrderPreservationError,
    RankInvarianceError,
    features_for_sample,
)
from aas_shortcut.create_problem import create_problem
from aas_shortcut.create_sampled_points import (
    create_sampled_points,
    ela_inputs,
    sample_seed,
)
from aas_shortcut.define_algorithm_portfolio import (
    ALGORITHMS,
    PORTFOLIO,
    algorithm_seed,
)
from aas_shortcut.define_paths import RUN_MANIFEST_NAME, STAGE_STORE_NAME, TRANSFORMS
from aas_shortcut.define_table_schema import (
    FEATURES_SCHEMA,
    VIEW_PERFORMANCE_SCHEMA,
    conform,
    validate_table,
)
from aas_shortcut.optimizers.define_optimizer_base import (
    BudgetedObjective,
    BudgetExhausted,
)
from aas_shortcut.predict_superiority import TARGET, r2_score, score_on
from aas_shortcut.read_write_tables import read_table
from aas_shortcut.resolve_stage_artifacts import REUSED, STAGE_LIST_NAME
from aas_shortcut.sample_monotone_transform import transform_for_instance
from tests.check_stage_artifacts import (
    STAGE_SCHEMAS,
    check_budget,
    check_feature_completeness,
    check_rank_invariance,
    check_unit_files,
    check_view_completeness,
    feature_stage_config,
    stored_units,
    view_benchmark_config,
)
from tests.conftest import (
    PERFORMANCE_SCHEMA,
    SUPERIORITY_SCHEMA,
    make_features,
    make_performance,
    published_binding,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS / "preanalysis"))

# A sample that is small but still above MIN_ELA_SAMPLE_SIZE, on the cheapest dimension,
# so the invariance tests can afford to run the real calculators.
SMALL_SAMPLE = 100
SMALL_DIM = 2

DEFAULT_BINDING = published_binding(CONDITION_SELECTION)


# The run the stored-artifact checks read: debug scale, ELA only (Deep-ELA needs
# vendored weights). The preanalysis is the cheapest experiment from an empty root.
DEBUG_RUN = ("--preset", "debug", "task.feature_sets=[ela]")

PREANALYSIS_ROOT = "preanalysis_debug"


@pytest.fixture(scope="module")
def debug_root(tmp_path_factory) -> Path:
    """An output root a debug run of the experiments filled from empty."""
    import run_preanalysis

    root = tmp_path_factory.mktemp("from-empty")
    assert run_preanalysis.main([*DEBUG_RUN, f"paths.output_root={root}"]) == 0
    return root


def stored_tables(root: Path, stage: str) -> list[tuple[dict, pd.DataFrame]]:
    """Every table of one stage's units in a run's store, beside the unit's record."""
    tables = [
        (record, read_table(path))
        for unit_stage, directory, record in stored_units(root / STAGE_STORE_NAME)
        if unit_stage == stage
        for path in sorted(directory.rglob("*.parquet"))
    ]
    assert tables, f"the run stored no {stage} unit"
    return tables


def report_files(root: Path) -> dict[str, bytes]:
    """Every file of a report root, by its path in it."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def fit_and_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    model_config: ModelConfig,
) -> tuple[float, float, int, int]:
    """Fit on ``train``; return ``(r2_train, r2_test, n_train, n_test)``."""
    model = fit_model(train, columns, model_config, TARGET)
    return (
        score_on(model, train, columns),
        score_on(model, test, columns),
        len(train),
        len(test),
    )


@pytest.fixture(scope="module")
def real_sample():
    """One real sample of a 2D MA-BBOB instance."""
    return create_sampled_points(
        base_seed=42,
        problem_class="bbob",
        function_id=1,
        instance_id=1,
        dim=SMALL_DIM,
        rep=1,
        sample_size=SMALL_SAMPLE,
    )


# ---------------------------------------------------------------------------
# Test 1 - rank invariance
# ---------------------------------------------------------------------------


class TestRankInvariance:
    """Test 1: rank features must agree between pre and post."""

    def test_real_features_are_invariant(self, real_sample):
        rows = pd.DataFrame(features_for_sample(real_sample))
        ranks = rows[rows["feature_variant"] == "rank"]
        wide = ranks.pivot_table(index="feature", columns="transform", values="value")

        assert not wide.empty
        assert np.allclose(
            wide["pre"].to_numpy(), wide["post"].to_numpy(), rtol=1e-7, atol=1e-9
        )

    def test_value_features_do_move(self, real_sample):
        """The contrast the study rests on: value features are *not* invariant."""
        rows = pd.DataFrame(features_for_sample(real_sample))
        values = rows[rows["feature_variant"] == "value"]
        wide = values.pivot_table(index="feature", columns="transform", values="value")

        assert not np.allclose(wide["pre"].to_numpy(), wide["post"].to_numpy())

    def test_violation_stops_the_pipeline(self, monkeypatch, real_sample):
        """A rank feature that moves must raise, not be recorded."""
        import aas_shortcut.calculate_features as module

        real = module.compute_ela_features
        calls = {"n": 0}

        def drifting(X, y, seed, rejections=None):
            calls["n"] += 1
            features = real(X, y, seed, rejections)
            # The fourth condition is (post, rank); perturb it.
            if calls["n"] == 4:
                key = next(iter(features))
                features[key] = features[key] + 1.0
            return features

        monkeypatch.setattr(module, "compute_ela_features", drifting)
        with pytest.raises(RankInvarianceError):
            module.features_for_sample(real_sample)

    def test_a_moved_rank_feature_in_a_stored_table_is_caught(self):
        assert check_rank_invariance(make_features(rank_invariant=True)) == []
        violations = check_rank_invariance(make_features(rank_invariant=False))
        assert violations and violations[0].kind == "rank_invariance"

    @pytest.mark.slow
    def test_the_stored_features_are_invariant(self, debug_root):
        for _, table in stored_tables(debug_root, "features"):
            assert check_rank_invariance(table) == []


# ---------------------------------------------------------------------------
# Test 2 - Deterministic execution
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Test 2: the same configuration must produce the same results twice."""

    def test_sample_seed_is_a_pure_function(self):
        first = sample_seed(42, "bbob", 1, 3, 10, 7)
        second = sample_seed(42, "bbob", 1, 3, 10, 7)
        assert first == second
        assert sample_seed(42, "bbob", 1, 3, 10, 8) != first

    def test_sample_seed_ignores_transform_and_encoding(self):
        """The shared-sample rule holds by construction, not by convention."""
        import inspect

        signature = inspect.signature(sample_seed)
        assert "transform" not in signature.parameters
        assert "encoding" not in signature.parameters

    def test_generated_sample_is_reproducible(self):
        first = create_sampled_points(42, "bbob", 1, 1, SMALL_DIM, 1, SMALL_SAMPLE)
        second = create_sampled_points(42, "bbob", 1, 1, SMALL_DIM, 1, SMALL_SAMPLE)
        assert np.array_equal(first.X_unit, second.X_unit)
        assert np.array_equal(first.y_pre, second.y_pre)
        assert np.array_equal(first.y_post, second.y_post)

    def test_sample_is_independent_of_the_global_rng(self):
        np.random.seed(1)
        first = create_sampled_points(42, "bbob", 1, 1, SMALL_DIM, 1, SMALL_SAMPLE)
        np.random.seed(999)
        second = create_sampled_points(42, "bbob", 1, 1, SMALL_DIM, 1, SMALL_SAMPLE)
        assert np.array_equal(first.X_unit, second.X_unit)

    def test_extracted_features_are_reproducible(self, real_sample):
        first = pd.DataFrame(features_for_sample(real_sample))
        second = pd.DataFrame(features_for_sample(real_sample))
        pd.testing.assert_frame_equal(first, second)

    @pytest.mark.slow
    def test_algorithm_results_are_reproducible(self):
        kwargs = {
            "binding": DEFAULT_BINDING,
            "base_seed": 42,
            "problem_class": "bbob",
            "function_id": 1,
            "instance_id": 1,
            "dim": SMALL_DIM,
            "trial": 1,
            "budget": 200,
        }
        first = pd.DataFrame(run_job(**kwargs))
        second = pd.DataFrame(run_job(**kwargs))
        pd.testing.assert_frame_equal(first, second)

    @pytest.mark.slow
    def test_a_rerun_computes_nothing_and_reports_the_same(self, debug_root):
        """A unit the store holds is read, never computed again."""
        import run_preanalysis

        report = debug_root / PREANALYSIS_ROOT
        before = report_files(report)
        rerun = [*DEBUG_RUN[:2], "--require-cached", *DEBUG_RUN[2:]]
        assert run_preanalysis.main([*rerun, f"paths.output_root={debug_root}"]) == 0
        listed = json.loads((report / STAGE_LIST_NAME).read_text(encoding="utf-8"))
        assert {unit["action"] for unit in listed["units"]} == {REUSED}
        after = report_files(report)
        # Only the two records of the run itself move: when it ran, and that it
        # read its units rather than computed them.
        moved = {
            name for name in {*before, *after} if before.get(name) != after.get(name)
        }
        assert moved <= {RUN_MANIFEST_NAME, STAGE_LIST_NAME}


# ---------------------------------------------------------------------------
# Test 3 - Artifact schema validation
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    """Test 3: every artifact must satisfy its declared schema."""

    def test_valid_tables_pass(self, features, performance, superiorities):
        assert validate_table(features, "features") == []
        assert validate_table(performance, PERFORMANCE_SCHEMA) == []
        assert validate_table(superiorities, SUPERIORITY_SCHEMA) == []

    @pytest.mark.slow
    def test_every_stored_table_satisfies_its_schema(self, debug_root):
        units = stored_units(debug_root / STAGE_STORE_NAME)
        expected = {"tune_pair", "features", "view_performance", "view_superiority"}
        assert {stage for stage, _, _ in units} >= expected
        for stage, directory, _ in units:
            for path in sorted(directory.rglob("*.parquet")):
                assert validate_table(read_table(path), STAGE_SCHEMAS[stage]) == []

    @pytest.mark.slow
    def test_every_stored_file_is_the_one_its_record_names(self, debug_root):
        for _, directory, record in stored_units(debug_root / STAGE_STORE_NAME):
            assert check_unit_files(directory, record) == []

    def test_duplicate_key_is_caught(self, features):
        doubled = pd.concat([features, features.head(1)], ignore_index=True)
        kinds = {v.kind for v in validate_table(doubled, "features")}
        assert "duplicate_key" in kinds

    def test_missing_value_is_caught(self, features):
        broken = features.copy()
        broken.loc[0, "value"] = np.nan
        kinds = {v.kind for v in validate_table(broken, "features")}
        assert {"missing_value", "nan"} & kinds

    def test_wrong_dtype_is_caught(self, features):
        broken = features.astype({"function_id": "int64"})
        kinds = {v.kind for v in validate_table(broken, "features")}
        assert "dtype" in kinds

    def test_unknown_category_is_caught(self, features):
        broken = features.copy()
        broken.loc[0, "transform"] = "sideways"
        kinds = {v.kind for v in validate_table(broken, "features")}
        assert "domain" in kinds

    def test_mixed_config_hash_is_caught(self, features):
        broken = features.copy()
        broken.loc[0, "config_hash"] = "deadbeefdeadbeef"
        kinds = {v.kind for v in validate_table(broken, "features")}
        assert "config_hash" in kinds


# ---------------------------------------------------------------------------
# Test 4 - Constant label sanity
# ---------------------------------------------------------------------------


class TestConstantLabel:
    """Test 4: a constant target must yield no measurable performance."""

    def test_r2_of_a_constant_target_is_zero(self):
        y = np.full(20, 0.7)
        assert r2_score(y, y) == 0.0
        assert r2_score(y, y + 0.1) == 0.0

    def test_model_on_a_constant_label_scores_zero(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
        frame["superiority"] = 0.5

        train, test = frame.iloc[:40], frame.iloc[40:]
        r2_train, r2_test, _, _ = fit_and_score(
            train, test, list("abcde"), ModelConfig()
        )
        assert r2_train == pytest.approx(0.0)
        assert r2_test == pytest.approx(0.0)

    def test_a_learnable_target_does_score(self):
        """The zero above must come from the constant label, not a broken fit."""
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(rng.normal(size=(200, 3)), columns=list("abc"))
        frame["superiority"] = frame["a"] * 2.0 + frame["b"]

        train, test = frame.iloc[:150], frame.iloc[150:]
        _, r2_test, _, _ = fit_and_score(train, test, list("abc"), ModelConfig())
        assert r2_test > 0.5


# ---------------------------------------------------------------------------
# Test 5 - Transformation order preservation
# ---------------------------------------------------------------------------


class TestOrderPreservation:
    """Test 5: the transformation must not change the ranking."""

    @pytest.mark.parametrize("function_id", [1, 8, 15, 21, 24])
    def test_rank_is_preserved_on_real_problems(self, function_id):
        sample = create_sampled_points(
            42,
            "bbob",
            function_id,
            1,
            SMALL_DIM,
            1,
            SMALL_SAMPLE,
        )
        assert np.array_equal(
            rankdata(sample.y_pre, method="average"),
            rankdata(sample.y_post, method="average"),
        )
        assert sample.ranks_agree()

    def test_transform_is_strictly_increasing(self):
        warp = transform_for_instance(42, "bbob", 1, 1, SMALL_DIM)
        span = warp.maximum_y - warp.optimum_y
        # Straddling the estimated maximum: the spline below it, the linear
        # continuation above, and the join between them.
        y = warp.optimum_y + np.linspace(0.0, 4.0, 2001) * span
        assert np.all(np.diff(warp(y)) > 0)

    def test_values_below_the_optimum_are_clamped(self):
        warp = transform_for_instance(42, "bbob", 1, 1, SMALL_DIM)
        transformed = warp(np.array([warp.optimum_y - 1e-9, warp.optimum_y]))
        assert transformed.tolist() == [0.0, 0.0]

    def test_a_broken_order_stops_the_pipeline(self, real_sample):
        from dataclasses import replace

        broken = replace(real_sample, y_post=real_sample.y_post[::-1].copy())
        assert not broken.ranks_agree()
        with pytest.raises(OrderPreservationError):
            features_for_sample(broken)


# ---------------------------------------------------------------------------
# Test 6 - Budget accounting
# ---------------------------------------------------------------------------


class TestBudgetAccounting:
    """Test 6: no run may exceed its budget, restarts included."""

    def test_budgeted_objective_refuses_to_overspend(self):
        objective = BudgetedObjective(lambda x: float(np.sum(x)), budget=5)
        for _ in range(5):
            objective(np.zeros(2))
        assert objective.evaluations == 5
        with pytest.raises(BudgetExhausted):
            objective(np.zeros(2))
        assert objective.exhausted

    @pytest.mark.slow
    @pytest.mark.parametrize("spec", PORTFOLIO, ids=lambda s: s.slug)
    @pytest.mark.parametrize("transform", TRANSFORMS)
    def test_every_algorithm_respects_the_budget(self, spec, transform):
        from aas_shortcut.transform_objective import FuncTransform

        budget = 250
        problem = create_problem(1, 1, SMALL_DIM)
        view = (
            FuncTransform(problem, transform_for_instance(42, "bbob", 1, 1, SMALL_DIM))
            if transform == "post"
            else problem
        )
        objective = BudgetedObjective(view, budget)
        seed = algorithm_seed(42, spec.name, 1, 1, SMALL_DIM, 1)

        report = spec.runner(
            objective, SMALL_DIM, seed, budget, problem.bounds.lb, problem.bounds.ub
        )

        assert objective.evaluations <= budget
        assert problem.state.evaluations <= budget
        assert problem.state.evaluations == objective.evaluations
        # The counters are separate quantities and each must be well formed.
        assert report.restarts >= 0
        assert report.multistarts >= 0
        assert report.native_reanneals >= 0

    @pytest.mark.slow
    def test_multistart_fires_for_the_local_solvers(self):
        """The multistart wrapper must actually restart, not run one start."""
        rows = run_job(DEFAULT_BINDING, 42, "bbob", 1, 1, SMALL_DIM, 1, 300)
        by_algorithm = {row["algorithm"]: row for row in rows}
        assert by_algorithm["BOBYQA"]["restarts"] > 0
        assert by_algorithm["Adaptive Nelder-Mead"]["restarts"] > 0

    @pytest.mark.slow
    def test_run_job_reports_every_algorithm_exactly_once(self):
        """One condition, one budget, one row per member."""
        rows = run_job(DEFAULT_BINDING, 42, "bbob", 1, 1, SMALL_DIM, 1, 300)
        keys = [(row["algorithm"], row["budget"]) for row in rows]
        assert len(keys) == len(set(keys)) == len(ALGORITHMS)
        assert {budget for _, budget in keys} == {300}
        for row in rows:
            assert row["evaluations_used"] == 300

    @pytest.mark.slow
    def test_a_longer_budget_is_a_separate_run_at_the_same_seed(self):
        """The seed does not depend on the budget, so the two differ only in it."""
        short = run_job(
            DEFAULT_BINDING,
            42,
            "bbob",
            1,
            1,
            SMALL_DIM,
            1,
            200,
            algorithms=("L-SHADE",),
        )[0]
        long = run_job(
            DEFAULT_BINDING,
            42,
            "bbob",
            1,
            1,
            SMALL_DIM,
            1,
            300,
            algorithms=("L-SHADE",),
        )[0]
        assert short["seed"] == long["seed"]
        assert short["evaluations_used"] == 200
        assert long["evaluations_used"] == 300

    def test_overspending_in_a_stored_table_is_caught(self, performance):
        assert check_budget(performance) == []

        broken = performance.copy()
        broken.loc[0, "evaluations_used"] = broken.loc[0, "budget"] + 1
        assert any(v.kind == "budget" for v in check_budget(broken))

    def test_absent_multistart_is_caught(self, performance):
        broken = performance.assign(restarts=0)
        assert any(v.kind == "multistart" for v in check_budget(broken))

    @pytest.mark.slow
    def test_no_stored_run_overspent_its_budget(self, debug_root):
        for _, table in stored_tables(debug_root, "view_performance"):
            assert check_budget(table) == []


# ---------------------------------------------------------------------------
# Test 7 - Result completeness
# ---------------------------------------------------------------------------


class TestCompleteness:
    """Test 7: no instance, transform, encoding, trial or algorithm may be missing."""

    @pytest.fixture
    def feature_config(self):
        return FeatureStageConfig(
            problem_class="bbob",
            dim=2,
            sampling_factor=50,
            num_reps=3,
            max_functions=1,
            max_instances=8,
            base_seed=42,
            scale="debug",
        )

    @pytest.fixture
    def view_benchmark(self):
        """The experiments' benchmark, over the grid ``make_performance`` fills."""
        return ViewBenchmarkConfig(
            problem_class="bbob",
            dim=2,
            max_functions=1,
            max_instances=8,
            num_trials=5,
            budget=200,
            base_seed=42,
            scale="debug",
            transforms=("pre",),
        )

    @pytest.fixture
    def view_performance(self):
        return conform(
            make_performance().assign(transform="pre"), VIEW_PERFORMANCE_SCHEMA
        )

    def test_complete_feature_table_passes(self, feature_config):
        assert check_feature_completeness(make_features(), feature_config) == []

    def test_missing_instance_is_caught(self, feature_config):
        table = make_features()
        broken = cast(pd.DataFrame, table[table["instance_id"] != 4])
        kinds = {v.kind for v in check_feature_completeness(broken, feature_config)}
        assert {"row_count", "incomplete"} & kinds

    def test_missing_transform_is_caught(self, feature_config):
        table = make_features()
        broken = cast(pd.DataFrame, table[table["transform"] != "post"])
        kinds = {v.kind for v in check_feature_completeness(broken, feature_config)}
        assert "incomplete" in kinds

    def test_missing_encoding_is_caught(self, feature_config):
        table = make_features()
        broken = cast(pd.DataFrame, table[table["feature_variant"] != "rank"])
        kinds = {v.kind for v in check_feature_completeness(broken, feature_config)}
        assert "incomplete" in kinds

    def test_complete_benchmark_table_passes(self, view_benchmark, view_performance):
        assert check_view_completeness(view_performance, view_benchmark) == []

    def test_missing_algorithm_is_caught(self, view_benchmark, view_performance):
        broken = view_performance[view_performance["algorithm"] != ALGORITHMS[-1]]
        kinds = {v.kind for v in check_view_completeness(broken, view_benchmark)}
        assert "incomplete" in kinds

    def test_expected_row_counts_match_the_grid(
        self, feature_config, view_benchmark, view_performance
    ):
        features = make_features()
        assert feature_config.expected_feature_rows(
            features["feature"].nunique()
        ) == len(features)
        assert view_benchmark.expected_rows() == len(view_performance)

    @pytest.mark.slow
    def test_every_stored_unit_holds_its_whole_grid(self, debug_root):
        for record, table in stored_tables(debug_root, "features"):
            config = feature_stage_config(record["settings"])
            assert check_feature_completeness(table, config) == []
        for record, table in stored_tables(debug_root, "view_performance"):
            config = view_benchmark_config(record["settings"])
            assert check_view_completeness(table, config) == []


# ---------------------------------------------------------------------------
# Test 8 - Numerical validity
# ---------------------------------------------------------------------------


class TestNumericalValidity:
    """Test 8: everything that reaches an analysis must be a usable number."""

    def test_real_features_are_finite(self, real_sample):
        rows = pd.DataFrame(features_for_sample(real_sample))
        assert np.isfinite(rows["value"].to_numpy()).all()
        assert not rows.empty

    def test_nan_is_caught(self, features):
        broken = features.copy()
        broken.loc[0, "value"] = np.nan
        assert any(v.kind == "nan" for v in validate_table(broken, FEATURES_SCHEMA))

    def test_inf_is_caught(self, features):
        broken = features.copy()
        broken.loc[0, "value"] = np.inf
        assert any(v.kind == "inf" for v in validate_table(broken, FEATURES_SCHEMA))

    def test_superiority_out_of_range_is_caught(self, superiorities):
        assert validate_table(superiorities, SUPERIORITY_SCHEMA) == []
        broken = superiorities.copy()
        broken.loc[0, "superiority"] = 1.4
        assert any(
            v.kind == "bounds" for v in validate_table(broken, SUPERIORITY_SCHEMA)
        )

    def test_superiorities_are_probabilities(self, superiorities):
        values = superiorities["superiority"].to_numpy()
        assert np.all((values >= 0.0) & (values <= 1.0))

    def test_performance_values_are_finite(self, performance):
        assert validate_table(performance, PERFORMANCE_SCHEMA) == []

    def test_features_are_not_all_constant(self, real_sample):
        """A feature set with no variance would carry no information at all."""
        rows = pd.DataFrame(features_for_sample(real_sample))
        for encoding in ("value", "rank"):
            subset = rows[rows["feature_variant"] == encoding]
            assert subset["value"].nunique() > 1

    def test_ela_inputs_are_normalised(self, real_sample):
        for transform in TRANSFORMS:
            for encoding in ("value", "rank"):
                _, y = ela_inputs(real_sample, transform, encoding)
                assert y.min() == pytest.approx(0.0)
                assert y.max() == pytest.approx(1.0)
