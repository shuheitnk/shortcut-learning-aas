"""The regression backends the prediction and selection experiments fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aas_shortcut.build_regression_model as brm
from aas_shortcut.build_regression_model import (
    FOREST_DEFAULTS,
    MODEL_KINDS,
    NEURAL_KINDS,
    RANDOM_FOREST,
    REALMLP,
    REALMLP_TD_DEFAULTS,
    VALIDATION_FRACTION,
    XGB_TD_DEFAULTS,
    XGBOOST,
    ConstantModel,
    EarlyStoppedXGBRegressor,
    ModelConfig,
    RealMLPTDRegressor,
    fit_model,
)

# Backends whose fit is fast enough for the default suite.
FAST_KINDS = (RANDOM_FOREST, XGBOOST)


def make_training_frame(
    n_rows: int = 120, n_features: int = 6, seed: int = 0
) -> tuple[pd.DataFrame, list[str]]:
    """Build a learnable problem on unequal scales; returns the frame and its columns.

    A backend that forgets to standardise its input is visible only on such scales.
    """
    rng = np.random.default_rng(seed)
    columns = [f"f{index}" for index in range(n_features)]
    X = rng.normal(size=(n_rows, n_features)) * np.logspace(-3, 3, n_features)
    target = X[:, 0] / 1e-3 + 0.5 * X[:, 1] / np.logspace(-3, 3, n_features)[1]
    frame = pd.DataFrame(X, columns=columns)
    frame["target"] = target + 0.01 * rng.normal(size=n_rows)
    return frame, columns


def small_config(kind: str) -> ModelConfig:
    """Return a configuration small enough to fit inside a unit test."""
    return ModelConfig(kind=kind, epochs=16)


class TestModelConfigValidation:
    def test_the_default_backend_is_the_one_the_schema_names(self):
        """The default backend matches ``task.model.kind`` in the schema."""
        from aas_shortcut.define_config_schema import ModelSchema

        assert ModelConfig().kind == ModelSchema().kind == REALMLP

    def test_an_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError, match="unknown model kind"):
            ModelConfig(kind="lightgbm")

    @pytest.mark.parametrize("kind", MODEL_KINDS)
    def test_every_kind_builds(self, kind: str):
        assert hasattr(ModelConfig(kind=kind).build(), "fit")


class TestFingerprint:
    """``as_dict`` decides which cached artifacts a configuration change invalidates."""

    def test_the_backend_is_part_of_the_fingerprint(self):
        assert ModelConfig(kind=XGBOOST).as_dict() != ModelConfig().as_dict()

    def test_a_field_the_backend_ignores_is_not(self):
        """Retuning a neural knob must not invalidate a cached forest artifact."""
        forest = ModelConfig(kind=RANDOM_FOREST)
        assert (
            ModelConfig(kind=RANDOM_FOREST, validation_fraction=0.5, epochs=2).as_dict()
            == forest.as_dict()
        )

    def test_a_field_the_backend_reads_is(self):
        assert (
            ModelConfig(kind=REALMLP, epochs=9).as_dict()
            != ModelConfig(kind=REALMLP).as_dict()
        )


class TestFitModel:
    @pytest.mark.parametrize("kind", FAST_KINDS)
    def test_every_backend_learns_a_learnable_target(self, kind: str):
        frame, columns = make_training_frame()
        model = fit_model(frame.iloc[:100], columns, small_config(kind), "target")

        held_out = frame.iloc[100:]
        predicted = model.predict(held_out[columns].to_numpy(dtype=np.float64))
        truth = held_out["target"].to_numpy(dtype=np.float64)
        residual = np.sum((truth - predicted) ** 2)
        total = np.sum((truth - truth.mean()) ** 2)
        assert 1.0 - residual / total > 0.5

    @pytest.mark.parametrize("kind", FAST_KINDS)
    def test_a_backend_is_reproducible_from_its_seed(self, kind: str):
        """Two fits of one configuration must agree bit for bit."""
        frame, columns = make_training_frame()
        config = small_config(kind)
        X = frame[columns].to_numpy(dtype=np.float64)

        first = fit_model(frame, columns, config, "target").predict(X)
        second = fit_model(frame, columns, config, "target").predict(X)
        assert np.array_equal(first, second)

    @pytest.mark.parametrize("kind", FAST_KINDS)
    def test_a_constant_target_falls_back_to_the_constant_model(self, kind: str):
        frame, columns = make_training_frame()
        frame["target"] = 3.0

        model = fit_model(frame, columns, small_config(kind), "target")
        assert isinstance(model, ConstantModel)
        assert np.allclose(model.predict(frame[columns]), 3.0)

    @pytest.mark.parametrize("kind", FAST_KINDS)
    def test_one_prediction_per_row(self, kind: str):
        frame, columns = make_training_frame()
        model = fit_model(frame, columns, small_config(kind), "target")
        assert model.predict(frame[columns].to_numpy(dtype=np.float64)).shape == (
            len(frame),
        )


def numpy_generator_state() -> np.ndarray:
    """The key and position of NumPy's legacy global generator."""
    inner = np.random.get_state(legacy=False)["state"]
    return np.append(np.asarray(inner["key"]), inner["pos"])


class TestRealMLPTD:
    """The main learner: pytabkit's RealMLP-TD at its regression defaults."""

    def test_it_is_a_candidate_and_a_neural_backend(self):
        """The augmented condition halves the epochs of neural backends only."""
        assert REALMLP in MODEL_KINDS
        assert REALMLP in NEURAL_KINDS

    def test_the_defaults_it_records_are_pytabkits(self):
        """Written out, and still what the installed pytabkit defaults to."""
        from pytabkit.models.sklearn.default_params import DefaultParams

        assert DefaultParams.RealMLP_TD_REG == REALMLP_TD_DEFAULTS

    def test_its_epochs_and_held_out_share_are_tds(self):
        """256 epochs and a fifth held out: TD's ``n_epochs`` and ``val_fraction``."""
        config = ModelConfig(kind=REALMLP)
        assert config.epochs == REALMLP_TD_DEFAULTS["n_epochs"] == 256
        assert config.validation_fraction == VALIDATION_FRACTION == 0.2

    def test_the_fingerprint_carries_tds_values_and_its_own(self):
        from dataclasses import replace

        base = ModelConfig(kind=REALMLP)
        recorded = base.as_dict()
        assert recorded["realmlp_td"] == REALMLP_TD_DEFAULTS
        assert (recorded["epochs"], recorded["validation_fraction"]) == (256, 0.2)
        assert replace(base, epochs=128).as_dict() != recorded
        # And not XGBoost's, so retuning XGBoost leaves a cached RealMLP-TD alone.
        assert replace(base, early_stopping_rounds=7).as_dict() == recorded

    def test_a_pytabkit_with_other_defaults_is_refused(self, monkeypatch):
        """A moved default would change the learner under an unchanged fingerprint."""
        from pytabkit.models.sklearn.default_params import DefaultParams

        moved = {**DefaultParams.RealMLP_TD_REG, "lr": 0.04}
        monkeypatch.setattr(DefaultParams, "RealMLP_TD_REG", moved)
        frame, columns = make_training_frame()
        with pytest.raises(RuntimeError, match="defaults differ"):
            fit_model(frame, columns, small_config(REALMLP), "target")

    @pytest.mark.slow
    def test_it_hands_pytabkit_nothing_but_the_two_departures(self, monkeypatch):
        """Every TD default reaches the fit; only the epochs and held-out rows differ.

        The held-out rows are the problem-level split, passed as ``val_idxs`` so that
        pytabkit does not draw random rows of its own.
        """
        from pytabkit import RealMLP_TD_Regressor

        seen: dict = {}
        original = RealMLP_TD_Regressor.fit

        def spy(self, X, y, **kwargs):
            seen["config"] = self.get_config()
            seen["val_idxs"] = kwargs.get("val_idxs")
            return original(self, X, y, **kwargs)

        monkeypatch.setattr(RealMLP_TD_Regressor, "fit", spy)
        frame, columns = make_training_frame(n_rows=200)
        X = frame[columns].to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy(dtype=np.float64)
        RealMLPTDRegressor(epochs=16, random_state=3).fit(X, y, validation_groups=TWICE)

        config = seen["config"]
        defaults = {k: v for k, v in REALMLP_TD_DEFAULTS.items() if k != "n_epochs"}
        assert {name: config[name] for name in defaults} == defaults
        assert config["n_epochs"] == 16
        assert (config["device"], config["n_threads"], config["random_state"]) == (
            "cpu",
            1,
            3,
        )
        train, valid = brm._validation_split(
            len(X), VALIDATION_FRACTION, np.random.default_rng(3), TWICE
        )
        assert np.array_equal(seen["val_idxs"], valid)
        assert set(TWICE[valid]).isdisjoint(TWICE[train])

    @pytest.mark.slow
    def test_it_learns_a_learnable_target(self):
        frame, columns = make_training_frame()
        model = fit_model(frame.iloc[:100], columns, small_config(REALMLP), "target")

        held_out = frame.iloc[100:]
        predicted = model.predict(held_out[columns].to_numpy(dtype=np.float64))
        truth = held_out["target"].to_numpy(dtype=np.float64)
        residual = np.sum((truth - predicted) ** 2)
        total = np.sum((truth - truth.mean()) ** 2)
        assert 1.0 - residual / total > 0.5

    @pytest.mark.slow
    def test_it_is_reproducible_from_its_seed(self):
        """Two fits of one configuration agree bit for bit."""
        frame, columns = make_training_frame()
        config = small_config(REALMLP)
        X = frame[columns].to_numpy(dtype=np.float64)

        first = fit_model(frame, columns, config, "target").predict(X)
        second = fit_model(frame, columns, config, "target").predict(X)
        assert np.array_equal(first.view(np.uint64), second.view(np.uint64))

    @pytest.mark.slow
    def test_it_leaves_the_global_generators_where_it_found_them(self):
        """Fitting must not shift another consumer's draws; pytabkit seeds torch's."""
        torch = pytest.importorskip("torch")
        frame, columns = make_training_frame()

        torch.manual_seed(1234)
        np.random.seed(1234)
        torch_before = torch.get_rng_state().clone()
        numpy_before = numpy_generator_state()
        fit_model(frame, columns, small_config(REALMLP), "target")
        assert torch.equal(torch.get_rng_state(), torch_before)
        assert np.array_equal(numpy_generator_state(), numpy_before)

    @pytest.mark.slow
    def test_one_prediction_per_row(self):
        frame, columns = make_training_frame()
        model = fit_model(frame, columns, small_config(REALMLP), "target")
        predicted = model.predict(frame[columns].to_numpy(dtype=np.float64))
        assert predicted.shape == (len(frame),)
        assert predicted.dtype == np.float64

    @pytest.mark.slow
    def test_predicting_on_a_different_width_is_rejected(self):
        frame, columns = make_training_frame()
        model = fit_model(frame, columns, small_config(REALMLP), "target")
        with pytest.raises(ValueError, match="was fitted on"):
            model.predict(np.zeros((3, len(columns) + 1)))

    def test_predicting_before_fitting_is_rejected(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            RealMLPTDRegressor().predict(np.zeros((3, 2)))


class TestTheEarlyStoppingSplitCanBeGrouped:
    """A problem's rows all fall on one side of the early-stopping split.

    The augmented condition carries every problem twice; a row split would hold out
    twins of training rows. A matrix with one row per problem must split exactly as
    a row split did, so the unaugmented conditions keep their results.
    """

    FRACTION = VALIDATION_FRACTION
    ROWS = 200

    @staticmethod
    def split(n_rows, fraction, seed, groups=None):
        return brm._validation_split(
            n_rows, fraction, np.random.default_rng(seed), groups
        )

    def test_one_row_per_problem_is_the_split_it_always_was(self):
        """Same arrays, from the same draw."""
        plain = self.split(self.ROWS, self.FRACTION, 7)
        grouped = self.split(
            self.ROWS, self.FRACTION, 7, np.arange(self.ROWS, dtype=np.int64)
        )
        assert np.array_equal(plain[0], grouped[0])
        assert np.array_equal(plain[1], grouped[1])

    def test_it_consumes_the_generator_the_same_way(self):
        """Bit-identity has to survive whatever draws next from the same rng."""
        left = np.random.default_rng(7)
        brm._validation_split(self.ROWS, self.FRACTION, left)
        right = np.random.default_rng(7)
        brm._validation_split(
            self.ROWS, self.FRACTION, right, np.arange(self.ROWS, dtype=np.int64)
        )
        assert (
            left.integers(0, 2**32, size=5).tolist()
            == right.integers(0, 2**32, size=5).tolist()
        )

    def test_a_duplicated_problem_lands_on_one_side(self):
        codes = np.repeat(np.arange(self.ROWS // 2), 2)
        train, valid = self.split(self.ROWS, self.FRACTION, 3, codes)
        for side in (train, valid):
            held = codes[side]
            counted = np.bincount(held, minlength=self.ROWS // 2)
            assert set(counted.tolist()) <= {0, 2}, "a problem was split in two"
        assert len(train) + len(valid) == self.ROWS
        assert not set(train.tolist()) & set(valid.tolist())

    def test_the_held_out_share_is_still_about_the_fraction(self):
        codes = np.repeat(np.arange(self.ROWS // 2), 2)
        _, valid = self.split(self.ROWS, self.FRACTION, 3, codes)
        assert len(valid) == pytest.approx(self.ROWS * self.FRACTION, abs=2)

    def test_codes_that_are_not_one_per_row_are_rejected(self):
        with pytest.raises(ValueError, match="one code per"):
            self.split(self.ROWS, self.FRACTION, 1, np.arange(5))

    def test_codes_that_are_not_numbered_from_zero_are_rejected(self):
        with pytest.raises(ValueError, match="numbered 0"):
            self.split(
                self.ROWS, self.FRACTION, 1, np.arange(1, self.ROWS + 1, dtype=np.int64)
            )

    def test_the_sizes_are_readable_without_a_seed(self):
        fitted, held = brm.validation_split_sizes(self.ROWS, self.FRACTION)
        assert fitted + held == self.ROWS
        assert held == round(self.ROWS * self.FRACTION)


class TestGroupingIsOptionalAndInert:
    """`fit_model` given the group columns, on a matrix with one row each."""

    @staticmethod
    def frame(n_rows: int) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            {
                "problem_class": "bbob",
                "function_id": np.arange(n_rows) % 4 + 1,
                "instance_id": np.arange(n_rows),
                "x0": rng.normal(size=n_rows),
                "x1": rng.normal(size=n_rows),
                "y": rng.normal(size=n_rows),
            }
        )

    def test_the_fit_is_the_one_it_was_without_the_groups(self):
        frame = self.frame(80)
        config = brm.ModelConfig(kind="realmlp", epochs=4)
        left = brm.fit_model(frame, ["x0", "x1"], config, "y")
        right = brm.fit_model(
            frame,
            ["x0", "x1"],
            config,
            "y",
            group_columns=("problem_class", "function_id", "instance_id"),
        )
        X = frame[["x0", "x1"]].to_numpy(dtype=np.float64)
        assert left.predict(X) == pytest.approx(right.predict(X))

    def test_a_missing_group_column_is_rejected(self):
        with pytest.raises(ValueError, match="no column"):
            brm.fit_model(
                self.frame(40),
                ["x0", "x1"],
                brm.ModelConfig(kind="realmlp", epochs=2),
                "y",
                group_columns=("rep",),
            )

    def test_the_forest_is_not_handed_the_groups(self):
        """It has no early-stopping split, and would reject the argument."""
        frame = self.frame(40)
        config = brm.ModelConfig(kind="random-forest")
        assert brm._group_fit_params(config, np.arange(40)) == {}
        brm.fit_model(
            frame,
            ["x0", "x1"],
            config,
            "y",
            group_columns=("problem_class", "function_id", "instance_id"),
        )


# The problem codes `fit_model` derives for a matrix that carries every problem
# twice, as the augmented condition's does.
TWICE = np.repeat(np.arange(100), 2)


class TestTheAppendixLearners:
    """The random forest and XGBoost of the appendix."""

    def test_the_forest_is_at_the_librarys_defaults(self):
        """Written out, and still what the installed library defaults to."""
        from sklearn.ensemble import RandomForestRegressor

        assert {
            name: RandomForestRegressor().get_params()[name] for name in FOREST_DEFAULTS
        } == FOREST_DEFAULTS
        built = ModelConfig(kind=RANDOM_FOREST, random_state=7).build().get_params()
        assert {name: built[name] for name in FOREST_DEFAULTS} == FOREST_DEFAULTS
        assert (built["random_state"], built["n_jobs"]) == (7, 1)

    def test_xgboost_is_at_the_published_xgb_td_defaults(self):
        """Holzmuller et al. (NeurIPS 2024), pytabkit's ``XGB_TD_REG``."""
        assert XGB_TD_DEFAULTS == {
            "n_estimators": 1000,
            "max_depth": 9,
            "learning_rate": 0.05,
            "min_child_weight": 2.0,
            "reg_lambda": 0.0,
            "subsample": 0.7,
            "tree_method": "hist",
            "max_bin": 256,
        }
        assert ModelConfig().early_stopping_rounds == 300

    def test_the_fingerprint_carries_xgboosts_stopping(self):
        base = ModelConfig(kind=XGBOOST)
        assert ModelConfig(kind=XGBOOST, early_stopping_rounds=50).as_dict() != (
            base.as_dict()
        )
        assert ModelConfig(kind=XGBOOST, validation_fraction=0.3).as_dict() != (
            base.as_dict()
        )
        assert base.as_dict()["xgb_td"] == XGB_TD_DEFAULTS

    def test_a_stop_without_waiting_is_rejected(self):
        with pytest.raises(ValueError, match="early_stopping_rounds"):
            ModelConfig(kind=XGBOOST, early_stopping_rounds=0)

    def test_xgboost_is_handed_the_groups(self):
        groups = np.arange(10)
        params = brm._group_fit_params(ModelConfig(kind=XGBOOST), groups)
        assert list(params) == [brm.GROUPS_PARAM]
        assert params[brm.GROUPS_PARAM] is groups

    def test_xgboost_holds_out_the_rows_realmlp_does(self, monkeypatch):
        """The same seed, share and problems: the split RealMLP-TD chooses on."""
        drawn = []
        split = brm._validation_split

        def recording(*args, **kwargs):
            drawn.append(split(*args, **kwargs))
            return drawn[-1]

        monkeypatch.setattr(brm, "_validation_split", recording)
        frame, columns = make_training_frame(n_rows=200)
        X = frame[columns].to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy(dtype=np.float64)
        realmlp = small_config(REALMLP)
        realmlp.build().fit(X, y, validation_groups=TWICE)
        ModelConfig(kind=XGBOOST).build().fit(X, y, validation_groups=TWICE)
        (realmlp_train, realmlp_valid), (xgb_train, xgb_valid) = drawn
        assert np.array_equal(realmlp_valid, xgb_valid)
        assert np.array_equal(realmlp_train, xgb_train)
        # Both copies of a problem are on the side the other one is.
        assert set(TWICE[xgb_valid]).isdisjoint(TWICE[xgb_train])

    def test_xgboost_is_the_xgb_td_fit_on_that_split(self):
        """Bit for bit the library's model, fitted and stopped on the split."""
        from xgboost import XGBRegressor

        frame, columns = make_training_frame(n_rows=200)
        X = frame[columns].to_numpy(dtype=np.float64)
        y = frame["target"].to_numpy(dtype=np.float64)
        wrapped = EarlyStoppedXGBRegressor(random_state=3).fit(
            X, y, validation_groups=TWICE
        )
        train, valid = brm._validation_split(
            len(X), VALIDATION_FRACTION, np.random.default_rng(3), TWICE
        )
        direct = XGBRegressor(
            **XGB_TD_DEFAULTS,
            early_stopping_rounds=300,
            random_state=3,
            n_jobs=1,
            verbosity=0,
        ).fit(X[train], y[train], eval_set=[(X[valid], y[valid])], verbose=False)
        assert np.array_equal(wrapped.predict(X), direct.predict(X))
        assert wrapped.best_iteration == direct.best_iteration
        assert wrapped.best_iteration < XGB_TD_DEFAULTS["n_estimators"] - 1

    def test_without_rows_to_hold_out_xgboost_runs_the_whole_schedule(self):
        frame, columns = make_training_frame(n_rows=20)
        model = EarlyStoppedXGBRegressor().fit(
            frame[columns].to_numpy(dtype=np.float64),
            frame["target"].to_numpy(dtype=np.float64),
        )
        assert model.best_iteration is None
        assert (
            model._model.get_booster().num_boosted_rounds()
            == (XGB_TD_DEFAULTS["n_estimators"])
        )
