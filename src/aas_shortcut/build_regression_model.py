"""Regression backends the prediction and selection experiments share."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Breiman's random forest, from scikit-learn.
RANDOM_FOREST = "random-forest"

# Gradient-boosted trees, from XGBoost.
XGBOOST = "xgboost"

# The main learner: RealMLP-TD (Holzmuller et al., "Better by default", NeurIPS
# 2024), pytabkit's implementation at its regression defaults.
REALMLP = "realmlp"

# Every backend :meth:`ModelConfig.build` accepts, in report order.
MODEL_KINDS: tuple[str, ...] = (RANDOM_FOREST, XGBOOST, REALMLP)

NEURAL_KINDS: tuple[str, ...] = (REALMLP,)


# One backend per run, so that a difference between two cells is never a
# difference between two models.
DEFAULT_CANDIDATES: tuple[str, ...] = (REALMLP,)

# With fewer held-out rows than this, nothing is held out.
MIN_VALIDATION_ROWS = 5

# Backends that hold rows out to choose an epoch or a round on.
EARLY_STOPPED_KINDS = frozenset({*NEURAL_KINDS, XGBOOST})

# pytabkit's ``val_fraction`` default, but held out as whole problems, not rows.
VALIDATION_FRACTION = 0.2

# pytabkit 1.7.3's ``DefaultParams.RealMLP_TD_REG``. Written out so that a pytabkit
# whose defaults moved stops the fit, and so that a unit's fingerprint carries them.
REALMLP_TD_DEFAULTS: dict[str, Any] = {
    "hidden_sizes": [256, 256, 256],
    "max_one_hot_cat_size": 9,
    "embedding_size": 8,
    "weight_param": "ntk",
    "weight_init_mode": "std",
    "bias_init_mode": "he+5",
    "bias_lr_factor": 0.1,
    "act": "mish",
    "use_parametric_act": True,
    "act_lr_factor": 0.1,
    "wd": 0.02,
    "wd_sched": "flat_cos",
    "bias_wd_factor": 0.0,
    "block_str": "w-b-a-d",
    "p_drop": 0.15,
    "p_drop_sched": "flat_cos",
    "add_front_scale": True,
    "scale_lr_factor": 6.0,
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
    "num_emb_type": "pbld",
    "plr_sigma": 0.1,
    "plr_hidden_1": 16,
    "plr_hidden_2": 4,
    "plr_lr_factor": 0.1,
    "clamp_output": True,
    "normalize_output": True,
    "lr": 0.2,
    "n_epochs": 256,
    "lr_sched": "coslog4",
    "opt": "adam",
    "sq_mom": 0.95,
}

# pytabkit's batch size for RealMLP; read only to count a fit's updates.
REALMLP_TD_BATCH_SIZE = 256

# scikit-learn's random-forest defaults, written out so that a library whose
# defaults moved cannot change the forest unnoticed.
FOREST_DEFAULTS: dict[str, Any] = {
    "n_estimators": 100,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 1.0,
    "bootstrap": True,
}

# XGB-TD: Holzmuller et al.'s tuned regression defaults (pytabkit's ``XGB_TD_REG``).
# Early stopping is part of them; it uses the same held-out problems as RealMLP-TD.
XGB_TD_DEFAULTS: dict[str, Any] = {
    "n_estimators": 1000,
    "max_depth": 9,
    "learning_rate": 0.05,
    "min_child_weight": 2.0,
    "reg_lambda": 0.0,
    "subsample": 0.7,
    "tree_method": "hist",
    "max_bin": 256,
}


class Regressor(Protocol):
    """The interface every backend and :class:`ConstantModel` satisfies."""

    def predict(self, X: Any) -> np.ndarray:
        """Return one prediction per row of ``X``."""
        ...


@dataclass(frozen=True)
class ModelConfig:
    """Which regressor the downstream stages fit, and how."""

    # Must match the default of ``task.model.kind``, so no run silently uses
    # another backend.
    kind: str = REALMLP
    # Each training run replaces it with its own seed
    # (`define_repetitions.MODEL_SEEDS`, which start from this value).
    random_state: int = 42
    # One, because the stages above parallelise over cells and folds.
    thread_count: int = 1

    # Rounds without improvement before XGBoost stops. Unlike RealMLP-TD's
    # epochs, it is not divided in the augmented condition.
    early_stopping_rounds: int = 300

    # The augmented condition divides it by the views it spends
    # (:mod:`aas_shortcut.match_update_budget`).
    epochs: int = REALMLP_TD_DEFAULTS["n_epochs"]
    # ``0.0`` holds nothing out: the last epoch is kept and XGBoost runs its whole
    # schedule.
    validation_fraction: float = VALIDATION_FRACTION

    def __post_init__(self) -> None:
        """Reject a configuration no backend could build."""
        if self.kind not in MODEL_KINDS:
            raise ValueError(
                f"unknown model kind {self.kind!r}; expected one of {list(MODEL_KINDS)}"
            )
        if self.early_stopping_rounds < 1:
            raise ValueError(
                f"early_stopping_rounds is {self.early_stopping_rounds}; XGBoost "
                "needs at least one round to wait for an improvement"
            )

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description, carrying only what ``kind`` actually reads."""
        common = {
            "kind": self.kind,
            "random_state": self.random_state,
        }
        if self.kind == RANDOM_FOREST:
            return {**common, "forest": dict(FOREST_DEFAULTS)}
        if self.kind == XGBOOST:
            return {
                **common,
                "xgb_td": dict(XGB_TD_DEFAULTS),
                "early_stopping_rounds": self.early_stopping_rounds,
                "validation_fraction": self.validation_fraction,
            }
        return {
            **common,
            "realmlp_td": dict(REALMLP_TD_DEFAULTS),
            "epochs": self.epochs,
            "validation_fraction": self.validation_fraction,
        }

    def build(self) -> Any:
        """Instantiate an unfitted regressor for :attr:`kind`."""
        if self.kind == RANDOM_FOREST:
            return self._build_random_forest()
        if self.kind == XGBOOST:
            return EarlyStoppedXGBRegressor(
                early_stopping_rounds=self.early_stopping_rounds,
                validation_fraction=self.validation_fraction,
                random_state=self.random_state,
                thread_count=self.thread_count,
            )
        return RealMLPTDRegressor(
            epochs=self.epochs,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
            thread_count=self.thread_count,
        )

    def _build_random_forest(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            **FOREST_DEFAULTS,
            random_state=self.random_state,
            n_jobs=self.thread_count,
        )


class RealMLPTDRegressor:
    """RealMLP-TD from pytabkit, choosing its epoch on held-out problems.

    Two departures from pytabkit's defaults: the held-out rows are whole problems,
    and ``epochs`` is divided in the augmented condition. The target is passed raw,
    since TD standardises it itself.
    """

    def __init__(
        self,
        epochs: int = REALMLP_TD_DEFAULTS["n_epochs"],
        validation_fraction: float = VALIDATION_FRACTION,
        random_state: int = 42,
        thread_count: int = 1,
    ) -> None:
        self.epochs = epochs
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.thread_count = thread_count
        self._model: Any = None

    def fit(self, X: Any, y: Any, validation_groups: Any = None) -> RealMLPTDRegressor:
        """Fit on the rows not held out, keeping the epoch the held-out ones chose."""
        try:
            import torch
            from pytabkit import RealMLP_TD_Regressor
            from pytabkit.models.sklearn.default_params import DefaultParams
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ImportError(
                "the realmlp backend needs the `pytabkit` package; run `uv sync`"
            ) from error
        if DefaultParams.RealMLP_TD_REG != REALMLP_TD_DEFAULTS:
            raise RuntimeError(
                "the installed pytabkit's RealMLP-TD regression defaults differ from "
                "the ones this study records (REALMLP_TD_DEFAULTS); install the "
                "locked environment with `uv sync --locked`"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        _, valid_index = _validation_split(
            len(X),
            self.validation_fraction,
            np.random.default_rng(self.random_state),
            None if validation_groups is None else np.asarray(validation_groups),
        )
        model = RealMLP_TD_Regressor(
            device="cpu",
            random_state=self.random_state,
            n_threads=self.thread_count,
            n_epochs=self.epochs,
            verbosity=0,
        )
        # pytabkit seeds the global generators; restore them so fitting cannot
        # shift another consumer's draws.
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        try:
            # An empty array holds nothing out; ``None`` would let pytabkit draw
            # its own random rows instead.
            model.fit(X, y, val_idxs=np.asarray(valid_index, dtype=np.int64))
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
        self._model = model
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Return one prediction per row of ``X``."""
        if self._model is None:
            raise RuntimeError("RealMLP-TD must be fitted before it can predict")
        X = np.asarray(X, dtype=np.float64)
        fitted = int(self._model.n_features_in_)
        if X.shape[1] != fitted:
            raise ValueError(
                f"RealMLP-TD was fitted on {fitted} feature(s) but was given "
                f"{X.shape[1]}"
            )
        return np.asarray(self._model.predict(X), dtype=np.float64).ravel()


class EarlyStoppedXGBRegressor:
    """XGBoost at the XGB-TD defaults, early-stopped on RealMLP-TD's held-out rows.

    The held-out rows are drawn as RealMLP-TD draws them, so both learners of one
    training run stop on the same problems.
    """

    def __init__(
        self,
        early_stopping_rounds: int = 300,
        validation_fraction: float = VALIDATION_FRACTION,
        random_state: int = 42,
        thread_count: int = 1,
    ) -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.thread_count = thread_count
        self._model: Any = None

    def fit(
        self, X: Any, y: Any, validation_groups: Any = None
    ) -> EarlyStoppedXGBRegressor:
        """Fit on the rows not held out, stopping on the held-out ones."""
        try:
            from xgboost import XGBRegressor
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise ImportError(
                "the xgboost backend needs the `xgboost` package; run `uv sync`"
            ) from error

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        train_index, valid_index = _validation_split(
            len(X),
            self.validation_fraction,
            np.random.default_rng(self.random_state),
            None if validation_groups is None else np.asarray(validation_groups),
        )
        stops = len(valid_index) > 0
        self._model = XGBRegressor(
            **XGB_TD_DEFAULTS,
            early_stopping_rounds=self.early_stopping_rounds if stops else None,
            random_state=self.random_state,
            n_jobs=self.thread_count,
            verbosity=0,
        )
        if stops:
            self._model.fit(
                X[train_index],
                y[train_index],
                eval_set=[(X[valid_index], y[valid_index])],
                verbose=False,
            )
        else:
            self._model.fit(X, y, verbose=False)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the best round the held-out rows chose."""
        return np.asarray(
            self._model.predict(np.asarray(X, dtype=np.float64)), dtype=np.float64
        )

    @property
    def best_iteration(self) -> int | None:
        """The round the early stopping kept, or ``None`` when it did not stop."""
        if self._model is None or self._model.early_stopping_rounds is None:
            return None
        return int(self._model.best_iteration)


def _validation_split(
    n_rows: int,
    fraction: float,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into a training and an early-stopping part.

    ``groups`` (codes in first-appearance order) keeps a problem's rows on one side,
    so an augmented matrix cannot stop on a row whose twin it was fitted on. With
    all codes distinct, the split equals the ungrouped one, so unaugmented results
    do not change.
    """
    if fraction <= 0.0:
        return np.arange(n_rows), np.empty(0, dtype=np.int64)

    n_valid = round(n_rows * fraction)
    if n_valid < MIN_VALIDATION_ROWS or n_rows - n_valid < MIN_VALIDATION_ROWS:
        return np.arange(n_rows), np.empty(0, dtype=np.int64)

    if groups is None:
        shuffled = rng.permutation(n_rows)
        return shuffled[n_valid:], shuffled[:n_valid]

    members = _rows_per_group(groups, n_rows)
    order = rng.permutation(len(members))
    valid: list[np.ndarray] = []
    train: list[np.ndarray] = []
    filled = 0
    for code in order:
        rows = members[code]
        if filled < n_valid:
            valid.append(rows)
            filled += len(rows)
        else:
            train.append(rows)
    if not train or filled >= n_rows:
        # Every code landed on one side, which no split can learn from.
        return np.arange(n_rows), np.empty(0, dtype=np.int64)
    return np.concatenate(train), np.concatenate(valid)


def validation_split_sizes(n_rows: int, fraction: float) -> tuple[int, int]:
    """How many rows a fit trains on and holds out, for a matrix of ``n_rows``.

    The sizes do not depend on the seed or the grouping.
    """
    train_index, valid_index = _validation_split(
        n_rows, fraction, np.random.default_rng(0)
    )
    return len(train_index), len(valid_index)


def _rows_per_group(groups: np.ndarray, n_rows: int) -> list[np.ndarray]:
    """Return each code's row positions, indexed by the code itself."""
    codes = np.asarray(groups)
    if codes.shape != (n_rows,):
        raise ValueError(
            f"the validation groups have shape {codes.shape}, not one code per "
            f"each of the {n_rows} row(s)"
        )
    total = int(codes.max()) + 1 if n_rows else 0
    if codes.min(initial=0) < 0 or not np.array_equal(
        np.unique(codes), np.arange(total)
    ):
        raise ValueError(
            "the validation groups must be numbered 0..n-1 in the order they "
            "first appear; pass them through `validation_groups`"
        )
    members = [np.flatnonzero(codes == code) for code in range(total)]
    return members


class ConstantModel:
    """Predicts the training mean, used when the target has no variance."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, X: Any) -> np.ndarray:
        """Return the constant, once per row of ``X``."""
        return np.full(len(X), self.value, dtype=np.float64)


# The fit parameter RealMLP-TD and XGBoost read their held-out problems from.
GROUPS_PARAM = "validation_groups"


def _group_fit_params(model_config: ModelConfig, groups: np.ndarray | None) -> dict:
    """Fit parameters that carry the group codes to the backends that hold rows out."""
    if groups is None or model_config.kind not in EARLY_STOPPED_KINDS:
        return {}
    return {GROUPS_PARAM: groups}


def fit_model(
    train: pd.DataFrame,
    columns: list[str],
    model_config: ModelConfig,
    target: str,
    group_columns: Sequence[str] = (),
) -> Regressor:
    """Fit one regressor, or a :class:`ConstantModel` when the target is constant.

    ``group_columns`` identify one problem; its rows stay on one side of the
    early-stopping split (see :func:`_validation_split`).
    """
    y_train = train[target].to_numpy(dtype=np.float64)

    scale = float(np.max(np.abs(y_train))) or 1.0
    if float(np.ptp(y_train)) <= 1e-12 * scale:
        LOGGER.warning(
            "the training target %r is constant at %g; fitting the constant model",
            target,
            float(y_train[0]) if y_train.size else float("nan"),
        )
        return ConstantModel(y_train.mean() if y_train.size else 0.0)

    groups = None
    if group_columns:
        missing = [name for name in group_columns if name not in train.columns]
        if missing:
            raise ValueError(
                f"the training matrix has no column(s) {missing} to group its "
                "early-stopping split by"
            )
        # First-appearance order, so one row per problem permutes as the rows do.
        codes, _ = pd.factorize(
            pd.MultiIndex.from_frame(cast(pd.DataFrame, train[list(group_columns)])),
            sort=False,
        )
        groups = np.asarray(codes, dtype=np.int64)

    model = model_config.build()
    model.fit(
        train[columns].to_numpy(dtype=np.float64),
        y_train,
        **_group_fit_params(model_config, groups),
    )
    return model
