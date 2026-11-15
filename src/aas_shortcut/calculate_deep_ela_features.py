"""The learned feature representation: Deep-ELA."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Upstream repository of the Deep-ELA package and its pretrained weights.
DEEP_ELA_REPOSITORY = "https://github.com/mvseiler/deep_ela.git"

# Commit the vendored checkout is pinned to.
DEEP_ELA_COMMIT = "4a5d4024b2fe3355de2ab6aa19a657a7c2a79300"

# Environment variable that overrides where the checkout is looked for.
VENDOR_ENV_VAR = "AAS_SHORTCUT_DEEP_ELA_HOME"

# Directory, relative to the project root, holding the vendored checkout.
VENDOR_SUBDIRECTORY = Path("vendor") / "deep_ela"

# ``large`` because the medium models cap ``d + m`` at 6 and this study needs
# ``10 + 1``; ``50d`` because it was pretrained on ``50 * d``-point samples, the
# study's sampling factor.
DEFAULT_MODEL = "large_50d_v1"

# The largest ``d + m`` the model accepts (the ``nu`` of the Deep-ELA paper's
# Table 1, stored as ``num_df`` in the checkpoint's hyperparameters).
MODEL_CAPACITY: dict[str, int] = {"large_50d_v1": 12}

# Features the model emits, from the ``features`` entry of its ``hparams.yaml``.
MODEL_FEATURE_COUNT: dict[str, int] = {"large_50d_v1": 48}

# Permutations of the input dimensions the model is averaged over.
DEFAULT_REPETITIONS = 10

# Prefix every Deep-ELA feature name carries, from upstream's own naming.
FEATURE_PREFIX = "deep_ela."

# Upstream reports a wall-clock timing alongside the features; it is dropped
# because it would make two identical runs differ.
RUNTIME_SUFFIX = "costs_runtime"


class DeepElaUnavailable(RuntimeError):
    """Raised when Deep-ELA features are requested but cannot be computed."""


def project_root() -> Path:
    """Return the directory the vendored checkout is looked for under."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def vendor_root() -> Path:
    """Return the path of the vendored Deep-ELA checkout."""
    override = os.environ.get(VENDOR_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / VENDOR_SUBDIRECTORY


@dataclass(frozen=True)
class Availability:
    """Whether Deep-ELA can run here, and why not when it cannot."""

    available: bool
    checkout: Path
    reason: str = ""

    def describe(self) -> str:
        """Render the status as a single actionable line."""
        if self.available:
            return f"Deep-ELA is available ({self.checkout})"
        return f"Deep-ELA is unavailable: {self.reason}"

    def require(self) -> None:
        """Raise :class:`DeepElaUnavailable`, with install steps, unless available."""
        if self.available:
            return
        raise DeepElaUnavailable(
            f"{self.reason}\n"
            "Install the backend with:\n"
            "  uv sync\n"
            "  uv run python scripts/install_deep_ela.py"
        )


def availability() -> Availability:
    """Report whether the vendored checkout and torch are both present."""
    checkout = vendor_root()
    package = checkout / "deep_ela"
    if not package.is_dir():
        return Availability(False, checkout, f"no Deep-ELA checkout at {checkout}")

    try:
        import torch  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        return Availability(False, checkout, f"torch is not installed ({error})")
    try:
        import lightning  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        return Availability(False, checkout, f"lightning is not installed ({error})")

    return Availability(True, checkout)


@dataclass(frozen=True)
class DeepElaConfig:
    """Everything that determines a Deep-ELA feature value."""

    model: str = DEFAULT_MODEL
    repetitions: int = DEFAULT_REPETITIONS
    device: str = "cpu"
    # torch threads.
    num_threads: int = 1

    def as_dict(self) -> dict:
        """Fingerprintable description of the configuration."""
        return {
            "model": self.model,
            "repetitions": self.repetitions,
            "device": self.device,
            "commit": DEEP_ELA_COMMIT,
        }

    def capacity(self) -> int:
        """Return the largest ``d + m`` the chosen model accepts."""
        try:
            return MODEL_CAPACITY[self.model]
        except KeyError:
            raise ValueError(
                f"unknown Deep-ELA model {self.model!r}; expected one of "
                f"{sorted(MODEL_CAPACITY)}"
            ) from None

    def validate(self, dim: int, n_objectives: int = 1) -> None:
        """Reject a dimensionality the chosen model was not built for.

        Upstream would otherwise fail with a broadcasting error deep inside
        ``predict``, hours into a sweep.
        """
        capacity = self.capacity()
        total = dim + n_objectives
        if total > capacity:
            larger = sorted(name for name, nu in MODEL_CAPACITY.items() if nu >= total)
            raise ValueError(
                f"Deep-ELA model {self.model!r} accepts d + m <= {capacity}, but "
                f"this configuration needs {dim} + {n_objectives} = {total}; "
                f"use one of {larger or 'no pretrained model — none is large enough'}"
            )


def _ensure_importable(checkout: Path) -> None:
    """Put the vendored checkout on ``sys.path`` exactly once."""
    entry = str(checkout)
    if entry not in sys.path:
        sys.path.insert(0, entry)


@lru_cache(maxsize=4)
def load_model(config: DeepElaConfig):
    """Load a pretrained Deep-ELA model (upstream wrapper), once per process."""
    status = availability()
    status.require()
    config.capacity()  # Reject an unknown model before touching the filesystem.

    _ensure_importable(status.checkout)
    import torch  # pyright: ignore[reportMissingImports]

    torch.set_num_threads(config.num_threads)

    import deep_ela  # pyright: ignore[reportMissingImports]

    loader = getattr(deep_ela, f"load_{config.model}")
    model = loader(device=config.device)
    LOGGER.info(
        "loaded Deep-ELA %s (d + m <= %d, %d repetitions)",
        config.model,
        config.capacity(),
        config.repetitions,
    )
    return model


def compute_deep_ela_features(
    X_df: pd.DataFrame,
    y_series: pd.Series,
    seed: int,
    config: DeepElaConfig | None = None,
) -> dict[str, float]:
    """Return the learned representation of one sample."""
    config = config or DeepElaConfig()
    model = load_model(config)

    import torch  # pyright: ignore[reportMissingImports]

    X = np.asarray(X_df, dtype=np.float64)
    y = np.asarray(y_series, dtype=np.float64)

    # ``fork_rng`` so that seeding does not leak into, or depend on, other torch
    # draws in the process.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        features = model(X, y, include_costs=False, repetitions=config.repetitions)

    values = {
        name: float(value)
        for name, value in features.items()
        if not name.endswith(RUNTIME_SUFFIX)
    }
    expected = MODEL_FEATURE_COUNT[config.model]
    if len(values) != expected:
        raise DeepElaUnavailable(
            f"Deep-ELA {config.model} returned {len(values)} feature(s), but "
            f"MODEL_FEATURE_COUNT declares {expected}; the completeness check "
            "and every expected row count are derived from that number"
        )
    return values


def is_deep_ela_feature(name: str) -> bool:
    """Whether a feature name belongs to the learned representation."""
    return name.startswith(FEATURE_PREFIX)
