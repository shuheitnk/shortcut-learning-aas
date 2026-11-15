"""Process-wide determinism switches."""

from __future__ import annotations

import os

# Environment variables that force single-threaded linear algebra.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_runtime(num_threads: int = 1) -> None:
    """Pin hash seeding and BLAS thread counts for reproducible runs."""
    os.environ.setdefault("PYTHONHASHSEED", "0")
    for name in THREAD_ENV_VARS:
        os.environ.setdefault(name, str(num_threads))


def thread_env_snapshot() -> dict[str, str | None]:
    """Return the determinism-relevant environment variables for manifests."""
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        **{name: os.environ.get(name) for name in THREAD_ENV_VARS},
    }
