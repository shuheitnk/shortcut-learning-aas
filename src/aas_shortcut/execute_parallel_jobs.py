"""Dispatching one stage's independent jobs, with a live estimate of what is left."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeVar

from joblib import Parallel, delayed, effective_n_jobs

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Shortest gap between two progress lines from one stage, in seconds.
MIN_LOG_INTERVAL = 5.0

# Longer than joblib's 300 s default: one reusable executor serves every stage, and
# a worker reaped between stages would be rebuilt, re-importing torch, for nothing.
IDLE_WORKER_TIMEOUT = 3600.0


def parallel_for(n_jobs: int, n_tasks: int) -> Parallel:
    """Return a pool sized for the work it is about to be given."""
    workers = max(1, min(effective_n_jobs(n_jobs), max(1, n_tasks)))
    if workers == 1:
        return Parallel(n_jobs=1)
    return Parallel(
        n_jobs=workers, backend="loky", idle_worker_timeout=IDLE_WORKER_TIMEOUT
    )


def _clock(seconds: float) -> str:
    """Render a duration as ``H:MM:SS``."""
    whole = int(max(seconds, 0.0))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


def format_progress(done: int, total: int, elapsed: float) -> str:
    """Render progress and a straight-line ETA as one line."""
    if total <= 0:
        return "0/0 (100%)"
    percent = 100.0 * done / total
    line = f"{done}/{total} ({percent:.0f}%) | {_clock(elapsed)} elapsed"
    if done <= 0:
        return line
    return f"{line} | ETA {_clock(elapsed / done * (total - done))}"


def _indexed(worker: Callable[..., T], index: int, task: Any) -> tuple[int, T]:
    """Run one task, carrying its position so the caller can reassemble in order."""
    return index, worker(*task)


def run_jobs(
    label: str,
    tasks: Sequence[Any],
    worker: Callable[..., T],
    *,
    n_jobs: int = 1,
    log_every: int = 0,
) -> Iterator[tuple[int, T]]:
    """Run ``worker(*task)`` for every task, yielding ``(index, result)`` in order."""
    total = len(tasks)
    if not total:
        return

    if log_every <= 0:
        log_every = max(1, total // 20)

    started = time.perf_counter()
    last_logged = started
    # Streaming, so workers never idle behind a slow sibling. Not
    # `generator_unordered`: in joblib 1.5.3 it can raise "Set changed size during
    # iteration" (joblib/joblib#1789).
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator",
        idle_worker_timeout=IDLE_WORKER_TIMEOUT,
    )(delayed(_indexed)(worker, index, task) for index, task in enumerate(tasks))
    for done, (index, value) in enumerate(results, start=1):
        now = time.perf_counter()
        if done == total or (
            done % log_every == 0 and now - last_logged >= MIN_LOG_INTERVAL
        ):
            last_logged = now
            LOGGER.info("%s: %s", label, format_progress(done, total, now - started))
        yield index, value
