"""Conventions shared by every parallel stage."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def as_rows(results: Any) -> list[Any]:
    """Materialise a ``Parallel`` result as a list, in submission order.

    ``None`` means a joblib dispatch failure; it is raised here rather than later as
    an unrelated error.
    """
    if results is None:
        raise TypeError("joblib returned no results; the backend dispatch failed")
    if not isinstance(results, Iterable):
        raise TypeError(f"expected an iterable of results, got {type(results)!r}")
    return list(results)
