"""Applying one instance's objective transformation to a problem."""

from __future__ import annotations

from aas_shortcut.sample_monotone_transform import MonotoneTransform


class FuncTransform:
    """Wrap a problem so calls return the monotonically transformed objective."""

    def __init__(self, problem, transform: MonotoneTransform) -> None:
        self.problem = problem
        self.transform = transform

    def __call__(self, x) -> float:
        """Evaluate the wrapped problem and transform the result."""
        return float(self.transform(self.problem(x)))

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped problem.

        Dunder lookups raise, so copy, pickle and ``inspect`` see a normal object.
        """
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self.problem, name)

    def __repr__(self) -> str:
        """Return a debug representation naming the wrapped problem."""
        return f"FuncTransform({self.problem!r}, transform={self.transform!r})"
