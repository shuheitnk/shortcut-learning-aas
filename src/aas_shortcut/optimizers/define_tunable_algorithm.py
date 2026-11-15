"""An optimizer plus the parameter grid a tuning searches."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aas_shortcut.optimizers.define_optimizer_base import Runner
from aas_shortcut.record_manifests import config_fingerprint

# Length of the ``config_id`` that keys a configuration's runs.
CONFIG_ID_LENGTH = 12


def config_id(params: Mapping[str, Any]) -> str:
    """Return the stable identifier of one parameter assignment."""
    return config_fingerprint(dict(params))[:CONFIG_ID_LENGTH]


@dataclass(frozen=True)
class TunableAlgorithm:
    """One optimizer, its default configuration and the grid to search."""

    name: str
    slug: str
    factory: Callable[..., Runner]
    defaults: Mapping[str, Any]
    grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        """Reject a grid that names a parameter the factory does not take."""
        unknown = [key for key in self.grid if key not in self.defaults]
        if unknown:
            raise ValueError(
                f"{self.name}: grid axis/axes {unknown} are not parameters of "
                f"this algorithm; expected a subset of {sorted(self.defaults)}"
            )

    def resolve(self, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return a complete parameter dict, filling gaps from the defaults.

        An undeclared parameter raises: ignoring it would give a distinct
        ``config_id`` to a configuration that behaves identically.
        """
        params = dict(params or {})
        unknown = [key for key in params if key not in self.defaults]
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {unknown}; expected a subset "
                f"of {sorted(self.defaults)}"
            )
        return {**dict(self.defaults), **params}

    def runner(self, params: Mapping[str, Any] | None = None) -> Runner:
        """Build the solver entry point for one parameter assignment."""
        return self.factory(**self.resolve(params))

    @property
    def default_config_id(self) -> str:
        """Identifier of the default configuration."""
        return config_id(self.resolve())

    def candidates(self) -> list[dict[str, Any]]:
        """Return every configuration to evaluate, the default first."""
        axes = sorted(self.grid)
        combinations = itertools.product(*(list(self.grid[axis]) for axis in axes))
        resolved = [
            self.resolve(),
            *(
                self.resolve(dict(zip(axes, values, strict=True)))
                for values in combinations
            ),
        ]

        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for params in resolved:
            identifier = config_id(params)
            if identifier in seen:
                continue
            seen.add(identifier)
            unique.append(params)
        return unique

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description of this entry."""
        return {
            "name": self.name,
            "slug": self.slug,
            "defaults": dict(self.defaults),
            "grid": {key: list(values) for key, values in sorted(self.grid.items())},
        }
