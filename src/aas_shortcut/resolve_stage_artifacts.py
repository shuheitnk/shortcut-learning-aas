"""Resolving a shared stage: read its artifact if it exists, compute it if not.

A unit lives in ``<output root>/_stages/<stage>/<fingerprint>/`` and is complete once
its ``stage.json`` exists. It is computed in ``<fingerprint>.partial/`` (so an
interrupted stage resumes) and published by one ``rename``; an exclusive lock on
``<fingerprint>.lock`` serialises two runs that need the same unit.
"""

from __future__ import annotations

import fcntl
import json
import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aas_shortcut.define_paths import STAGE_STORE_NAME, RunPaths
from aas_shortcut.read_write_tables import read_table
from aas_shortcut.record_manifests import (
    git_state,
    make_jsonable,
    sha256_file,
    stage_fingerprint,
)

LOGGER = logging.getLogger(__name__)


# Bumped when the layout of ``stage.json`` changes incompatibly.
STAGE_RECORD_VERSION = 1

# The record that makes a unit complete.
STAGE_RECORD_NAME = "stage.json"

# Lists, in an experiment's run root, every unit it resolved.
STAGE_LIST_NAME = "stages.json"

# Suffixes beside a unit's directory: the scratch it is computed in, and its lock.
PARTIAL_SUFFIX = ".partial"
LOCK_SUFFIX = ".lock"

# A stage's scratch rows, which a complete unit must not hold.
PARTS_SUFFIX = ".parts"

# When a unit is absent: compute it, refuse, or refuse and report it as a plan.
COMPUTE = "compute"
REQUIRE_CACHED = "require-cached"
PLAN = "plan"
POLICIES = (COMPUTE, REQUIRE_CACHED, PLAN)

REUSED, COMPUTED, MISSING = "reused", "computed", "missing"


class MissingStageArtifact(RuntimeError):
    """Raised when a resolution that may not compute finds no complete unit."""


class StageRecordError(RuntimeError):
    """Raised when a unit on disk does not match what claims to have produced it."""


def unit_fingerprint(
    settings: Mapping[str, Any], inputs: Mapping[str, str] | None = None
) -> str:
    """Fingerprint one unit: its settings, and the fingerprints of what it reads.

    A unit without inputs is fingerprinted by its settings alone, as
    ``stage_fingerprint`` always did; inputs chain in, so an upstream change makes a
    new unit downstream.
    """
    if not inputs:
        return stage_fingerprint(dict(settings))
    return stage_fingerprint({**dict(settings), "inputs": dict(sorted(inputs.items()))})


@dataclass(frozen=True)
class StageUnit:
    """One unit of a stage: what decides its content, and what it was made from.

    ``provenance`` is recorded but not fingerprinted, so two equal values from
    different sources stay one unit.
    """

    stage: str
    settings: Mapping[str, Any]
    inputs: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Return the unit's identity, also the ``config_hash`` its rows carry."""
        return unit_fingerprint(self.settings, self.inputs)


@dataclass(frozen=True)
class StageArtifact:
    """A complete unit on disk."""

    unit: StageUnit
    path: Path
    record: Mapping[str, Any]

    @property
    def fingerprint(self) -> str:
        """Return the unit's fingerprint."""
        return self.unit.fingerprint

    @property
    def run_paths(self) -> RunPaths:
        """Return the unit's directory as a run root."""
        return RunPaths(self.path)

    def file(self, relative: str | Path) -> Path:
        """Return one of the unit's files, which its record must list."""
        name = Path(relative).as_posix()
        if name not in self.record["files"]:
            held = sorted(self.record["files"])
            raise FileNotFoundError(f"{self.path} holds no {name}; it records {held}")
        return self.path / name

    def read_table(self, relative: str | Path) -> pd.DataFrame:
        """Read one of the unit's tables."""
        return read_table(self.file(relative))


@dataclass
class StageStore:
    """The shared stages' artifacts under one output root."""

    root: Path
    policy: str = COMPUTE
    # Every resolution, in order, for the run root's list.
    resolved: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Reject an unknown policy before anything is resolved."""
        if self.policy not in POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; expected one of {list(POLICIES)}"
            )
        self.root = Path(self.root)

    @classmethod
    def under(cls, output_root: Path | str, policy: str = COMPUTE) -> StageStore:
        """Return the store of one output root."""
        return cls(Path(output_root).expanduser().resolve() / STAGE_STORE_NAME, policy)

    def directory(self, unit: StageUnit) -> Path:
        """Return where the complete unit lives."""
        return self.root / unit.stage / unit.fingerprint

    def partial(self, unit: StageUnit) -> Path:
        """Return where the unit is computed until it is complete."""
        return self.root / unit.stage / f"{unit.fingerprint}{PARTIAL_SUFFIX}"

    def lock_path(self, unit: StageUnit) -> Path:
        """Return the file whose exclusive lock serialises work on the unit."""
        return self.root / unit.stage / f"{unit.fingerprint}{LOCK_SUFFIX}"

    def lookup(self, unit: StageUnit) -> StageArtifact | None:
        """Return the complete unit, or ``None`` when there is none.

        A unit that disagrees with its own record is damaged and raises rather than
        being read.
        """
        directory = self.directory(unit)
        path = directory / STAGE_RECORD_NAME
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        expected = {
            "stage": unit.stage,
            "fingerprint": unit.fingerprint,
            "settings": make_jsonable(dict(unit.settings)),
            "inputs": make_jsonable(dict(unit.inputs)),
        }
        found = {key: record.get(key) for key in expected}
        if found != expected:
            differing = sorted(key for key in expected if found[key] != expected[key])
            raise StageRecordError(
                f"{path} records other {differing} than the unit resolved to it"
            )
        for name, described in record["files"].items():
            file = directory / name
            if not file.exists() or file.stat().st_size != described["size_bytes"]:
                raise StageRecordError(
                    f"{directory}: {name} is missing or not the size its record holds"
                )
        return StageArtifact(unit, directory, record)

    def resolve(
        self, unit: StageUnit, compute: Callable[[RunPaths, str], object]
    ) -> StageArtifact:
        """Return the unit, computing it first if it does not exist.

        ``compute`` receives the scratch run root and the fingerprint (the rows'
        ``config_hash``) and writes the unit's files there.
        """
        found = self.lookup(unit)
        if found is not None:
            self._note(unit, REUSED, found.path)
            return found
        if self.policy != COMPUTE:
            self._note(unit, MISSING, self.directory(unit))
            raise MissingStageArtifact(
                f"{unit.stage} {unit.fingerprint} is not in {self.root} and this run "
                f"may not compute it ({self.policy})"
            )
        with self._locked(unit):
            # Another run may have finished it while this one waited for the lock.
            found = self.lookup(unit)
            if found is not None:
                self._note(unit, REUSED, found.path)
                return found
            scratch = self.partial(unit)
            scratch.mkdir(parents=True, exist_ok=True)
            LOGGER.info("%s %s: computing in %s", unit.stage, unit.fingerprint, scratch)
            compute(RunPaths(scratch), unit.fingerprint)
            artifact = self._complete(unit, scratch)
        self._note(unit, COMPUTED, artifact.path)
        return artifact

    def resolve_together(
        self,
        units: Sequence[StageUnit],
        compute: Callable[[Sequence[StageUnit], Mapping[str, RunPaths]], object],
    ) -> dict[str, StageArtifact]:
        """Return several units of one stage, computing the missing ones together.

        For cells whose fits share one worker pool. ``compute`` receives the units
        still missing once their locks are held, and their scratch roots.
        """
        found = {unit.fingerprint: self.lookup(unit) for unit in units}
        missing = [unit for unit in units if found[unit.fingerprint] is None]
        if missing and self.policy != COMPUTE:
            for unit in missing:
                self._note(unit, MISSING, self.directory(unit))
            raise MissingStageArtifact(
                f"{len(missing)} {missing[0].stage} unit(s) are not in {self.root} "
                f"and this run may not compute them ({self.policy})"
            )
        computed: set[str] = set()
        if missing:
            with ExitStack() as stack:
                # One order for every run, so two runs cannot each hold a lock
                # the other waits for.
                for unit in sorted(missing, key=lambda unit: unit.fingerprint):
                    stack.enter_context(self._locked(unit))
                still = [unit for unit in missing if self.lookup(unit) is None]
                if still:
                    scratches = {unit.fingerprint: self.partial(unit) for unit in still}
                    for scratch in scratches.values():
                        scratch.mkdir(parents=True, exist_ok=True)
                    LOGGER.info(
                        "%s: computing %d unit(s) together", still[0].stage, len(still)
                    )
                    compute(
                        still,
                        {fp: RunPaths(scratch) for fp, scratch in scratches.items()},
                    )
                    for unit in still:
                        self._complete(unit, scratches[unit.fingerprint])
                        computed.add(unit.fingerprint)
        artifacts: dict[str, StageArtifact] = {}
        for unit in units:
            artifact = self.lookup(unit)
            if artifact is None:  # pragma: no cover - completed just above
                raise StageRecordError(f"{unit.stage} {unit.fingerprint} vanished")
            artifacts[unit.fingerprint] = artifact
            action = COMPUTED if unit.fingerprint in computed else REUSED
            self._note(unit, action, artifact.path)
        return artifacts

    def write_list(self, run_root: Path | str) -> Path:
        """Write the units this store resolved into a run root and return the path."""
        path = Path(run_root) / STAGE_LIST_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"store": str(self.root), "units": self.resolved},
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
        return path

    def _note(self, unit: StageUnit, action: str, path: Path) -> None:
        """Record one resolution, for the run root's list and for a plan."""
        self.resolved.append(
            {
                "stage": unit.stage,
                "fingerprint": unit.fingerprint,
                "action": action,
                "path": str(path),
            }
        )
        LOGGER.info("%s %s: %s", unit.stage, unit.fingerprint, action)

    @contextmanager
    def _locked(self, unit: StageUnit) -> Iterator[None]:
        """Hold the unit's exclusive lock; released when the process ends anyway."""
        path = self.lock_path(unit)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _complete(self, unit: StageUnit, scratch: Path) -> StageArtifact:
        """Record what the scratch directory holds and make it the unit.

        An unfinished shard (``.parts`` left behind) must never become a complete unit.
        """
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(scratch.rglob("*")):
            relative = path.relative_to(scratch)
            if any(part.endswith(PARTS_SUFFIX) for part in relative.parts):
                raise StageRecordError(
                    f"{scratch} still holds the parts of an unfinished shard: "
                    f"{relative}"
                )
            if path.is_file() and path.name != STAGE_RECORD_NAME:
                files[relative.as_posix()] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
        if not files:
            raise StageRecordError(f"{scratch} holds no file; nothing was written")
        record = {
            "schema_version": STAGE_RECORD_VERSION,
            "stage": unit.stage,
            "fingerprint": unit.fingerprint,
            "settings": make_jsonable(dict(unit.settings)),
            "inputs": make_jsonable(dict(unit.inputs)),
            "provenance": make_jsonable(dict(unit.provenance)),
            # Fixed values, kept so every record has the same layout.
            "origin": COMPUTED,
            "source": {},
            "files": files,
            "completed_utc": datetime.now(UTC).isoformat(),
            "git": git_state(Path.cwd()),
        }
        temporary = scratch / f"{STAGE_RECORD_NAME}.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(scratch / STAGE_RECORD_NAME)
        directory = self.directory(unit)
        # A complete unit is never replaced; only a write behind the lock reaches this.
        if directory.exists():
            raise StageRecordError(f"{directory} appeared while {scratch} was computed")
        scratch.rename(directory)
        LOGGER.info("%s %s: complete in %s", unit.stage, unit.fingerprint, directory)
        return StageArtifact(unit, directory, record)
