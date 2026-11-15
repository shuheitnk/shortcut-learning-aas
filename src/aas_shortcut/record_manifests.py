"""Run manifests and determinism helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from aas_shortcut.configure_runtime import thread_env_snapshot
from aas_shortcut.define_paths import MANIFEST_SUFFIX, RUN_MANIFEST_NAME

# Bumped whenever the manifest layout changes incompatibly.
MANIFEST_SCHEMA_VERSION = 1

# Files under a run root that describe artifacts rather than being one.
SIDECAR_SUFFIXES = (MANIFEST_SUFFIX, ".content.json", ".reuse.json")

# Directories under a run root that hold an interrupted stage's scratch rows.
PARTS_SUFFIX = ".parts"

# Fingerprint characters kept in a table's ``config_hash`` column.
STAGE_FINGERPRINT_LENGTH = 16


def make_jsonable(value: Any) -> Any:
    """Convert configuration values into JSON-serialisable structures."""
    if isinstance(value, range):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [make_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(make_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in sorted(value.items())}
    if hasattr(value, "item"):
        # NumPy scalars.
        try:
            return value.item()
        except ValueError:
            pass
    return value


def config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint of an experiment configuration."""
    payload = json.dumps(
        make_jsonable(config),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stage_fingerprint(config: dict[str, Any]) -> str:
    """Fingerprint a stage configuration, shortened for a table column."""
    return config_fingerprint(config)[:STAGE_FINGERPRINT_LENGTH]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Describe a file for a manifest, tolerating files that do not exist."""
    path = Path(path)
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if record["exists"]:
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    return record


def installed_packages() -> dict[str, str]:
    """Collect installed distribution versions, sorted case-insensitively."""
    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def package_origins() -> dict[str, str]:
    """Record the commit of every dependency installed from a repository."""
    origins: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        try:
            raw = dist.read_text("direct_url.json")
        except OSError:  # pragma: no cover - unreadable metadata
            continue
        if not raw:
            continue
        try:
            recorded = json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - malformed metadata
            continue
        commit = (recorded.get("vcs_info") or {}).get("commit_id")
        if commit:
            origins[name] = f"{recorded.get('url', '')}@{commit}"
    return dict(sorted(origins.items(), key=lambda item: item[0].lower()))


def git_state(cwd: Path) -> dict[str, Any] | None:
    """Capture the Git commit and dirty state, or ``None`` outside a repository."""
    if not (Path(cwd) / ".git").exists():
        return None

    def run_git(args: list[str]) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = run_git(["rev-parse", "HEAD"])
        status = run_git(["status", "--short"])
    except (OSError, subprocess.CalledProcessError):
        return None
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def environment_record() -> dict[str, Any]:
    """Collect the runtime environment description embedded in every manifest."""
    cwd = Path.cwd()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "cwd": str(cwd),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": thread_env_snapshot(),
        "git": git_state(cwd),
        "packages": installed_packages(),
        "package_origins": package_origins(),
    }


def write_manifest(
    manifest_path: Path,
    experiment: str,
    config: dict[str, Any],
    inputs: Iterable[Path] = (),
    outputs: Iterable[Path] = (),
    source_files: Iterable[Path] = (),
    status: str = "started",
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the JSON manifest describing one stage invocation."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment": experiment,
        "status": status,
        "config": make_jsonable(config),
        "config_fingerprint": config_fingerprint(config),
        "diagnostics": make_jsonable(diagnostics or {}),
        "source_files": [file_record(Path(path)) for path in source_files],
        "inputs": [file_record(Path(path)) for path in inputs],
        "outputs": [file_record(Path(path)) for path in outputs],
        "environment": environment_record(),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def artifact_files(root: Path) -> list[Path]:
    """Return every result file under a run root, in a stable order.

    Sidecars, the run manifest and an unfinished stage's parts are left out.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != RUN_MANIFEST_NAME
        and not path.name.endswith(SIDECAR_SUFFIXES)
        and not any(part.endswith(PARTS_SUFFIX) for part in path.parts)
    )


def write_run_manifest(
    run_root: Path,
    experiment: str,
    config: dict[str, Any],
    inputs: Iterable[Path] = (),
    status: str = "completed",
    diagnostics: dict[str, Any] | None = None,
) -> Path:
    """Record what an entry point left in its run root, and what produced it."""
    run_root = Path(run_root)
    path = run_root / RUN_MANIFEST_NAME
    write_manifest(
        path,
        experiment,
        config,
        inputs=inputs,
        outputs=artifact_files(run_root),
        status=status,
        diagnostics=diagnostics,
    )
    return path
