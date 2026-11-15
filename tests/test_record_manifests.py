"""Tests for manifests and fingerprints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aas_shortcut import record_manifests as repro
from aas_shortcut.configure_runtime import (
    THREAD_ENV_VARS,
    configure_runtime,
    thread_env_snapshot,
)


class TestConfigureRuntime:
    def test_sets_thread_limits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in THREAD_ENV_VARS:
            monkeypatch.delenv(name, raising=False)
        configure_runtime()
        for name in THREAD_ENV_VARS:
            assert thread_env_snapshot()[name] == "1"

    def test_existing_values_are_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator override from the shell must win."""
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        configure_runtime()
        assert thread_env_snapshot()["OMP_NUM_THREADS"] == "8"

    def test_snapshot_covers_hash_seed(self) -> None:
        assert "PYTHONHASHSEED" in thread_env_snapshot()


class TestMakeJsonable:
    def test_range_is_summarised_not_expanded(self) -> None:
        """A 1000-instance sweep must not produce a 1000-element manifest entry."""
        assert repro.make_jsonable(range(1, 1001)) == {
            "start": 1,
            "stop": 1001,
            "step": 1,
        }

    def test_path_becomes_a_string(self) -> None:
        assert isinstance(repro.make_jsonable(Path("a/b")), str)

    def test_tuple_becomes_a_list(self) -> None:
        assert repro.make_jsonable((1, 2)) == [1, 2]

    def test_set_is_sorted(self) -> None:
        assert repro.make_jsonable({3, 1, 2}) == [1, 2, 3]

    def test_dict_keys_are_sorted_and_stringified(self) -> None:
        assert list(repro.make_jsonable({"b": 1, "a": 2})) == ["a", "b"]

    def test_numpy_scalars_become_python_scalars(self) -> None:
        value = repro.make_jsonable(np.float64(1.5))
        assert value == 1.5
        assert type(value) is float

    def test_nested_structures_are_converted(self) -> None:
        converted = repro.make_jsonable({"runs": range(1, 3), "out": Path("x")})
        assert converted == {"out": "x", "runs": {"start": 1, "stop": 3, "step": 1}}

    def test_output_is_json_serialisable(self) -> None:
        payload = {"a": range(1, 3), "b": Path("p"), "c": {1, 2}, "d": np.int64(4)}
        json.dumps(repro.make_jsonable(payload))


class TestConfigFingerprint:
    def test_key_order_does_not_matter(self) -> None:
        assert repro.config_fingerprint({"a": 1, "b": 2}) == repro.config_fingerprint(
            {"b": 2, "a": 1}
        )

    def test_value_change_changes_the_fingerprint(self) -> None:
        assert repro.config_fingerprint({"dim": 2}) != repro.config_fingerprint(
            {"dim": 10}
        )

    def test_added_key_changes_the_fingerprint(self) -> None:
        assert repro.config_fingerprint({"dim": 2}) != repro.config_fingerprint(
            {"dim": 2, "seed": 1}
        )

    def test_equivalent_ranges_agree(self) -> None:
        assert repro.config_fingerprint(
            {"runs": range(1, 5)}
        ) == repro.config_fingerprint({"runs": range(1, 5)})

    def test_different_ranges_disagree(self) -> None:
        assert repro.config_fingerprint(
            {"runs": range(1, 5)}
        ) != repro.config_fingerprint({"runs": range(1, 6)})

    def test_is_a_sha256_hex_digest(self) -> None:
        fingerprint = repro.config_fingerprint({"a": 1})
        assert len(fingerprint) == 64
        assert set(fingerprint) <= set("0123456789abcdef")


class TestFileDigests:
    def test_sha256_matches_a_known_value(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.csv"
        # Bytes, not text: write_text() would use the platform's line ending.
        path.write_bytes(b"a,b,c\n1,2,3\n")
        assert (
            repro.sha256_file(path)
            == "9284ed4fd7fe1346904656f329db6cc49c0e7ae5b8279bff37f96bc6eb59baad"
        )

    def test_identical_content_hashes_alike(self, tmp_path: Path) -> None:
        (tmp_path / "a").write_bytes(b"x" * 5_000_000)
        (tmp_path / "b").write_bytes(b"x" * 5_000_000)
        assert repro.sha256_file(tmp_path / "a") == repro.sha256_file(tmp_path / "b")

    def test_file_record_of_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        path.write_text("hello")
        record = repro.file_record(path)
        assert record["exists"] is True
        assert record["size_bytes"] == 5
        assert len(record["sha256"]) == 64

    def test_file_record_of_missing_file(self, tmp_path: Path) -> None:
        record = repro.file_record(tmp_path / "missing.csv")
        assert record["exists"] is False
        assert "sha256" not in record


class TestEnvironmentRecord:
    def test_contains_the_expected_sections(self) -> None:
        record = repro.environment_record()
        for key in ("timestamp_utc", "command", "cwd", "python", "platform"):
            assert key in record
        assert "environment" in record
        assert "packages" in record

    def test_records_this_project(self) -> None:
        assert "shortcut-learning-aas" in repro.environment_record()["packages"]

    def test_git_state_is_none_outside_a_repository(self, tmp_path: Path) -> None:
        assert repro.git_state(tmp_path) is None


class TestWriteManifest:
    def test_writes_a_readable_manifest(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "out" / "x.manifest.json"
        data = tmp_path / "data.csv"
        data.write_text("a\n1\n")

        repro.write_manifest(
            manifest_path,
            "feature_extraction",
            {"dim": 2, "runs": range(1, 3)},
            outputs=[data],
            source_files=[data],
            status="finished",
        )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["experiment"] == "feature_extraction"
        assert manifest["status"] == "finished"
        assert manifest["schema_version"] == repro.MANIFEST_SCHEMA_VERSION
        assert manifest["config"]["dim"] == 2
        assert manifest["outputs"][0]["exists"] is True
        assert manifest["source_files"][0]["sha256"]

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "deep" / "nested" / "m.json"
        repro.write_manifest(manifest_path, "x", {})
        assert manifest_path.exists()

    def test_fingerprint_matches_the_config(self, tmp_path: Path) -> None:
        config = {"dim": 10, "seed": 42}
        repro.write_manifest(tmp_path / "m.json", "x", config)
        manifest = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert manifest["config_fingerprint"] == repro.config_fingerprint(config)

    def test_missing_outputs_are_recorded_not_fatal(self, tmp_path: Path) -> None:
        """A `started` manifest is written before the outputs exist."""
        repro.write_manifest(
            tmp_path / "m.json", "x", {}, outputs=[tmp_path / "later.csv"]
        )
        manifest = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert manifest["outputs"][0]["exists"] is False


class TestPackageOrigins:
    """Which build of a dependency produced an artifact, where it can be told."""

    def test_the_pinned_dependency_is_recorded_with_its_commit(self) -> None:
        origins = repro.package_origins()
        assert "pflacco" in origins, (
            "pflacco is pinned to a commit and must be recorded as such; a build "
            "installed any other way would change every handcrafted feature"
        )
        assert "@" in origins["pflacco"]

    def test_no_origin_carries_a_local_path(self) -> None:
        """No origin records a local path, such as the project's install directory."""
        for name, origin in repro.package_origins().items():
            assert not origin.startswith("file:"), f"{name} records a local path"
            assert origin.startswith(("http://", "https://", "git+")), name

    def test_the_environment_record_carries_them(self) -> None:
        assert "package_origins" in repro.environment_record()


class TestRunManifest:
    """What an entry point leaves in its run root is traceable to what made it."""

    def _root(self, tmp_path: Path) -> Path:
        root = tmp_path / "experiment3"
        (root / "metrics").mkdir(parents=True)
        (root / "figures").mkdir()
        (root / "metrics" / "view_selection.parquet").write_bytes(b"table")
        (root / "figures" / "fig16.png").write_bytes(b"figure")
        (root / "metrics" / "view_selection.parquet.content.json").write_text("{}")
        (root / "metrics" / "grid.parquet.manifest.json").write_text("{}")
        parts = root / "metrics" / "unfinished.parts"
        parts.mkdir()
        (parts / "part-000000.parquet").write_bytes(b"scratch")
        return root

    def test_every_result_file_is_listed(self, tmp_path: Path) -> None:
        root = self._root(tmp_path)
        names = [path.name for path in repro.artifact_files(root)]
        assert names == ["fig16.png", "view_selection.parquet"]

    def test_sidecars_and_scratch_parts_are_not_results(self, tmp_path: Path) -> None:
        root = self._root(tmp_path)
        listed = {path.name for path in repro.artifact_files(root)}
        assert not {name for name in listed if name.endswith(".json")}
        assert "part-000000.parquet" not in listed

    def test_the_manifest_records_a_digest_of_each_result(self, tmp_path: Path) -> None:
        root = self._root(tmp_path)
        path = repro.write_run_manifest(root, "experiment3", {"seed": 42})
        manifest = json.loads(path.read_text(encoding="utf-8"))
        digests = {
            Path(entry["path"]).name: entry["sha256"] for entry in manifest["outputs"]
        }
        assert digests["view_selection.parquet"] == repro.sha256_file(
            root / "metrics" / "view_selection.parquet"
        )
        assert manifest["config_fingerprint"] == repro.config_fingerprint({"seed": 42})
        assert manifest["status"] == "completed"

    def test_a_rewrite_does_not_list_the_previous_manifest(
        self, tmp_path: Path
    ) -> None:
        root = self._root(tmp_path)
        repro.write_run_manifest(root, "experiment3", {})
        path = repro.write_run_manifest(root, "experiment3", {})
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert all(
            Path(entry["path"]).name != path.name for entry in manifest["outputs"]
        )

    def test_a_missing_root_lists_nothing(self, tmp_path: Path) -> None:
        assert repro.artifact_files(tmp_path / "absent") == []
