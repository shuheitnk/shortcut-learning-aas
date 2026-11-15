"""The stage store: fingerprints, resolution, completion, resumption and the lock."""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import pandas as pd
import pytest

from aas_shortcut.define_paths import STAGE_STORE_NAME, RunPaths
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.record_manifests import stage_fingerprint
from aas_shortcut.resolve_stage_artifacts import (
    COMPUTED,
    MISSING,
    PLAN,
    REQUIRE_CACHED,
    REUSED,
    STAGE_LIST_NAME,
    STAGE_RECORD_NAME,
    MissingStageArtifact,
    StageRecordError,
    StageStore,
    StageUnit,
    unit_fingerprint,
)

SETTINGS = {"stage": "toy", "budget": 100, "functions": [1, 2]}


def _rows(config_hash: str, values=(0.25, 0.75)) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "problem_class": "bbob",
            "function_id": [1, 2],
            "instance_id": 1,
            "dimension": 2,
            "transform": "pre",
            "pair": "a__b",
            "algorithm_a": "a",
            "algorithm_b": "b",
            "pair_type": "G1",
            "n_trials": 3,
            "superiority": list(values),
            "scale": "debug",
            "config_hash": config_hash,
        }
    )


def _write_toy(paths: RunPaths, config_hash: str) -> None:
    write_table(_rows(config_hash), paths.metric("toy").data, "view_superiority")


class TestFingerprint:
    def test_a_unit_that_reads_nothing_keeps_its_stage_fingerprint(self):
        assert unit_fingerprint(SETTINGS) == stage_fingerprint(SETTINGS)
        assert StageUnit("toy", SETTINGS).fingerprint == stage_fingerprint(SETTINGS)

    def test_every_setting_moves_the_fingerprint(self):
        base = unit_fingerprint(SETTINGS)
        for key, value in (("budget", 101), ("functions", [1, 3]), ("stage", "x")):
            assert unit_fingerprint({**SETTINGS, key: value}) != base

    def test_an_input_chains_its_fingerprint_into_the_unit(self):
        alone = unit_fingerprint(SETTINGS)
        chained = unit_fingerprint(SETTINGS, {"performance": "0123456789abcdef"})
        moved = unit_fingerprint(SETTINGS, {"performance": "fedcba9876543210"})
        assert len({alone, chained, moved}) == 3

    def test_the_order_inputs_are_named_in_does_not_matter(self):
        one = unit_fingerprint(SETTINGS, {"a": "1", "b": "2"})
        other = unit_fingerprint(SETTINGS, {"b": "2", "a": "1"})
        assert one == other

    def test_provenance_is_recorded_but_never_fingerprinted(self):
        plain = StageUnit("toy", SETTINGS)
        traced = StageUnit("toy", SETTINGS, provenance={"tune_pair": "c6588116"})
        assert plain.fingerprint == traced.fingerprint


class TestResolve:
    def test_a_missing_unit_is_computed_and_then_reused(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS, provenance={"source": "test"})
        calls = []

        def compute(paths: RunPaths, config_hash: str) -> None:
            calls.append(config_hash)
            _write_toy(paths, config_hash)

        first = store.resolve(unit, compute)
        second = store.resolve(unit, compute)
        assert calls == [unit.fingerprint]
        assert first.path == second.path == store.directory(unit)
        assert first.path == tmp_path.resolve() / STAGE_STORE_NAME / "toy" / (
            unit.fingerprint
        )
        assert [entry["action"] for entry in store.resolved] == [COMPUTED, REUSED]

    def test_the_rows_carry_the_unit_fingerprint(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS)
        artifact = store.resolve(unit, _write_toy)
        table = artifact.read_table("metrics/toy.parquet")
        assert set(table["config_hash"]) == {unit.fingerprint}

    def test_the_record_describes_the_unit(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit(
            "toy",
            SETTINGS,
            inputs={"performance": "0123456789abcdef"},
            provenance={"tuning": "c6588116a8aaf6f2"},
        )
        artifact = store.resolve(unit, _write_toy)
        record = json.loads((artifact.path / STAGE_RECORD_NAME).read_text())
        assert record["stage"] == "toy"
        assert record["fingerprint"] == unit.fingerprint
        assert record["settings"] == SETTINGS
        assert record["inputs"] == {"performance": "0123456789abcdef"}
        assert record["provenance"] == {"tuning": "c6588116a8aaf6f2"}
        assert record["origin"] == COMPUTED
        assert set(record["files"]) == {"metrics/toy.parquet"}
        assert len(record["files"]["metrics/toy.parquet"]["sha256"]) == 64

    def test_no_scratch_is_left_once_the_unit_is_complete(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS)
        store.resolve(unit, _write_toy)
        assert not store.partial(unit).exists()

    @pytest.mark.parametrize("policy", [REQUIRE_CACHED, PLAN])
    def test_a_run_that_may_not_compute_stops_at_a_missing_unit(self, tmp_path, policy):
        store = StageStore.under(tmp_path, policy=policy)
        unit = StageUnit("toy", SETTINGS)
        with pytest.raises(MissingStageArtifact, match=unit.fingerprint):
            store.resolve(unit, _write_toy)
        assert not store.partial(unit).exists()
        assert store.resolved[-1]["action"] == MISSING

    def test_a_run_that_may_not_compute_still_reads_what_exists(self, tmp_path):
        unit = StageUnit("toy", SETTINGS)
        StageStore.under(tmp_path).resolve(unit, _write_toy)
        cached = StageStore.under(tmp_path, policy=REQUIRE_CACHED)
        assert cached.resolve(unit, _write_toy).path == cached.directory(unit)

    def test_an_unknown_policy_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="policy"):
            StageStore.under(tmp_path, policy="sometimes")

    def test_an_interrupted_computation_resumes_from_its_scratch(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS)
        seen = []

        def interrupted(paths: RunPaths, config_hash: str) -> None:
            (paths.root / "half.txt").write_text("first half")
            raise KeyboardInterrupt

        def resumed(paths: RunPaths, config_hash: str) -> None:
            seen.append((paths.root / "half.txt").read_text())
            _write_toy(paths, config_hash)

        with pytest.raises(KeyboardInterrupt):
            store.resolve(unit, interrupted)
        assert store.partial(unit).exists()
        artifact = store.resolve(unit, resumed)
        assert seen == ["first half"]
        assert set(artifact.record["files"]) == {"half.txt", "metrics/toy.parquet"}

    def test_unfinished_parts_never_become_a_unit(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS)

        def leaves_parts(paths: RunPaths, config_hash: str) -> None:
            parts = paths.root / "metrics" / "toy.parts"
            parts.mkdir(parents=True)
            (parts / "part-000000.parquet").write_bytes(b"rows")

        with pytest.raises(StageRecordError, match="unfinished"):
            store.resolve(unit, leaves_parts)
        assert not store.directory(unit).exists()

    def test_a_computation_that_writes_nothing_is_an_error(self, tmp_path):
        store = StageStore.under(tmp_path)
        with pytest.raises(StageRecordError, match="no file"):
            store.resolve(StageUnit("toy", SETTINGS), lambda paths, config_hash: None)

    def test_a_damaged_unit_is_refused_rather_than_read(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS)
        artifact = store.resolve(unit, _write_toy)
        (artifact.path / "metrics" / "toy.parquet").write_bytes(b"truncated")
        with pytest.raises(StageRecordError, match="size"):
            store.lookup(unit)

    def test_a_record_for_other_settings_is_refused(self, tmp_path):
        store = StageStore.under(tmp_path)
        unit = StageUnit("toy", SETTINGS)
        artifact = store.resolve(unit, _write_toy)
        path = artifact.path / STAGE_RECORD_NAME
        record = json.loads(path.read_text())
        record["settings"]["budget"] = 7
        path.write_text(json.dumps(record))
        with pytest.raises(StageRecordError, match="settings"):
            store.lookup(unit)

    def test_a_file_the_unit_does_not_hold_is_not_served(self, tmp_path):
        artifact = StageStore.under(tmp_path).resolve(
            StageUnit("toy", SETTINGS), _write_toy
        )
        with pytest.raises(FileNotFoundError, match="records"):
            artifact.file("metrics/other.parquet")


class TestList:
    def test_the_run_root_lists_every_unit_it_resolved(self, tmp_path):
        store = StageStore.under(tmp_path / "outputs")
        one = StageUnit("toy", SETTINGS)
        two = StageUnit("toy", {**SETTINGS, "budget": 7})
        store.resolve(one, _write_toy)
        store.resolve(two, _write_toy)
        store.resolve(one, _write_toy)
        path = store.write_list(tmp_path / "outputs" / "experiment2")
        assert path.name == STAGE_LIST_NAME
        listed = json.loads(path.read_text())["units"]
        assert [(u["fingerprint"], u["action"]) for u in listed] == [
            (one.fingerprint, COMPUTED),
            (two.fingerprint, COMPUTED),
            (one.fingerprint, REUSED),
        ]


def _resolve_slowly(root: str, log: str) -> str:
    """Resolve the toy unit, taking long enough for a second process to queue."""

    def compute(paths: RunPaths, config_hash: str) -> None:
        with Path(log).open("a") as handle:
            handle.write("computed\n")
        time.sleep(0.5)
        _write_toy(paths, config_hash)

    artifact = StageStore.under(root).resolve(StageUnit("toy", SETTINGS), compute)
    return str(artifact.path)


def _resolve_into(root: str, log: str, queue) -> None:
    queue.put(_resolve_slowly(root, log))


class TestLock:
    def test_two_processes_compute_one_unit_once(self, tmp_path):
        context = multiprocessing.get_context("fork")
        log = tmp_path / "calls.log"
        queue = context.Queue()
        workers = [
            context.Process(target=_resolve_into, args=(str(tmp_path), str(log), queue))
            for _ in range(2)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0
        paths = {queue.get(timeout=5) for _ in workers}
        assert log.read_text().splitlines() == ["computed"]
        assert len(paths) == 1


class TestResolveTogether:
    """Several units of one stage, the missing ones computed in one call."""

    @staticmethod
    def _units() -> list[StageUnit]:
        cells = ("a", "b", "c")
        return [StageUnit("toy", {**SETTINGS, "cell": cell}) for cell in cells]

    def test_only_the_missing_units_are_computed_and_all_are_returned(self, tmp_path):
        store = StageStore.under(tmp_path)
        units = self._units()
        store.resolve(units[1], _write_toy)
        handed = []

        def compute(missing, scratches):
            handed.append(sorted(unit.fingerprint for unit in missing))
            for unit in missing:
                _write_toy(scratches[unit.fingerprint], unit.fingerprint)

        artifacts = store.resolve_together(units, compute)
        assert handed == [sorted(units[i].fingerprint for i in (0, 2))]
        assert set(artifacts) == {unit.fingerprint for unit in units}
        actions = [entry["action"] for entry in store.resolved[-3:]]
        assert actions == [COMPUTED, REUSED, COMPUTED]

    def test_nothing_is_computed_when_every_unit_exists(self, tmp_path):
        store = StageStore.under(tmp_path)
        units = self._units()
        for unit in units:
            store.resolve(unit, _write_toy)

        def compute(missing, scratches):
            raise AssertionError("nothing should be computed")

        assert len(store.resolve_together(units, compute)) == 3

    def test_a_run_that_may_not_compute_stops_before_computing_any(self, tmp_path):
        units = self._units()
        StageStore.under(tmp_path).resolve(units[0], _write_toy)
        cached = StageStore.under(tmp_path, policy=REQUIRE_CACHED)
        with pytest.raises(MissingStageArtifact, match="2 toy unit"):
            cached.resolve_together(units, lambda missing, scratches: None)
        assert not any(cached.partial(unit).exists() for unit in units)
