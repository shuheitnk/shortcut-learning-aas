"""The parquet I/O layer: round trips, part files and the stale-state guard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut import read_write_tables
from aas_shortcut.define_paths import RunPaths
from aas_shortcut.define_table_schema import SCHEMAS
from aas_shortcut.read_write_tables import PartWriter, read_table, write_table


class TestTableIO:
    def test_round_trips_exactly(self, tmp_path, features):
        path = tmp_path / "features.parquet"
        write_table(features, path, "features")
        pd.testing.assert_frame_equal(read_table(path), features)

    def test_refuses_to_write_an_invalid_table(self, tmp_path, features):
        broken = features.copy()
        broken.loc[0, "value"] = np.inf
        with pytest.raises(ValueError, match="schema violation"):
            write_table(broken, tmp_path / "bad.parquet", "features")
        assert not (tmp_path / "bad.parquet").exists()

    def test_missing_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_table(tmp_path / "absent.parquet")


class TestPartWriter:
    @pytest.fixture
    def writer(self, run_paths: RunPaths) -> PartWriter:
        return PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])

    def test_empty_writer_returns_a_typed_empty_table(self, writer):
        existing = writer.existing()
        assert existing.empty
        assert list(existing.columns) == SCHEMAS["features"].column_order

    def test_parts_accumulate_then_consolidate(self, writer, features):
        half = len(features) // 2
        writer.append(features.iloc[:half])
        writer.append(features.iloc[half:])

        assert writer.parts_dir.exists()
        assert len(writer.existing()) == len(features)

        consolidated = writer.consolidate()
        assert len(consolidated) == len(features)
        assert writer.shard.exists()
        assert not writer.parts_dir.exists()

    def test_resume_continues_part_numbering(self, run_paths, features):
        first = PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])
        first.append(features.iloc[:10])

        resumed = PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])
        resumed.append(features.iloc[10:20])

        assert len(resumed.existing()) == 20
        assert len(list(resumed.parts_dir.glob("part-*.parquet"))) == 2

    def test_empty_batch_writes_nothing(self, writer):
        assert writer.append([]) == 0
        assert not writer.parts_dir.exists()

    def test_consolidation_is_order_independent(self, run_paths, features):
        forward = PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])
        forward.append(features.iloc[:20])
        forward.append(features.iloc[20:])
        expected = forward.consolidate()

        backward = PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])
        backward.append(features.iloc[20:])
        backward.append(features.iloc[:20])
        actual = backward.consolidate()

        pd.testing.assert_frame_equal(expected, actual)

    def test_prepare_runs_before_validation(self, writer, features):
        broken = features.copy()
        broken.loc[broken["feature"] == broken["feature"].iloc[0], "value"] = np.nan
        writer.append(broken)

        from aas_shortcut.calculate_features import drop_nonfinite_features

        consolidated = writer.consolidate(prepare=drop_nonfinite_features)
        assert len(consolidated) < len(broken)
        assert np.isfinite(consolidated["value"].to_numpy()).all()

    def test_discard_removes_everything(self, writer, features):
        writer.append(features)
        writer.consolidate()
        writer.discard()
        assert not writer.shard.exists()
        assert not writer.parts_dir.exists()


class TestStaleStateGuard:
    """A changed configuration must never be served old results."""

    def _writer(self, run_paths, fingerprint):
        return PartWriter(
            run_paths.features("bbob", 2),
            SCHEMAS["features"],
            fingerprint=fingerprint,
        )

    def test_a_matching_finished_shard_is_reused(self, run_paths, features):
        """`consolidate` stamps its own content fingerprint; no manifest is needed."""
        writer = self._writer(run_paths, "abc")
        writer.append(features)
        writer.consolidate()

        cached = self._writer(run_paths, "abc").reuse_or_reset()
        assert cached is not None
        assert len(cached) == len(features)

    def test_a_stale_finished_shard_is_discarded(self, run_paths, features):
        writer = self._writer(run_paths, "abc")
        writer.append(features)
        writer.consolidate()

        fresh = self._writer(run_paths, "different")
        assert fresh.reuse_or_reset() is None
        assert not fresh.shard.exists()
        assert fresh.existing().empty

    def test_a_shard_without_a_content_fingerprint_is_discarded(
        self, run_paths, features
    ):
        """A shard without its fingerprint sidecar (as after a crash) is not reused."""
        writer = self._writer(run_paths, "abc")
        writer.append(features)
        writer.consolidate()
        writer._content_fingerprint_path.unlink()

        fresh = self._writer(run_paths, "abc")
        assert fresh.reuse_or_reset() is None
        assert not fresh.shard.exists()

    def test_matching_parts_are_kept_for_resume(self, run_paths, features):
        writer = self._writer(run_paths, "abc")
        writer.append(features.iloc[:10])

        resumed = self._writer(run_paths, "abc")
        assert resumed.reuse_or_reset() is None
        assert len(resumed.existing()) == 10

    def test_stale_parts_are_discarded(self, run_paths, features):
        writer = self._writer(run_paths, "abc")
        writer.append(features.iloc[:10])

        fresh = self._writer(run_paths, "different")
        assert fresh.reuse_or_reset() is None
        assert fresh.existing().empty

    def test_without_a_fingerprint_nothing_is_discarded(self, run_paths, features):
        writer = self._writer(run_paths, "abc")
        writer.append(features)
        writer.consolidate()

        plain = PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])
        assert plain.reuse_or_reset() is not None


class TestConfigHashStamp:
    """A shard's rows must agree with the fingerprint its sidecar records.

    A grown shard reuses rows stamped with an older ``config_hash``; without a
    restamp the merged table describes two configurations and fails the schema check.
    """

    def _writer(self, run_paths, fingerprint):
        return PartWriter(
            run_paths.features("bbob", 2),
            SCHEMAS["features"],
            fingerprint=fingerprint,
        )

    def test_growing_a_shard_with_stale_stamps_still_writes(self, run_paths, features):
        """500 reused rows plus 500 new ones end up under one fingerprint."""
        stale = features.assign(config_hash="an-earlier-fingerprint")
        self._writer(run_paths, "shape").append(stale)
        self._writer(run_paths, "shape").consolidate()

        grown = self._writer(run_paths, "shape")
        grown.append(features.assign(instance_id=features["instance_id"] + 100))

        consolidated = grown.consolidate()
        assert set(consolidated["config_hash"]) == {"shape"}
        assert len(consolidated) == 2 * len(features)

    def test_the_pandas_path_restamps_too(self, run_paths, features):
        """`prepare` forces the in-memory path; both must hold the invariant."""
        writer = self._writer(run_paths, "shape")
        writer.append(features.assign(config_hash="an-earlier-fingerprint"))

        consolidated = writer.consolidate(prepare=lambda table: table)
        assert set(consolidated["config_hash"]) == {"shape"}

    def test_the_rows_agree_with_the_sidecar(self, run_paths, features):
        writer = self._writer(run_paths, "shape")
        writer.append(features.assign(config_hash="an-earlier-fingerprint"))
        consolidated = writer.consolidate()

        recorded = self._writer(run_paths, "shape")._read_content_sidecar()
        assert set(consolidated["config_hash"]) == {recorded["config_fingerprint"]}

    def test_an_unfingerprinted_writer_leaves_the_column_alone(
        self, run_paths, features
    ):
        """Without a fingerprint there is no authority to restamp towards."""
        writer = PartWriter(run_paths.features("bbob", 2), SCHEMAS["features"])
        writer.append(features.assign(config_hash="whatever-the-caller-set"))

        consolidated = writer.consolidate()
        assert set(consolidated["config_hash"]) == {"whatever-the-caller-set"}


class TestDuplicateRecovery:
    """A part directory holding the same work twice must still consolidate.

    Every unit is seeded from its own coordinates, so a repeat is bit-identical; a
    repeat that disagrees is a defect and must still reach the schema check.
    """

    def _writer(self, run_paths, fingerprint="shape"):
        return PartWriter(
            run_paths.features("bbob", 2),
            SCHEMAS["features"],
            fingerprint=fingerprint,
        )

    def test_identical_repeats_are_collapsed(self, run_paths, features):
        writer = self._writer(run_paths)
        writer.append(features)
        writer.append(features)

        consolidated = writer.consolidate()
        assert len(consolidated) == len(features)
        assert not consolidated.duplicated(
            subset=list(SCHEMAS["features"].key_columns)
        ).any()

    def test_the_collapsed_values_are_the_originals(self, run_paths, features):
        writer = self._writer(run_paths)
        writer.append(features)
        writer.append(features)

        key = list(SCHEMAS["features"].key_columns)
        consolidated = writer.consolidate().sort_values(key).reset_index(drop=True)
        expected = features.sort_values(key).reset_index(drop=True)
        pd.testing.assert_series_equal(
            consolidated["value"], expected["value"], check_dtype=False
        )

    def test_conflicting_repeats_still_fail(self, run_paths, features):
        """Two values under one key is a defect, not an interrupted run."""
        writer = self._writer(run_paths)
        writer.append(features)
        writer.append(features.assign(value=features["value"] + 1.0))

        with pytest.raises(ValueError, match="duplicate_key"):
            writer.consolidate()

    def test_the_pandas_path_collapses_too(self, run_paths, features):
        writer = self._writer(run_paths)
        writer.append(features)
        writer.append(features)

        consolidated = writer.consolidate(prepare=lambda table: table)
        assert len(consolidated) == len(features)


class TestGrouperLimits:
    """The duplicate-key check must not take the process down.

    Past 2 GiB, pyarrow 15's ``Table.group_by`` overflows its 32-bit offsets and dies
    on SIGSEGV, so large tables are checked with pandas instead.
    """

    def _writer(self, run_paths) -> PartWriter:
        return PartWriter(
            run_paths.features("bbob", 2), SCHEMAS["features"], fingerprint="shape"
        )

    def test_prepare_never_reads_the_parts_into_arrow(self, run_paths, features):
        writer = self._writer(run_paths)
        writer.append(features)

        def refuse(self):  # pragma: no cover - the point is that it is not called
            raise AssertionError("the Arrow path is unreachable when prepare is given")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PartWriter, "_arrow_parts", refuse)
            consolidated = writer.consolidate(prepare=lambda table: table)
        assert len(consolidated) == len(features)

    def test_an_oversized_key_falls_back_to_pandas(
        self, run_paths, features, monkeypatch
    ):
        monkeypatch.setattr(read_write_tables, "GROUPER_KEY_BUDGET", 0)
        writer = self._writer(run_paths)
        writer.append(features)
        writer.append(features)

        consolidated = writer.consolidate()
        assert len(consolidated) == len(features)
        assert not consolidated.duplicated(
            subset=list(SCHEMAS["features"].key_columns)
        ).any()


class TestFlushSize:
    """How often a stage checkpoints, and how many files that leaves behind."""

    def test_small_stages_keep_the_cadence_they_asked_for(self):
        assert read_write_tables.plan_flush_size(50_000, 25) == 25

    def test_a_huge_stage_widens_rather_than_leaving_a_file_per_25_jobs(self):
        # The full-scale MA-BBOB benchmark: 50 000 instances x 31 trials x 6 members.
        flush = read_write_tables.plan_flush_size(9_300_000, 25)
        assert flush > 25
        assert -(-9_300_000 // flush) <= read_write_tables.MAX_PART_FILES

    def test_the_cadence_is_never_narrowed(self):
        assert read_write_tables.plan_flush_size(10, 1000) == 1000

    def test_nothing_pending_is_not_a_division_by_zero(self):
        assert read_write_tables.plan_flush_size(0, 25) == 25
