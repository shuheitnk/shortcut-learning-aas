"""Reading and writing the long-format result tables."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from aas_shortcut.define_paths import Shard
from aas_shortcut.define_table_schema import (
    SCHEMAS,
    TableSchema,
    conform,
    describe_violations,
)
from aas_shortcut.define_table_schema import validate_table as _validate_table

LOGGER = logging.getLogger(__name__)

PARQUET_ENGINE = "pyarrow"

# ``zstd`` is deterministic at a fixed level and about halves the size versus snappy.
PARQUET_COMPRESSION = "zstd"

# Encoded-key budget for the Arrow grouper, in bytes. ``Table.group_by`` uses 32-bit
# offsets and segfaults past 2 GiB rather than raising; half the limit leaves room
# for padding. A shard above it takes the pandas path, with the same result.
GROUPER_KEY_BUDGET = 1 << 30

# Part files one shard may accumulate before :func:`plan_flush_size` widens the
# flush cadence, since a resume opens every part file.
MAX_PART_FILES = 2000


def plan_flush_size(
    pending: int, save_every: int, max_parts: int = MAX_PART_FILES
) -> int:
    """Return how many finished jobs to buffer between flushes to disk."""
    floor = max(1, save_every)
    if pending <= 0 or max_parts <= 0:
        return floor
    return max(floor, -(-pending // max_parts))


def read_table(path: Path | str) -> pd.DataFrame:
    """Read one long-format parquet table; a missing table raises."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such table: {path}")
    return pd.read_parquet(path, engine=PARQUET_ENGINE)


def write_table(
    df: pd.DataFrame,
    path: Path | str,
    schema: TableSchema | str,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Conform a table to its schema, validate it and write it as parquet."""
    if isinstance(schema, str):
        schema = SCHEMAS[schema]

    conformed = conform(df, schema)
    if validate:
        violations = _validate_table(conformed, schema)
        if violations:
            raise ValueError(
                f"Refusing to write {path}: {len(violations)} schema "
                f"violation(s)\n{describe_violations(violations)}"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conformed.to_parquet(
        path,
        engine=PARQUET_ENGINE,
        compression=PARQUET_COMPRESSION,
        index=False,
    )
    return conformed


def _sort_key(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Sort a table by its key so two runs producing the same rows agree."""
    key = [column for column in schema.key_columns if column in df.columns]
    return df.sort_values(key, kind="stable").reset_index(drop=True)


@dataclass
class PartWriter:
    """Incremental builder for one long-format shard."""

    shard: Shard
    schema: TableSchema
    # Fingerprint of the configuration producing this shard.
    fingerprint: str | None = None
    # Recorded beside the shard for callers that reuse at a finer grain (e.g. each
    # member's ``config_id``). Not part of the cache key.
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _next_part: int = 0

    def __post_init__(self) -> None:
        """Resolve the part directory and continue numbering after a resume."""
        if isinstance(self.schema, str):  # pragma: no cover - defensive
            self.schema = SCHEMAS[self.schema]
        self._next_part = len(list(self._existing_parts()))

    @property
    def parts_dir(self) -> Path:
        """Directory holding this shard's part files."""
        return self.shard.data.with_suffix(".parts")

    @property
    def _fingerprint_path(self) -> Path:
        """Sidecar recording which configuration the parts belong to."""
        return self.parts_dir / "fingerprint.json"

    @property
    def _content_fingerprint_path(self) -> Path:
        """Sidecar recording which configuration the consolidated shard belongs to."""
        return self.shard.data.with_suffix(self.shard.data.suffix + ".content.json")

    def _existing_parts(self) -> Iterable[Path]:
        if not self.parts_dir.exists():
            return []
        return sorted(self.parts_dir.glob("part-*.parquet"))

    def _files_on_disk(self) -> list[Path]:
        """Every parquet file holding this shard's rows: its parts, then the shard."""
        files = list(self._existing_parts())
        if self.shard.exists():
            files.append(self.shard.data)
        return files

    def _stored_fingerprint(self) -> str | None:
        """Fingerprint recorded beside the parts, if any."""
        return self._read_parts_sidecar().get("config_fingerprint")

    def _read_content_sidecar(self) -> dict[str, Any]:
        """Return the finished shard's sidecar, or ``{}`` when there is none."""
        if not self._content_fingerprint_path.exists():
            return {}
        with self._content_fingerprint_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _read_parts_sidecar(self) -> dict[str, Any]:
        """Return the part directory's sidecar, or ``{}`` when there is none."""
        if not self._fingerprint_path.exists():
            return {}
        with self._fingerprint_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def reuse_or_reset(self) -> pd.DataFrame | None:
        """Return the cached table if it matches the fingerprint, else reset to None."""
        if self.fingerprint is None:
            return read_table(self.shard.data) if self.shard.exists() else None

        if self.shard.exists():
            recorded = self._read_content_sidecar().get("config_fingerprint")
            if recorded == self.fingerprint:
                return read_table(self.shard.data)
            LOGGER.info(
                "%s was produced by a different configuration; discarding it",
                self.shard.data,
            )
            self.discard()
            return None

        stored = self._stored_fingerprint()
        if self.parts_dir.exists() and stored != self.fingerprint:
            LOGGER.info(
                "%s holds parts from a different configuration; discarding them",
                self.parts_dir,
            )
            self.discard()
        return None

    def existing(self, columns: Sequence[str] | None = None) -> pd.DataFrame:
        """Return everything written so far, from parts or the final file."""
        wanted = list(columns) if columns is not None else None
        frames = [
            pd.read_parquet(part, engine=PARQUET_ENGINE, columns=wanted)
            for part in self._existing_parts()
        ]
        if self.shard.exists():
            frames.append(
                pd.read_parquet(self.shard.data, engine=PARQUET_ENGINE, columns=wanted)
            )
        if not frames:
            empty = pd.DataFrame(
                {column: [] for column in (wanted or self.schema.columns)}
            )
            return empty if wanted else conform(empty, self.schema)
        return pd.concat(frames, ignore_index=True)

    def append(self, rows: Sequence[dict] | pd.DataFrame) -> int:
        """Write one batch of rows as a new part file; an empty batch writes nothing."""
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        if df.empty:
            return 0

        # Whole-table checks (duplicate keys, row counts) wait for consolidation.
        conformed = conform(df, self.schema)
        conformed.to_parquet(
            self._claim_part_path(),
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION,
            index=False,
        )
        return len(conformed)

    def _claim_part_path(self) -> Path:
        """Reserve the next part file's path, creating the directory and sidecar."""
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        if self.fingerprint is not None and not self._fingerprint_path.exists():
            with self._fingerprint_path.open("w", encoding="utf-8") as handle:
                # So parts left by an interrupted run can still be checked and reused.
                json.dump(
                    {
                        "config_fingerprint": self.fingerprint,
                        "metadata": dict(self.metadata),
                    },
                    handle,
                )
        path = self.parts_dir / f"part-{self._next_part:06d}.parquet"
        self._next_part += 1
        return path

    def _arrow_parts(self) -> pa.Table | None:
        """Read every part into one Arrow table, or ``None`` if there are none."""
        parts = self._files_on_disk()
        if not parts:
            return None
        # Lets a part written before a schema column was added still concatenate.
        return pa.concat_tables(
            [pq.read_table(part) for part in parts], promote_options="default"
        )

    def _restamp(self, table: pd.DataFrame) -> pd.DataFrame:
        """Stamp every row with this writer's fingerprint."""
        if self.fingerprint is None or "config_hash" not in self.schema.columns:
            return table
        if table.empty or "config_hash" not in table.columns:
            return table

        stale = table["config_hash"].astype("string") != self.fingerprint
        n_stale = int(stale.sum())
        if not n_stale:
            return table

        LOGGER.info(
            "%s: restamping %d row(s) written under an earlier fingerprint %s to %s",
            self.shard.data.name,
            n_stale,
            sorted(set(table.loc[stale, "config_hash"].dropna().astype(str))),
            self.fingerprint,
        )
        return table.assign(config_hash=self.fingerprint)

    def _restamp_arrow(self, arrow: pa.Table) -> pa.Table:
        """:meth:`_restamp` for the Arrow path, which never leaves Arrow."""
        if self.fingerprint is None or "config_hash" not in arrow.schema.names:
            return arrow

        index = arrow.schema.get_field_index("config_hash")
        column = arrow.column(index)
        distinct = {
            str(value) for value in column.unique().to_pylist() if value is not None
        }
        if distinct == {self.fingerprint}:
            return arrow

        LOGGER.info(
            "%s: restamping row(s) written under an earlier fingerprint %s to %s",
            self.shard.data.name,
            sorted(distinct - {self.fingerprint}),
            self.fingerprint,
        )
        stamp = pa.array(
            [self.fingerprint] * arrow.num_rows, type=arrow.schema.field(index).type
        )
        return arrow.set_column(index, arrow.schema.field(index), stamp)

    def _deduplicate(self, table: pd.DataFrame) -> pd.DataFrame:
        """Collapse rows that repeat a key, provided they agree on everything else."""
        key = [column for column in self.schema.key_columns if column in table.columns]
        if not key or table.empty:
            return table

        repeated = table.duplicated(subset=key, keep=False)
        n_repeated = int(repeated.sum())
        if not n_repeated:
            return table

        # Collapsing on the full row leaves one copy per key iff every copy agreed.
        collapsed = cast(pd.DataFrame, table[repeated].drop_duplicates())
        if len(collapsed) != len(
            cast(pd.DataFrame, collapsed.drop_duplicates(subset=key))
        ):
            LOGGER.warning(
                "%s: %d row(s) repeat a key with conflicting values; leaving them "
                "for validation to reject",
                self.shard.data.name,
                n_repeated,
            )
            return table

        LOGGER.info(
            "%s: dropping %d duplicate row(s) left by an interrupted run; the "
            "repeats agree value for value",
            self.shard.data.name,
            n_repeated - len(collapsed),
        )
        return pd.concat(
            [cast(pd.DataFrame, table[~repeated]), collapsed], ignore_index=True
        )

    def _has_repeated_keys(self, arrow: pa.Table) -> bool:
        """Whether any key appears more than once, decided without leaving Arrow."""
        key = [
            column for column in self.schema.key_columns if column in arrow.schema.names
        ]
        if not key or arrow.num_rows == 0:
            return False
        encoded = sum(arrow.column(column).nbytes for column in key)
        if encoded > GROUPER_KEY_BUDGET:
            LOGGER.info(
                "%s: %d row(s) of key encode to about %.1f GiB, over what the Arrow "
                "grouper addresses; consolidating through pandas instead",
                self.shard.data.name,
                arrow.num_rows,
                encoded / (1 << 30),
            )
            return True
        counts = arrow.group_by(key).aggregate([(key[0], "count")])
        column = counts.column(f"{key[0]}_count")
        return bool(pc.max(column).as_py() > 1)

    def consolidate(
        self,
        *,
        validate: bool = True,
        prepare: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Merge the parts into the final shard and remove the part directory."""
        # `prepare` needs the pandas path, so skip reading into Arrow then.
        arrow = self._arrow_parts() if prepare is None else None
        # An interrupted run can leave a unit of work in two parts; only the pandas
        # path can collapse them.
        if arrow is not None and self._has_repeated_keys(arrow):
            del arrow
            arrow = None
        if arrow is None:
            merged = self._deduplicate(self._restamp(self.existing()))
            merged = _sort_key(merged, self.schema)
            if prepare is not None:
                merged = _sort_key(prepare(merged), self.schema)
            written = write_table(
                merged, self.shard.data, self.schema, validate=validate
            )
        else:
            arrow = self._restamp_arrow(arrow)
            names = set(arrow.schema.names)
            key = [column for column in self.schema.key_columns if column in names]
            if key:
                arrow = arrow.sort_by([(column, "ascending") for column in key])
            self.shard.data.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                arrow,
                self.shard.data,
                compression=PARQUET_COMPRESSION,
            )
            del arrow
            # Read back: the file is the artifact, and this avoids a second copy.
            written = read_table(self.shard.data)
            if validate:
                violations = _validate_table(written, self.schema)
                if violations:
                    raise ValueError(
                        f"Refusing to keep {self.shard.data}: {len(violations)} "
                        f"schema violation(s)\n{describe_violations(violations)}"
                    )

        if self.parts_dir.exists():
            shutil.rmtree(self.parts_dir)
            self._next_part = 0
        if self.fingerprint is not None:
            with self._content_fingerprint_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "config_fingerprint": self.fingerprint,
                        "metadata": dict(self.metadata),
                    },
                    handle,
                )
        LOGGER.info("consolidated %s (%d rows)", self.shard.data, len(written))
        return written

    def discard(self) -> None:
        """Delete any partial state, used when a stage restarts from scratch."""
        self._content_fingerprint_path.unlink(missing_ok=True)
        if self.parts_dir.exists():
            shutil.rmtree(self.parts_dir)
        self.shard.data.unlink(missing_ok=True)
        self._next_part = 0
