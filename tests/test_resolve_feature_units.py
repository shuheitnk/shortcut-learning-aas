"""Feature units: where their shards live, and the column check at resolution."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from aas_shortcut.calculate_features import FeatureStageConfig
from aas_shortcut.define_feature_columns import ELA_COLUMNS, UndeclaredColumnLoss
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.record_manifests import stage_fingerprint
from aas_shortcut.resolve_feature_units import (
    feature_table,
    feature_unit,
    resolve_feature_unit,
)
from aas_shortcut.resolve_stage_artifacts import REQUIRE_CACHED, StageStore

CONFIG = FeatureStageConfig(
    problem_class="bbob",
    dim=2,
    sampling_factor=50,
    num_reps=1,
    max_functions=1,
    max_instances=1,
    base_seed=42,
    # The label the column check reads: the paper's declaration.
    scale="full",
    feature_sets=("ela",),
)


def _shard(config_hash: str, features=ELA_COLUMNS) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "problem_class": "bbob",
                "function_id": 1,
                "instance_id": 1,
                "dimension": 2,
                "transform": transform,
                "feature_set": "ela",
                "feature_variant": variant,
                "rep": 1,
                "sample_seed": 7,
                "sample_size": 100,
                "feature": name,
                "value": 0.5,
                "scale": "full",
                "config_hash": config_hash,
            }
            for transform in ("pre", "post")
            for variant in ("value", "rank")
            for name in features
        ]
    )


def _populate(store: StageStore, dropped=("ela_meta.quad_simple.cond",), **shard):
    relative = feature_table(CONFIG, deployment=False)

    def write(paths, config_hash):
        write_table(_shard(config_hash, **shard), paths.root / relative, "features")
        manifest = {
            "diagnostics": {
                "dropped_features": {
                    "nonfinite": list(dropped),
                    "unbalanced": [],
                    "n_dropped": len(dropped),
                }
            }
        }
        (paths.root / f"{relative}.manifest.json").write_text(json.dumps(manifest))

    store.resolve(feature_unit(CONFIG), write)


class TestUnit:
    def test_a_feature_unit_keeps_its_stage_fingerprint(self):
        assert feature_unit(CONFIG).fingerprint == stage_fingerprint(CONFIG.as_dict())

    def test_the_two_shards_are_where_the_stage_writes_them(self):
        assert feature_table(CONFIG, deployment=False) == (
            "features/features_bbob_2d.parquet"
        )
        assert feature_table(CONFIG, deployment=True) == (
            "features/deployment_features_bbob_2d.parquet"
        )


class TestResolution:
    def test_a_unit_that_lost_only_the_declared_columns_is_served(self, tmp_path):
        store = StageStore.under(tmp_path)
        _populate(store)
        cached = StageStore.under(tmp_path, policy=REQUIRE_CACHED)
        artifact, table = resolve_feature_unit(cached, CONFIG, deployment=False)
        assert set(table["feature"]) == set(ELA_COLUMNS)
        assert artifact.path == store.directory(feature_unit(CONFIG))

    def test_a_unit_that_lost_a_paper_column_stops_the_run(self, tmp_path):
        store = StageStore.under(tmp_path)
        kept = tuple(name for name in ELA_COLUMNS if name != "ic.m0")
        _populate(store, dropped=("ela_meta.quad_simple.cond", "ic.m0"), features=kept)
        with pytest.raises(UndeclaredColumnLoss, match=r"ic\.m0"):
            resolve_feature_unit(
                StageStore.under(tmp_path, policy=REQUIRE_CACHED),
                CONFIG,
                deployment=False,
            )
