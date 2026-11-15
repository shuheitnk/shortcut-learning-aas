"""The feature stage: its completeness guard, fingerprint and shard records.

The stage is resumable, so the guard decides whether the parts on disk cover the
whole grid.
"""

from __future__ import annotations

import pathlib
from dataclasses import replace
from typing import cast

import pandas as pd
import pytest

from aas_shortcut.calculate_features import (
    FeatureStageConfig,
    IncompleteShardError,
    assert_complete,
)
from aas_shortcut.define_paths import Shard
from aas_shortcut.sample_monotone_transform import (
    HELD_OUT_DOMAIN,
    WARP_ID,
    WarpConfig,
)
from tests.conftest import make_features


@pytest.fixture
def config() -> FeatureStageConfig:
    """The configuration the synthetic feature table corresponds to."""
    return FeatureStageConfig(
        problem_class="bbob",
        dim=2,
        sampling_factor=50,
        num_reps=3,
        max_functions=1,
        max_instances=8,
        base_seed=42,
        scale="debug",
    )


@pytest.fixture
def shard(tmp_path) -> Shard:
    return Shard(tmp_path / "features_mabbob_2d.parquet")


class TestAssertComplete:
    def test_a_complete_shard_passes_through(self, config, shard):
        table = make_features()
        assert assert_complete(table, config, shard) is table

    def test_a_missing_instance_is_refused(self, config, shard):
        table = make_features()
        short = cast(pd.DataFrame, table[table["instance_id"] != 4])
        with pytest.raises(IncompleteShardError, match="incomplete"):
            assert_complete(short, config, shard)

    def test_a_missing_repetition_is_refused(self, config, shard):
        table = make_features()
        short = cast(pd.DataFrame, table[table["rep"] != 3])
        with pytest.raises(IncompleteShardError):
            assert_complete(short, config, shard)

    def test_an_empty_shard_is_refused(self, config, shard):
        """Zero features would otherwise make the expected row count zero too."""
        with pytest.raises(IncompleteShardError):
            assert_complete(make_features().iloc[:0], config, shard)

    def test_a_dropped_feature_is_not_an_incompleteness(self, config, shard):
        """The sanitizer removes a feature everywhere; the grid is still complete."""
        table = make_features()
        kept = cast(pd.DataFrame, table[table["feature"] != table["feature"].iloc[0]])
        assert assert_complete(kept, config, shard) is kept


class TestWarpFingerprint:
    """A shard measured under one transformation must not be served under another."""

    @staticmethod
    def config(**overrides):
        return FeatureStageConfig(
            problem_class="bbob",
            dim=2,
            sampling_factor=50,
            num_reps=1,
            max_functions=24,
            max_instances=5,
            base_seed=42,
            scale="debug",
            **overrides,
        )

    def test_the_default_hashes_as_it_did_before_the_warp_was_a_parameter(self):
        """The default warp is fingerprinted by the shared warp id."""
        assert self.config().as_dict()["transform"] == WARP_ID

    def test_another_family_is_another_configuration(self):
        a = self.config()
        b = self.config(warp=WarpConfig(family="power"))
        assert a.as_dict() != b.as_dict()

    def test_another_seed_domain_is_another_configuration(self):
        a = self.config()
        b = self.config(warp=WarpConfig(domain=HELD_OUT_DOMAIN))
        assert a.as_dict() != b.as_dict()


class TestDeepElaIsRequiredEarly:
    """A missing backend must be reported before the tuning is spent."""

    def test_validate_refuses_when_the_backend_is_absent(self, monkeypatch, config):
        import aas_shortcut.calculate_features as module
        from aas_shortcut.calculate_deep_ela_features import (
            Availability,
            DeepElaUnavailable,
        )

        monkeypatch.setattr(
            module,
            "availability",
            lambda: Availability(False, pathlib.Path("vendor/deep_ela"), "no checkout"),
        )
        wanted = replace(config, feature_sets=("ela", "deep-ela"))
        with pytest.raises(DeepElaUnavailable):
            wanted.validate()

    def test_the_handcrafted_set_never_asks_for_it(self, monkeypatch, config):
        import aas_shortcut.calculate_features as module

        def explode():
            raise AssertionError("availability() must not be consulted")

        monkeypatch.setattr(module, "availability", explode)
        config.validate()


class TestPflaccoIsThePinnedBuild:
    """Every handcrafted feature value depends on which pflacco is installed."""

    def test_the_installed_pflacco_is_the_commit_pyproject_pins(self):
        """The installed pflacco comes from the pinned commit.

        Deep-ELA's installer republishes itself as ``pflacco`` with the same version
        string, so only the wheel's recorded origin tells the two apart.
        """
        import json
        import tomllib
        from importlib import metadata

        pyproject = tomllib.loads(
            pathlib.Path("pyproject.toml").read_text(encoding="utf-8")
        )
        specifiers = [
            requirement
            for requirement in pyproject["project"]["dependencies"]
            if requirement.startswith("pflacco")
        ]
        assert len(specifiers) == 1, "pflacco must be pinned exactly once"
        pinned = specifiers[0].rsplit("@", maxsplit=1)[-1].strip()

        origin = metadata.distribution("pflacco").read_text("direct_url.json")
        assert origin is not None, (
            "the installed pflacco records no origin, so it did not come from the "
            "pinned commit; reinstall with `uv sync --locked`"
        )
        commit = json.loads(origin).get("vcs_info", {}).get("commit_id")
        assert commit == pinned, (
            f"pflacco is installed from {commit}, not the pinned {pinned}; every "
            "handcrafted feature value would differ. Reinstall with "
            "`uv sync --locked` and do not run Deep-ELA's own installer."
        )


class TestParallelismDoesNotChangeTheResult:
    """`task.n_jobs` splits jobs across processes and must do nothing else."""

    @pytest.mark.slow
    def test_two_workers_produce_the_same_shard_as_one(self, tmp_path):

        from aas_shortcut.calculate_features import run_feature_stage
        from aas_shortcut.define_paths import RunPaths

        # The smallest grid with something to split. `n_jobs` is not in
        # `as_dict()`, so both runs share one fingerprint.
        config = FeatureStageConfig(
            problem_class="bbob",
            dim=2,
            sampling_factor=50,
            num_reps=1,
            max_functions=2,
            max_instances=1,
            base_seed=42,
            scale="debug",
        )
        serial = run_feature_stage(
            config, RunPaths.from_root(tmp_path / "serial"), "fingerprint"
        )
        parallel = run_feature_stage(
            replace(config, n_jobs=2),
            RunPaths.from_root(tmp_path / "parallel"),
            "fingerprint",
        )
        pd.testing.assert_frame_equal(serial, parallel)


class TestTheShardRecordsWhatItDropped:
    """Every feature shard carries a manifest of the features the sanitizer dropped."""

    @pytest.mark.slow
    def test_a_manifest_is_written_beside_the_shard(self, tmp_path):
        import json

        from aas_shortcut.calculate_features import run_feature_stage
        from aas_shortcut.define_paths import RunPaths, manifest_path_for

        config = FeatureStageConfig(
            problem_class="bbob",
            dim=2,
            sampling_factor=50,
            num_reps=1,
            max_functions=1,
            max_instances=1,
            base_seed=42,
            scale="debug",
        )
        run_paths = RunPaths.from_root(tmp_path / "run")
        shard = run_paths.deployment_features(config.problem_class, config.dim)
        run_feature_stage(config, run_paths, "fingerprint", shard=shard)

        manifest = manifest_path_for(shard.data)
        assert manifest.exists(), "the deployment shard recorded nothing"
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        dropped = recorded["diagnostics"]["dropped_features"]
        assert set(dropped) == {"nonfinite", "unbalanced", "n_dropped"}
        assert dropped["n_dropped"] == len(
            set(dropped["nonfinite"]) | set(dropped["unbalanced"])
        )
