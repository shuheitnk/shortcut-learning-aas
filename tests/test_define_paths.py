"""The experiment vocabulary and the artifact layout."""

from __future__ import annotations

import pytest

from aas_shortcut.define_paths import (
    CONCAT_VARIANT,
    FEATURE_ENCODINGS,
    FEATURE_VARIANTS,
    PROBLEM_CLASSES,
    RUN_MANIFEST_NAME,
    SCALES,
    TRANSFORMS,
    RunPaths,
    Shard,
    fingerprint_variant,
    fingerprint_variants,
    manifest_path_for,
    problem_slug,
    representation_path,
    run_root_name,
    shard_stem,
    validate_encoding,
    validate_feature_variant,
    validate_problem_class,
    validate_transform,
)


class TestVocabulary:
    def test_transforms_are_pre_and_post(self):
        assert TRANSFORMS == ("pre", "post")

    def test_concat_is_not_a_stored_encoding(self):
        """The concatenation is assembled at analysis time."""
        assert "concat" not in FEATURE_ENCODINGS
        assert validate_feature_variant("concat") == "concat"

    def test_concat_is_the_third_variant(self):
        assert FEATURE_VARIANTS == ("value", "rank", CONCAT_VARIANT)

    def test_the_scales_are_the_debug_and_the_papers(self):
        assert SCALES == ("debug", "full")

    @pytest.mark.parametrize(
        ("validator", "bad"),
        [
            (validate_transform, "middle"),
            (validate_encoding, "concat"),
            (validate_feature_variant, "ordinal"),
            (validate_problem_class, "cec"),
        ],
    )
    def test_unknown_values_are_rejected(self, validator, bad):
        with pytest.raises(ValueError, match="expected one of"):
            validator(bad)


class TestPathLayout:
    def test_shards_never_collide(self, run_paths: RunPaths):
        paths = {
            shard(problem_class, dim).data
            for shard in (run_paths.features, run_paths.view_superiority)
            for problem_class in PROBLEM_CLASSES
            for dim in (2, 10)
        }
        assert len(paths) == 4  # 2 shard kinds x 2 dimensions

    def test_stem_encodes_suite_and_dimension(self):
        assert shard_stem("features", "bbob", 10) == "features_bbob_10d"

    def test_slug_is_filename_safe(self):
        assert "-" not in problem_slug("bbob")

    def test_manifest_sits_beside_the_data(self, run_paths: RunPaths):
        shard = run_paths.features("bbob", 2)
        assert manifest_path_for(shard.data).parent == shard.data.parent

    def test_mkdirs_creates_every_directory(self, run_paths: RunPaths):
        for directory in (
            run_paths.features_dir,
            run_paths.performance_dir,
            run_paths.superiority_dir,
            run_paths.mean_superiority_dir,
            run_paths.metrics_dir,
            run_paths.figures_dir,
        ):
            assert directory.is_dir()


class TestShard:
    def test_exists_reflects_the_data_file(self, tmp_path):
        shard = Shard(tmp_path / "x.parquet")
        assert not shard.exists()
        shard.data.write_bytes(b"")
        assert shard.exists()


class TestRunRootName:
    """A rehearsal writes beside the reported artifacts, never over them."""

    def test_the_reported_scale_keeps_the_bare_name(self):
        assert run_root_name("experiment3", "full", "full") == "experiment3"

    @pytest.mark.parametrize("preset", ["debug"])
    def test_every_other_scale_is_suffixed_with_its_preset(self, preset):
        assert run_root_name("experiment3", preset, "full") == f"experiment3_{preset}"

    def test_no_two_scales_share_a_root(self):
        roots = {run_root_name("experiment3", preset, "full") for preset in SCALES}
        assert len(roots) == len(SCALES)

    def test_the_run_manifest_is_not_mistaken_for_a_shard_manifest(self):
        """Artifact validation reads every ``*.manifest.json`` as one stage's."""
        assert not RUN_MANIFEST_NAME.endswith(".manifest.json")


class TestRepresentationPath:
    def test_handcrafted_ela_keeps_the_bare_name(self, tmp_path):
        assert (
            representation_path(tmp_path / "board.csv", "ela") == tmp_path / "board.csv"
        )

    def test_deep_ela_sits_beside_it(self, tmp_path):
        assert representation_path(tmp_path / "board.csv", "deep-ela").name == (
            "board_deep-ela.csv"
        )

    def test_an_unknown_representation_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="feature set"):
            representation_path(tmp_path / "board.csv", "pooled")


class TestFingerprintSpelling:
    """Fingerprints spell ``concat`` as ``combined``.

    A fingerprint is a measurement's identity; a new spelling would orphan artifacts.
    """

    def test_the_pinned_spelling_is_the_historical_one(self):
        assert fingerprint_variant(CONCAT_VARIANT) == "combined"

    def test_the_other_variants_are_untouched(self):
        assert fingerprint_variant("value") == "value"
        assert fingerprint_variant("rank") == "rank"

    def test_a_sequence_keeps_its_order(self):
        assert fingerprint_variants(FEATURE_VARIANTS) == ["value", "rank", "combined"]
