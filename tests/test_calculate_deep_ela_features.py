"""Tests for the Deep-ELA feature representation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.build_feature_matrix import build_feature_matrix, feature_columns
from aas_shortcut.calculate_deep_ela_features import (
    DEFAULT_MODEL,
    MODEL_CAPACITY,
    MODEL_FEATURE_COUNT,
    DeepElaConfig,
    availability,
    is_deep_ela_feature,
)
from aas_shortcut.calculate_features import (
    FeatureStageConfig,
    RankInvarianceError,
    features_for_sample,
)
from aas_shortcut.classify_feature_invariance import is_structurally_invariant
from aas_shortcut.create_sampled_points import create_sampled_points, ela_inputs
from aas_shortcut.define_paths import DEEP_ELA_FEATURE_SET, ELA_FEATURE_SET
from tests.conftest import (
    DEEP_FEATURE_NAMES,
    FEATURE_NAMES,
    deep_ela_feature_names,
    make_features,
)

requires_backend = pytest.mark.skipif(
    not availability().available,
    reason="Deep-ELA backend not installed (uv sync, then install_deep_ela)",
)


class TestModelChoice:
    """Only the large family admits the study's dimensionality."""

    def test_the_default_model_takes_ten_dimensions(self):
        DeepElaConfig().validate(dim=10)

    def test_the_capacity_is_exactly_the_boundary(self):
        capacity = MODEL_CAPACITY[DEFAULT_MODEL]
        DeepElaConfig().validate(dim=capacity - 1)
        with pytest.raises(ValueError):
            DeepElaConfig().validate(dim=capacity)

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="unknown Deep-ELA model"):
            DeepElaConfig(model="enormous_99d_v9").validate(dim=2)

    def test_the_commit_is_part_of_the_fingerprint(self):
        """Different weights are different features; the cache must see that."""
        assert "commit" in DeepElaConfig().as_dict()


class TestFeatureNaming:
    def test_learned_and_handcrafted_names_never_collide(self):
        assert all(is_deep_ela_feature(name) for name in deep_ela_feature_names())
        assert not any(is_deep_ela_feature(name) for name in FEATURE_NAMES)

    def test_no_learned_feature_is_structurally_invariant(self):
        """The structural count is a handcrafted-only claim."""
        assert not any(
            is_structurally_invariant(name) for name in deep_ela_feature_names()
        )


class TestFeatureStageConfig:
    def _config(self, **overrides) -> FeatureStageConfig:
        base = {
            "problem_class": "bbob",
            "dim": 10,
            "sampling_factor": 50,
            "num_reps": 3,
            "max_functions": 3,
            "max_instances": 2,
            "base_seed": 42,
            "scale": "debug",
        }
        return FeatureStageConfig(**{**base, **overrides})

    def test_the_learned_settings_are_fingerprinted_only_when_requested(self):
        handcrafted = self._config().as_dict()
        both = self._config(
            feature_sets=(ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET)
        ).as_dict()
        assert "deep_ela" not in handcrafted
        assert "deep_ela" in both

    def test_the_feature_sets_are_part_of_the_fingerprint(self):
        assert self._config().as_dict()["feature_sets"] == [ELA_FEATURE_SET]

    def test_an_unknown_feature_set_is_refused(self):
        with pytest.raises(ValueError, match="feature set"):
            self._config(feature_sets=("shallow-ela",)).validate()

    def test_no_feature_set_is_refused(self):
        with pytest.raises(ValueError, match="at least one feature set"):
            self._config(feature_sets=()).validate()

    def test_a_dimension_too_large_for_the_model_is_refused(self):
        with pytest.raises(ValueError, match="d \\+ m"):
            self._config(
                feature_sets=(DEEP_ELA_FEATURE_SET,),
                dim=MODEL_CAPACITY[DEFAULT_MODEL],
            ).validate()


class TestFeatureSetAxis:
    """A matrix must hold one representation, never a silent mixture."""

    @pytest.fixture
    def both(self) -> pd.DataFrame:
        return make_features(
            n_instances=4,
            n_reps=2,
            feature_sets=(ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET),
        )

    def test_each_set_yields_only_its_own_features(self, both):
        handcrafted = build_feature_matrix(
            both, "pre", "value", feature_set=ELA_FEATURE_SET
        )
        learned = build_feature_matrix(
            both, "pre", "value", feature_set=DEEP_ELA_FEATURE_SET
        )
        assert len(feature_columns(handcrafted)) == len(FEATURE_NAMES)
        assert len(feature_columns(learned)) == len(DEEP_FEATURE_NAMES)
        assert not set(feature_columns(handcrafted)) & set(feature_columns(learned))

    def test_concat_stays_within_one_representation(self, both):
        concat = build_feature_matrix(
            both, "pre", "concat", feature_set=DEEP_ELA_FEATURE_SET
        )
        assert len(feature_columns(concat)) == 2 * len(DEEP_FEATURE_NAMES)

    def test_an_absent_set_is_an_error_not_an_empty_matrix(self, both):
        handcrafted_only = both[both["feature_set"] == ELA_FEATURE_SET]
        with pytest.raises(ValueError, match="deep-ela"):
            build_feature_matrix(
                handcrafted_only, "pre", "value", feature_set=DEEP_ELA_FEATURE_SET
            )


@pytest.fixture(scope="module")
def conditions() -> dict:
    """All four conditions of one real sample, at the study's dimensionality."""
    sample = create_sampled_points(42, "bbob", 1, 1, 10, 1, 500)
    from aas_shortcut.calculate_deep_ela_features import compute_deep_ela_features

    return {
        (transform, encoding): compute_deep_ela_features(
            *ela_inputs(sample, transform, encoding), seed=sample.sample_seed
        )
        for transform in ("pre", "post")
        for encoding in ("value", "rank")
    }


@requires_backend
class TestPretrainedModel:
    """The claims that can only be made by running the real model."""

    def _vector(self, conditions, key) -> np.ndarray:
        names = sorted(conditions[key])
        return np.array([conditions[key][name] for name in names])

    def test_it_emits_the_declared_number_of_features(self, conditions):
        assert len(conditions[("pre", "value")]) == MODEL_FEATURE_COUNT[DEFAULT_MODEL]

    def test_the_rank_features_are_bit_identical_across_the_transform(self, conditions):
        """Test 1, for the learned representation."""
        pre = self._vector(conditions, ("pre", "rank"))
        post = self._vector(conditions, ("post", "rank"))
        assert np.array_equal(pre, post)

    def test_the_value_features_do_move(self, conditions):
        """Otherwise the comparison would have nothing to measure."""
        pre = self._vector(conditions, ("pre", "value"))
        post = self._vector(conditions, ("post", "value"))
        assert not np.allclose(pre, post)

    def test_the_two_encodings_differ(self, conditions):
        value = self._vector(conditions, ("pre", "value"))
        rank = self._vector(conditions, ("pre", "rank"))
        assert not np.allclose(value, rank)

    def test_every_feature_is_finite(self, conditions):
        for values in conditions.values():
            assert np.isfinite(np.array(list(values.values()))).all()

    def test_the_same_seed_reproduces_the_same_features(self):
        from aas_shortcut.calculate_deep_ela_features import compute_deep_ela_features

        sample = create_sampled_points(42, "bbob", 2, 1, 5, 1, 200)
        X, y = ela_inputs(sample, "pre", "value")
        first = compute_deep_ela_features(X, y, seed=7)
        second = compute_deep_ela_features(X, y, seed=7)
        assert first == second

    def test_the_stage_computes_both_sets_from_one_sample(self):
        sample = create_sampled_points(42, "bbob", 1, 1, 5, 1, 200)
        rows = features_for_sample(
            sample, feature_sets=(ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET)
        )
        table = pd.DataFrame(rows)
        assert set(table["feature_set"]) == {ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET}
        # One shared sample, so every condition reports the same sample.
        assert table["sample_seed"].nunique() == 1
        assert table["sample_size"].nunique() == 1

    def test_the_inline_invariance_check_covers_the_learned_set(self, monkeypatch):
        """A learned rank feature that moved must stop the pipeline too."""
        import aas_shortcut.calculate_features as extraction

        real = extraction.compute_deep_ela_features
        state = {"calls": 0}

        def drifting(X, y, seed, config=None):
            state["calls"] += 1
            values = real(X, y, seed, config)
            if state["calls"] == 4:  # the (post, rank) condition
                return {name: value + 1.0 for name, value in values.items()}
            return values

        monkeypatch.setattr(extraction, "compute_deep_ela_features", drifting)
        sample = create_sampled_points(42, "bbob", 1, 1, 5, 1, 200)
        with pytest.raises(RankInvarianceError, match="deep-ela"):
            features_for_sample(sample, feature_sets=(DEEP_ELA_FEATURE_SET,))
