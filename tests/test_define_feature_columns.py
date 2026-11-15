"""The paper's fixed column lists, and the three checks that keep them true.

Test 1 matches the declaration against the calculators; test 2 is the check run
when a feature unit is resolved; test 3 is the record of non-finite observations
that test 2 reads.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.calculate_deep_ela_features import DeepElaConfig
from aas_shortcut.calculate_features import (
    FeatureDiagnostics,
    compute_ela_features,
    sanitize_features,
)
from aas_shortcut.create_sampled_points import create_sampled_points, ela_inputs
from aas_shortcut.define_feature_columns import (
    DEEP_ELA_COLUMNS,
    DEEP_ELA_MODEL,
    ELA_COLUMNS,
    EXCLUDED_BY_SCALE,
    EXCLUDED_COLUMNS,
    UndeclaredColumnLoss,
    check_feature_unit,
    excluded_columns,
    paper_columns,
)
from aas_shortcut.define_paths import DEEP_ELA_FEATURE_SET, ELA_FEATURE_SET
from tests.conftest import deep_ela_feature_names


def _shard(features: dict[str, list[str]]) -> pd.DataFrame:
    """A feature table holding each representation's named columns once."""
    return pd.DataFrame(
        [
            {"feature_set": feature_set, "feature": name, "value": 0.0}
            for feature_set, names in features.items()
            for name in names
        ]
    )


def _diagnostics(nonfinite=(), unbalanced=()) -> dict:
    return {
        "dropped_features": {
            "nonfinite": list(nonfinite),
            "unbalanced": list(unbalanced),
            "n_dropped": len(set(nonfinite) | set(unbalanced)),
        }
    }


class TestDeclaration:
    """Test 1: the declaration is what the calculators produce, minus two columns."""

    def test_the_lists_hold_59_and_48_columns(self):
        assert len(ELA_COLUMNS) == len(set(ELA_COLUMNS)) == 59
        assert len(DEEP_ELA_COLUMNS) == len(set(DEEP_ELA_COLUMNS)) == 48

    def test_the_ela_classes_are_the_documented_counts(self):
        """The 59 ELA columns split into the documented class counts."""
        counts = Counter(name.split(".")[0] for name in ELA_COLUMNS)
        assert counts == {
            "disp": 16,
            "ela_level": 9,
            "ela_meta": 8,
            "pca": 8,
            "fitness_distance": 6,
            "nbc": 5,
            "ic": 4,
            "ela_distr": 3,
        }

    def test_the_ela_calculators_produce_the_declared_and_the_excluded_columns(
        self,
    ):
        sample = create_sampled_points(42, "bbob", 1, 1, 2, 1, 100)
        X_df, y_series = ela_inputs(sample, "pre", "value")
        produced = set(compute_ela_features(X_df, y_series, seed=0))
        assert produced == set(ELA_COLUMNS) | set(EXCLUDED_COLUMNS)
        assert not set(ELA_COLUMNS) & set(EXCLUDED_COLUMNS)

    def test_deep_ela_produces_the_declared_coordinates(self):
        names = deep_ela_feature_names(DeepElaConfig(model=DEEP_ELA_MODEL))
        assert sorted(names) == list(DEEP_ELA_COLUMNS)

    def test_the_lists_are_in_the_order_the_matrices_read(self):
        assert list(ELA_COLUMNS) == sorted(ELA_COLUMNS)
        assert list(DEEP_ELA_COLUMNS) == sorted(DEEP_ELA_COLUMNS)

    def test_paper_columns_merge_the_representations_by_name(self):
        both = paper_columns([ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET])
        assert both == tuple(sorted({*ELA_COLUMNS, *DEEP_ELA_COLUMNS}))
        assert paper_columns([ELA_FEATURE_SET]) == ELA_COLUMNS

    def test_an_unknown_representation_is_rejected(self):
        with pytest.raises(ValueError, match="feature set"):
            paper_columns(["handmade"])


class TestCheckAtResolution:
    """Test 2: a shard may lose the two declared columns and nothing else."""

    def test_losing_only_the_declared_columns_passes(self):
        shard = _shard({ELA_FEATURE_SET: list(ELA_COLUMNS)})
        check_feature_unit(shard, _diagnostics(nonfinite=EXCLUDED_COLUMNS), "power")

    def test_a_shard_that_still_carries_an_excluded_column_passes(self):
        """The training shard keeps `ic.eps_ratio`; the paper just does not read it."""
        shard = _shard({ELA_FEATURE_SET: [*ELA_COLUMNS, "ic.eps_ratio"]})
        diagnostics = _diagnostics(nonfinite=["ela_meta.quad_simple.cond"])
        check_feature_unit(shard, diagnostics, "training")

    def test_a_paper_column_lost_as_non_finite_is_an_error(self):
        kept = [name for name in ELA_COLUMNS if name != "ic.m0"]
        shard = _shard({ELA_FEATURE_SET: kept})
        diagnostics = _diagnostics(nonfinite=["ela_meta.quad_simple.cond", "ic.m0"])
        with pytest.raises(UndeclaredColumnLoss, match=r"ic\.m0"):
            check_feature_unit(shard, diagnostics, "knots-dense")

    def test_a_paper_column_lost_as_unbalanced_is_an_error(self):
        shard = _shard({ELA_FEATURE_SET: list(ELA_COLUMNS)})
        with pytest.raises(UndeclaredColumnLoss, match=r"pca\.expl_var\.cor_x"):
            check_feature_unit(
                shard, _diagnostics(unbalanced=["pca.expl_var.cor_x"]), "spline"
            )

    def test_a_paper_column_absent_without_a_record_is_an_error(self):
        kept = [name for name in DEEP_ELA_COLUMNS if not name.endswith(".X7")]
        shard = _shard({DEEP_ELA_FEATURE_SET: kept})
        with pytest.raises(UndeclaredColumnLoss, match=r"X7"):
            check_feature_unit(shard, _diagnostics(), "power")

    def test_each_representation_is_checked_against_its_own_list(self):
        shard = _shard(
            {
                ELA_FEATURE_SET: list(ELA_COLUMNS),
                DEEP_ELA_FEATURE_SET: list(DEEP_ELA_COLUMNS),
            }
        )
        check_feature_unit(shard, _diagnostics(), "both")


class TestNonFiniteObservations:
    """Test 3: every non-finite observation is recorded, not only its column."""

    @staticmethod
    def _table() -> pd.DataFrame:
        rows = []
        for instance in (1, 2):
            for transform in ("pre", "post"):
                for name in ("ic.m0", "ela_meta.quad_simple.cond"):
                    rows.append(
                        {
                            "problem_class": "bbob",
                            "function_id": 3,
                            "instance_id": instance,
                            "dimension": 10,
                            "transform": transform,
                            "feature_set": ELA_FEATURE_SET,
                            "feature_variant": "value",
                            "rep": 1,
                            "feature": name,
                            "value": 1.0,
                        }
                    )
        table = pd.DataFrame(rows)
        cond = table["feature"] == "ela_meta.quad_simple.cond"
        table.loc[cond & (table["instance_id"] == 2), "value"] = math.inf
        table.loc[
            (table["feature"] == "ic.m0")
            & (table["instance_id"] == 1)
            & (table["transform"] == "post"),
            "value",
        ] = np.nan
        return table

    def test_each_failing_observation_is_named(self):
        diagnostics = FeatureDiagnostics()
        cleaned = sanitize_features(self._table(), diagnostics)
        assert set(cleaned["feature"]) == set()
        recorded = diagnostics.as_dict()["nonfinite_observations"]
        assert recorded["ic.m0"] == {
            "n_observations": 1,
            "observations": [
                {
                    "problem_class": "bbob",
                    "function_id": 3,
                    "instance_id": 1,
                    "dimension": 10,
                    "transform": "post",
                    "feature_set": ELA_FEATURE_SET,
                    "feature_variant": "value",
                    "rep": 1,
                    "value": "nan",
                }
            ],
        }
        cond = recorded["ela_meta.quad_simple.cond"]
        assert cond["n_observations"] == 2
        assert {(o["transform"], o["value"]) for o in cond["observations"]} == {
            ("pre", "inf"),
            ("post", "inf"),
        }

    def test_the_record_leaves_the_dropped_columns_as_they_were(self):
        diagnostics = FeatureDiagnostics()
        sanitize_features(self._table(), diagnostics)
        dropped = diagnostics.as_dict()["dropped_features"]
        assert dropped == {
            "nonfinite": ["ela_meta.quad_simple.cond", "ic.m0"],
            "unbalanced": [],
            "n_dropped": 2,
        }

    def test_the_recorded_loss_stops_the_resolution(self):
        """What test 2 reads is what this record wrote."""
        diagnostics = FeatureDiagnostics()
        sanitize_features(self._table(), diagnostics)
        kept = _shard({ELA_FEATURE_SET: [n for n in ELA_COLUMNS if n != "ic.m0"]})
        with pytest.raises(UndeclaredColumnLoss, match=r"ic\.m0"):
            check_feature_unit(kept, diagnostics.as_dict(), "synthetic")

    def test_a_finite_table_records_nothing(self):
        table = self._table()
        table["value"] = 1.0
        diagnostics = FeatureDiagnostics()
        assert len(sanitize_features(table, diagnostics)) == len(table)
        assert diagnostics.as_dict()["nonfinite_observations"] == {}


class TestScales:
    """The debug scale follows the same rule with its own, declared, result."""

    def test_every_scale_excludes_columns_the_calculators_produce(self):
        produced = set(ELA_COLUMNS) | set(EXCLUDED_COLUMNS)
        for scale, excluded in EXCLUDED_BY_SCALE.items():
            assert set(excluded) <= produced, scale
            ela = paper_columns([ELA_FEATURE_SET], scale)
            assert set(ela) | set(excluded) == produced, scale
            assert not set(ela) & set(excluded), scale

    def test_the_paper_scale_is_the_declared_59(self):
        assert paper_columns([ELA_FEATURE_SET], "full") == ELA_COLUMNS

    def test_the_debug_scale_reads_57_columns_with_the_quadratic_condition(self):
        """At 100 points in two dimensions the condition number stays finite."""
        debug = paper_columns([ELA_FEATURE_SET], "debug")
        assert len(debug) == 57
        assert "ela_meta.quad_simple.cond" in debug
        assert not {"ela_level.lda_qda_10", "ic.eps_ratio"} & set(debug)
        assert paper_columns([DEEP_ELA_FEATURE_SET], "debug") == DEEP_ELA_COLUMNS

    def test_a_scale_without_a_declaration_cannot_be_read(self):
        with pytest.raises(UndeclaredColumnLoss, match="smoke"):
            excluded_columns("smoke")

    def test_the_check_reads_the_scale_it_is_given(self):
        debug = list(paper_columns([ELA_FEATURE_SET], "debug"))
        shard = _shard({ELA_FEATURE_SET: debug})
        diagnostics = _diagnostics(
            nonfinite=["ela_level.lda_qda_25", "ela_level.lda_qda_50"]
        )
        check_feature_unit(shard, diagnostics, "debug training", "debug")
        with pytest.raises(UndeclaredColumnLoss, match="lda_qda_25"):
            check_feature_unit(shard, diagnostics, "debug training", "full")
