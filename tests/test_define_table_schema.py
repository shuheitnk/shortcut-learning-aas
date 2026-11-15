"""Long-format table schemas and their validator."""

from __future__ import annotations

import numpy as np
import pytest

from aas_shortcut.define_table_schema import (
    SCHEMAS,
    conform,
    describe_violations,
    validate_table,
)


class TestConform:
    def test_applies_order_and_dtypes(self, features):
        shuffled = features[list(reversed(features.columns))]
        conformed = conform(shuffled, SCHEMAS["features"])
        assert list(conformed.columns) == SCHEMAS["features"].column_order
        assert str(conformed["function_id"].dtype) == "int32"

    def test_missing_column_raises(self, features):
        with pytest.raises(ValueError, match="missing column"):
            conform(features.drop(columns=["rep"]), SCHEMAS["features"])

    def test_extra_column_raises(self, features):
        with pytest.raises(ValueError, match="unexpected column"):
            conform(features.assign(extra=1), SCHEMAS["features"])


class TestValidateTable:
    def test_expected_row_count(self, features):
        assert validate_table(features, "features", expected_rows=len(features)) == []
        violations = validate_table(
            features, "features", expected_rows=len(features) + 1
        )
        assert any(v.kind == "row_count" for v in violations)

    def test_required_values(self, features):
        violations = validate_table(
            features, "features", required_values={"rep": [1, 2, 3, 99]}
        )
        assert any(v.kind == "incomplete" for v in violations)

    def test_describe_is_multiline(self, features):
        broken = features.copy()
        broken.loc[0, "value"] = np.nan
        text = describe_violations(validate_table(broken, "features"))
        assert "[features]" in text

    def test_every_schema_key_is_a_column(self):
        for schema in SCHEMAS.values():
            assert set(schema.key_columns) <= set(schema.columns)
            assert set(schema.finite_columns) <= set(schema.columns)
            assert set(schema.domains) <= set(schema.columns)
