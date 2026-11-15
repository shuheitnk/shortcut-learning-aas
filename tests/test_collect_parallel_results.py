"""The convention every parallel stage's results come back through."""

from __future__ import annotations

import pytest

from aas_shortcut.collect_parallel_results import as_rows


class TestAsRows:
    def test_a_generator_is_materialised_in_submission_order(self):
        """A generator comes back as a list in submission order."""
        assert as_rows(index for index in range(5)) == [0, 1, 2, 3, 4]

    def test_a_list_comes_back_unchanged(self):
        rows = [{"a": 1}, {"a": 2}]
        assert as_rows(rows) == rows

    def test_an_empty_result_is_an_empty_list_rather_than_an_error(self):
        """An empty result is an empty list, not an error."""
        assert as_rows([]) == []

    def test_none_is_rejected(self):
        """A None result (a failed joblib dispatch) is rejected where it happens."""
        with pytest.raises(TypeError, match="dispatch failed"):
            as_rows(None)

    def test_a_non_iterable_is_rejected(self):
        with pytest.raises(TypeError, match="expected an iterable"):
            as_rows(42)
