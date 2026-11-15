"""Tests for the repetitions a reported number is the median over.

A cell's number is the median over eleven regressor seeds: an odd count, with
repetition 0 the study's own run, and no other index accepted.
"""

from __future__ import annotations

import pytest

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.define_repetitions import (
    MODEL_SEEDS,
    REPETITIONS,
    model_for,
    repetitions,
)


class TestTheRepetitions:
    def test_there_is_an_odd_number_of_them(self):
        """So a cell's median is one repetition's own value, not an interpolation."""
        assert REPETITIONS % 2 == 1

    def test_the_first_is_the_run_the_study_already_measured(self):
        assert repetitions()[0] == 0
        assert model_for(ModelConfig(), 0) == ModelConfig()

    def test_each_repetition_moves_the_regressor(self):
        assert len(set(MODEL_SEEDS)) == REPETITIONS

    @pytest.mark.parametrize("index", [-1, REPETITIONS, REPETITIONS + 5])
    def test_an_index_outside_the_study_is_refused(self, index):
        with pytest.raises(ValueError, match="not one of"):
            model_for(ModelConfig(), index)
