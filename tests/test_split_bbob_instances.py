"""The instance-level train/test splits of BBOB."""

from __future__ import annotations

import pytest

from aas_shortcut.split_bbob_instances import (
    MAX_INSTANCE,
    TEST_INSTANCES,
    TRAIN_INSTANCES,
    TUNING_INSTANCES,
    check_tuning_instances,
)


class TestRanges:
    """The cut the experiments share."""

    def test_the_two_blocks_are_disjoint(self):
        """No test instance is in the training block."""
        assert not set(TRAIN_INSTANCES) & set(TEST_INSTANCES)

    def test_the_grid_covers_both_blocks(self):
        both = (*TRAIN_INSTANCES, *TEST_INSTANCES)
        assert max(both) == MAX_INSTANCE
        assert sorted(both) == list(range(1, MAX_INSTANCE + 1))

    def test_the_held_out_block_is_the_smaller_one(self):
        """The held-out block, measured more than once, is the smaller one."""
        assert len(TEST_INSTANCES) < len(TRAIN_INSTANCES)

    def test_the_training_block_starts_where_the_test_block_ends(self):
        """The training block starts right after the test block."""
        assert min(TRAIN_INSTANCES) == max(TEST_INSTANCES) + 1

    def test_the_tuning_instance_is_not_on_the_test_side(self):
        """The default tuning instance lies in the test block, so the study moves it.

        A held-out problem must never help choose the parameters it is measured under.
        """
        from aas_shortcut.tune_algorithms import PairConditions

        assert set(PairConditions().instance_ids) <= set(TEST_INSTANCES)
        assert min(TRAIN_INSTANCES) not in set(PairConditions().instance_ids)


@pytest.mark.parametrize("role", ["train", "test"])
def test_neither_block_is_empty(role):
    block = TRAIN_INSTANCES if role == "train" else TEST_INSTANCES
    assert block


class TestTuningInstances:
    """The instance the portfolio is tuned on never lies in the held-out block."""

    def test_the_study_tunes_on_the_first_training_instance(self):
        assert (min(TRAIN_INSTANCES),) == TUNING_INSTANCES
        assert not set(TUNING_INSTANCES) & set(TEST_INSTANCES)

    def test_a_training_instance_is_accepted(self):
        assert check_tuning_instances([11]) == (11,)

    @pytest.mark.parametrize("leaked", [1, 10])
    def test_a_held_out_instance_is_rejected(self, leaked):
        with pytest.raises(ValueError, match="held-out block"):
            check_tuning_instances([11, leaked])

    def test_no_instance_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            check_tuning_instances([])

    def test_the_configured_default_is_the_study_tuning_instance(self):
        from tests.conftest import compose_config

        assert tuple(compose_config().task.tuning_instance_ids) == TUNING_INSTANCES

    def test_the_tuning_refuses_to_tune_on_the_held_out_block(self, run_paths):
        from aas_shortcut.resolve_tuning_stages import pair_tuning_unit
        from aas_shortcut.run_experiment_spec import build_context
        from tests.conftest import compose_config

        cfg = compose_config("preset=debug", "task.tuning_instance_ids=[1]")
        with pytest.raises(ValueError, match="held-out block"):
            pair_tuning_unit(build_context(cfg, run_paths))
