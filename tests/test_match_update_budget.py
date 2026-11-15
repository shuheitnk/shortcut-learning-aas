"""Whether the two training conditions take the same number of parameter updates.

The augmented condition carries every problem twice, so its epochs are halved. The
two then differ by at most one step per epoch, as RealMLP-TD drops partial batches.
"""

from __future__ import annotations

import pytest

from aas_shortcut.build_regression_model import (
    RANDOM_FOREST,
    REALMLP,
    XGBOOST,
    ModelConfig,
)
from aas_shortcut.match_update_budget import (
    BASELINE_VIEWS,
    describe_budget,
    learning_settings_of,
    matched_config,
    steps_per_epoch,
    total_updates,
)

# The training block of Experiments 4 and 5: 24 functions x 100 instances, and
# the same again under each instance's own transformation.
PLAIN_ROWS = 2400
AUGMENTED_ROWS = 4800


class TestTheStepsOneEpochTakes:
    def test_it_counts_the_full_batches_of_the_rows_the_loop_fits_on(self):
        """The held-out part comes off first; the last, partial batch is dropped."""
        config = ModelConfig()
        # 2400 rows, 20% held out, batches of 256: 1920 / 256 rounded down.
        assert steps_per_epoch(PLAIN_ROWS, config) == 7
        assert steps_per_epoch(AUGMENTED_ROWS, config) == 15

    def test_doubling_the_rows_doubles_them_up_to_one_step(self):
        config = ModelConfig()
        doubled = steps_per_epoch(AUGMENTED_ROWS, config)
        assert doubled - 2 * steps_per_epoch(PLAIN_ROWS, config) in (0, 1)


class TestTheMatchedBudget:
    def test_the_unaugmented_condition_is_untouched(self):
        """One view leaves the configuration unchanged, keeping bit-identity."""
        config = ModelConfig()
        assert matched_config(config, BASELINE_VIEWS) == config

    def test_two_views_halve_the_epochs(self):
        """RealMLP-TD's 256 epochs become 128."""
        config = ModelConfig()
        matched = matched_config(config, 2)
        assert (config.epochs, matched.epochs) == (256, 128)

    def test_the_two_conditions_end_at_most_one_step_per_epoch_apart(self):
        """256 epochs of 7 steps and 128 of 15 differ by at most one step per epoch."""
        config = ModelConfig()
        plain = total_updates(PLAIN_ROWS, matched_config(config, 1))
        augmented = total_updates(AUGMENTED_ROWS, matched_config(config, 2))
        assert (plain, augmented) == (1792, 1920)
        assert 0 <= augmented - plain <= matched_config(config, 2).epochs

    def test_the_unmatched_budget_is_twice_as_long(self):
        """Unmatched, the augmented fit takes at least twice the updates."""
        config = ModelConfig()
        assert total_updates(AUGMENTED_ROWS, config) >= 2 * total_updates(
            PLAIN_ROWS, config
        )

    def test_nothing_but_the_budget_moves(self):
        config = ModelConfig()
        matched = matched_config(config, 2)
        for name in (
            "kind",
            "validation_fraction",
            "random_state",
            "thread_count",
            "early_stopping_rounds",
        ):
            assert getattr(matched, name) == getattr(config, name)
        assert matched.as_dict()["realmlp_td"] == config.as_dict()["realmlp_td"]

    def test_a_budget_that_does_not_divide_is_rejected(self):
        """Rounding would leave the conditions on different update counts."""
        with pytest.raises(ValueError, match="not divisible"):
            matched_config(ModelConfig(epochs=201), 2)

    def test_no_views_at_all_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            matched_config(ModelConfig(), 0)

    def test_the_line_it_logs_names_the_updates(self):
        line = describe_budget(AUGMENTED_ROWS, ModelConfig(), 2)
        assert "4800 row(s)" in line
        assert "15 step(s) x 128 epoch(s) = 1920 update(s)" in line


# The held-out split a unit records for a learner that holds rows out: whole
# problems, at pytabkit's share.
SPLIT = {"unit": "problem", "fraction": 0.2}


class TestWhatAUnitRecords:
    """The settings a fit depends on beyond ``ModelConfig.as_dict``, per learner."""

    def test_realmlp_records_the_divided_epochs_and_the_split(self):
        assert learning_settings_of(ModelConfig(kind=REALMLP), 2) == {
            "epoch_divisor": 2,
            "early_stopping_split": SPLIT,
        }

    def test_xgboost_records_the_split_and_no_divided_schedule(self):
        assert learning_settings_of(ModelConfig(kind=XGBOOST), 2) == {
            "early_stopping_split": SPLIT
        }

    def test_the_forest_depends_on_neither(self):
        assert learning_settings_of(ModelConfig(kind=RANDOM_FOREST), 2) == {}

    def test_xgboosts_stopping_is_not_divided_with_the_epochs(self):
        """The boosting schedule is not divided by the views."""
        config = ModelConfig(kind=XGBOOST)
        assert matched_config(config, 2).early_stopping_rounds == (
            config.early_stopping_rounds
        )
