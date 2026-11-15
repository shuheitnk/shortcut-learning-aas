"""The selection cells' training conditions, split and table names."""

from __future__ import annotations

import pandas as pd
import pytest

from aas_shortcut.calculate_mean_superiority import compute_mean_superiorities
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.measure_selection_transfer import (
    PROBLEM_SHARD,
    SEED_DEPLOYMENT_SHARD,
    SPLIT,
    TRAINING_AUGMENTED,
    TRAINING_CONDITIONS,
    TRAINING_PLAIN,
    TRAINING_VIEWS,
    training_views,
)
from aas_shortcut.split_bbob_instances import TEST_INSTANCES, TRAIN_INSTANCES
from tests.conftest import make_features, make_performance
from tests.test_select_algorithms import DIM

# A training block and a test block small enough to fit in a test, disjoint as the
# study's are and in the same direction: the low ids are held out. Shared with the
# selection cells' tests.
TEST = (1, 2)
TRAIN = tuple(range(3, 23))
N_FUNCTIONS = 3
N_TRIALS = 4


def make_view_features(n_reps: int = 1) -> pd.DataFrame:
    """Features for both blocks under both views, rank-invariant by construction."""
    return pd.concat(
        [
            make_features(
                problem_class="bbob",
                function_id=function_id,
                n_instances=max(TRAIN),
                n_reps=n_reps,
                dim=DIM,
                seed=function_id,
            )
            for function_id in range(1, N_FUNCTIONS + 1)
        ],
        ignore_index=True,
    )


def make_view_performance(budget: int, seed_offset: int = 0) -> pd.DataFrame:
    """Build a performance table carrying both views.

    The ``post`` half uses a different seed, so the two views disagree on the winner,
    as a heterogeneous portfolio does under a warp.
    """
    blocks = []
    for index, transform in enumerate(TRANSFORMS):
        rows = pd.concat(
            [
                make_performance(
                    problem_class="bbob",
                    function_id=function_id,
                    n_instances=max(TRAIN),
                    n_trials=N_TRIALS,
                    dim=DIM,
                    budget=budget,
                    seed=function_id + seed_offset + 100 * index,
                )
                for function_id in range(1, N_FUNCTIONS + 1)
            ],
            ignore_index=True,
        )
        blocks.append(rows.assign(transform=transform))
    return pd.concat(blocks, ignore_index=True)


def make_view_mean_superiorities(performance: pd.DataFrame) -> pd.DataFrame:
    """Reduce a two-view performance table to a mean superiority per view."""
    from aas_shortcut.calculate_mean_superiority import VIEW_GROUP_COLUMNS

    ranks = compute_mean_superiorities(performance, group_columns=VIEW_GROUP_COLUMNS)
    return ranks.assign(scale="debug", config_hash="hash")


class TestTrainingViews:
    def test_plain_fits_on_the_untransformed_view_alone(self):
        assert TRAINING_VIEWS[TRAINING_PLAIN] == (TRANSFORMS[0],)

    def test_augmented_fits_on_both(self):
        assert TRAINING_VIEWS[TRAINING_AUGMENTED] == TRANSFORMS

    def test_a_block_is_resolved_under_the_views_its_conditions_read(self):
        assert training_views((TRAINING_PLAIN,)) == (TRANSFORMS[0],)
        assert training_views(TRAINING_CONDITIONS) == TRANSFORMS

    def test_an_unknown_condition_is_rejected(self):
        with pytest.raises(ValueError, match="unknown training condition"):
            training_views(("sideways",))


class TestSplit:
    def test_it_fits_on_the_training_block_and_selects_on_the_test_block(self):
        assert SPLIT.train_instances == TRAIN_INSTANCES
        assert SPLIT.test_instances == TEST_INSTANCES


class TestTables:
    def test_a_cell_keeps_its_runs_deployments_and_their_problems(self):
        """The tables are named as Experiment 4's are."""
        assert SEED_DEPLOYMENT_SHARD == "deployment_scores"
        assert PROBLEM_SHARD == "problem_scores"
