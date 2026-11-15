"""Tests for Experiment 4: the superiority regression, augmented, on held-out families.

Two audit points share an untransformed half bit for bit and differ in the
transformed one; the rank encoding and the G1 target do not move between the views,
which makes G1 x rank the control.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

import aas_shortcut.match_update_budget as budget_module
import aas_shortcut.measure_augmented_regression as regression_module
from aas_shortcut.augment_training_samples import REP_COLUMN
from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.compare_training_runs import per_seed
from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS
from aas_shortcut.define_paths import ELA_FEATURE_SET, TRANSFORMS
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_reporting_scopes import (
    SCOPE_CONTROL,
    SCOPE_UNSEEN,
    UNSEEN_FAMILIES,
)
from aas_shortcut.define_table_schema import DEPLOYMENT_SCORE_SCHEMA, validate_table
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.match_update_budget import total_updates
from aas_shortcut.measure_augmented_regression import (
    BOARD_LEVEL_COLUMNS,
    DEPLOYMENT_SHARD,
    AugmentedRegressionConfig,
    best_variant,
    board_columns,
    board_levels,
    group_board,
    group_spread,
    held_out_blocks,
    predictors_of,
    run_augmented_regression_stage,
    score_seed,
    seed_levels,
    training_matrix,
)
from aas_shortcut.measure_representation_transfer import VARIANTS, group_of
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_CONDITIONS,
    TRAINING_PLAIN,
)
from aas_shortcut.read_write_tables import read_table
from tests.conftest import make_features
from tests.test_measure_representation_transfer import (
    G1,
    G3,
    make_view_superiorities,
)

TRAIN = tuple(range(3, 23))
TEST = (1, 2)
INSTANCES = len(TRAIN) + len(TEST)

# Three deployments rather than eleven: what is tested is that they are not
# collapsed, not how many there are.
DEPLOYMENTS = 3

RUNS = MODEL_SEEDS[:1]

CONTROL_LABEL = "spline/warpheld"
OTHER_LABEL = "power/warpheld"
LABELS = (CONTROL_LABEL, OTHER_LABEL)


@pytest.fixture
def train_features() -> pd.DataFrame:
    """The training block's single sample, both views and both encodings."""
    return make_features(
        problem_class="bbob", function_id=1, n_instances=INSTANCES, n_reps=1
    )


@pytest.fixture
def test_features() -> dict[str, pd.DataFrame]:
    """Two audit points' held-out blocks: one untransformed half, two warps.

    ``value_shift`` moves only the transformed half, and the untransformed half uses
    the same seed, so the two points agree on ``pre`` exactly.
    """
    return {
        label: make_features(
            problem_class="bbob",
            function_id=1,
            n_instances=INSTANCES,
            n_reps=DEPLOYMENTS,
            value_shift=shift,
        )
        for label, shift in zip(LABELS, (1.5, 4.0), strict=True)
    }


@pytest.fixture
def train_superiorities() -> pd.DataFrame:
    """Superiorities of the three pairs, per instance and view. G1's cannot move."""
    return make_view_superiorities(n_instances=INSTANCES)


@pytest.fixture
def test_superiorities(train_superiorities) -> dict[str, pd.DataFrame]:
    return dict.fromkeys(LABELS, train_superiorities)


@pytest.fixture
def config() -> AugmentedRegressionConfig:
    """A configuration small enough to fit in a unit test."""
    return AugmentedRegressionConfig(
        audits=LABELS,
        train_instances=TRAIN,
        test_instances=TEST,
        candidates=("random-forest",),
        model=ModelConfig(kind="random-forest"),
        # One training run per cell: the stage is under test, not the median.
        n_seeds=1,
    )


class TestConfig:
    def test_an_unknown_representation_is_rejected(self):
        with pytest.raises(ValueError, match="unknown feature variant"):
            AugmentedRegressionConfig(audits=LABELS, variants=("value", "ordinal"))

    def test_an_unknown_training_condition_is_rejected(self):
        with pytest.raises(ValueError, match="unknown training condition"):
            AugmentedRegressionConfig(audits=LABELS, trainings=("plain", "widened"))

    def test_an_unknown_backend_is_rejected(self):
        with pytest.raises(ValueError, match="unknown candidate backend"):
            AugmentedRegressionConfig(audits=LABELS, candidates=("lightgbm",))

    def test_a_run_with_no_audit_point_is_rejected(self):
        """A run with no audit point is rejected."""
        with pytest.raises(ValueError, match="at least one audit point"):
            AugmentedRegressionConfig()

    def test_overlapping_blocks_are_rejected(self):
        with pytest.raises(ValueError, match="both sides of the split"):
            AugmentedRegressionConfig(
                audits=LABELS, train_instances=(1, 2, 3), test_instances=(3, 4)
            )

    def test_the_fingerprint_covers_the_points_and_the_conditions(self):
        """A table built from three families is another measurement, not a part."""
        five = AugmentedRegressionConfig(audits=(*LABELS, "piecewise/warpheld"))
        two = AugmentedRegressionConfig(audits=LABELS)
        assert five.as_dict() != two.as_dict()
        assert (
            two.as_dict()
            != AugmentedRegressionConfig(
                audits=LABELS, trainings=(TRAINING_PLAIN,)
            ).as_dict()
        )
        assert "criterion" not in two.as_dict()

    def test_the_fingerprint_names_the_training_runs(self):
        """Every run is fitted and kept; no cross-validation chooses one."""
        assert AugmentedRegressionConfig(audits=LABELS).as_dict()["model_seeds"] == (
            list(MODEL_SEEDS)
        )
        three = AugmentedRegressionConfig(audits=LABELS, n_seeds=3)
        assert three.model_seeds == MODEL_SEEDS[:3]
        assert three.as_dict() != AugmentedRegressionConfig(audits=LABELS).as_dict()

    def test_a_seed_count_outside_the_study_is_refused(self):
        with pytest.raises(ValueError, match="n_seeds"):
            AugmentedRegressionConfig(audits=LABELS, n_seeds=0)
        with pytest.raises(ValueError, match="n_seeds"):
            AugmentedRegressionConfig(audits=LABELS, n_seeds=len(MODEL_SEEDS) + 1)


class TestTrainingMatrix:
    def test_augmenting_adds_rows_and_changes_nothing_else(
        self, train_features, train_superiorities, config
    ):
        """The plain rows are the first half of the augmented ones, bit for bit."""
        plain = training_matrix(
            train_features,
            train_superiorities,
            "value",
            TRAINING_PLAIN,
            G1.slug,
            config,
        )
        augmented = training_matrix(
            train_features,
            train_superiorities,
            "value",
            TRAINING_AUGMENTED,
            G1.slug,
            config,
        )
        assert len(plain) == len(TRAIN)
        assert len(augmented) == 2 * len(TRAIN)
        pd.testing.assert_frame_equal(augmented.head(len(plain)), plain)

    def test_each_row_carries_the_superiority_its_own_view_earned(
        self, train_features, train_superiorities, config
    ):
        """G3's target moves with the view, so the added rows are not duplicates."""
        augmented = training_matrix(
            train_features,
            train_superiorities,
            "value",
            TRAINING_AUGMENTED,
            G3.slug,
            config,
        )
        by_view = augmented.groupby("transform")["superiority"].mean()
        assert by_view["post"] != pytest.approx(by_view["pre"])

    def test_the_invariant_pair_is_augmented_with_its_own_targets(
        self, train_features, train_superiorities, config
    ):
        """G1's superiority cannot move, so its added rows carry the same targets."""
        augmented = training_matrix(
            train_features,
            train_superiorities,
            "rank",
            TRAINING_AUGMENTED,
            G1.slug,
            config,
        )
        views = {
            name: rows.sort_values("instance_id")
            for name, rows in augmented.groupby("transform")
        }
        assert np.array_equal(
            views["pre"]["superiority"].to_numpy(),
            views["post"]["superiority"].to_numpy(),
        )
        columns = predictors_of(augmented)
        assert np.array_equal(
            views["pre"][columns].to_numpy(), views["post"][columns].to_numpy()
        )

    def test_the_identifiers_are_not_predictors(
        self, train_features, train_superiorities, config
    ):
        matrix = training_matrix(
            train_features,
            train_superiorities,
            "value",
            TRAINING_PLAIN,
            G1.slug,
            config,
        )
        for column in ("superiority", "instance_id", "transform"):
            assert column not in predictors_of(matrix)


class TestTestBlocks:
    def test_a_deployment_sample_stays_its_own_row(
        self, test_features, test_superiorities, config
    ):
        """Collapsing them would report a practitioner who sampled eleven times."""
        blocks = held_out_blocks(
            test_features, test_superiorities, "value", G1.slug, config
        )
        block = blocks[CONTROL_LABEL]["post"]
        assert len(block) == len(TEST) * DEPLOYMENTS
        assert sorted(block[REP_COLUMN].unique()) == list(range(1, DEPLOYMENTS + 1))

    def test_every_audit_point_is_built(
        self, test_features, test_superiorities, config
    ):
        blocks = held_out_blocks(
            test_features, test_superiorities, "value", G1.slug, config
        )
        assert sorted(blocks) == sorted(LABELS)
        assert sorted(blocks[OTHER_LABEL]) == sorted(TRANSFORMS)

    def test_the_points_share_their_untransformed_half(
        self, test_features, test_superiorities, config
    ):
        """What an audit changes is the warp; if `pre` moved it changed the problem."""
        blocks = held_out_blocks(
            test_features, test_superiorities, "value", G1.slug, config
        )
        columns = predictors_of(blocks[CONTROL_LABEL]["pre"])
        keys = ["instance_id", REP_COLUMN]
        left = blocks[CONTROL_LABEL]["pre"].sort_values(keys)[columns].to_numpy()
        right = blocks[OTHER_LABEL]["pre"].sort_values(keys)[columns].to_numpy()
        assert np.array_equal(left, right)

    def test_the_points_differ_in_their_transformed_half(
        self, test_features, test_superiorities, config
    ):
        blocks = held_out_blocks(
            test_features, test_superiorities, "value", G1.slug, config
        )
        columns = predictors_of(blocks[CONTROL_LABEL]["post"])
        keys = ["instance_id", REP_COLUMN]
        left = blocks[CONTROL_LABEL]["post"].sort_values(keys)[columns].to_numpy()
        right = blocks[OTHER_LABEL]["post"].sort_values(keys)[columns].to_numpy()
        assert not np.array_equal(left, right)

    def test_the_rank_encoding_is_the_same_under_every_point_and_view(
        self, test_features, test_superiorities, config
    ):
        """No family can move a matrix computed from the ranks alone."""
        blocks = held_out_blocks(
            test_features, test_superiorities, "rank", G1.slug, config
        )
        keys = ["instance_id", REP_COLUMN]
        columns = predictors_of(blocks[CONTROL_LABEL]["pre"])
        reference = blocks[CONTROL_LABEL]["pre"].sort_values(keys)[columns].to_numpy()
        for label in LABELS:
            for transform in TRANSFORMS:
                assert np.array_equal(
                    blocks[label][transform].sort_values(keys)[columns].to_numpy(),
                    reference,
                )


@pytest.fixture
def run_root(tmp_path):
    """The run root the stage writes its table into."""
    from aas_shortcut.define_paths import RunPaths

    run_paths = RunPaths.from_root(tmp_path / "experiment4")
    run_paths.mkdirs()
    return run_paths


@pytest.fixture
def deployments(
    run_root,
    train_features,
    train_superiorities,
    test_features,
    test_superiorities,
    config,
) -> pd.DataFrame:
    """One small run of the stage, over both training conditions and two points."""
    written = run_augmented_regression_stage(
        train_features,
        train_superiorities,
        test_features,
        test_superiorities,
        run_root,
        "debug",
        "0123456789abcdef",
        config,
    )
    assert not written.empty, "the stage has to have run for its shard to exist"
    return read_table(run_root.metric(DEPLOYMENT_SHARD).data)


@pytest.fixture
def levels(deployments) -> pd.DataFrame:
    """Every family's two levels, as the tables read them."""
    return seed_levels(deployments, RUNS)


class TestStage:
    def test_the_stage_writes_the_deployments_alone(self, deployments, run_root):
        """No table of one chosen run sits beside them."""
        assert not validate_table(deployments, DEPLOYMENT_SCORE_SCHEMA)
        written = {path.name for path in run_root.root.rglob("*.parquet")}
        assert written == {f"{DEPLOYMENT_SHARD}.parquet"}

    def test_every_cell_is_scored_on_every_point_and_view(self, levels, config):
        cells = levels.groupby(
            ["group", "feature_variant", "training", "audit", "metric"], observed=True
        ).size()
        assert set(cells) == {1}
        assert len(cells) == (
            len(config.pairs)
            * len(config.variants)
            * len(config.trainings)
            * len(config.audits)
            * len(TEST_METRICS)
        )

    def test_every_row_names_its_family(self, deployments):
        assert set(deployments["audit"]) == set(LABELS)


class TestReading:
    def test_the_columns_pair_a_representation_with_its_two_conditions(self):
        """The two numbers a reader compares have to sit beside each other."""
        assert board_columns()[:2] == [
            ("value", TRAINING_PLAIN),
            ("value", TRAINING_AUGMENTED),
        ]

    def test_a_winner_is_named_per_group_and_condition(self, deployments):
        for training in TRAINING_CONDITIONS:
            winners = best_variant(deployments, training, SCOPE_CONTROL, seeds=RUNS)
            assert set(winners) == {"G1", "G2", "G3"}

    def test_the_winner_is_read_over_the_range_asked_for(self):
        """`rank` leads on the unseen families only, `value` on the known one."""
        runs = family_runs(UNSEEN_SCORES)
        lead = runs["feature_variant"] == "rank"
        known = runs["audit"] == CONTROL_LABEL
        runs.loc[lead & ~known, "value"] += 0.1
        runs.loc[(runs["feature_variant"] == "value") & known, "value"] += 0.1
        unseen = best_variant(runs, TRAINING_PLAIN, SCOPE_UNSEEN)
        assert set(unseen.values()) == {"rank"}
        assert set(best_variant(runs, TRAINING_PLAIN, SCOPE_CONTROL).values()) == {
            "value"
        }


# Each family's transformed score: the four unseen families run 0.38 down to 0.32
# and the known one sits above, so a five-family mean matches neither range.
UNSEEN_SCORES: dict[str, float] = {
    CONTROL_LABEL: 0.40,
    **{label: 0.38 - 0.02 * index for index, label in enumerate(UNSEEN_FAMILIES)},
}


def family_runs(
    post: dict[str, float], pre: float = 0.8, seeds=MODEL_SEEDS
) -> pd.DataFrame:
    """A per-deployment table where every run of a family scores the family's number.

    The untransformed score cannot depend on the family, so it is one number.
    """
    rows = [
        {
            "feature_set": ELA_FEATURE_SET,
            "group": group_of(pair),
            "pair": pair.slug,
            "model": "random-forest",
            "n_features": 4,
            "feature_variant": variant,
            "training": training,
            "audit": audit,
            "seed": seed,
            "rep": rep,
            "metric": metric,
            "value": pre if transform == TRANSFORMS[0] else value,
            "n_test": len(TEST),
            "scale": "debug",
            "config_hash": "0123456789abcdef",
        }
        for pair in FOCUS_PAIRS
        for variant in VARIANTS
        for training in TRAINING_CONDITIONS
        for audit, value in post.items()
        for transform, metric in TEST_METRICS.items()
        for seed in seeds
        for rep in range(1, DEPLOYMENTS + 1)
    ]
    return pd.DataFrame(rows)


def moving_runs() -> pd.DataFrame:
    """`family_runs` with two unseen families moving with the run.

    The median of the runs' four-family means then differs from the mean of the
    families' medians, so the order of the two reductions is visible.
    """
    runs = family_runs(UNSEEN_SCORES)
    step = runs["seed"] - MODEL_SEEDS[0]
    falling = runs["audit"] == UNSEEN_FAMILIES[0]
    rising = runs["audit"] == UNSEEN_FAMILIES[1]
    runs.loc[falling, "value"] += 0.01 * (10 - step[falling])
    runs.loc[rising, "value"] += 0.001 * step[rising] ** 2
    return runs


class TestTheMainTable:
    """Three algorithm groups by six cells, over one range."""

    def test_it_is_three_groups_by_six_cells(self, deployments):
        page = group_board(deployments, SCOPE_CONTROL, seeds=RUNS)
        assert list(page.index) == ["G1", "G2", "G3"]
        assert list(page.columns) == [
            column for column in board_columns() if column in page.columns
        ]

    def test_a_known_cell_is_the_known_family_s_own_score(self, levels, deployments):
        """One family: its number, not a mean that moved it."""
        page = group_board(deployments, SCOPE_CONTROL, seeds=RUNS)
        own = levels[
            (levels["group"] == "G2")
            & (levels["audit"] == CONTROL_LABEL)
            & (levels["metric"] == TEST_METRICS["post"])
        ]
        for (variant, training), value in page.loc["G2"].items():
            row = own[
                (own["feature_variant"] == variant) & (own["training"] == training)
            ]
            assert value == row["value"].item()

    def test_an_unseen_cell_is_the_mean_over_the_four_unseen_families(self):
        page = group_board(family_runs(UNSEEN_SCORES), SCOPE_UNSEEN)
        unseen = [UNSEEN_SCORES[label] for label in UNSEEN_FAMILIES]
        assert page.to_numpy() == pytest.approx(np.mean(unseen))

    def test_the_known_family_stays_out_of_the_unseen_mean(self):
        page = group_board(family_runs(UNSEEN_SCORES), SCOPE_UNSEEN)
        five = np.mean(list(UNSEEN_SCORES.values()))
        assert not np.isclose(page.to_numpy(), five).any()

    def test_a_cell_is_the_median_of_the_runs_range_means(self):
        """Each run's mean over the range first, then their median."""
        runs = moving_runs()
        page = group_board(runs, SCOPE_UNSEEN)
        per_run = per_seed(runs, SCOPE_UNSEEN, TEST_METRICS["post"])
        assert page.to_numpy() == pytest.approx(per_run["value"].median())
        levels = seed_levels(runs)
        unseen = levels[
            levels["audit"].isin(UNSEEN_FAMILIES)
            & (levels["metric"] == TEST_METRICS["post"])
        ]
        other_order = unseen.groupby("audit")["value"].first().mean()
        assert not np.isclose(page.to_numpy(), other_order).any()

    def test_the_spread_is_the_widest_unseen_family_minus_the_narrowest(self):
        width = group_spread(seed_levels(family_runs(UNSEEN_SCORES)), SCOPE_UNSEEN)
        unseen = [UNSEEN_SCORES[label] for label in UNSEEN_FAMILIES]
        assert width.to_numpy() == pytest.approx(max(unseen) - min(unseen))

    def test_the_spread_is_over_the_families_own_medians(self):
        levels = seed_levels(moving_runs())
        width = group_spread(levels, SCOPE_UNSEEN)
        unseen = levels[
            levels["audit"].isin(UNSEEN_FAMILIES)
            & (levels["metric"] == TEST_METRICS["post"])
        ]
        medians = unseen.groupby("audit")["value"].first()
        assert width.to_numpy() == pytest.approx(medians.max() - medians.min())

    def test_the_untransformed_block_cannot_depend_on_the_family(self):
        """Five copies of one untransformed score are one number, on the main table."""
        levels = seed_levels(family_runs(UNSEEN_SCORES))
        width = group_spread(levels, SCOPE_UNSEEN, TRANSFORMS[0])
        assert width.to_numpy().max() == pytest.approx(0.0, abs=1e-12)

    def test_a_range_missing_one_of_its_families_is_rejected(self, deployments):
        """The fixture scores two families; the unseen range needs four."""
        with pytest.raises(ValueError, match="needs the families"):
            group_board(deployments, SCOPE_UNSEEN, seeds=RUNS)

    def test_a_cell_missing_a_training_run_is_rejected(self):
        runs = family_runs(UNSEEN_SCORES, seeds=MODEL_SEEDS[:-1])
        with pytest.raises(ValueError, match="every training run"):
            group_board(runs, SCOPE_UNSEEN)


class TestTheBoardLevels:
    """The main table's numbers at full precision, one row per cell."""

    @staticmethod
    def _tables() -> tuple[pd.DataFrame, pd.DataFrame]:
        columns = pd.MultiIndex.from_tuples(
            [("value", TRAINING_PLAIN), ("rank", TRAINING_AUGMENTED)]
        )
        mean = pd.DataFrame(
            [[0.123456789, 0.5], [0.25, 0.75]], index=["G1", "G2"], columns=columns
        )
        width = pd.DataFrame(
            [[0.01, 0.02], [0.03, 0.04]], index=["G1", "G2"], columns=columns
        )
        return mean, width

    def test_one_row_per_cell_at_full_precision(self):
        mean, width = self._tables()
        rows = board_levels(mean, width, SCOPE_UNSEEN, "post")
        assert list(rows.columns) == list(BOARD_LEVEL_COLUMNS)
        assert len(rows) == 4
        first = rows.iloc[0]
        assert (first["group"], first["feature_variant"], first["training"]) == (
            "G1",
            "value",
            TRAINING_PLAIN,
        )
        assert first["level"] == 0.123456789
        assert first["spread"] == 0.01
        assert set(rows["scope"]) == {SCOPE_UNSEEN}

    def test_a_range_of_one_family_has_no_spread(self):
        mean, _ = self._tables()
        spread = board_levels(mean, None, SCOPE_UNSEEN, "pre")["spread"]
        assert np.isnan(spread.to_numpy(dtype=float)).all()

    def test_two_tables_of_different_shape_are_rejected(self):
        mean, width = self._tables()
        with pytest.raises(ValueError, match="the same table"):
            board_levels(mean, width.iloc[:1], SCOPE_UNSEEN, "post")


def common_features(shards: dict[str, pd.DataFrame], source: pd.DataFrame) -> set[str]:
    """The features every setting and the training block all carry."""
    shared = set(source["feature"].unique())
    for features in shards.values():
        shared &= set(features["feature"].unique())
    return shared


class TestEveryFamilyIsReadOnOneColumnSet:
    """A family that loses a feature narrows the others and the training block too.

    Otherwise the families' rows would differ in predictor sets as well as in family.
    """

    @staticmethod
    def _narrow(frames, train, missing):
        """Drop one feature from one family, then take the common set."""
        held = {
            label: frame[frame["feature"] != missing] if label == OTHER_LABEL else frame
            for label, frame in frames.items()
        }
        common = common_features(held, train)
        return (
            train[train["feature"].isin(common)],
            {
                label: frame[frame["feature"].isin(common)]
                for label, frame in held.items()
            },
            common,
        )

    @pytest.mark.parametrize("variant", ["value", "rank", "concat"])
    def test_the_training_matrix_and_every_block_carry_the_same_predictors(
        self,
        train_features,
        train_superiorities,
        test_features,
        test_superiorities,
        variant,
    ):
        missing = sorted(train_features["feature"].unique())[0]
        train, held, common = self._narrow(test_features, train_features, missing)
        config = AugmentedRegressionConfig(
            audits=LABELS,
            features=tuple(sorted(common)),
            train_instances=TRAIN,
            test_instances=TEST,
            candidates=("random-forest",),
            n_seeds=1,
        )
        matrix = training_matrix(
            train, train_superiorities, variant, TRAINING_PLAIN, G1.slug, config
        )
        blocks = held_out_blocks(held, test_superiorities, variant, G1.slug, config)
        expected = predictors_of(matrix)
        assert expected, "the narrowing must not empty the representation"
        for label, views in blocks.items():
            for transform, block in views.items():
                assert predictors_of(block) == expected, (label, transform)

    def test_the_common_set_enters_the_fingerprint(self, train_features):
        """A run over a subset of the families is another measurement, not a part."""
        missing = sorted(train_features["feature"].unique())[0]
        whole = AugmentedRegressionConfig(audits=LABELS, features=("a", "b"))
        short = AugmentedRegressionConfig(audits=LABELS, features=("a",))
        assert whole.as_dict() != short.as_dict()
        assert missing not in short.as_dict()["features"]


class TestTheLevelsTheTablesRead:
    """What replaces the deployed run's score in every table."""

    def test_a_family_s_level_is_the_median_of_its_runs_medians(self):
        runs = moving_runs()
        runs["value"] += 0.0001 * runs["rep"]
        levels = seed_levels(runs)
        keys = ["group", "feature_variant", "training", "audit", "metric"]
        expected = (
            runs.groupby([*keys, "seed"])["value"].median().groupby(keys).median()
        )
        found = levels.set_index(keys)["value"]
        assert found.sort_index().to_numpy() == pytest.approx(
            expected.sort_index().to_numpy()
        )

    def test_a_cell_missing_a_training_run_is_refused(self):
        with pytest.raises(ValueError, match="every training run"):
            seed_levels(family_runs(UNSEEN_SCORES, seeds=MODEL_SEEDS[:-1]))


class TestTheDeploymentsBehindTheMedianAreKept:
    """What the representations' paired test is taken over.

    A median cannot be tested against a median, so every sample of every training run
    is kept; one deployment must be one sample shared by the three representations.
    """

    CELL: ClassVar[list[str]] = [
        "feature_set",
        "feature_variant",
        "training",
        "audit",
        "pair",
        "model",
        "metric",
    ]

    def test_the_table_satisfies_its_schema(self, deployments):
        assert not validate_table(deployments, DEPLOYMENT_SCORE_SCHEMA)

    def test_every_test_score_keeps_its_deployments(self, deployments):
        counted = deployments.groupby([*self.CELL, "seed"], observed=True)["rep"].agg(
            ["size", "nunique"]
        )
        assert set(counted["size"]) == {DEPLOYMENTS}
        assert set(counted["nunique"]) == {DEPLOYMENTS}

    def test_one_deployment_is_one_sample_across_the_representations(self, deployments):
        """The same deployment indices are scored in every representation of a cell."""
        block = ["group", "training", "audit", "metric", "seed"]
        scored = deployments.groupby([*block, "feature_variant"], observed=True)[
            "rep"
        ].agg(lambda reps: tuple(sorted(reps)))
        per_block = scored.groupby(block, observed=True).nunique()
        assert set(per_block) == {1}
        assert set(scored) == {tuple(range(1, DEPLOYMENTS + 1))}

    def test_the_two_views_are_the_only_scores(self, deployments):
        """Nothing is scored on the training block; only the two held-out views."""
        assert set(deployments["metric"]) == set(TEST_METRICS.values())
        assert "deployed" not in deployments.columns


class TestEveryTrainingRunReachesTheHeldOutBlock:
    """Every run is scored on every block; none is chosen."""

    SEEDS = 3

    @pytest.fixture
    def several(
        self,
        run_root,
        train_features,
        train_superiorities,
        test_features,
        test_superiorities,
        config,
    ):
        """One cell, fitted under three training runs rather than one."""
        cell = (G3, "value", TRAINING_PLAIN, config.candidates[0])
        configured = replace(config, n_seeds=self.SEEDS, audits=(CONTROL_LABEL,))
        written = run_augmented_regression_stage(
            train_features,
            train_superiorities,
            test_features,
            {CONTROL_LABEL: test_superiorities[CONTROL_LABEL]},
            run_root,
            "debug",
            "hash",
            configured,
            cells=[cell],
        )
        return written, configured

    def test_every_run_is_scored_on_the_block(self, several):
        written, _ = several
        assert set(written["seed"]) == set(MODEL_SEEDS[: self.SEEDS])
        per_run = written.groupby(["seed", "metric"], observed=True)["rep"].nunique()
        assert set(per_run) == {DEPLOYMENTS}

    def test_a_run_scores_the_same_alone_as_in_the_pass(
        self,
        several,
        train_features,
        train_superiorities,
        test_features,
        test_superiorities,
    ):
        """A run depends on its own seed alone, not on the runs beside it."""
        written, configured = several
        train = training_matrix(
            train_features,
            train_superiorities,
            "value",
            TRAINING_PLAIN,
            G3.slug,
            configured,
        )
        blocks = held_out_blocks(
            {CONTROL_LABEL: test_features[CONTROL_LABEL]},
            {CONTROL_LABEL: test_superiorities[CONTROL_LABEL]},
            "value",
            G3.slug,
            configured,
        )
        seed = MODEL_SEEDS[1]
        alone = pd.DataFrame(
            score_seed(
                train,
                blocks,
                predictors_of(train),
                G3,
                "value",
                TRAINING_PLAIN,
                configured.candidates[0],
                configured,
                seed,
            )
        )
        keys = ["metric", "rep"]
        ours = alone.set_index(keys)["value"].sort_index().to_numpy()
        theirs = (
            written[written["seed"] == seed]
            .set_index(keys)["value"]
            .sort_index()
            .to_numpy()
        )
        assert np.array_equal(ours.view(np.uint64), theirs.view(np.uint64))


class TestEveryRunSpendsTheMatchedBudget:
    """Matched updates on the fits the stage makes.

    The augmented condition fits on twice the rows, so its epochs are divided by the
    views it spends.
    """

    @staticmethod
    def budgets_handed_to(monkeypatch, **kwargs) -> list:
        """Run the stage and return every fit's ``(training rows, settings)``.

        Batches of two, so that the row count decides how many steps an epoch takes;
        at the real batch size these fixtures fit in one step either way.
        """
        monkeypatch.setattr(budget_module, "REALMLP_TD_BATCH_SIZE", 2)
        seen: list = []
        original = regression_module.fit_model

        def spy(train, columns, model_config, target, **options):
            seen.append((len(train), model_config, options.get("group_columns")))
            return original(train, columns, model_config, target, **options)

        monkeypatch.setattr(regression_module, "fit_model", spy)
        run_augmented_regression_stage(**kwargs)
        return seen

    @pytest.fixture
    def arguments(
        self,
        run_root,
        train_features,
        train_superiorities,
        test_features,
        test_superiorities,
        config,
    ):
        return {
            "train_features": train_features,
            "train_superiorities": train_superiorities,
            "test_features": test_features,
            "test_superiorities": test_superiorities,
            "run_paths": run_root,
            "scale": "debug",
            "config_hash": "0123456789abcdef",
            "config": replace(
                config,
                model=ModelConfig(
                    kind="random-forest",
                    epochs=200,
                    # No held-out split: on a matrix this small it would break the
                    # proportion for the fixture's size, not for the budget.
                    validation_fraction=0.0,
                ),
            ),
        }

    def test_every_fit_spends_the_same_updates(self, monkeypatch, arguments):
        """Every fit spends one budget of optimiser steps, not of epochs.

        Asserted as epochs times the rows an epoch batches over, so the baseline
        budget can change without rewriting the test.
        """
        seen = self.budgets_handed_to(monkeypatch, **arguments)
        assert seen, "the stage fitted nothing"
        rows = {count for count, _, _ in seen}
        assert len(rows) == 2, (
            "the augmented side spends no extra views on this fixture, so "
            "agreeing update counts would say nothing"
        )
        spent = {total_updates(count, config) for count, config, _ in seen}
        assert len(spent) == 1, f"the fits spend {sorted(spent)} updates"

    def test_every_fit_groups_its_early_stopping_split(self, monkeypatch, arguments):
        """A problem's augmented copy stays on its own side of the split."""
        seen = self.budgets_handed_to(monkeypatch, **arguments)
        assert {groups for _, _, groups in seen} == {regression_module.GROUP_COLUMNS}
