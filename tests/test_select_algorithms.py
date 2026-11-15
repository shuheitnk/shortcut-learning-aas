"""Tests for the algorithm-selection stage and its SBS baseline.

The comparison is fair only if both methods get the same evaluations and the selector
sees one sample per deployment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytest

from aas_shortcut.build_regression_model import ModelConfig
from aas_shortcut.create_sampled_points import create_sampled_points
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_paths import (
    DEEP_ELA_FEATURE_SET,
    ELA_FEATURE_SET,
    ORACLE_VARIANT,
)
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_table_schema import SELECTION_SCHEMA, validate_table
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.select_algorithms import (
    SelectionConfig,
    SelectionSplit,
    assign_trials,
    best_sampled_values,
    config_for,
    fit_member,
    instance_winners,
    oracle_choices,
    oracle_rows,
    prepare_selection,
    problem_rows,
    rank_single_best_solver,
    score_selector,
    seed_deployment_rows,
    select_per_repetition,
    single_best_solver,
    training_matrix,
)
from tests.conftest import (
    CONFIG_HASH,
    make_features,
    make_mean_superiorities,
    make_performance,
    make_sbs_performance,
)

DIM = 2
SAMPLE_SIZE = 100
SEARCH_BUDGET = 200
TOTAL_BUDGET = SEARCH_BUDGET + SAMPLE_SIZE


def make_bbob_tables(
    n_functions: int = 3,
    n_instances: int = 2,
    n_trials: int = 4,
    n_reps: int | None = None,
):
    """Build BBOB features, performance and mean superiorities with the real schema."""
    n_reps = n_trials if n_reps is None else n_reps
    features = pd.concat(
        [
            make_features(
                problem_class="bbob",
                function_id=function_id,
                n_instances=n_instances,
                n_reps=n_reps,
                dim=DIM,
                seed=function_id,
                feature_sets=(ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET),
            )
            for function_id in range(1, n_functions + 1)
        ],
        ignore_index=True,
    )
    performance = pd.concat(
        [
            make_performance(
                problem_class="bbob",
                function_id=function_id,
                n_instances=n_instances,
                n_trials=n_trials,
                dim=DIM,
                budget=SEARCH_BUDGET,
                seed=function_id,
            )
            for function_id in range(1, n_functions + 1)
        ],
        ignore_index=True,
    )
    return features, performance, make_mean_superiorities(performance)


# Both sides BBOB, split by instance as in every experiment.
STAGE_TRAIN = (3, 4, 5, 6, 7, 8)
STAGE_TEST = (1, 2)
STAGE_SPLIT = SelectionSplit("bbob", "bbob", STAGE_TRAIN, STAGE_TEST)


def run_selection(
    features_by_class,
    mean_superiorities_by_class,
    selected_performance,
    sbs_performance,
    run_paths,
    scale,
    config_hash,
    dim,
    base_seed,
    sample_size,
    config,
    *,
    sbs_mean_superiorities=None,
    split=None,
    shard_name="selection",
):
    """Run the selection on one block and write the oracle's and each cell's rows.

    Each cell keeps its first deployment's rows, one per held-out problem.
    """
    prepared = prepare_selection(
        features_by_class,
        mean_superiorities_by_class,
        selected_performance,
        sbs_performance,
        dim,
        base_seed,
        sample_size,
        config,
        sbs_mean_superiorities=sbs_mean_superiorities,
        split=split,
    )
    block = prepared.block
    rows = list(oracle_rows(block, config)) if config.include_oracle else []
    for feature_set in config.feature_sets:
        for variant in config.feature_variants:
            matrix, columns = training_matrix(
                prepared.train_features,
                prepared.train_mean_superiorities,
                variant,
                feature_set,
                config,
            )
            models = {
                name: fit_member(matrix, columns, config, name)
                for name in config.algorithms
            }
            scored = score_selector(
                block, models, columns, variant, feature_set, config
            )
            rows.extend(scored.runs[min(scored.runs)])
    table = pd.DataFrame(rows).assign(scale=scale, config_hash=config_hash)
    return write_table(table, run_paths.metric(shard_name).data, SELECTION_SCHEMA)


def fixed_ranking(
    mean_superiorities: pd.DataFrame, order: Sequence[str]
) -> pd.DataFrame:
    """Rewrite a mean-superiority table so ``order`` ranks every instance."""
    last = len(order) - 1
    score_of = {name: float(last - index) / last for index, name in enumerate(order)}
    return mean_superiorities.assign(
        mean_superiority=mean_superiorities["algorithm"].map(score_of).astype("float64")
    )


def sbs_condition(performance: pd.DataFrame) -> pd.DataFrame:
    """Build the ``sbs`` shard that goes with a ``selection`` shard."""
    return make_sbs_performance(performance, TOTAL_BUDGET, penalty=0.25)


class TestSingleBestSolver:
    def test_it_picks_the_winner_of_the_only_instance(self):
        mean_superiorities = pd.DataFrame(
            {
                "instance_id": [1, 1, 1],
                "algorithm": ["BIPOP-CMA-ES", "SHADE", "L-SHADE"],
                "mean_superiority": [0.0, 0.75, 0.5],
            }
        )
        candidates = ("BIPOP-CMA-ES", "SHADE", "L-SHADE")
        assert single_best_solver(mean_superiorities, candidates) == "SHADE"

    def test_it_averages_scores_rather_than_counting_instance_wins(self):
        """The SBS maximises the mean score, not the count of instance wins."""
        mean_superiorities = pd.DataFrame(
            {
                "instance_id": [1, 1, 2, 2, 3, 3],
                "algorithm": ["BIPOP-CMA-ES", "SHADE"] * 3,
                "mean_superiority": [0.6, 0.4, 0.6, 0.4, 0.0, 1.0],
            }
        )
        candidates = ("BIPOP-CMA-ES", "SHADE")
        wins = instance_winners(mean_superiorities, candidates).value_counts()
        assert int(wins["BIPOP-CMA-ES"]) > int(wins["SHADE"])
        assert single_best_solver(mean_superiorities, candidates) == "SHADE"

    def test_the_instance_winner_is_the_largest_mean_superiority_there(self):
        mean_superiorities = pd.DataFrame(
            {
                "instance_id": [1, 1, 2, 2],
                "algorithm": ["BIPOP-CMA-ES", "SHADE"] * 2,
                "mean_superiority": [1.0, 0.0, 0.2, 0.8],
            }
        )
        winners = instance_winners(mean_superiorities, ("BIPOP-CMA-ES", "SHADE"))
        assert list(winners) == ["BIPOP-CMA-ES", "SHADE"]

    def test_a_tie_is_broken_by_portfolio_order(self):
        mean_superiorities = pd.DataFrame(
            {
                "instance_id": [1, 1],
                "algorithm": ["SHADE", "BIPOP-CMA-ES"],
                "mean_superiority": [0.5, 0.5],
            }
        )
        assert single_best_solver(mean_superiorities, ("BIPOP-CMA-ES", "SHADE")) == (
            "BIPOP-CMA-ES"
        )

    def test_it_refuses_a_member_it_has_no_scores_for(self, mean_superiorities):
        with pytest.raises(ValueError, match="no mean superiorities"):
            single_best_solver(mean_superiorities, ("not-an-algorithm",))


class TestTheSingleBestSolversRanking:
    """The ranking table the SBS choice is read from."""

    @staticmethod
    def scores() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "instance_id": [1, 1, 1, 2, 2, 2],
                "algorithm": ["SHADE", "BIPOP-CMA-ES", "L-SHADE"] * 2,
                "mean_superiority": [0.9, 0.5, 0.1, 0.5, 0.7, 0.3],
            }
        )

    def test_one_row_per_member_in_portfolio_order(self):
        candidates = ("BIPOP-CMA-ES", "SHADE", "L-SHADE")
        ranking = rank_single_best_solver(self.scores(), candidates)
        assert list(ranking["algorithm"]) == list(candidates)
        assert list(ranking["n_problems"]) == [2, 2, 2]
        assert ranking["mean_superiority"].tolist() == pytest.approx([0.6, 0.7, 0.2])

    def test_it_flags_the_member_the_choice_names(self):
        candidates = ("BIPOP-CMA-ES", "SHADE", "L-SHADE")
        ranking = rank_single_best_solver(self.scores(), candidates)
        assert ranking.loc[ranking["is_sbs"], "algorithm"].tolist() == ["SHADE"]
        assert single_best_solver(self.scores(), candidates) == "SHADE"

    def test_a_tie_flags_the_earlier_member_alone(self):
        tied = pd.DataFrame(
            {
                "instance_id": [1, 1],
                "algorithm": ["SHADE", "BIPOP-CMA-ES"],
                "mean_superiority": [0.5, 0.5],
            }
        )
        ranking = rank_single_best_solver(tied, ("BIPOP-CMA-ES", "SHADE"))
        assert ranking["is_sbs"].tolist() == [True, False]


class TestAssignTrials:
    """Which sampling repetition each algorithm trial is attributed to."""

    def test_one_repetition_covers_every_trial(self):
        """One repetition per instance is paired with every trial."""
        assignment = assign_trials([1], [1, 2, 3, 4, 5])
        assert list(assignment["trial"]) == [1, 2, 3, 4, 5]
        assert set(assignment["rep"]) == {1}

    def test_equal_counts_reproduce_the_one_to_one_pairing(self):
        """With as many repetitions as trials, repetition t goes with trial t."""
        assignment = assign_trials([1, 2, 3], [1, 2, 3])
        assert list(assignment["rep"]) == list(assignment["trial"])

    def test_fewer_repetitions_than_trials_cycle(self):
        assignment = assign_trials([1, 2], [1, 2, 3, 4, 5])
        assert list(assignment["rep"]) == [1, 2, 1, 2, 1]

    def test_every_trial_is_attributed_exactly_once(self):
        assignment = assign_trials([1, 2, 3], [1, 2, 3, 4, 5, 6, 7])
        assert len(assignment) == 7
        assert assignment["trial"].is_unique

    @pytest.mark.parametrize(("reps", "trials"), [([], [1]), ([1], [])])
    def test_an_empty_side_is_rejected(self, reps, trials):
        with pytest.raises(ValueError, match="cannot pair"):
            assign_trials(reps, trials)


class TestSelectPerRepetition:
    """Each repetition's sample gets its own selection."""

    def test_every_repetition_gets_its_own_choice(self):
        features, _, _ = make_bbob_tables(n_functions=1, n_instances=2, n_trials=4)
        columns = ["value__ela_meta.lin_simple.adj_r2"]

        class First:
            """Predicts the feature itself, so the choice tracks the data."""

            def predict(self, X):
                return X[:, 0]

        class Constant:
            def __init__(self, value):
                self.value = value

            def predict(self, X):
                return np.full(len(X), self.value)

        models = {"BIPOP-CMA-ES": First(), "SHADE": Constant(2.0)}
        chosen = select_per_repetition(
            features,
            models,
            columns,
            "value",
            ELA_FEATURE_SET,
            ("BIPOP-CMA-ES", "SHADE"),
        )
        assert set(chosen["rep"]) == {1, 2, 3, 4}
        assert len(chosen) == 2 * 4
        # Instance 1's feature sits near 1.0 and instance 2's near 2.0, so the selector
        # must not return the same algorithm everywhere.
        assert chosen["selected_algorithm"].nunique() == 2

    def test_a_missing_feature_column_is_an_error(self):
        features, _, _ = make_bbob_tables(n_functions=1, n_instances=2, n_trials=2)
        with pytest.raises(ValueError, match="missing"):
            select_per_repetition(
                features,
                {"BIPOP-CMA-ES": object()},
                ["value__not_a_feature"],
                "value",
                ELA_FEATURE_SET,
                ("BIPOP-CMA-ES",),
            )


class TestBestSampledValues:
    """The re-evaluated sample must be the same sample the feature stage used."""

    def test_it_reproduces_the_sampled_minimum(self):
        instances = pd.DataFrame({"function_id": [1], "instance_id": [1]})
        table = best_sampled_values(42, "bbob", instances, DIM, [1, 2], SAMPLE_SIZE)
        assert list(table["rep"]) == [1, 2]
        for rep in (1, 2):
            sample = create_sampled_points(42, "bbob", 1, 1, DIM, rep, SAMPLE_SIZE)
            recorded = float(table[table["rep"] == rep]["best_sampled_y"].iloc[0])
            assert recorded == float(np.min(sample.y_pre))


class TestSelectionModelAndTarget:
    """The regressor and its target must be visible to the caching layer."""

    def test_the_backend_is_part_of_the_fingerprint(self):
        """Changing the backend changes the fingerprint, so no cached run is reused."""
        default = SelectionConfig().as_dict()
        swapped = SelectionConfig(model=ModelConfig(kind="random-forest")).as_dict()
        assert default != swapped


class TestSelectionSplit:
    """The split that decides which rows train the selector and which test it."""

    def test_both_sides_are_bbob_by_default(self):
        split = SelectionSplit()
        assert split.train_problem_class == "bbob"
        assert split.test_problem_class == "bbob"

    def test_a_within_suite_split_asks_for_that_suite_once(self):
        """A within-suite split hands in one table, not the same one twice."""
        split = SelectionSplit("bbob", "bbob", (1, 2), (3, 4))
        assert split.required_classes() == ("bbob",)

    def test_an_instance_split_cuts_one_table_in_two(self):
        frame = pd.DataFrame({"instance_id": [1, 2, 3, 4], "x": [0.0] * 4})
        split = SelectionSplit("bbob", "bbob", (1, 2), (3, 4))
        assert list(split.train({"bbob": frame})["instance_id"]) == [1, 2]
        assert list(split.test({"bbob": frame})["instance_id"]) == [3, 4]

    def test_no_instance_reaches_both_sides(self):
        frame = pd.DataFrame({"instance_id": [1, 2, 3, 4], "x": [0.0] * 4})
        split = SelectionSplit("bbob", "bbob", (1, 2), (3, 4))
        train = set(split.train({"bbob": frame})["instance_id"])
        assert not train & set(split.test({"bbob": frame})["instance_id"])

    def test_a_split_the_shard_cannot_serve_is_rejected(self):
        """A split the shard does not cover raises instead of scoring nothing."""
        frame = pd.DataFrame({"instance_id": [1, 2], "x": [0.0, 0.0]})
        split = SelectionSplit("bbob", "bbob", (1, 2), (11, 12))
        with pytest.raises(ValueError, match="does not cover the split"):
            split.test({"bbob": frame})

    def test_the_split_is_part_of_the_fingerprint(self):
        """The split is in the fingerprint, so two splits never share a cached table."""
        assert (
            SelectionSplit().as_dict()
            != SelectionSplit("bbob", "bbob", (1, 2), (3, 4)).as_dict()
        )


class TestSelectionStage:
    @pytest.fixture
    def tables(self):
        features, performance, mean_superiorities = make_bbob_tables(n_instances=8)
        train_ranks = cast(
            pd.DataFrame,
            mean_superiorities[mean_superiorities["instance_id"].isin(STAGE_TRAIN)],
        )
        return {
            "features": {"bbob": features},
            "mean_superiorities": {"bbob": mean_superiorities},
            "selected_performance": performance,
            "sbs_performance": sbs_condition(performance),
            "sbs": single_best_solver(train_ranks),
        }

    def _run(self, tables, run_paths, sbs_mean_superiorities=None, **overrides):
        settings = {
            "feature_variants": ("value", "rank"),
            "feature_sets": (ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET),
            "model": ModelConfig(),
            "include_best_sampled_point": False,
            **overrides,
        }
        config = SelectionConfig(**settings)
        return run_selection(
            tables["features"],
            tables["mean_superiorities"],
            tables["selected_performance"],
            tables["sbs_performance"],
            run_paths,
            "debug",
            CONFIG_HASH,
            DIM,
            42,
            SAMPLE_SIZE,
            config,
            sbs_mean_superiorities=sbs_mean_superiorities,
            split=STAGE_SPLIT,
        )

    def test_the_sbs_is_named_from_the_training_block_alone(self, tables, run_paths):
        """The SBS is chosen from the training block, even given the whole shard."""
        on_train, on_everything = ALGORITHMS[0], ALGORITHMS[1]
        ranks = tables["mean_superiorities"]["bbob"]
        held_out = ranks["instance_id"].isin(STAGE_TEST)
        # The first member is best on the training block and catastrophic on the
        # held-out one, so including the held-out rows would name the second.
        by_member = {on_train: (1.0, 0.0), on_everything: (0.9, 0.9)}
        whole_shard = ranks.assign(
            mean_superiority=[
                by_member.get(str(name), (0.2, 0.2))[1 if test else 0]
                for name, test in zip(ranks["algorithm"], held_out, strict=True)
            ]
        )
        assert single_best_solver(whole_shard, ALGORITHMS) == on_everything
        assert single_best_solver(whole_shard[~held_out], ALGORITHMS) == on_train

        written = self._run(tables, run_paths, sbs_mean_superiorities=whole_shard)
        assert set(written["sbs_algorithm"]) == {on_train}

    def test_the_selector_only_ever_learns_from_the_training_block(
        self, tables, run_paths
    ):
        """With opposite rankings per block, the selector follows the training block."""
        best_on_train, best_on_test = ALGORITHMS[0], ALGORITHMS[-1]
        ranks = tables["mean_superiorities"]["bbob"]
        held_out = ranks["instance_id"].isin(STAGE_TEST)
        tables = {
            **tables,
            "mean_superiorities": {
                "bbob": pd.concat(
                    [
                        fixed_ranking(ranks[~held_out], ALGORITHMS),
                        fixed_ranking(ranks[held_out], ALGORITHMS[::-1]),
                    ],
                    ignore_index=True,
                )
            },
        }

        written = self._run(tables, run_paths)
        learned = written[written["feature_variant"] != ORACLE_VARIANT]
        oracle = written[written["feature_variant"] == ORACLE_VARIANT]

        assert set(learned["modal_algorithm"]) == {best_on_train}
        assert set(oracle["modal_algorithm"]) == {best_on_test}

    def test_it_writes_a_valid_table(self, tables, run_paths):
        written = self._run(tables, run_paths)
        assert not written.empty
        assert validate_table(written, "selection") == []
        assert run_paths.metric("selection").exists()

    def test_it_covers_every_set_and_variant(self, tables, run_paths):
        written = self._run(tables, run_paths)
        assert set(written["feature_set"]) == {ELA_FEATURE_SET, DEEP_ELA_FEATURE_SET}
        assert set(written["feature_variant"]) == {"value", "rank", ORACLE_VARIANT}

    def test_the_oracle_takes_the_best_ranked_member_of_each_instance(self, tables):
        """The Oracle picks each instance's best-ranked member."""
        ranks = tables["mean_superiorities"]["bbob"]
        chosen = oracle_choices(ranks, reps=[0])

        best = ranks.loc[
            ranks.groupby(["function_id", "instance_id"])["mean_superiority"].idxmax()
        ]
        expected = dict(
            zip(
                zip(best["function_id"], best["instance_id"], strict=True),
                best["algorithm"],
                strict=True,
            )
        )
        got = dict(
            zip(
                zip(chosen["function_id"], chosen["instance_id"], strict=True),
                chosen["selected_algorithm"],
                strict=True,
            )
        )
        assert got == expected

    def test_the_oracle_commits_to_one_algorithm_per_instance(self, tables, run_paths):
        """The Oracle chooses the same algorithm for an instance in every deployment."""
        written = self._run(tables, run_paths)
        oracle = written[written["feature_variant"] == ORACLE_VARIANT]
        assert not oracle.empty
        assert (oracle["n_selected_algorithms"] == 1).all()
        assert (oracle["modal_share"] == 1.0).all()

    def test_both_methods_are_given_the_same_budget(self, tables, run_paths):
        written = self._run(tables, run_paths)
        assert (written["selection_budget"] == written["sbs_budget"]).all()
        assert set(written["selection_budget"]) == {TOTAL_BUDGET}

    def test_it_refuses_an_unequal_budget(self, tables, run_paths):
        """Unequal selector and SBS budgets are refused."""
        tables = dict(tables)
        tables["sbs_performance"] = tables["sbs_performance"].assign(
            budget=TOTAL_BUDGET + 1
        )
        with pytest.raises(ValueError, match="same budget"):
            self._run(tables, run_paths)

    def test_it_refuses_a_condition_holding_two_budgets(self, tables, run_paths):
        """An SBS shard holding two budgets is refused."""
        tables = dict(tables)
        tables["sbs_performance"] = pd.concat(
            [tables["sbs_performance"], tables["selected_performance"]],
            ignore_index=True,
        )
        with pytest.raises(ValueError, match="exactly one budget"):
            self._run(tables, run_paths)

    def test_the_recorded_sbs_is_the_training_sets(self, tables, run_paths):
        written = self._run(tables, run_paths)
        assert set(written["sbs_algorithm"]) == {tables["sbs"]}

    def test_superiorities_are_probabilities(self, tables, run_paths):
        written = self._run(tables, run_paths)
        assert written["superiority_vs_sbs"].between(0.0, 1.0).all()

    def test_the_sample_credit_can_only_help(self, tables, run_paths):
        """Crediting the sample's best point can never worsen the selection."""
        without = self._run(tables, run_paths)
        with_credit = self._run(tables, run_paths, include_best_sampled_point=True)
        key = ["function_id", "instance_id", "feature_set", "feature_variant"]
        merged = without.merge(with_credit, on=key, suffixes=("_off", "_on"))
        assert not merged.empty
        assert (
            merged["median_best_y_selection_on"]
            <= merged["median_best_y_selection_off"] + 1e-12
        ).all()


class TestSingleRepetitionDeployment:
    """One sample per instance, that choice measured over every trial."""

    N_TRIALS = 4

    @pytest.fixture
    def tables(self):
        features, performance, mean_superiorities = make_bbob_tables(
            n_instances=8, n_trials=self.N_TRIALS, n_reps=1
        )
        train_ranks = cast(
            pd.DataFrame,
            mean_superiorities[mean_superiorities["instance_id"].isin(STAGE_TRAIN)],
        )
        return {
            "features": {"bbob": features},
            "mean_superiorities": {"bbob": mean_superiorities},
            "selected_performance": performance,
            "sbs_performance": sbs_condition(performance),
            "sbs": single_best_solver(train_ranks),
        }

    def _run(self, tables, run_paths, **overrides):
        config = SelectionConfig(
            feature_variants=("rank",),
            model=ModelConfig(),
            **{"include_best_sampled_point": False, **overrides},
        )
        return run_selection(
            tables["features"],
            tables["mean_superiorities"],
            tables["selected_performance"],
            tables["sbs_performance"],
            run_paths,
            "debug",
            CONFIG_HASH,
            DIM,
            42,
            SAMPLE_SIZE,
            config,
            split=STAGE_SPLIT,
        )

    def test_the_superiority_vs_sbs_uses_every_trial(self, tables, run_paths):
        """With one feature sample, the comparison still uses every trial."""
        written = self._run(tables, run_paths)
        assert not written.empty
        assert set(written["n_trials"]) == {self.N_TRIALS}

    def test_one_observation_means_one_choice_per_instance(self, tables, run_paths):
        written = self._run(tables, run_paths)
        assert (written["n_selected_algorithms"] == 1).all()
        assert written["modal_share"].sub(1.0).abs().max() < 1e-12

    def test_the_sample_credit_still_applies_to_every_trial(self, tables, run_paths):
        """The one sample is credited to all of its trials, not just the first."""
        without = self._run(tables, run_paths)
        with_credit = self._run(tables, run_paths, include_best_sampled_point=True)
        assert set(with_credit["n_trials"]) == {self.N_TRIALS}
        merged = without.merge(
            with_credit,
            on=["function_id", "instance_id", "feature_variant"],
            suffixes=("_off", "_on"),
        )
        assert not merged.empty
        assert (
            merged["median_best_y_selection_on"]
            <= merged["median_best_y_selection_off"] + 1e-12
        ).all()


class TestConfigFor:
    """One training run's configuration: the cell's, re-seeded."""

    @pytest.fixture
    def config(self) -> SelectionConfig:
        return SelectionConfig(
            algorithms=ALGORITHMS[:2],
            model=ModelConfig(kind="random-forest"),
        )

    def test_each_repetition_gets_its_own_model_seed(self, config):
        seeds = {config_for(config, seed).model.random_state for seed in MODEL_SEEDS}
        assert seeds == set(MODEL_SEEDS)

    def test_nothing_else_of_the_cell_moves(self, config):
        """Re-seeding changes nothing of the cell but the model seed."""
        moved = config_for(config, MODEL_SEEDS[3])
        assert moved.algorithms == config.algorithms
        assert moved.feature_variants == config.feature_variants
        assert moved.model.kind == config.model.kind

    def test_a_seed_outside_the_study_is_refused(self, config):
        with pytest.raises(ValueError, match="not one of the study"):
            config_for(config, 9999)


class TestTrainingRuns:
    """Every training run is fitted and kept; none is chosen."""

    def test_a_cell_fits_the_study_s_eleven_runs_by_default(self):
        assert SelectionConfig().model_seeds == MODEL_SEEDS

    def test_a_smaller_count_is_the_first_of_the_seeds(self):
        assert SelectionConfig(n_seeds=3).model_seeds == MODEL_SEEDS[:3]


class TestDeploymentRecords:
    """What one fitted selector's deployments write: their means, and every problem."""

    @pytest.fixture
    def scored(self, run_paths):
        features, performance, mean_superiorities = make_bbob_tables(n_instances=8)
        config = SelectionConfig(
            feature_variants=("value",),
            feature_sets=(ELA_FEATURE_SET,),
            model=ModelConfig(),
            include_best_sampled_point=False,
        )
        prepared = prepare_selection(
            {"bbob": features},
            {"bbob": mean_superiorities},
            performance,
            sbs_condition(performance),
            DIM,
            42,
            SAMPLE_SIZE,
            config,
            split=STAGE_SPLIT,
        )
        matrix, columns = training_matrix(
            prepared.train_features,
            prepared.train_mean_superiorities,
            "value",
            ELA_FEATURE_SET,
            config,
        )
        models = {
            name: fit_member(matrix, columns, config, name)
            for name in config.algorithms
        }
        return score_selector(
            prepared.block, models, columns, "value", ELA_FEATURE_SET, config
        )

    def test_every_deployment_is_kept(self, scored):
        assert len(scored.runs) == len(scored.earned) > 1

    def test_one_record_per_deployment_with_no_deployed_mark(self, scored):
        rows = seed_deployment_rows(scored, ELA_FEATURE_SET, "value", "pre", DIM, 42)
        assert [row["rep"] for row in rows] == sorted(scored.earned)
        assert all("deployed" not in row for row in rows)
        assert {row["seed"] for row in rows} == {42}

    def test_one_problem_row_per_problem_and_deployment(self, scored):
        rows = problem_rows(scored, ELA_FEATURE_SET, "value", "pre", 42)
        problems = len(STAGE_TEST) * 3
        assert len(rows) == problems * len(scored.runs)
        assert {row["rep"] for row in rows} == set(scored.runs)

    def test_a_deployment_s_mean_is_the_mean_of_its_problem_rows(self, scored):
        """A deployment's mean equals the mean of its problem rows, bit for bit."""
        means = {
            row["rep"]: row["superiority_vs_sbs"]
            for row in seed_deployment_rows(
                scored, ELA_FEATURE_SET, "value", "pre", DIM, 42
            )
        }
        problems = pd.DataFrame(
            problem_rows(scored, ELA_FEATURE_SET, "value", "pre", 42)
        )
        for rep, rows in problems.groupby("rep"):
            assert (
                float(np.mean(rows["superiority_vs_sbs"].tolist()))
                == means[int(cast(int, rep))]
            )
