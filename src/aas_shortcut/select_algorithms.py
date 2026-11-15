"""Stage ``selection``: whether landscape analysis pays for itself."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from aas_shortcut.augment_training_samples import AUGMENTED_ID_COLUMNS
from aas_shortcut.build_feature_matrix import (
    INSTANCE_COLUMNS,
    build_feature_matrix,
    feature_columns,
)
from aas_shortcut.build_regression_model import ModelConfig, fit_model
from aas_shortcut.calculate_superiority import vargha_delaney
from aas_shortcut.create_sampled_points import create_sampled_points
from aas_shortcut.define_algorithm_portfolio import ALGORITHMS
from aas_shortcut.define_paths import (
    ELA_FEATURE_SET,
    FEATURE_VARIANTS,
    ORACLE_VARIANT,
    fingerprint_variants,
)
from aas_shortcut.define_repetitions import MODEL_SEEDS, REPETITIONS, model_for
from aas_shortcut.match_update_budget import matched_config
from aas_shortcut.predict_ranking import join_mean_superiorities, target_column
from aas_shortcut.predict_superiority import restrict_to_common_features
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS

LOGGER = logging.getLogger(__name__)

# Objective view the selector is trained under.
TRANSFORM = "pre"


def view_of(table: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Narrow a table to one view of the objective, if it carries several."""
    if "transform" not in table.columns:
        return table
    rows = table[table["transform"] == transform]
    if rows.empty:
        held = sorted(str(value) for value in table["transform"].unique())
        raise ValueError(f"the table holds no {transform!r} row; it covers {held}")
    return rows.drop(columns=["transform"])


@dataclass(frozen=True)
class SelectionSplit:
    """Which rows train the selector and which rows it is measured on."""

    train_problem_class: str = PROBLEM_CLASS
    test_problem_class: str = PROBLEM_CLASS
    train_instances: tuple[int, ...] | None = None
    test_instances: tuple[int, ...] | None = None

    def required_classes(self) -> tuple[str, ...]:
        """Suites a caller must supply, without repeating a shared one."""
        return tuple(dict.fromkeys((self.train_problem_class, self.test_problem_class)))

    def train(self, by_class: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return the training rows of a per-suite table."""
        return _restrict(by_class[self.train_problem_class], self.train_instances)

    def test(self, by_class: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return the test rows of a per-suite table."""
        return _restrict(by_class[self.test_problem_class], self.test_instances)

    def as_dict(self) -> dict:
        """Fingerprintable description of the split."""
        return {
            "train_problem_class": self.train_problem_class,
            "test_problem_class": self.test_problem_class,
            "train_instances": list(self.train_instances or ()),
            "test_instances": list(self.test_instances or ()),
        }


def _restrict(frame: pd.DataFrame, instances: tuple[int, ...] | None) -> pd.DataFrame:
    """Narrow a table to a set of instance ids, or leave it whole if ``None``."""
    if instances is None:
        return frame
    narrowed = frame[frame["instance_id"].isin(instances)]
    if narrowed.empty:
        raise ValueError(
            f"no row of the table has an instance id in {sorted(instances)}; "
            "the shard does not cover the split this run asks for"
        )
    return narrowed


@dataclass(frozen=True)
class SelectionConfig:
    """Settings of the selection experiment."""

    feature_variants: tuple[str, ...] = FEATURE_VARIANTS
    feature_sets: tuple[str, ...] = (ELA_FEATURE_SET,)
    algorithms: tuple[str, ...] = ALGORITHMS
    # The regressor fitted once per member; the same for every representation.
    model: ModelConfig = field(default_factory=ModelConfig)
    # Credit the selection method with the best point its sample evaluated.
    include_best_sampled_point: bool = True
    # Score the perfect selector (the Oracle) alongside the learned ones.
    include_oracle: bool = True
    # Views the selector is fitted on: ``pre``, or both for an augmented cell, each
    # with the target its own view earns.
    train_transforms: tuple[str, ...] = (TRANSFORM,)
    # View the test block is both described by and scored under; they must match.
    test_transform: str = TRANSFORM
    # Workers for the sample re-evaluation pass.
    n_jobs: int = 1
    # Training runs per cell (the first model seeds); tables report their median.
    n_seeds: int = REPETITIONS

    @property
    def model_seeds(self) -> tuple[int, ...]:
        """The training runs every cell fits, in the study's order."""
        return MODEL_SEEDS[: self.n_seeds]

    def as_dict(self) -> dict:
        """Fingerprintable description of the configuration."""
        return {
            "stage": "selection",
            # Omitted at the single-view default, which keeps that fingerprint stable.
            **(
                {}
                if self.train_transforms == (TRANSFORM,)
                and self.test_transform == TRANSFORM
                else {
                    "train_transforms": list(self.train_transforms),
                    "test_transform": self.test_transform,
                }
            ),
            "feature_variants": fingerprint_variants(self.feature_variants),
            "feature_sets": list(self.feature_sets),
            "algorithms": list(self.algorithms),
            "include_best_sampled_point": self.include_best_sampled_point,
            "include_oracle": self.include_oracle,
            "model": self.model.as_dict(),
            # So a shard fitted on another target is never reused for this one.
            "target": {"kind": "mean-superiority"},
        }


def instance_winners(
    mean_superiorities: pd.DataFrame,
    algorithms: Sequence[str] = ALGORITHMS,
) -> pd.Series:
    """Return the member with the best mean superiority on each instance.

    Ties go to the earlier member in portfolio order.
    """
    candidates = mean_superiorities[
        mean_superiorities["algorithm"].isin(list(algorithms))
    ]
    if candidates.empty:
        raise ValueError(f"no mean superiorities for any of {list(algorithms)}")

    key = [column for column in INSTANCE_COLUMNS if column in candidates.columns]
    if "instance_id" not in key:
        key = ["instance_id"]
    order = {name: index for index, name in enumerate(algorithms)}
    ranked = candidates.assign(
        _tiebreak=candidates["algorithm"].map(order).astype("int64")
    ).sort_values(
        ["mean_superiority", "_tiebreak"], ascending=[False, True], kind="stable"
    )
    best = ranked.groupby(key, observed=True, as_index=False).first()
    return best.set_index(key)["algorithm"]


def rank_single_best_solver(
    mean_superiorities: pd.DataFrame,
    algorithms: Sequence[str] = ALGORITHMS,
) -> pd.DataFrame:
    """Return each member's aggregate over the block, flagging the single best solver.

    As in ASlib (Kerschke et al.), the single best solver has the best aggregate over
    the problems (largest average mean superiority, ties to portfolio order), not the
    most problems won.
    """
    candidates = mean_superiorities[
        mean_superiorities["algorithm"].isin(list(algorithms))
    ]
    if candidates.empty:
        raise ValueError(f"no mean superiorities for any of {list(algorithms)}")

    aggregate = cast(
        pd.Series,
        candidates.groupby("algorithm", observed=True)["mean_superiority"].mean(),
    )
    available = [name for name in algorithms if name in aggregate.index]
    best = min(
        available, key=lambda name: (-float(aggregate[name]), algorithms.index(name))
    )
    return pd.DataFrame(
        [
            {
                "algorithm": name,
                "mean_superiority": float(aggregate[name]),
                "n_problems": int((candidates["algorithm"] == name).sum()),
                "is_sbs": name == best,
            }
            for name in available
        ]
    )


def single_best_solver(
    mean_superiorities: pd.DataFrame,
    algorithms: Sequence[str] = ALGORITHMS,
) -> str:
    """Return the member flagged by :func:`rank_single_best_solver`."""
    ranking = rank_single_best_solver(mean_superiorities, algorithms)
    best = ranking[ranking["is_sbs"]].iloc[0]
    LOGGER.info(
        "single best solver: %s (average mean superiority %.4f over %d problem(s))",
        best["algorithm"],
        float(best["mean_superiority"]),
        int(best["n_problems"]),
    )
    return str(best["algorithm"])


def oracle_choices(
    test_mean_superiorities: pd.DataFrame,
    reps: Sequence[int],
    algorithms: Sequence[str] = ALGORITHMS,
) -> pd.DataFrame:
    """Choose, per test instance, the member that actually performed best."""
    chosen = (
        instance_winners(test_mean_superiorities, algorithms)
        .rename("selected_algorithm")
        .reset_index()
    )
    LOGGER.info(
        "oracle: %d instance(s), %d distinct choice(s)",
        len(chosen),
        int(chosen["selected_algorithm"].nunique()),
    )
    return chosen.merge(pd.DataFrame({"rep": [int(rep) for rep in reps]}), how="cross")


def assign_trials(reps: Sequence[int], trials: Sequence[int]) -> pd.DataFrame:
    """Map each algorithm trial to the sampling repetition that selected for it."""
    reps = sorted(reps)
    trials = sorted(trials)
    if not reps or not trials:
        raise ValueError(
            f"cannot pair {len(reps)} repetition(s) with {len(trials)} trial(s)"
        )
    return pd.DataFrame(
        {
            "rep": [reps[index % len(reps)] for index in range(len(trials))],
            "trial": trials,
        }
    )


def _best_sampled_job(
    base_seed: int,
    problem_class: str,
    function_id: int,
    instance_id: int,
    dim: int,
    rep: int,
    sample_size: int,
) -> dict:
    """Return the best raw objective value one sampling repetition evaluated."""
    sample = create_sampled_points(
        base_seed=base_seed,
        problem_class=problem_class,
        function_id=function_id,
        instance_id=instance_id,
        dim=dim,
        rep=rep,
        sample_size=sample_size,
    )
    return {
        "function_id": int(function_id),
        "instance_id": int(instance_id),
        "rep": int(rep),
        "best_sampled_y": float(np.min(sample.y_pre)),
    }


def best_sampled_values(
    base_seed: int,
    problem_class: str,
    instances: pd.DataFrame,
    dim: int,
    reps: Sequence[int],
    sample_size: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Re-evaluate every sampling repetition and record the best point it found."""
    keys = instances[["function_id", "instance_id"]].drop_duplicates()
    jobs = [
        (int(function_id), int(instance_id), int(rep))
        for function_id, instance_id in keys.itertuples(index=False, name=None)
        for rep in reps
    ]
    started = time.perf_counter()
    rows = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_best_sampled_job)(
            base_seed, problem_class, function_id, instance_id, dim, rep, sample_size
        )
        for function_id, instance_id, rep in jobs
    )
    LOGGER.info(
        "selection: re-evaluated %d sampling repetition(s) in %.1fs",
        len(jobs),
        time.perf_counter() - started,
    )
    return pd.DataFrame(rows)


def training_matrix(
    train_features: pd.DataFrame,
    mean_superiorities: pd.DataFrame,
    variant: str,
    feature_set: str,
    config: SelectionConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble the rows one cell is fitted on, and name its predictor columns.

    Separate from the fit so that a cell's training runs share one matrix.
    """
    blocks: list[pd.DataFrame] = []
    for transform in config.train_transforms:
        wide = build_feature_matrix(
            train_features, transform, variant, feature_set=feature_set
        )
        blocks.append(
            join_mean_superiorities(
                wide,
                view_of(mean_superiorities, transform),
                config.algorithms,
                validate="1:1",
            )
        )
    matrix = blocks[0] if len(blocks) == 1 else pd.concat(blocks, ignore_index=True)

    targets = {target_column(name) for name in config.algorithms}
    identifiers = {*INSTANCE_COLUMNS, *AUGMENTED_ID_COLUMNS}
    columns = [
        column
        for column in feature_columns(matrix)
        if column not in targets and column not in identifiers
    ]
    return matrix, columns


def config_for(config: SelectionConfig, seed: int) -> SelectionConfig:
    """Return a cell's configuration with one training run's model seed."""
    if seed not in MODEL_SEEDS:
        raise ValueError(
            f"seed {seed} is not one of the study's repetitions {list(MODEL_SEEDS)}"
        )
    return replace(config, model=model_for(config.model, MODEL_SEEDS.index(seed)))


def fit_member(
    matrix: pd.DataFrame, columns: list[str], config: SelectionConfig, name: str
) -> object:
    """Fit one member's regressor on a cell's training rows.

    The epoch budget is divided by the number of training views, so an augmented
    matrix gets the same number of updates as a one-view fit.
    """
    return fit_model(
        matrix,
        columns,
        matched_config(config.model, len(config.train_transforms)),
        target_column(name),
        group_columns=INSTANCE_COLUMNS,
    )


def select_per_repetition(
    test_features: pd.DataFrame,
    models: dict[str, object],
    columns: list[str],
    variant: str,
    feature_set: str,
    algorithms: Sequence[str],
    transform: str = TRANSFORM,
) -> pd.DataFrame:
    """Choose one algorithm per ``(instance, repetition)``, from one view."""
    rows: list[pd.DataFrame] = []
    for rep, group in test_features.groupby("rep", observed=True):
        matrix = build_feature_matrix(
            group, transform, variant, feature_set=feature_set
        )
        missing = [column for column in columns if column not in matrix.columns]
        if missing:
            raise ValueError(
                f"repetition {rep} is missing {len(missing)} feature column(s) the "
                f"selector was fitted on, e.g. {missing[:3]}"
            )
        X = matrix[columns].to_numpy(dtype=np.float64)
        predicted = np.column_stack([models[name].predict(X) for name in algorithms])
        # ``argmax`` breaks a tie by portfolio order, deterministically.
        chosen = [algorithms[index] for index in np.argmax(predicted, axis=1)]
        rows.append(
            matrix[list(INSTANCE_COLUMNS)].assign(
                rep=int(cast(int, rep)), selected_algorithm=chosen
            )
        )
    return pd.concat(rows, ignore_index=True)


class Deployments(NamedTuple):
    """What one fitted selector earned on every deployment of one block."""

    # By deployment: one row per held-out problem.
    runs: dict[int, list[dict]]
    # By deployment: the mean superiority over the held-out problems.
    earned: dict[int, float]


def _deploy(
    test_features: pd.DataFrame,
    models: dict[str, object],
    columns: list[str],
    variant: str,
    feature_set: str,
    config: SelectionConfig,
    *,
    reps: Sequence[int],
    trials: Sequence[int],
    at_search_budget: pd.DataFrame,
    best_sampled: pd.DataFrame | None,
    on_sbs: pd.DataFrame,
    identity: dict,
    sbs_algorithm: str,
    selection_budget: int,
    sbs_budget: int,
) -> Deployments:
    """Deploy one fitted selector once per sampling repetition.

    A deployment is one sample of the held-out problem, so each repetition gets every
    trial.
    """
    runs: dict[int, list[dict]] = {}
    for rep in reps:
        deployment = cast(pd.DataFrame, test_features[test_features["rep"] == rep])
        selected = select_per_repetition(
            deployment,
            models,
            columns,
            variant,
            feature_set,
            config.algorithms,
            config.test_transform,
        )
        outcomes = _outcomes(
            selected, assign_trials([rep], trials), at_search_budget, best_sampled
        )
        runs[rep] = _instance_rows(
            outcomes,
            on_sbs,
            {**identity, "feature_variant": variant},
            sbs_algorithm,
            selection_budget,
            sbs_budget,
        )

    means = {
        rep: float(np.mean([row["superiority_vs_sbs"] for row in rows]))
        for rep, rows in runs.items()
    }
    if len(runs) > 1:
        LOGGER.info(
            "selection: %d deployment(s), mean superiority %.4f-%.4f",
            len(runs),
            min(means.values()),
            max(means.values()),
        )
    return Deployments(runs, means)


def _outcomes(
    selected: pd.DataFrame,
    assignment: pd.DataFrame,
    performance: pd.DataFrame,
    best_sampled: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach the objective value the selection method actually achieved."""
    expanded = selected.merge(assignment, on="rep", how="inner")

    runs = performance[
        ["function_id", "instance_id", "trial", "algorithm", "best_y"]
    ].rename(columns={"algorithm": "selected_algorithm"})

    joined = expanded.merge(
        runs,
        on=["function_id", "instance_id", "trial", "selected_algorithm"],
        how="left",
        validate="1:1",
    )
    unmatched = joined["best_y"].isna()
    if bool(unmatched.any()):
        example = joined.loc[unmatched].head(1).to_dict("records")
        raise ValueError(
            f"{int(unmatched.sum())} selected run(s) are absent from the "
            f"performance table, e.g. {example}"
        )

    if best_sampled is not None:
        # Every trial of a repetition is credited with that one sample's best point.
        joined = joined.merge(
            best_sampled,
            on=["function_id", "instance_id", "rep"],
            how="left",
            validate="m:1",
        )
        # A missing value would silently drop the credit, so it is an error.
        if bool(joined["best_sampled_y"].isna().any()):
            raise ValueError("some trial has no sample best value")
        joined["achieved_y"] = np.minimum(joined["best_y"], joined["best_sampled_y"])
    else:
        joined["achieved_y"] = joined["best_y"]
    return joined


def _instance_rows(
    outcomes: pd.DataFrame,
    sbs_runs: pd.DataFrame,
    identity: dict,
    sbs_algorithm: str,
    selection_budget: int,
    sbs_budget: int,
) -> list[dict]:
    """Score one ``(set, variant)`` combination, instance by instance."""
    rows: list[dict] = []
    key = ["problem_class", "function_id", "instance_id"]

    for (problem_class, function_id, instance_id), group in outcomes.groupby(
        key, observed=True, sort=True
    ):
        sbs = sbs_runs[
            (sbs_runs["function_id"] == function_id)
            & (sbs_runs["instance_id"] == instance_id)
        ]
        if sbs.empty:
            LOGGER.warning(
                "no SBS runs for f%s i%s; skipping the instance",
                function_id,
                instance_id,
            )
            continue

        ours = group["achieved_y"].to_numpy(dtype=np.float64)
        theirs = sbs["best_y"].to_numpy(dtype=np.float64)
        counts = group["selected_algorithm"].value_counts()

        rows.append(
            {
                **identity,
                "problem_class": str(problem_class),
                "function_id": int(function_id),
                "instance_id": int(instance_id),
                "modal_algorithm": str(counts.index[0]),
                "modal_share": float(counts.iloc[0] / len(group)),
                "n_selected_algorithms": len(counts),
                "sbs_algorithm": sbs_algorithm,
                "n_trials": int(min(len(ours), len(theirs))),
                "selection_budget": selection_budget,
                "sbs_budget": sbs_budget,
                # Smaller is better, so this is P(selection beats the SBS).
                "superiority_vs_sbs": vargha_delaney(ours, theirs),
                "median_best_y_selection": float(np.median(ours)),
                "median_best_y_sbs": float(np.median(theirs)),
            }
        )
    return rows


def _single_budget(performance: pd.DataFrame, condition: str) -> int:
    """Return the one budget a condition's performance shard was recorded at.

    More than one would mean two conditions were concatenated, so it raises.
    """
    budgets = sorted(int(value) for value in performance["budget"].unique())
    if len(budgets) != 1:
        raise ValueError(
            f"the {condition!r} performance table must hold exactly one budget; "
            f"it has {budgets}"
        )
    return budgets[0]


def _test_features_of_view(test_features: pd.DataFrame, transform: str) -> pd.DataFrame:
    """Narrow the test features to the view being scored, keeping the column."""
    if "transform" not in test_features.columns:
        return test_features
    held = sorted(str(value) for value in test_features["transform"].unique())
    kept = test_features[test_features["transform"] == transform]
    if kept.empty:
        raise ValueError(
            f"the feature shard holds no {transform!r} row for the test block; "
            f"it covers {held}"
        )
    return kept


def _matched_budgets(
    selected_performance: pd.DataFrame, sbs_performance: pd.DataFrame, sample_size: int
) -> tuple[int, int]:
    """Return the two methods' total budgets, having checked that they agree."""
    search_budget = _single_budget(selected_performance, "selection")
    sbs_budget = _single_budget(sbs_performance, "sbs")
    selection_budget = search_budget + sample_size
    if selection_budget != sbs_budget:
        raise ValueError(
            f"the two methods were not given the same budget: selection spends "
            f"{sample_size} on features plus {search_budget} on search = "
            f"{selection_budget}, the SBS was given {sbs_budget}"
        )
    return selection_budget, sbs_budget


@dataclass(frozen=True)
class SelectionBlock:
    """Everything scoring reads about one test block in one view, but the selector."""

    test_features: pd.DataFrame
    test_mean_superiorities: pd.DataFrame
    at_search_budget: pd.DataFrame
    on_sbs: pd.DataFrame
    sbs_algorithm: str
    reps: list[int]
    trials: list[int]
    best_sampled: pd.DataFrame | None
    selection_budget: int
    sbs_budget: int
    dim: int


@dataclass(frozen=True)
class PreparedSelection:
    """The training side a selector is fitted on, and the block it is scored on."""

    train_features: pd.DataFrame
    train_mean_superiorities: pd.DataFrame
    block: SelectionBlock


def prepare_selection(
    features_by_class: dict[str, pd.DataFrame],
    mean_superiorities_by_class: dict[str, pd.DataFrame],
    selected_performance: pd.DataFrame,
    sbs_performance: pd.DataFrame,
    dim: int,
    base_seed: int,
    sample_size: int,
    config: SelectionConfig,
    *,
    sbs_mean_superiorities: pd.DataFrame | None = None,
    split: SelectionSplit | None = None,
    best_sampled: pd.DataFrame | None = None,
) -> PreparedSelection:
    """Split the tables into the training side and the scored test block.

    ``best_sampled`` may be passed in: it is on the raw objective, so it is the same
    for every view and family of one test block.
    """
    split = split or SelectionSplit()
    for problem_class in split.required_classes():
        if problem_class not in features_by_class:
            raise KeyError(f"the selection experiment needs {problem_class} features")

    features_by_class = restrict_to_common_features(features_by_class)
    train_features = split.train(features_by_class)
    test_features = split.test(features_by_class)
    # The training side keeps every view; the test side is the one being scored.
    train_mean_superiorities = split.train(mean_superiorities_by_class)
    test_mean_superiorities = view_of(
        split.test(mean_superiorities_by_class), config.test_transform
    )
    test_features = _test_features_of_view(test_features, config.test_transform)
    selected_performance = view_of(selected_performance, config.test_transform)
    sbs_performance = view_of(sbs_performance, config.test_transform)

    if sbs_mean_superiorities is None:
        LOGGER.warning(
            "no SBS-budget mean superiorities given; naming the single best solver "
            "from the search-budget mean superiorities instead"
        )
        sbs_mean_superiorities = train_mean_superiorities
    else:
        # A held-out problem must take no part in choosing its own baseline.
        sbs_mean_superiorities = _restrict(
            sbs_mean_superiorities, split.train_instances
        )
    # Read off the untransformed training block, so the baseline is the same
    # whichever view the test block is scored under.
    sbs_algorithm = single_best_solver(
        view_of(sbs_mean_superiorities, TRANSFORM), config.algorithms
    )
    selection_budget, sbs_budget = _matched_budgets(
        selected_performance, sbs_performance, sample_size
    )

    at_search_budget = selected_performance
    on_sbs = cast(
        pd.DataFrame, sbs_performance[sbs_performance["algorithm"] == sbs_algorithm]
    )
    if on_sbs.empty:
        raise ValueError(
            f"the sbs performance table holds no run for the single best solver "
            f"{sbs_algorithm!r} at budget {sbs_budget}"
        )

    reps = sorted(test_features["rep"].unique())
    trials = sorted(at_search_budget["trial"].unique())
    LOGGER.info(
        "selection: %d sampling repetition(s) deployed over %d trial(s)",
        len(reps),
        len(trials),
    )

    if config.include_best_sampled_point and best_sampled is None:
        best_sampled = best_sampled_values(
            base_seed,
            split.test_problem_class,
            test_features,
            dim,
            reps,
            sample_size,
            config.n_jobs,
        )
    elif not config.include_best_sampled_point:
        best_sampled = None
    block = SelectionBlock(
        test_features=test_features,
        test_mean_superiorities=test_mean_superiorities,
        at_search_budget=at_search_budget,
        on_sbs=on_sbs,
        sbs_algorithm=sbs_algorithm,
        reps=reps,
        trials=trials,
        best_sampled=best_sampled,
        selection_budget=selection_budget,
        sbs_budget=sbs_budget,
        dim=dim,
    )
    return PreparedSelection(train_features, train_mean_superiorities, block)


def oracle_rows(block: SelectionBlock, config: SelectionConfig) -> list[dict]:
    """Score the perfect selector on a block, once per feature set."""
    oracle = _outcomes(
        oracle_choices(block.test_mean_superiorities, block.reps, config.algorithms),
        assign_trials(block.reps, block.trials),
        block.at_search_budget,
        block.best_sampled,
    )
    # The Oracle reads no features; it repeats per feature set only for the table key.
    return [
        row
        for feature_set in config.feature_sets
        for row in _instance_rows(
            oracle,
            block.on_sbs,
            {
                "dimension": block.dim,
                "feature_set": feature_set,
                "feature_variant": ORACLE_VARIANT,
            },
            block.sbs_algorithm,
            block.selection_budget,
            block.sbs_budget,
        )
    ]


def score_selector(
    block: SelectionBlock,
    models: dict[str, object],
    columns: list[str],
    variant: str,
    feature_set: str,
    config: SelectionConfig,
) -> Deployments:
    """Deploy one fitted selector on a block, once per sampling repetition."""
    return _deploy(
        block.test_features,
        models,
        columns,
        variant,
        feature_set,
        config,
        reps=block.reps,
        trials=block.trials,
        at_search_budget=block.at_search_budget,
        best_sampled=block.best_sampled,
        on_sbs=block.on_sbs,
        identity={"dimension": block.dim, "feature_set": feature_set},
        sbs_algorithm=block.sbs_algorithm,
        selection_budget=block.selection_budget,
        sbs_budget=block.sbs_budget,
    )


def seed_deployment_rows(
    scored: Deployments,
    feature_set: str,
    variant: str,
    transform: str,
    dim: int,
    seed: int,
) -> list[dict]:
    """Return one row per deployment: the mean over the held-out problems."""
    return [
        {
            "feature_set": feature_set,
            "feature_variant": variant,
            "transform": transform,
            "dimension": dim,
            "seed": int(seed),
            "rep": int(rep),
            "superiority_vs_sbs": float(value),
        }
        for rep, value in sorted(scored.earned.items())
    ]


def problem_rows(
    scored: Deployments,
    feature_set: str,
    variant: str,
    transform: str,
    seed: int,
) -> list[dict]:
    """Return what one training run's selector earned per problem and deployment."""
    return [
        {
            "feature_set": feature_set,
            "feature_variant": variant,
            "transform": transform,
            "seed": int(seed),
            "rep": int(rep),
            "problem_class": row["problem_class"],
            "function_id": row["function_id"],
            "instance_id": row["instance_id"],
            "dimension": row["dimension"],
            "superiority_vs_sbs": row["superiority_vs_sbs"],
        }
        for rep, rows in sorted(scored.runs.items())
        for row in rows
    ]
