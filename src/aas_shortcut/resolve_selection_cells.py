"""Selection cells: one unit per feature set, encoding and training condition.

A cell's unit holds, for every family's held-out block under both views, each
training run's deployment scores and per-problem superiorities; each family's oracle
rows are a unit of their own. Each training run is fitted once and scored on every
family and view, since a selector depends only on the training side. Missing cells
are fitted in one pass; a cell's result does not depend on which cells share it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
from joblib import delayed

from aas_shortcut.bind_portfolio_parameters import CONDITION_SBS, CONDITION_SELECTION
from aas_shortcut.compare_selection_runs import FAMILY_COLUMN
from aas_shortcut.define_experiment_context import ExperimentContext
from aas_shortcut.define_feature_columns import paper_columns
from aas_shortcut.define_paths import TRANSFORMS, RunPaths
from aas_shortcut.define_table_schema import (
    SELECTION_DEPLOYMENT_SCORE_SCHEMA,
    SELECTION_PROBLEM_SCORE_SCHEMA,
    SELECTION_SCHEMA,
    VIEW_SELECTION_SCHEMA,
    conform,
)
from aas_shortcut.execute_parallel_jobs import parallel_for
from aas_shortcut.match_update_budget import learning_settings_of
from aas_shortcut.measure_selection_transfer import (
    PROBLEM_SHARD,
    SEED_DEPLOYMENT_SHARD,
    SPLIT,
    TRAINING_CONDITIONS,
    TRAINING_VIEWS,
)
from aas_shortcut.read_write_tables import write_table
from aas_shortcut.resolve_portfolio_blocks import (
    CONDITIONS,
    SelectionHeldOut,
    SelectionTraining,
    selection_inputs,
    training_mean_superiority_role,
)
from aas_shortcut.resolve_stage_artifacts import StageArtifact, StageStore, StageUnit
from aas_shortcut.select_algorithms import (
    SelectionBlock,
    SelectionConfig,
    best_sampled_values,
    config_for,
    fit_member,
    oracle_rows,
    prepare_selection,
    problem_rows,
    score_selector,
    seed_deployment_rows,
    training_matrix,
)
from aas_shortcut.split_bbob_instances import PROBLEM_CLASS

LOGGER = logging.getLogger(__name__)

# The oracle's table of every held-out problem's superiority.
SELECTION_SHARD = "view_selection"


@dataclass(frozen=True)
class SelectionCell:
    """One selector: a feature set, an encoding, a training condition."""

    feature_set: str
    variant: str
    training: str


def cells_of(
    config: SelectionConfig, trainings: Sequence[str] = TRAINING_CONDITIONS
) -> list[SelectionCell]:
    """Return every cell of a configuration, in scoring order."""
    return [
        SelectionCell(feature_set, variant, training)
        for training in trainings
        for feature_set in config.feature_sets
        for variant in config.feature_variants
    ]


def cell_config(config: SelectionConfig, cell: SelectionCell) -> SelectionConfig:
    """Return the configuration narrowed to one cell and its training views."""
    return replace(
        config,
        feature_sets=(cell.feature_set,),
        feature_variants=(cell.variant,),
        train_transforms=TRAINING_VIEWS[cell.training],
    )


def selection_learning(config: SelectionConfig, training: str) -> dict[str, Any]:
    """Return the fit settings the configuration does not carry, for the fingerprint."""
    return learning_settings_of(config.model, len(TRAINING_VIEWS[training]))


@dataclass(frozen=True)
class SelectionSetting:
    """What every unit of one call shares besides its configuration."""

    # The fixed paper columns the selectors read.
    features: tuple[str, ...]
    dim: int
    base_seed: int
    # Points of one sample: paid from the selector's budget, and its best value is
    # credited.
    sample_size: int

    def as_dict(self) -> dict[str, Any]:
        """Fingerprintable description of the setting."""
        return {
            "features": sorted(self.features),
            "dim": self.dim,
            "base_seed": self.base_seed,
            "sample_size": self.sample_size,
            "split": SPLIT.as_dict(),
        }


def cell_inputs(inputs: Mapping[str, str], training: str) -> dict[str, str]:
    """Return the input fingerprints a selection cell of one training condition reads.

    The cell reads its condition's views at the search budget and the ``pre`` view
    at the total budget (for the SBS). The training features keep both views, since
    they are one unit.
    """
    read = {
        training_mean_superiority_role(view, CONDITION_SELECTION)
        for view in TRAINING_VIEWS[training]
    }
    read.add(training_mean_superiority_role(TRANSFORMS[0], CONDITION_SBS))
    missing = sorted(read - set(inputs))
    if missing:
        raise ValueError(
            f"a {training!r} cell reads the training block's {missing}, which the "
            "block was not resolved with"
        )
    unread = {
        training_mean_superiority_role(view, condition)
        for view in TRANSFORMS
        for condition in CONDITIONS
    } - read
    return {role: value for role, value in inputs.items() if role not in unread}


def selection_cell_unit(
    config: SelectionConfig,
    cell: SelectionCell,
    setting: SelectionSetting,
    families: Sequence[str],
    inputs: Mapping[str, str],
) -> StageUnit:
    """Return the unit of one selection cell, scored on ``families`` (by label)."""
    settings = {
        **cell_config(config, cell).as_dict(),
        "training": cell.training,
        "model_seeds": list(config.model_seeds),
        "families": list(families),
        **setting.as_dict(),
        "learning": selection_learning(config, cell.training),
        "records": [SEED_DEPLOYMENT_SHARD, PROBLEM_SHARD],
    }
    return StageUnit(
        "view_selection", settings, inputs=cell_inputs(inputs, cell.training)
    )


def oracle_unit(
    config: SelectionConfig,
    label: str,
    setting: SelectionSetting,
    trainings: Sequence[str],
    inputs: Mapping[str, str],
) -> StageUnit:
    """Return the unit of one family's oracle rows."""
    settings = {
        "stage": "selection_oracle",
        "family": label,
        "feature_sets": list(config.feature_sets),
        "trainings": list(trainings),
        "transforms": list(TRANSFORMS),
        "algorithms": list(config.algorithms),
        "include_best_sampled_point": config.include_best_sampled_point,
        **setting.as_dict(),
    }
    return StageUnit("selection_oracle", settings, inputs=dict(inputs))


def family_table(label: str, shard: str) -> str:
    """Where a cell's unit holds one family's table."""
    return f"families/{label.replace('/', '_')}/metrics/{shard}.parquet"


def oracle_table() -> str:
    """Where an oracle unit holds its rows."""
    return f"metrics/{SELECTION_SHARD}.parquet"


@dataclass(frozen=True)
class FamilyData:
    """One family's held-out block, as a selector is scored on it."""

    features: pd.DataFrame
    mean_superiorities: pd.DataFrame
    selected_performance: pd.DataFrame
    sbs_performance: pd.DataFrame
    # The fingerprints the family's oracle reads.
    inputs: Mapping[str, str]


@dataclass(frozen=True)
class SelectionData:
    """What every cell of one call is fitted on and scored against."""

    # Training block, both views; mean superiorities at the search budget.
    features: pd.DataFrame
    mean_superiorities: pd.DataFrame
    # Mean superiorities at the total budget, which name the single best solver.
    sbs_mean_superiorities: pd.DataFrame
    # By family label, in the order given.
    blocks: Mapping[str, FamilyData]
    # The fingerprint of every unit a cell reads, by role.
    inputs: Mapping[str, str]


def selection_data(
    training: SelectionTraining,
    blocks: Sequence[SelectionHeldOut],
    features: Sequence[str],
) -> SelectionData:
    """Build what a configuration's cells are fitted on and scored against."""
    kept = list(features)
    source = training.feature_table
    train = source["instance_id"].isin(list(SPLIT.train_instances or ()))
    sbs_role = training_mean_superiority_role(TRANSFORMS[0], CONDITION_SBS)
    sbs_pre = training.inputs[sbs_role]
    families: dict[str, FamilyData] = {}
    for block in blocks:
        table = block.feature_table
        test = table["instance_id"].isin(list(SPLIT.test_instances or ()))
        families[block.label] = FamilyData(
            features=cast(pd.DataFrame, table[test & table["feature"].isin(kept)]),
            mean_superiorities=block.mean_superiorities(CONDITION_SELECTION),
            selected_performance=block.performance(CONDITION_SELECTION),
            sbs_performance=block.performance(CONDITION_SBS),
            inputs={**block.inputs, sbs_role: sbs_pre},
        )
    return SelectionData(
        features=cast(pd.DataFrame, source[train & source["feature"].isin(kept)]),
        mean_superiorities=training.mean_superiorities(CONDITION_SELECTION),
        sbs_mean_superiorities=training.mean_superiorities(CONDITION_SBS),
        blocks=families,
        inputs=selection_inputs(training, blocks),
    )


def _block(
    data: SelectionData,
    label: str,
    config: SelectionConfig,
    setting: SelectionSetting,
    best_sampled: pd.DataFrame | None,
) -> SelectionBlock:
    """Return one family's block in the view ``config`` scores."""
    family = data.blocks[label]
    return prepare_selection(
        {PROBLEM_CLASS: pd.concat([data.features, family.features], ignore_index=True)},
        {
            PROBLEM_CLASS: pd.concat(
                [data.mean_superiorities, family.mean_superiorities], ignore_index=True
            )
        },
        family.selected_performance,
        family.sbs_performance,
        setting.dim,
        setting.base_seed,
        setting.sample_size,
        config,
        sbs_mean_superiorities=data.sbs_mean_superiorities,
        split=SPLIT,
        best_sampled=best_sampled,
    ).block


def _best_sampled(
    data: SelectionData, config: SelectionConfig, setting: SelectionSetting
) -> pd.DataFrame | None:
    """Return the sample's best values on the held-out problems, once for all blocks.

    They are on the raw objective, so every family and view shares them.
    """
    if not config.include_best_sampled_point:
        return None
    test = next(iter(data.blocks.values())).features
    test = cast(pd.DataFrame, test[test["transform"] == TRANSFORMS[0]])
    return best_sampled_values(
        setting.base_seed,
        SPLIT.test_problem_class,
        test,
        setting.dim,
        sorted(test["rep"].unique()),
        setting.sample_size,
        config.n_jobs,
    )


@dataclass(frozen=True)
class CellResult:
    """What one cell's training runs earned, family by family."""

    # By family label: every training run's deployments, both views, in scoring
    # order; ``config_hash`` left blank.
    records: Mapping[str, pd.DataFrame]
    # By family label: every problem of every one of those deployments.
    problems: Mapping[str, pd.DataFrame]


def score_cells(
    data: SelectionData,
    config: SelectionConfig,
    setting: SelectionSetting,
    cells: Sequence[SelectionCell],
    scale: str,
) -> dict[SelectionCell, CellResult]:
    """Fit every cell's training runs once and score them on every block."""
    seeds = config.model_seeds

    # Every training run's members are fitted in one pool.
    matrices = {
        cell: training_matrix(
            data.features,
            data.mean_superiorities,
            cell.variant,
            cell.feature_set,
            cell_config(config, cell),
        )
        for cell in cells
    }
    jobs = [
        (cell, seed, member)
        for cell in cells
        for seed in seeds
        for member in config.algorithms
    ]
    LOGGER.info(
        "selection: %d cell(s) x %d seed(s) x %d member(s) on %d worker(s)",
        len(cells),
        len(seeds),
        len(config.algorithms),
        config.n_jobs,
    )
    fitted = parallel_for(config.n_jobs, len(jobs))(
        delayed(fit_member)(
            matrices[cell][0],
            matrices[cell][1],
            config_for(cell_config(config, cell), seed),
            member,
        )
        for cell, seed, member in jobs
    )
    models: dict[tuple[SelectionCell, int], dict[str, object]] = {}
    for (cell, seed, member), model in zip(jobs, fitted, strict=True):
        models.setdefault((cell, seed), {})[member] = model

    best_sampled = _best_sampled(data, config, setting)
    blocks = {
        (label, view): _block(
            data, label, replace(config, test_transform=view), setting, best_sampled
        )
        for label in data.blocks
        for view in TRANSFORMS
    }

    results: dict[SelectionCell, CellResult] = {}
    for cell in cells:
        recorded: dict[str, list[dict]] = {label: [] for label in data.blocks}
        problems: dict[str, list[dict]] = {label: [] for label in data.blocks}
        for seed in seeds:
            fitted_cell = config_for(cell_config(config, cell), seed)
            for label in data.blocks:
                for view in TRANSFORMS:
                    scored = score_selector(
                        blocks[(label, view)],
                        models[(cell, seed)],
                        matrices[cell][1],
                        cell.variant,
                        cell.feature_set,
                        replace(fitted_cell, test_transform=view),
                    )
                    recorded[label].extend(
                        {**row, "training": cell.training}
                        for row in seed_deployment_rows(
                            scored,
                            cell.feature_set,
                            cell.variant,
                            view,
                            setting.dim,
                            seed,
                        )
                    )
                    problems[label].extend(
                        {**row, "training": cell.training}
                        for row in problem_rows(
                            scored, cell.feature_set, cell.variant, view, seed
                        )
                    )
        results[cell] = CellResult(
            records={
                label: pd.DataFrame(rows).assign(scale=scale, config_hash="")
                for label, rows in recorded.items()
            },
            problems={
                label: pd.DataFrame(rows).assign(scale=scale, config_hash="")
                for label, rows in problems.items()
            },
        )
    return results


def write_cell(root: Path, result: CellResult, fingerprint: str) -> None:
    """Write one cell's tables into its unit, stamped with its fingerprint."""
    for label, table in result.records.items():
        write_table(
            table.assign(config_hash=fingerprint),
            root / family_table(label, SEED_DEPLOYMENT_SHARD),
            SELECTION_DEPLOYMENT_SCORE_SCHEMA,
        )
    for label, table in result.problems.items():
        write_table(
            table.assign(config_hash=fingerprint),
            root / family_table(label, PROBLEM_SHARD),
            SELECTION_PROBLEM_SCORE_SCHEMA,
        )


def resolve_selection_cells(
    store: StageStore,
    config: SelectionConfig,
    cells: Sequence[SelectionCell],
    data: SelectionData,
    setting: SelectionSetting,
    scale: str,
) -> dict[SelectionCell, StageArtifact]:
    """Return every cell, fitting only the missing ones, together."""
    families = list(data.blocks)
    units = {
        cell: selection_cell_unit(config, cell, setting, families, data.inputs)
        for cell in cells
    }

    def compute(
        missing: Sequence[StageUnit], scratches: Mapping[str, RunPaths]
    ) -> None:
        wanted = [cell for cell, unit in units.items() if unit in missing]
        results = score_cells(data, config, setting, wanted, scale)
        for cell in wanted:
            fingerprint = units[cell].fingerprint
            write_cell(Path(scratches[fingerprint].root), results[cell], fingerprint)

    artifacts = store.resolve_together(list(units.values()), compute)
    return {cell: artifacts[unit.fingerprint] for cell, unit in units.items()}


def oracle_rows_of(
    data: SelectionData,
    label: str,
    config: SelectionConfig,
    setting: SelectionSetting,
    scale: str,
    trainings: Sequence[str],
    best_sampled: pd.DataFrame | None,
) -> pd.DataFrame:
    """Return one family's oracle rows, both views, in a fixed order."""
    parts = []
    for training in trainings:
        for view in TRANSFORMS:
            block = _block(
                data,
                label,
                replace(
                    config,
                    train_transforms=TRAINING_VIEWS[training],
                    test_transform=view,
                ),
                setting,
                best_sampled,
            )
            parts.append(
                conform(
                    pd.DataFrame(oracle_rows(block, config)).assign(
                        scale=scale, config_hash=""
                    ),
                    SELECTION_SCHEMA,
                ).assign(transform=view, training=training)
            )
    return pd.concat(parts, ignore_index=True)


def resolve_oracles(
    store: StageStore,
    config: SelectionConfig,
    data: SelectionData,
    setting: SelectionSetting,
    scale: str,
    trainings: Sequence[str] = TRAINING_CONDITIONS,
) -> dict[str, StageArtifact]:
    """Return every family's oracle rows, by label, computing what is missing."""
    best: list[pd.DataFrame | None] = []
    artifacts: dict[str, StageArtifact] = {}
    for label, family in data.blocks.items():
        unit = oracle_unit(config, label, setting, trainings, family.inputs)

        def compute(paths: Any, config_hash: str, label: str = label) -> None:
            if not best:
                best.append(_best_sampled(data, config, setting))
            rows = oracle_rows_of(
                data, label, config, setting, scale, trainings, best[0]
            )
            write_table(
                rows.assign(config_hash=config_hash),
                paths.metric(SELECTION_SHARD).data,
                VIEW_SELECTION_SCHEMA,
            )

        artifacts[label] = store.resolve(unit, compute)
    return artifacts


def assemble_oracles(
    oracles: Mapping[str, StageArtifact], labels: Sequence[str]
) -> pd.DataFrame:
    """Return every family's oracle rows, both views and conditions, by family."""
    return pd.concat(
        [
            oracles[label].read_table(oracle_table()).assign(**{FAMILY_COLUMN: label})
            for label in labels
        ],
        ignore_index=True,
    )


def assemble_problems(
    cells: Mapping[SelectionCell, StageArtifact], labels: Sequence[str]
) -> pd.DataFrame:
    """Return every cell's per-problem rows on every family, with a family column."""
    return pd.concat(
        [
            artifact.read_table(family_table(label, PROBLEM_SHARD)).assign(
                **{FAMILY_COLUMN: label}
            )
            for label in labels
            for artifact in cells.values()
        ],
        ignore_index=True,
    )


def assemble_records(
    cells: Mapping[SelectionCell, StageArtifact], labels: Sequence[str]
) -> pd.DataFrame:
    """Return every cell's deployment rows on every family, with a family column."""
    return pd.concat(
        [
            artifact.read_table(family_table(label, SEED_DEPLOYMENT_SHARD)).assign(
                **{FAMILY_COLUMN: label}
            )
            for label in labels
            for artifact in cells.values()
        ],
        ignore_index=True,
    )


@dataclass(frozen=True)
class FamilySelection:
    """Experiment 5's selection units, resolved once, and the tables they hold."""

    cells: Mapping[SelectionCell, StageArtifact]
    oracles: Mapping[str, StageArtifact]
    # The blocks' families, in the blocks' order.
    labels: tuple[str, ...]

    def records(self) -> pd.DataFrame:
        """The cells' records, joined over every family."""
        return assemble_records(self.cells, self.labels)

    def oracle(self) -> pd.DataFrame:
        """The oracles' rows, joined over every family."""
        return assemble_oracles(self.oracles, self.labels)

    def problems(self, labels: Sequence[str]) -> pd.DataFrame:
        """The cells' per-problem rows on the families named."""
        return assemble_problems(self.cells, labels)


def resolve_family_records(
    store: StageStore,
    context: ExperimentContext,
    training: SelectionTraining,
    blocks: Sequence[SelectionHeldOut],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Experiment 5's cell records and oracle rows, joined over families."""
    units = resolve_family_selection(store, context, training, blocks)
    return units.records(), units.oracle()


def resolve_family_selection(
    store: StageStore,
    context: ExperimentContext,
    training: SelectionTraining,
    blocks: Sequence[SelectionHeldOut],
) -> FamilySelection:
    """Resolve every selection cell of Experiment 5, and every family's oracle.

    Experiment 5 and the appendix read the selection through here, so they cannot
    name different units.
    """
    config = context.selection_config()
    features = paper_columns(context.feature_sets, context.scale)
    setting = SelectionSetting(
        features=tuple(features),
        dim=context.dim,
        base_seed=context.seed,
        sample_size=context.feature_config(PROBLEM_CLASS).sample_size,
    )
    data = selection_data(training, blocks, features)
    cells = resolve_selection_cells(
        store, config, cells_of(config), data, setting, context.scale
    )
    oracles = resolve_oracles(store, config, data, setting, context.scale)
    return FamilySelection(cells, oracles, tuple(block.label for block in blocks))
