"""Whether two selectors differ by more than their randomness.

A seed's value is the median over its deployments of the mean superiority over
the SBS on the transformed block; a table's level is the median over seeds, so the
tests and the tables read the same observations. The tests are Experiment 4's
(`compare_training_runs`): representations paired by seed (Wilcoxon), training
conditions unpaired (exact Mann-Whitney), Holm within each family.

Selection values tie (seeds that choose alike score alike), so ties and zero
differences are decided within `compare_training_runs.TIE_TOLERANCE`, since the
same superiority summed in different orders differs in the last bits.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from aas_shortcut.compare_training_runs import (
    ALPHA,
    CONDITIONS,
    EFFECT_COLUMN,
    PAIRED,
    PAIRED_EFFECT,
    SEED_COLUMN,
    SIZE_COLUMN,
    UNIT,
    UNPAIRED,
    UNPAIRED_EFFECT,
    compare_paired_runs,
    compare_unpaired_runs,
    holm,
    median_over_seeds,
)
from aas_shortcut.define_paths import TRANSFORMS
from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_reporting_scopes import SCOPE_COLUMN, scope_mean

CELL_COLUMNS: tuple[str, ...] = ("feature_set", "feature_variant", "training")

# The family a record was scored on; a cell's unit keeps it in the record's
# place, not in a column (`family_table`).
FAMILY_COLUMN = "family"

# The tests read the transformed block, as Experiment 4's do.
TESTED_VIEW = TRANSFORMS[1]

PROBLEM_COLUMNS: tuple[str, ...] = (
    "problem_class",
    "function_id",
    "instance_id",
    "dimension",
)


def seed_values(
    records: pd.DataFrame,
    by: Sequence[str] = (),
    view: str = TESTED_VIEW,
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Reduce every seed's deployments to their median, on one view."""
    rows = records[records["transform"].astype(str) == view]
    if rows.empty:
        raise ValueError(f"the records hold no {view!r} deployment")
    keys = [*CELL_COLUMNS, *by, SEED_COLUMN]
    values = (
        rows.groupby(keys, observed=True)["superiority_vs_sbs"].median().reset_index()
    )
    values = values.rename(columns={"superiority_vs_sbs": "value"})
    held = values.groupby([*CELL_COLUMNS, *by], observed=True)[SEED_COLUMN].agg(set)
    short = held[held.map(lambda found: not set(seeds) <= found)]
    if not short.empty:
        raise ValueError(
            f"{len(short)} cell(s) do not carry every training run, e.g. "
            f"{short.index[0]} with {sorted(short.iloc[0])}"
        )
    return values


def selection_runs(records: pd.DataFrame, scope: str) -> pd.DataFrame:
    """One value per seed of every cell: the mean over the range's families."""
    per_family = seed_values(records, by=(FAMILY_COLUMN,))
    return scope_mean(per_family, [*CELL_COLUMNS, SEED_COLUMN], scope, FAMILY_COLUMN)


def selection_levels(records: pd.DataFrame, scopes: Sequence[str]) -> pd.DataFrame:
    """Each cell's level over each range: the median of its eleven seeds' values."""
    frames = [
        median_over_seeds(selection_runs(records, scope), CELL_COLUMNS).assign(
            **{SCOPE_COLUMN: scope}
        )
        for scope in scopes
    ]
    return pd.concat(frames, ignore_index=True)


def view_levels(
    records: pd.DataFrame,
    by: Sequence[str] = (FAMILY_COLUMN,),
    views: Sequence[str] = TRANSFORMS,
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Each cell's level on each view of each family: the median over its seeds."""
    frames = [
        median_over_seeds(
            seed_values(records, by, view, seeds), [*CELL_COLUMNS, *by]
        ).assign(transform=view)
        for view in views
    ]
    return pd.concat(frames, ignore_index=True)


def problem_levels(
    problems: pd.DataFrame,
    by: Sequence[str] = (FAMILY_COLUMN,),
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Each problem's level in each cell: the median of its seeds' median deployment."""
    keys = [*CELL_COLUMNS, *by, "transform", *PROBLEM_COLUMNS]
    per_seed = (
        problems.groupby([*keys, SEED_COLUMN], observed=True)["superiority_vs_sbs"]
        .median()
        .reset_index()
    )
    held = per_seed.groupby(keys, observed=True)[SEED_COLUMN].agg(set)
    short = held[held.map(lambda found: not set(seeds) <= found)]
    if not short.empty:
        raise ValueError(
            f"{len(short)} problem(s) do not carry every training run, e.g. "
            f"{short.index[0]} with {sorted(short.iloc[0])}"
        )
    return (
        per_seed.groupby(keys, observed=True)["superiority_vs_sbs"]
        .median()
        .reset_index()
    )


def compare_selection_representations(runs: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Test the three representations against each other, paired by seed.

    Holm corrects over the three pairs of one (feature set, training condition).
    """
    rows: list[dict] = []
    for (feature_set, training), cell in runs.groupby(
        ["feature_set", "training"], observed=True
    ):
        wide = cell.pivot_table(
            index=SEED_COLUMN, columns="feature_variant", values="value"
        )
        variants = list(wide.columns)
        family: list[dict] = []
        for index, left in enumerate(variants):
            for right in variants[index + 1 :]:
                verdict, p_value, size = compare_paired_runs(
                    wide[left].to_numpy(), wide[right].to_numpy(), (left, right)
                )
                family.append(
                    {
                        "feature_set": feature_set,
                        SCOPE_COLUMN: scope,
                        "training": training,
                        "left": left,
                        "right": right,
                        "test": PAIRED,
                        "unit": UNIT,
                        "n_pairs": len(wide),
                        "higher": verdict,
                        "p": p_value,
                        EFFECT_COLUMN: PAIRED_EFFECT,
                        SIZE_COLUMN: size,
                    }
                )
        for row, adjusted in zip(
            family, holm([one["p"] for one in family]), strict=True
        ):
            row["p_holm"] = adjusted
            row["significant"] = adjusted < ALPHA
        rows.extend(family)
    return pd.DataFrame(rows)


def compare_selection_trainings(
    runs: pd.DataFrame, scope: str, conditions: tuple[str, str] = CONDITIONS
) -> pd.DataFrame:
    """Test augmentation against its absence, in every representation, unpaired.

    Holm corrects over the three representations of one feature set.
    """
    rows: list[dict] = []
    for feature_set, block in runs.groupby("feature_set", observed=True):
        family: list[dict] = []
        for variant, cell in block.groupby("feature_variant", observed=True):
            samples = {
                name: cell[cell["training"] == name]
                .sort_values(SEED_COLUMN)["value"]
                .to_numpy()
                for name in conditions
            }
            verdict, p_value, size = compare_unpaired_runs(
                samples[conditions[0]], samples[conditions[1]], conditions
            )
            family.append(
                {
                    "feature_set": feature_set,
                    SCOPE_COLUMN: scope,
                    "feature_variant": variant,
                    "test": UNPAIRED,
                    "unit": UNIT,
                    f"n_{conditions[0]}": len(samples[conditions[0]]),
                    f"n_{conditions[1]}": len(samples[conditions[1]]),
                    "higher": verdict,
                    "p": p_value,
                    EFFECT_COLUMN: UNPAIRED_EFFECT,
                    SIZE_COLUMN: size,
                }
            )
        for row, adjusted in zip(
            family, holm([one["p"] for one in family]), strict=True
        ):
            row["p_holm"] = adjusted
            row["significant"] = adjusted < ALPHA
        rows.extend(family)
    return pd.DataFrame(rows)
