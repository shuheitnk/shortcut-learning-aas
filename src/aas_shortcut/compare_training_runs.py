"""Exact tests of whether two cells of Experiment 4 differ beyond seed randomness.

The unit is the training run: one seed's median over its eleven deployments,
averaged over one range's families, as for the reported number. Representations
share a seed's split and minibatch order (those draws read only the row count), so
they pair by seed (Wilcoxon signed-rank); augmentation doubles the rows, so training
conditions do not pair (Mann-Whitney U). Each range is corrected on its own.

P-values are counted over every arrangement, ties at their midrank, because scipy's
exact distributions assume no ties and superiorities are discrete enough to tie.
Ties and zeros are decided within TIE_TOLERANCE, since float sums of one rational
can differ in the last bits.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from functools import cache

import numpy as np
import pandas as pd
from scipy import stats

from aas_shortcut.define_repetitions import MODEL_SEEDS
from aas_shortcut.define_reporting_scopes import (
    SCOPE_COLUMN,
    measurable_scopes,
    scope_mean,
)
from aas_shortcut.evaluate_two_view_cell import TEST_METRICS
from aas_shortcut.measure_selection_transfer import (
    TRAINING_AUGMENTED,
    TRAINING_PLAIN,
)

LOGGER = logging.getLogger(__name__)


# Columns naming a cell once deployments and families are reduced away.
CELL_COLUMNS: tuple[str, ...] = ("feature_set", "group", "feature_variant", "training")

SEED_COLUMN = "seed"

# One observation of either test is a training run, not one of its deployments.
UNIT = "model_seed"

PAIRED = "wilcoxon"
UNPAIRED = "mann-whitney"

# Direction when neither side ranks above the other.
TIED = "tied"

# The size of the difference, and which statistic that size is under the row's test.
SIZE_COLUMN = "median_difference"
EFFECT_COLUMN = "effect_kind"

# Wilcoxon pairs by seed, so it reports the median of the paired differences;
# Mann-Whitney does not pair, so it reports the difference of the medians.
PAIRED_EFFECT = "median of paired differences"
UNPAIRED_EFFECT = "difference of medians"

ALPHA = 0.05

# Most arrangements an exact p-value is counted over; eleven runs a side take
# C(22, 11) = 705,432 splits or 2^11 sign assignments.
MAX_ARRANGEMENTS = 2**24

# Two values closer than this are one value. A selection's value is a rational on a
# grid (a multiple of 1/1,845,120, about 5.4e-7, for the unseen range), while float
# summation errors stay below 1e-13; 1e-9 sits between. R^2 is continuous, its seeds
# about 1e-3 apart, so the rule does not affect it.
TIE_TOLERANCE = 1e-9

# A gap below ROUNDING_BOUND is rounding and one above REAL_BOUND is real; a gap
# between them breaks the tolerance's assumptions, so the test stops.
ROUNDING_BOUND = 1e-12
REAL_BOUND = 1e-8


def per_seed(
    deployments: pd.DataFrame,
    scope: str,
    metric: str = TEST_METRICS["post"],
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Reduce the per-deployment table to one value per training run, over a range.

    The median over a run's deployments, then the mean over the range's families,
    as for the main table's cell. Every cell must carry every seed.
    """
    rows = deployments[deployments["metric"] == metric]
    if rows.empty:
        raise ValueError(f"the deployment table holds no {metric!r} row")

    keys = [*CELL_COLUMNS, "audit", SEED_COLUMN]
    per_family = rows.groupby(keys, observed=True)["value"].median().reset_index()
    reduced = scope_mean(per_family, [*CELL_COLUMNS, SEED_COLUMN], scope, "audit")

    expected = set(seeds)
    held = reduced.groupby(list(CELL_COLUMNS), observed=True)[SEED_COLUMN].agg(set)
    short = held[held.map(lambda found: not expected <= found)]
    if not short.empty:
        raise ValueError(
            f"{len(short)} cell(s) do not carry every training run, e.g. "
            f"{short.index[0]} with {sorted(short.iloc[0])}; the planned "
            f"comparison is over all {len(expected)} of them"
        )
    return reduced


def median_over_seeds(
    runs: pd.DataFrame, keys: Sequence[str] = CELL_COLUMNS
) -> pd.DataFrame:
    """Reduce one value per training run to the main table's cell: their median.

    Tables and tests so read the same eleven numbers; eleven is odd, so the median is
    one run's own value.
    """
    return runs.groupby(list(keys), observed=True)["value"].median().reset_index()


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values, in the order they were given."""
    order = np.argsort(np.asarray(p_values, dtype=float), kind="stable")
    total = len(order)
    adjusted = np.empty(total, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (total - rank) * float(p_values[index])
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return [float(value) for value in adjusted]


def _direction(size: float, names: tuple[str, str]) -> str:
    """Which side is above the other, read off the sign of the difference."""
    if size > 0.0:
        return names[0]
    if size < 0.0:
        return names[1]
    return TIED


def _snapped(size: float) -> float:
    """A difference, or zero when it is within the tolerance of none."""
    return 0.0 if abs(size) < TIE_TOLERANCE else size


def _check_separation(values: np.ndarray, what: str) -> None:
    """Stop if two of the values are neither clearly one value nor clearly two."""
    gaps = np.abs(values[:, None] - values[None, :])[np.triu_indices(values.size, 1)]
    between = gaps[(gaps >= ROUNDING_BOUND) & (gaps <= REAL_BOUND)]
    if between.size:
        raise ValueError(
            f"{between.size} difference(s) between {what} lie between "
            f"{ROUNDING_BOUND:g} and {REAL_BOUND:g} (e.g. {between[0]:.3g}): too "
            "large to be rounding and too small to be a difference, so whether "
            "they tie cannot be told"
        )


def _doubled_midranks(values: np.ndarray) -> np.ndarray:
    """Twice each value's midrank, as integers, values within the tolerance tied.

    Doubling makes every rank sum an integer, so comparisons with the observed one
    are exact.
    """
    order = np.argsort(values, kind="stable")
    starts = np.ones(values.size, dtype=bool)
    starts[1:] = np.diff(values[order]) >= TIE_TOLERANCE
    groups = np.empty(values.size, dtype=np.int64)
    groups[order] = np.cumsum(starts)
    return np.rint(2.0 * stats.rankdata(groups)).astype(np.int64)


@cache
def _splits(total: int, chosen: int) -> np.ndarray:
    """Every choice of ``chosen`` of ``total`` positions, one row each.

    Read-only, so the shared cache cannot be written through.
    """
    count = math.comb(total, chosen)
    if count > MAX_ARRANGEMENTS:
        raise ValueError(f"{count} splits of {total} runs are too many to count out")
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(total), chosen)),
        dtype=np.int16,
        count=count * chosen,
    )
    flat.setflags(write=False)
    return flat.reshape(count, chosen)


def _two_sided(statistics: np.ndarray, observed: int) -> float:
    """Twice the smaller one-sided share of the counted statistics, capped at 1.

    scipy's definition for its exact tests; both shares include the observed value.
    """
    below = int(np.count_nonzero(statistics <= observed))
    above = int(np.count_nonzero(statistics >= observed))
    return min(1.0, 2 * min(below, above) / statistics.size)


def _signed_rank_p(differences: np.ndarray) -> float:
    """Exact two-sided Wilcoxon p, over every assignment of signs.

    Zero differences are dropped first, as scipy's ``zero_method="wilcox"`` does.
    """
    kept = differences[np.abs(differences) >= TIE_TOLERANCE]
    if 2**kept.size > MAX_ARRANGEMENTS:
        raise ValueError(f"{kept.size} differences are too many to count out")
    ranks = _doubled_midranks(np.abs(kept))
    signs = (np.arange(2**kept.size)[:, None] >> np.arange(kept.size)) & 1
    return _two_sided(signs @ ranks, int(ranks[kept > 0].sum()))


def _rank_sum_p(left: np.ndarray, right: np.ndarray) -> float:
    """Exact two-sided Mann-Whitney p, over every split of the pooled runs."""
    ranks = _doubled_midranks(np.concatenate([left, right]))
    splits = _splits(ranks.size, left.size)
    sums = np.zeros(len(splits), dtype=np.int64)
    for column in splits.T:
        sums += ranks[column]
    return _two_sided(sums, int(ranks[: left.size].sum()))


def compare_paired_runs(
    left: np.ndarray, right: np.ndarray, names: tuple[str, str]
) -> tuple[str, float, float]:
    """Exact Wilcoxon signed-rank test of two cells' runs, paired by seed.

    Returns the direction, the p-value and the median paired difference (positive
    when ``names[0]`` is higher). Experiment 1 runs the same test.
    """
    differences = left - right
    # Each difference against zero, and the differences' sizes against each other.
    _check_separation(np.append(np.abs(differences), 0.0), "the paired differences")
    size = _snapped(float(np.median(differences)))
    if not (np.abs(differences) >= TIE_TOLERANCE).any():
        # Every pair identical: nothing to rank, so the cells are not told apart.
        return TIED, 1.0, size
    return _direction(size, names), _signed_rank_p(differences), size


def compare_unpaired_runs(
    left: np.ndarray, right: np.ndarray, names: tuple[str, str]
) -> tuple[str, float, float]:
    """Exact Mann-Whitney test of two unpaired samples of runs.

    Returns the direction, the p-value and the difference of the medians (positive
    when ``names[0]`` is higher). The exact floor for eleven a side is 2 / C(22, 11)
    = 2.8e-6, where scipy's normal approximation stops at 8.2e-5. Experiment 5 runs
    the same test.
    """
    pooled = np.concatenate([left, right])
    _check_separation(pooled, "the pooled runs")
    size = _snapped(float(np.median(left) - np.median(right)))
    if np.ptp(pooled) < TIE_TOLERANCE:
        # Every run one value: nothing to rank, and no side above the other.
        return TIED, 1.0, size
    return _direction(size, names), _rank_sum_p(left, right), size


def compare_representations(runs: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Test the three representations against each other, within every cell.

    Holm corrects over the three comparisons of one ``(group, training)`` within one
    scope; ``scope`` is only recorded on the rows.
    """
    rows: list[dict] = []
    for (feature_set, group, training), cell in runs.groupby(
        ["feature_set", "group", "training"], observed=True
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
                        "group": group,
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


def _measurable(audits: Sequence[str]) -> list[str]:
    """The ranges a run's families cover, each one it does not cover logged."""
    covered, lacking = measurable_scopes(audits)
    for scope, missing in lacking.items():
        LOGGER.warning(
            "the run does not audit %s, so the %r range is not measurable from it",
            missing,
            scope,
        )
    return covered


def compare_representations_by_scope(
    deployments: pd.DataFrame,
    audits: Sequence[str],
    metric: str = TEST_METRICS["post"],
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Test the representations over both ranges of families, into one table."""
    parts = [
        compare_representations(
            per_seed(deployments, scope, metric=metric, seeds=seeds), scope
        )
        for scope in _measurable(audits)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def compare_training_conditions(
    runs: pd.DataFrame,
    scope: str,
    conditions: tuple[str, str] = (TRAINING_AUGMENTED, TRAINING_PLAIN),
) -> pd.DataFrame:
    """Test augmentation against its absence, in every cell of the table.

    Holm corrects over the ``(group, representation)`` cells of one feature set
    within one scope; ``scope`` is only recorded on the rows.
    """
    rows: list[dict] = []
    for feature_set, block in runs.groupby("feature_set", observed=True):
        family: list[dict] = []
        for (group, variant), cell in block.groupby(
            ["group", "feature_variant"], observed=True
        ):
            samples = {
                name: cell[cell["training"] == name]["value"].to_numpy()
                for name in conditions
            }
            verdict, p_value, size = compare_unpaired_runs(
                samples[conditions[0]], samples[conditions[1]], conditions
            )
            family.append(
                {
                    "feature_set": feature_set,
                    SCOPE_COLUMN: scope,
                    "group": group,
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


def compare_training_conditions_by_scope(
    deployments: pd.DataFrame,
    audits: Sequence[str],
    conditions: tuple[str, str] = (TRAINING_AUGMENTED, TRAINING_PLAIN),
    metric: str = TEST_METRICS["post"],
    seeds: Sequence[int] = MODEL_SEEDS,
) -> pd.DataFrame:
    """Test augmentation against its absence over both ranges, into one table."""
    parts = [
        compare_training_conditions(
            per_seed(deployments, scope, metric=metric, seeds=seeds), scope, conditions
        )
        for scope in _measurable(audits)
    ]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# The direction and the sign of SIZE_COLUMN follow this order.
CONDITIONS: tuple[str, str] = (TRAINING_AUGMENTED, TRAINING_PLAIN)
