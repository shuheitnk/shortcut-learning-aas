"""The heterogeneous optimizer portfolio."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from aas_shortcut.derive_seeds import derive_seed
from aas_shortcut.optimizers.build_cmaes import make_bipop_cmaes
from aas_shortcut.optimizers.build_differential_evolution import make_shade
from aas_shortcut.optimizers.build_dual_annealing import make_dual_annealing
from aas_shortcut.optimizers.build_nelder_mead import make_nelder_mead
from aas_shortcut.optimizers.build_nlopt_local import make_bobyqa
from aas_shortcut.optimizers.define_optimizer_base import (
    EPS_X,
    MIN_LOCAL_BUDGET_FACTOR,
    Runner,
)

# SHADE's and L-SHADE's initial population, as a multiple of ``dim``.
SHADE_POPULATION_FACTOR = 18.0

# Largest population factor the tuning grid may select.
MAX_SHADE_POPULATION_FACTOR = 50.0

# Values the ``F``/``CR`` memory starts from, as published.
SHADE_F0 = 0.5
SHADE_CR0 = 0.5

# Information-use groups.
GROUP_RANK_SEARCH = "G1"
GROUP_RANK_SELECTION = "G2"
GROUP_VALUE_SEARCH = "G3"

# Groups in order of increasing dependence on objective magnitudes.
GROUPS: tuple[str, ...] = (
    GROUP_RANK_SEARCH,
    GROUP_RANK_SELECTION,
    GROUP_VALUE_SEARCH,
)

# The within-group pair types the study compares.
FOCUS_PAIR_TYPES: tuple[str, ...] = ("G1_G1", "G2_G2", "G3_G3")


def pair_type_of(first_group: str, second_group: str) -> str:
    """Return the ``Gi_Gj`` pair type of two groups, in canonical order."""
    low, high = sorted((first_group, second_group), key=GROUPS.index)
    return f"{low}_{high}"


# --------------------------------------------------------------------------
# Execution primitives
# --------------------------------------------------------------------------


def algorithm_seed(base_seed: int, algorithm: str, *parts: object) -> int:
    """Derive a per-algorithm seed that no other component shares."""
    return derive_seed("algorithm", base_seed, algorithm, *parts)


# --------------------------------------------------------------------------
# The portfolio's members
# --------------------------------------------------------------------------


# G1 — rank-based search.
run_bipop_cmaes: Runner = make_bipop_cmaes()

# Nelder-Mead with Gao-Han's dimension-dependent coefficients, behind the shared
# multistart wrapper.
run_nelder_mead: Runner = make_nelder_mead(adaptive=True)

# G2 — rank-based selection with value-based adaptation.
run_shade: Runner = make_shade(
    population_factor=SHADE_POPULATION_FACTOR,
    f0=SHADE_F0,
    cr0=SHADE_CR0,
    lpsr=False,
)

# L-SHADE is SHADE plus linear population size reduction, driven by the evaluation
# counter alone.
run_lshade: Runner = make_shade(
    population_factor=SHADE_POPULATION_FACTOR,
    f0=SHADE_F0,
    cr0=SHADE_CR0,
    lpsr=True,
)

# G3 — value-based search.
run_bobyqa: Runner = make_bobyqa(xtol_rel=EPS_X)

# Dual Annealing on SciPy's own components, with the toroidal wrap in its visiting
# distribution replaced by the portfolio's projection rule, and its own restart
# criterion driving the episodes.
run_dual_annealing: Runner = make_dual_annealing()


# --------------------------------------------------------------------------
# Portfolio registry
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgorithmSpec:
    """One portfolio member."""

    name: str
    slug: str
    # One of :data:`GROUPS`.
    group: str
    # ``(objective, dim, seed, budget, lb, ub) -> restarts`` at the member's
    # *published* configuration.
    runner: Runner


# The portfolio, ordered by group and therefore by how much of the objective each
# member uses.
PORTFOLIO: tuple[AlgorithmSpec, ...] = (
    # G1 — rank-based search
    AlgorithmSpec("BIPOP-CMA-ES", "bipop-cmaes", GROUP_RANK_SEARCH, run_bipop_cmaes),
    AlgorithmSpec(
        "Adaptive Nelder-Mead",
        "adaptive-nelder-mead",
        GROUP_RANK_SEARCH,
        run_nelder_mead,
    ),
    # G2 — rank-based selection with value-based adaptation
    AlgorithmSpec("SHADE", "shade", GROUP_RANK_SELECTION, run_shade),
    AlgorithmSpec("L-SHADE", "l-shade", GROUP_RANK_SELECTION, run_lshade),
    # G3 — value-based search
    AlgorithmSpec("BOBYQA", "bobyqa", GROUP_VALUE_SEARCH, run_bobyqa),
    AlgorithmSpec(
        "Dual Annealing", "dual-annealing", GROUP_VALUE_SEARCH, run_dual_annealing
    ),
)

ALGORITHMS: tuple[str, ...] = tuple(spec.name for spec in PORTFOLIO)

# Bumped whenever a portfolio member starts producing different results.
PORTFOLIO_REVISION = 6


# Filename- and identifier-safe slug per algorithm.
ALGORITHM_SLUGS: dict[str, str] = {spec.name: spec.slug for spec in PORTFOLIO}

# Invariance group per algorithm.
ALGORITHM_GROUP: dict[str, str] = {spec.name: spec.group for spec in PORTFOLIO}


@dataclass(frozen=True)
class AlgorithmPair:
    """One unordered pair of portfolio members and its group pair type."""

    first: str
    second: str

    @property
    def slug(self) -> str:
        """Identifier used in artifact axes, e.g. ``shade__l-shade``."""
        return f"{ALGORITHM_SLUGS[self.first]}__{ALGORITHM_SLUGS[self.second]}"

    @property
    def pair_type(self) -> str:
        """The ``Gi_Gj`` type of this pair, e.g. ``G1_G3``."""
        return pair_type_of(ALGORITHM_GROUP[self.first], ALGORITHM_GROUP[self.second])

    @property
    def is_focus(self) -> bool:
        """Whether this is one of the within-group pair types the study compares."""
        return self.pair_type in FOCUS_PAIR_TYPES


def algorithm_pairs() -> list[AlgorithmPair]:
    """Return every unordered pair of portfolio members, in a fixed order."""
    return [AlgorithmPair(a, b) for a, b in itertools.combinations(ALGORITHMS, 2)]


PAIRS: tuple[AlgorithmPair, ...] = tuple(algorithm_pairs())

# The pairs the hypotheses are about.
FOCUS_PAIRS: tuple[AlgorithmPair, ...] = tuple(pair for pair in PAIRS if pair.is_focus)


def minimum_fe_budget(dim: int) -> int:
    """Return the smallest evaluation budget the portfolio can run with."""
    return max(
        round(MAX_SHADE_POPULATION_FACTOR * dim),
        MIN_LOCAL_BUDGET_FACTOR * (dim + 1),
    )
