"""The tuning search space ``Theta``."""

from __future__ import annotations

from aas_shortcut.define_algorithm_portfolio import (
    MAX_SHADE_POPULATION_FACTOR,
    SHADE_CR0,
    SHADE_F0,
    SHADE_POPULATION_FACTOR,
)
from aas_shortcut.optimizers.build_cmaes import SIGMA0_FACTOR, make_bipop_cmaes
from aas_shortcut.optimizers.build_differential_evolution import (
    SHADE_MEMORY_SIZE,
    make_shade,
)
from aas_shortcut.optimizers.build_dual_annealing import (
    ACCEPT_PARAM,
    INITIAL_TEMP,
    VISIT_PARAM,
    make_dual_annealing,
)
from aas_shortcut.optimizers.build_nelder_mead import make_nelder_mead
from aas_shortcut.optimizers.build_nlopt_local import make_bobyqa
from aas_shortcut.optimizers.define_optimizer_base import EPS_X, Runner
from aas_shortcut.optimizers.define_tunable_algorithm import TunableAlgorithm

# Initial simplex size of Nelder-Mead, as a fraction of the box width.
SIMPLEX_FACTOR = 0.05

# The restart-threshold axis, shared by every member so none gets a finer restart
# schedule. It bounds a normalised displacement for five members, and for Dual
# Annealing the ratio of the annealing temperature to ``T_0``.
EPS_X_GRID = (1e-8, 1e-6, EPS_X)


def _bobyqa(*, initial_step_factor: float | None, eps_x: float) -> Runner:
    """Build BOBYQA with ``eps_x`` mapped onto NLopt's ``xtol_rel``."""
    return make_bobyqa(initial_step_factor=initial_step_factor, xtol_rel=eps_x)


def _shade(
    *,
    population_factor: float,
    f0: float,
    cr0: float,
    p: float | None,
    eps_x: float,
) -> Runner:
    """Build SHADE. ``lpsr`` is bound off: it is what makes this member SHADE."""
    return make_shade(
        population_factor=population_factor,
        f0=f0,
        cr0=cr0,
        p=p,
        memory_size=SHADE_MEMORY_SIZE,
        lpsr=False,
        eps_x=eps_x,
    )


def _lshade(
    *,
    population_factor: float,
    f0: float,
    cr0: float,
    p: float | None,
    eps_x: float,
) -> Runner:
    """Build L-SHADE. ``lpsr`` is bound on, and is not an axis."""
    return make_shade(
        population_factor=population_factor,
        f0=f0,
        cr0=cr0,
        p=p,
        memory_size=SHADE_MEMORY_SIZE,
        lpsr=True,
        eps_x=eps_x,
    )


# Axes and candidate values shared by SHADE and L-SHADE.
_SHADE_DEFAULTS = {
    "population_factor": SHADE_POPULATION_FACTOR,
    "f0": SHADE_F0,
    "cr0": SHADE_CR0,
    "p": None,
    "eps_x": EPS_X,
}
_SHADE_GRID = {
    # 18 is Tanabe and Fukunaga's published 18n; the range spans a small population that
    # converges fast to one that explores.
    "population_factor": (5.0, 10.0, 18.0, 30.0, MAX_SHADE_POPULATION_FACTOR),
    "f0": (0.3, 0.4, 0.5, 0.6, 0.7),
    "cr0": (0.3, 0.4, 0.5, 0.6, 0.7),
    "eps_x": EPS_X_GRID,
}


# One entry per portfolio member, keyed by the
# name :data:`~aas_shortcut.define_algorithm_portfolio.ALGORITHMS` uses.
TUNABLE: dict[str, TunableAlgorithm] = {
    "BIPOP-CMA-ES": TunableAlgorithm(
        name="BIPOP-CMA-ES",
        slug="bipop-cmaes",
        factory=make_bipop_cmaes,
        defaults={
            "sigma0_factor": SIGMA0_FACTOR,
            "lambda0_factor": None,
            "eps_x": EPS_X,
        },
        grid={
            "sigma0_factor": (0.1, 0.2, 0.3, 0.4, 0.5),
            # A multiple of CMA-ES's own 4 + floor(3 ln d); ``None`` is that default.
            "lambda0_factor": (None, 2.0, 3.0, 5.0, 8.0),
            "eps_x": EPS_X_GRID,
        },
        notes="Rank-based search under BIPOP's two-regime population schedule.",
    ),
    "Adaptive Nelder-Mead": TunableAlgorithm(
        name="Adaptive Nelder-Mead",
        slug="adaptive-nelder-mead",
        factory=make_nelder_mead,
        defaults={"simplex_factor": SIMPLEX_FACTOR, "eps_x": EPS_X},
        grid={
            "simplex_factor": (0.01, 0.025, 0.05, 0.1, 0.25),
            "eps_x": EPS_X_GRID,
        },
        notes="Gao-Han coefficients behind the shared multistart wrapper.",
    ),
    "SHADE": TunableAlgorithm(
        name="SHADE",
        slug="shade",
        factory=_shade,
        defaults=_SHADE_DEFAULTS,
        grid=_SHADE_GRID,
        notes="Rank-based selection with value-based F/CR adaptation.",
    ),
    "L-SHADE": TunableAlgorithm(
        name="L-SHADE",
        slug="l-shade",
        factory=_lshade,
        defaults=_SHADE_DEFAULTS,
        grid=_SHADE_GRID,
        notes="SHADE plus linear population size reduction.",
    ),
    "BOBYQA": TunableAlgorithm(
        name="BOBYQA",
        slug="bobyqa",
        factory=_bobyqa,
        defaults={"initial_step_factor": None, "eps_x": EPS_X},
        grid={
            # rho_beg as a fraction of the box width; ``None`` is NLopt's own choice,
            # which is the published behaviour.
            "initial_step_factor": (None, 0.05, 0.1, 0.2, 0.4),
            "eps_x": EPS_X_GRID,
        },
        notes="Powell's BOBYQA; eps_x maps onto NLopt's xtol_rel.",
    ),
    "Dual Annealing": TunableAlgorithm(
        name="Dual Annealing",
        slug="dual-annealing",
        factory=make_dual_annealing,
        defaults={
            "initial_temp": INITIAL_TEMP,
            "visit": VISIT_PARAM,
            "accept": ACCEPT_PARAM,
            "eps_x": EPS_X,
        },
        grid={
            # SciPy admits (0.01, 5e4] for the temperature, (1, 3] for the visiting
            # exponent, (-1e4, -5] for the acceptance parameter and (0, 1) for the
            # re-annealing ratio `eps_x` becomes, so every value below is inside its
            # solver's domain.
            "initial_temp": (500.0, 1000.0, 5230.0, 15000.0, 40000.0),
            "visit": (1.5, 2.0, 2.62, 2.85, 2.95),
            "accept": (-5.0, -15.0, -50.0, -150.0, -500.0),
            "eps_x": EPS_X_GRID,
        },
        notes=(
            "Value-based search: a Metropolis rule on objective differences. eps_x "
            "is its re-annealing ratio; the L-BFGS-B phase uses a fixed tolerance."
        ),
    ),
}
