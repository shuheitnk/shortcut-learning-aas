"""What restarts each portfolio member, declared once for the tests that check it.

The optimizers implement the mechanisms; this states them, so the tests can check
the code against the declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

# The study replaces the member's own restart handling, if it had any, with the
# criterion below.
RESEARCH_SIDE = "research-side"

# The member's published mechanism kept, but with the study tuning its trigger
# threshold, on the same footing as ``eps_x`` for the research-side members.
NATIVE_TUNED = "native-tuned"


@dataclass(frozen=True)
class RestartMechanism:
    """How one member decides to restart, and where that decision comes from."""

    algorithm: str
    policy: str
    # ``tau_A``: where the threshold comes from.
    threshold_source: str
    # The member's own restart handling, and what was done with it.
    native_handling: str

    @property
    def uses_eps_x_for_restart(self) -> bool:
        """Whether ``eps_x`` is this member's restart threshold."""
        return self.threshold_source.startswith("eps_x")

    @property
    def keeps_native_mechanism(self) -> bool:
        """Whether the member restarts through its own published mechanism."""
        return self.policy == NATIVE_TUNED

    @property
    def threshold_is_tuned(self) -> bool:
        """Whether ``tau_A`` is chosen by tuning rather than taken as published."""
        return self.policy in (RESEARCH_SIDE, NATIVE_TUNED)


# One entry per portfolio member, in portfolio order.
MECHANISMS: dict[str, RestartMechanism] = {
    "BIPOP-CMA-ES": RestartMechanism(
        algorithm="BIPOP-CMA-ES",
        policy=RESEARCH_SIDE,
        threshold_source="eps_x",
        native_handling=(
            "modcma's own restart strategies are disabled "
            "(RestartStrategy.NONE); the BIPOP population schedule is driven by "
            "this criterion instead"
        ),
    ),
    "Adaptive Nelder-Mead": RestartMechanism(
        algorithm="Adaptive Nelder-Mead",
        policy=RESEARCH_SIDE,
        threshold_source="eps_x",
        native_handling="no published restart mechanism; multistart supplies it",
    ),
    "SHADE": RestartMechanism(
        algorithm="SHADE",
        policy=RESEARCH_SIDE,
        threshold_source="eps_x",
        native_handling="no published restart mechanism",
    ),
    "L-SHADE": RestartMechanism(
        algorithm="L-SHADE",
        policy=RESEARCH_SIDE,
        threshold_source="eps_x",
        native_handling=(
            "no published restart mechanism; LPSR is a population schedule, not a "
            "restart, and is continued across restarts rather than reset"
        ),
    ),
    "BOBYQA": RestartMechanism(
        algorithm="BOBYQA",
        policy=RESEARCH_SIDE,
        threshold_source="eps_x, passed to NLopt as xtol_rel",
        native_handling="no published restart mechanism; multistart supplies it",
    ),
    "Dual Annealing": RestartMechanism(
        algorithm="Dual Annealing",
        policy=NATIVE_TUNED,
        threshold_source=(
            "eps_x, as the re-annealing ratio: SciPy's mechanism fires at "
            "T(t) <= eps_x * T_0. Same axis and same three values as the other five"
        ),
        native_handling=(
            "SciPy's re-annealing is kept as the mechanism -- the study defines no "
            "displacement criterion for this member -- and its trigger is eps_x, "
            "tuned as it is for the other five rather than left at SciPy's 2e-5"
        ),
    ),
}


def mechanism_for(algorithm: str) -> RestartMechanism:
    """Return one member's declared restart mechanism (KeyError if undeclared)."""
    try:
        return MECHANISMS[algorithm]
    except KeyError:
        raise KeyError(
            f"no restart mechanism declared for {algorithm!r}; every portfolio member "
            "must have one in MECHANISMS"
        ) from None
