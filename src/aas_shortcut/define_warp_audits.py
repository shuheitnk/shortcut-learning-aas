"""The five transformation families a held-out block is audited under.

Shared by Experiments 4 and 5. The first is the control: the training family,
drawn from a seed domain the training never saw, which separates "a different
family" from "a different draw".
"""

from __future__ import annotations

from dataclasses import dataclass

from aas_shortcut.sample_monotone_transform import (
    HELD_OUT_DOMAIN,
    KNOT_SCALE,
    PIECEWISE_FAMILY,
    POWER_FAMILY,
    SPLINE_FAMILY,
    SPLINE_KNOTS,
    WarpConfig,
)

# Control-point ranges outside the training one (`SPLINE_KNOTS`). With three or
# four points the not-a-knot spline is a single quadratic or cubic, so sparse
# curves bend at most twice; dense ones wander.
SPARSE_KNOTS = (3, 5)
DENSE_KNOTS = (15, 21)


@dataclass(frozen=True)
class Audit:
    """One transformation the held-out block is scored under."""

    name: str
    family: str
    domain: str
    # Only the spline family reads this; the others reject a non-default value.
    knots: tuple[int, int] = SPLINE_KNOTS
    # What puts this point outside the training family, for the printed plan.
    structure: str = ""
    # The strength and the redraw of the maps; the five families keep the defaults.
    alpha: float = KNOT_SCALE
    draw: int = 0

    def run_name(self, prefix: str) -> str:
        """Return the run root of one experiment's copy of this point."""
        return f"{prefix}_{self.name}_{self.domain}".replace("-", "_")

    @property
    def label(self) -> str:
        """Return how the audit is named in the tables and the figures."""
        return f"{self.name}/{self.domain}"

    @property
    def warp(self) -> WarpConfig:
        """Return the warp setting this point's ``post`` view is drawn under."""
        return WarpConfig(
            family=self.family,
            domain=self.domain,
            knots=self.knots,
            alpha=self.alpha,
            draw=self.draw,
        )

    def overrides(self) -> list[str]:
        """Return the Hydra overrides that put a run under this point's maps."""
        return [
            f"task.warp_family={self.family}",
            f"task.warp_domain={self.domain}",
            f"task.warp_knots=[{self.knots[0]},{self.knots[1]}]",
        ]


# In figure order: the control, then the two that keep the spline construction,
# then the two that change it.
AUDITS: tuple[Audit, ...] = (
    Audit("spline", SPLINE_FAMILY, HELD_OUT_DOMAIN, SPLINE_KNOTS, "same family"),
    Audit(
        "knots-sparse",
        SPLINE_FAMILY,
        HELD_OUT_DOMAIN,
        SPARSE_KNOTS,
        f"{SPARSE_KNOTS[0]}-{SPARSE_KNOTS[1] - 1} control points",
    ),
    Audit(
        "knots-dense",
        SPLINE_FAMILY,
        HELD_OUT_DOMAIN,
        DENSE_KNOTS,
        f"{DENSE_KNOTS[0]}-{DENSE_KNOTS[1] - 1} control points",
    ),
    Audit(
        "power",
        POWER_FAMILY,
        HELD_OUT_DOMAIN,
        structure="derivative unbounded at the optimum",
    ),
    Audit(
        "piecewise-linear",
        PIECEWISE_FAMILY,
        HELD_OUT_DOMAIN,
        structure="not differentiable at the breakpoints",
    ),
)

# Same family and control points, a draw the augmentation did not see.
CONTROL = AUDITS[0]


# The appendix's further test blocks of the control family: the identity map, so
# that only the rescaling to [0, 1] remains, and four more draws of the maps.
IDENTITY_CONTROL = Audit(
    "spline-alpha0",
    SPLINE_FAMILY,
    HELD_OUT_DOMAIN,
    structure="the identity map: the rescaling alone",
    alpha=0.0,
)
REDRAWS: tuple[Audit, ...] = tuple(
    Audit(
        f"spline-draw{draw}",
        SPLINE_FAMILY,
        HELD_OUT_DOMAIN,
        structure=f"draw {draw} of the control family's maps",
        draw=draw,
    )
    for draw in range(1, 5)
)
APPENDIX_BLOCKS: tuple[Audit, ...] = (IDENTITY_CONTROL, *REDRAWS)
