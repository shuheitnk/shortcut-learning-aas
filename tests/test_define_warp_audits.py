"""The transformations a held-out block is audited under."""

from __future__ import annotations

from aas_shortcut.define_warp_audits import (
    APPENDIX_BLOCKS,
    AUDITS,
    IDENTITY_CONTROL,
    REDRAWS,
)
from aas_shortcut.sample_monotone_transform import WarpConfig


def test_the_five_families_keep_their_draws():
    for audit in AUDITS:
        warp = audit.warp
        assert (warp.alpha, warp.draw) == (1.0, 0)
        assert warp == WarpConfig(
            family=audit.family, domain=audit.domain, knots=audit.knots
        )


def test_the_identity_control_is_the_control_family_at_zero_strength():
    control = AUDITS[0]
    assert IDENTITY_CONTROL.warp == WarpConfig(
        family=control.family, domain=control.domain, alpha=0.0
    )


def test_the_redraws_differ_from_the_control_only_in_the_draw():
    control = AUDITS[0].warp
    assert [audit.warp.draw for audit in REDRAWS] == [1, 2, 3, 4]
    for audit in REDRAWS:
        assert audit.warp.identifier != control.identifier
        assert audit.warp == WarpConfig(
            family=control.family, domain=control.domain, draw=audit.draw
        )


def test_every_block_has_a_label_of_its_own():
    labels = [audit.label for audit in (*AUDITS, *APPENDIX_BLOCKS)]
    assert len(set(labels)) == len(labels) == 10
