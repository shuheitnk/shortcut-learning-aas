"""The members of the three focus pairs, which the pair benchmark runs."""

from __future__ import annotations

from aas_shortcut.define_algorithm_portfolio import FOCUS_PAIRS

# Every member of the three focus pairs, in portfolio order and without repeats
# (all six members, since the pairs are within-group).
MEMBERS: tuple[str, ...] = tuple(
    dict.fromkeys(name for pair in FOCUS_PAIRS for name in (pair.first, pair.second))
)
