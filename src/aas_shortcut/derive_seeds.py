"""Deriving explicit seeds."""

from __future__ import annotations

import hashlib

# Exclusive upper bound of every derived seed.
SEED_MODULUS = 2**31 - 1


def derive_seed(*parts: object) -> int:
    """Derive a reproducible seed from the coordinates of a unit of work."""
    key = "\x1f".join(str(part) for part in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MODULUS
