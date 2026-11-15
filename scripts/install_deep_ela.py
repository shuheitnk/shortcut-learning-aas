"""Fetch the Deep-ELA package and its pretrained weights into vendor/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from aas_shortcut.calculate_deep_ela_features import (
    DEEP_ELA_COMMIT,
    DEEP_ELA_REPOSITORY,
    vendor_root,
)


def clone(destination: Path, *, force: bool) -> None:
    """Clone the pinned Deep-ELA commit into ``destination``."""
    if destination.exists():
        if not force:
            print(f"{destination} already exists; pass --force to re-clone")
            return
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {DEEP_ELA_REPOSITORY} into {destination}")
    subprocess.run(
        ["git", "clone", "--quiet", DEEP_ELA_REPOSITORY, str(destination)], check=True
    )
    # A pinned commit, not a branch: the pretrained weights are the feature extractor.
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "--quiet", DEEP_ELA_COMMIT],
        check=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Fetch the vendored checkout and report whether it can be imported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="delete and re-clone an existing checkout"
    )
    args = parser.parse_args(argv)

    destination = vendor_root()
    clone(destination, force=args.force)

    from aas_shortcut.calculate_deep_ela_features import availability

    status = availability()
    print(status.describe())
    return 0 if status.available else 1


if __name__ == "__main__":
    sys.exit(main())
