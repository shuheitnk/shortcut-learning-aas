"""Appendix B - the family of strictly increasing maps, one per BBOB function.

Needs no experiment: each curve is computed from its instance's coordinates.

uv run python scripts/appendix_warp/draw_appendix_warp.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from aas_shortcut.create_figures.draw_warp_figure import (
    WARP_FORMATS,
    figure_monotone_warps,
)
from aas_shortcut.define_paths import RunPaths

DEFAULT_OUT = Path("outputs/appendix_warp")


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the figure and write it. Returns a process exit status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--formats", nargs="+", default=list(WARP_FORMATS))
    args = parser.parse_args(argv)

    written = figure_monotone_warps(RunPaths.from_root(args.out), args.formats)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
