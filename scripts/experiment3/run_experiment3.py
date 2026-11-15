"""Experiment 3 - Heterogeneous-Portfolio Selection under a Monotone Transformation.

Reads Experiment 5's non-augmented cells on the control family and writes the
report and figures.

uv run python scripts/experiment3/run_experiment3.py task.n_jobs=28
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from aas_shortcut.run_selection_experiment import run_selection_experiment

SPEC_PATH = Path(__file__).with_name("config.yaml")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment. Returns a process exit status."""
    return run_selection_experiment(argv, SPEC_PATH)


if __name__ == "__main__":
    sys.exit(main())
