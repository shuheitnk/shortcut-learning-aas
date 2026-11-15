"""Check that a change to the restart specification left every trajectory alone.

Run "capture before.json" before the change and "capture after.json" after it,
then "compare before.json after.json"; it exits 0 when no trajectory moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from aas_shortcut.create_problem import create_problem
from aas_shortcut.define_algorithm_portfolio import PORTFOLIO
from aas_shortcut.optimizers.define_optimizer_base import BudgetedObjective
from aas_shortcut.sample_monotone_transform import transform_for_instance

# Small enough to run in a minute, wide enough that a changed proposal shows up:
# every member, both objective views, several function classes and two seeds.
DIM = 10
BUDGET = 4000
FUNCTION_IDS = (1, 5, 8, 15, 21, 24)
SEEDS = (42, 7)

# Any difference in these fields is a trajectory change.
TRAJECTORY_FIELDS = (
    "evaluations_used",
    "best_y",
    "x_digest",
    "y_digest",
    "projections",
)


def _digest(values: list[float]) -> str:
    """Return a short digest of a full sequence of floats."""
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()[
        :16
    ]


def _one_run(spec, function_id: int, seed: int, view: str, lb, ub) -> dict[str, object]:
    """Fingerprint one run; a function of its own so each run has fresh recorders."""
    problem = create_problem(function_id, 1, DIM)
    warp = transform_for_instance(SEEDS[0], "bbob", function_id, 1, DIM)
    proposals: list[float] = []
    values: list[float] = []

    def objective(x) -> float:
        proposals.extend(np.asarray(x, dtype=np.float64).ravel().tolist())
        raw = float(problem(x))
        seen = raw if view == "pre" else float(warp(raw))
        values.append(seen)
        return seen

    wrapped = BudgetedObjective(objective, BUDGET, bounds=(lb, ub))
    report = spec.runner(wrapped, DIM, seed, BUDGET, lb, ub)
    return {
        "restarts": int(report.restarts),
        "multistarts": int(report.multistarts),
        "native_reanneals": int(report.native_reanneals),
        "evaluations_used": wrapped.evaluations,
        "best_y": min(values) if values else None,
        "x_digest": _digest(proposals),
        "y_digest": _digest(values),
        "projections": wrapped.projections,
    }


def capture() -> dict[str, dict[str, object]]:
    """Fingerprint every member's trajectory under both objective views."""
    lb, ub = np.full(DIM, -5.0), np.full(DIM, 5.0)
    records: dict[str, dict[str, object]] = {}

    for spec in PORTFOLIO:
        for function_id in FUNCTION_IDS:
            for seed in SEEDS:
                for view in ("pre", "post"):
                    key = f"{spec.slug}|f{function_id}|s{seed}|{view}"
                    records[key] = _one_run(spec, function_id, seed, view, lb, ub)
    return records


def _restart_count(record: dict) -> int:
    """Read the restart count from a capture of either revision."""
    if "restarts" in record:
        return int(record["restarts"])
    return int(record["restarts_returned"])


def compare(before: dict, after: dict) -> int:
    """Report what moved between two captures; return 0 when nothing did."""
    if before.keys() != after.keys():
        print("the two captures do not cover the same runs", file=sys.stderr)
        return 1

    moved = [
        key
        for key in before
        if any(before[key][f] != after[key][f] for f in TRAJECTORY_FIELDS)
    ]
    print(f"runs compared                      : {len(before)}")
    print(f"trajectories that moved            : {len(moved)}")
    for key in moved[:10]:
        print(f"    {key}: {before[key]} -> {after[key]}")

    counters = {
        key: (_restart_count(before[key]), _restart_count(after[key]))
        for key in before
        if _restart_count(before[key]) != _restart_count(after[key])
    }
    print(f"reported restart counts changed    : {len(counters)}")
    if counters:
        by_member = Counter(key.split("|")[0] for key in counters)
        deltas = sorted({after - before for before, after in counters.values()})
        print(f"    by member                      : {dict(by_member)}")
        print(f"    deltas                         : {deltas}")
    print(
        "\nVERDICT: "
        + (
            "diagnostic-only; existing performance artifacts stay valid"
            if not moved
            else "TRAJECTORIES MOVED -- results must be recomputed"
        )
    )
    return 1 if moved else 0


def main() -> int:
    """Run the requested sub-command and return the process exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture_parser = sub.add_parser("capture", help="write a trajectory fingerprint")
    capture_parser.add_argument("path", type=Path)
    compare_parser = sub.add_parser("compare", help="compare two fingerprints")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    args = parser.parse_args()

    if args.command == "capture":
        records = capture()
        args.path.write_text(json.dumps(records, indent=0, sort_keys=True))
        print(f"captured {len(records)} run fingerprint(s) -> {args.path}")
        return 0
    return compare(
        json.loads(args.before.read_text()), json.loads(args.after.read_text())
    )


if __name__ == "__main__":
    raise SystemExit(main())
