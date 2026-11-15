# Development and experiment commands. `just` on its own lists them.
#
# Thread limits are exported here, not set from Python: the BLAS backend reads
# them once at load time, and a multi-threaded reduction is order-dependent.

set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

export PYTHONHASHSEED := "0"
export OMP_NUM_THREADS := "1"
export OPENBLAS_NUM_THREADS := "1"
export MKL_NUM_THREADS := "1"
export VECLIB_MAXIMUM_THREADS := "1"
export NUMEXPR_NUM_THREADS := "1"

# List the available recipes.
default:
    @just --list

# Reconstruct the locked environment.
sync:
    uv sync --locked

# Fetch the Deep-ELA package and its pretrained weights into vendor/.
vendor-deep-ela:
    uv run python scripts/install_deep_ela.py

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

# Rewrite the sources in Ruff's format.
format:
    uv run ruff format .

# Formatting, lint and static types. No tests, so it stays fast enough for a save hook.
lint:
    uv run ruff format --check .
    uv run ruff check .
    uv run pyright

# The fast tests: everything not marked `slow`.
test:
    uv run python -m pytest -m "not slow"

# Every test, including the end-to-end determinism ones.
test-all:
    uv run python -m pytest

# The full quality gate. Contains no experiment.
check: lint test

# ---------------------------------------------------------------------------
# The paper's six experiments
# ---------------------------------------------------------------------------
#
# Conditions come from each experiment's config.yaml and the preset; extra
# arguments go to Hydra, e.g. `just experiment 2 --preset debug task.n_jobs=8`.
# A preset other than `full` appends `_<preset>` to the run root, so a rehearsal
# never overwrites a reported artifact.

# Run the preanalysis.
preanalysis *args:
    uv run python scripts/preanalysis/run_preanalysis.py {{args}}

# Run one of Experiments 1 to 5.
experiment number *args:
    uv run python scripts/experiment{{number}}/run_experiment{{number}}.py {{args}}

# Long-running even at `debug`; `full` takes days. Arguments are split on
# whitespace.
#
# Run all six experiments at one scale.
experiments preset="debug" *args:
    uv run python -c "import subprocess, sys;         [subprocess.run([sys.executable, path, '--preset', '{{preset}}', *'{{args}}'.split()], check=True)          for path in ['scripts/preanalysis/run_preanalysis.py'] + ['scripts/experiment' + str(n) + '/run_experiment' + str(n) + '.py' for n in (1, 2, 3, 4, 5)]]"

# Run after the six experiments: the table reads RealMLP's cells too.
#
# Refit the experiments with the appendix's learners and compare the verdicts.
learners preset="debug" *args:
    uv run python -c "import subprocess, sys;         [subprocess.run([sys.executable, path, '--preset', '{{preset}}', '--learner', learner, *'{{args}}'.split()], check=True)          for learner in ('random-forest', 'xgboost')          for path in ('scripts/experiment4/run_experiment4.py', 'scripts/experiment5/run_experiment5.py',                       'scripts/experiment1/run_experiment1.py', 'scripts/experiment1/report_predictor_comparison.py',                       'scripts/experiment2/run_experiment2.py', 'scripts/experiment3/run_experiment3.py')]"
    just report-learners {{preset}} {{args}}

# Read from the stage store alone; a missing cell stops it.
#
# The appendix's table of verdicts, into outputs/learner_agreement.
report-learners preset="debug" *args:
    uv run python scripts/report_learner_agreement.py --preset {{preset}} {{args}}

# Run the LOFO appendix.
lofo *args:
    uv run python scripts/appendix_lofo/run_appendix_lofo.py {{args}}

# Draw Appendix B's figure of the transformation family. Needs no experiment.
warp *args:
    uv run python scripts/appendix_warp/draw_appendix_warp.py {{args}}

# Multi-day; narrow the workers with `just reproduce task.n_jobs=<N>`.
#
# Regenerate the results the paper reports.
reproduce *args:
    just experiments full {{args}}
    just report 1 predictor_comparison
    just learners full {{args}}

# e.g. `just report 1 predictor_comparison`, after the experiments.
#
# Run scripts/experiment<N>/report_<name>.py.
report number name *args:
    uv run python scripts/experiment{{number}}/report_{{name}}.py {{args}}

# `capture` a record before a code change and one after it, then `compare` the two.
#
# Whether a code change left every portfolio member's trajectory bit-identical.
verify-trajectories *args:
    uv run python scripts/verify_trajectory_identity.py {{args}}

# Whether the documents quote the numbers the report files hold.
check-numbers *args:
    uv run python scripts/check_document_numbers.py {{args}}

# Run the bootstrap appendix (refits Experiment 4's cells for their predictions).
bootstrap *args:
    uv run python scripts/appendix_bootstrap/run_appendix_bootstrap.py {{args}}

# Run the appendix's checks of the paper's statements.
checks *args:
    uv run python scripts/appendix_checks/run_appendix_checks.py {{args}}

# Run the appendix's further test blocks (identity map, four redraws).
blocks *args:
    uv run python scripts/appendix_blocks/run_appendix_blocks.py {{args}}

# Run the appendix's plain-4800 training condition (instances 11-210, untransformed).
doubled *args:
    uv run python scripts/appendix_doubled/run_appendix_doubled.py {{args}}
