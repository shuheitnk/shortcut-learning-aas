# Experimental settings

Every setting the preanalysis, Experiments 1–5 and the appendices ran with, and
where each one is fixed.

## 1. Conventions

* **Source** is a configuration key (composed by Hydra from
  `src/aas_shortcut/configs/` and `scripts/<experiment>/config.yaml`) or a
  constant or function of the package (`split_bbob_instances.TRAIN_INSTANCES` is
  in `src/aas_shortcut/split_bbob_instances.py`).
* Values are those of the `full` preset, the paper's scale.
* *Computed* values are results of the pipeline. Each names its report file and
  is checked against it by `just check-numbers`.

## 2. Configuration

### 2.1 How a run is composed

| Layer | File | What it fixes |
| --- | --- | --- |
| Root | `src/aas_shortcut/configs/config.yaml` | `seed: 42`, `num_threads: 1`, output paths |
| Scale | `src/aas_shortcut/configs/preset/full.yaml` | dimension, sample size, budget, trials, functions, instances |
| Task | `src/aas_shortcut/configs/task/pipeline.yaml` | representations, feature sets, Deep-ELA model, learner |
| Experiment | `scripts/<experiment>/config.yaml` (`overrides`) | report root name, instance grid, tuning instance, regression learner |

Every key is validated against `define_config_schema.py`. Each entry point
records the resolved configuration in its report root (`run_manifest.json`).

| Key | Value | Source |
| --- | --- | --- |
| `seed` | 42 (every other seed is derived from it, §16) | `configs/config.yaml` |
| `num_threads` | 1 (BLAS, torch and Deep-ELA included) | `configs/config.yaml` |
| `preset.dim` | 10 | `preset/full.yaml` |
| `preset.sampling_factor` | 50 (sample of 50·d = 500 points) | `preset/full.yaml` |
| `preset.fe_budget_factor` | 1000 (search budget 1000·d = 10,000 evaluations) | `preset/full.yaml` |
| `preset.num_feature_reps` | 1 (one sample per training problem) | `preset/full.yaml` |
| `preset.num_trials` | 31 | `preset/full.yaml` |
| `preset.max_functions` | 24 | `preset/full.yaml` |
| `preset.max_instances` | 10; 110 in Experiments 1–5 (`overrides`) | `preset/full.yaml`, `scripts/experiment<N>/config.yaml` |
| `task.tuning_instance_ids` | [11] | `scripts/experiment<N>/config.yaml`; the preanalysis takes the schema's default, `split_bbob_instances.TUNING_INSTANCES` |
| `task.feature_variants` | `value`, `rank`, `concat` | `task/pipeline.yaml` |
| `task.feature_sets` | `ela`, `deep-ela` | `task/pipeline.yaml` |
| `task.deep_ela_model` | `large_50d_v1` | `task/pipeline.yaml` |
| `task.deep_ela_repetitions` | 10 | `task/pipeline.yaml` |
| `task.include_best_sampled_point` | true (§7.3) | `task/pipeline.yaml` |
| `task.model.kind` | `realmlp`, the main learner (§9.1), for the selection | `task/pipeline.yaml` |
| `task.model_selection.candidates` | `[realmlp]`, the main learner, for the regression | `task/pipeline.yaml` |
| `task.model.epochs`, `validation_fraction` | 256, 0.2 | `define_config_schema.ModelSchema` |
| `task.model.early_stopping_rounds` | 300 (XGBoost only, §14) | `build_regression_model.ModelConfig` |
| `task.warp_family`, `warp_domain`, `warp_knots` | `spline`, `warp`, [5, 9]; the map's amplitude (1.0) and draw (0) are not settings | `sample_monotone_transform.DEFAULT_WARP` |
| `task.n_jobs` | −1 (every core). Results do not depend on it (§16) | `task/pipeline.yaml` |

### 2.2 Scales

The feature columns are declared per scale
(`define_feature_columns.EXCLUDED_BY_SCALE`), since a small sample makes other
features non-finite; hence only these two presets.

| Preset | d | Sample | Functions × instances | Search budget | Trials |
| --- | --- | --- | --- | --- | --- |
| `full` | 10 | 500 | 24 × 10 (× 110 for Experiments 1–5) | 10,000 | 31 |
| `debug` | 2 | 100 | 3 × 2 | 200 | 3 |

## 3. Problems and splits

| Setting | Value | Source |
| --- | --- | --- |
| Suite | BBOB f1–f24 through `ioh` 0.3.17 | `create_problem`, `split_bbob_instances.PROBLEM_CLASS` |
| Dimension | d = 10 | `preset.dim` |
| Search space | [−5, 5]^10 | `ioh`'s BBOB bounds |
| Training block | instances 11–110: 2,400 problems | `split_bbob_instances.TRAIN_INSTANCES` |
| Test block | instances 1–10: 240 problems | `split_bbob_instances.TEST_INSTANCES` |
| Tuning instance | 11, the first of the training block. An id inside the test block is refused | `split_bbob_instances.TUNING_INSTANCES`, `check_tuning_instances` |

## 4. Objective transformation

A strictly increasing map `g` is applied to the objective on the normalized scale
`z = (y − y*) / (ŷmax − y*)` (`sample_monotone_transform.MonotoneTransform`):

* `y*` is the problem's optimum; `ŷmax` is the largest objective value among
  10,000 uniform points in the box (`MAX_ESTIMATE_SAMPLE_SIZE`, seed domain
  `warpmax`).
* The transformed objective is `g(z)` for `z ≤ 1` and `z` above it; `z < 0` from
  numerical noise is clamped to 0.
* One map per problem, seeded by the problem alone
  (`warp_seed(domain, 42, "bbob", f, i, d)`), so every trial and feature sample
  of the problem sees the same map.
* The augmentation draws maps in the seed domain `warp`, every test block in
  `warpheld`, so no test problem sees a map the training saw (`WARP_DOMAINS`).
  `draw = 0` throughout.
* A map's *strength* is `max |g(z) − z|` on the grid
  (`MonotoneTransform.strength`), written per problem to
  `preanalysis/warp_strength.csv` (test block) and `experiment4/warp_strength.csv`
  (augmentation).

| Family | Construction | Source |
| --- | --- | --- |
| `spline` (the study's family) | Control points equally spaced on [0, 1], their number uniform in {5, …, 8}; heights ~ N(0, α²) with α = 1; a cubic spline through them (not-a-knot) is the log-density; its exponential is integrated by the trapezoid rule on a 2,001-point grid and normalized to g(0) = 0, g(1) = 1; read between grid points by PCHIP | `_spline_warp`, `SPLINE_KNOTS = (5, 9)`, `KNOT_SCALE`, `GRID_SIZE` |
| `knots-sparse` | The spline construction with 3–4 control points | `define_warp_audits.SPARSE_KNOTS = (3, 5)` |
| `knots-dense` | The spline construction with 15–20 control points | `define_warp_audits.DENSE_KNOTS = (15, 21)` |
| `power` | g(z) = z^p with p = exp(N(0, α²)), tabulated on the 2,001 squares of an even grid and read by PCHIP | `_power_warp` |
| `piecewise-linear` | Interior breakpoints uniform on (0, 1), their number uniform in {4, …, 7}; slopes exp(N(0, α²)); normalized; read by linear interpolation (not differentiable at the breakpoints) | `_piecewise_linear_warp`, `PIECEWISE_BREAKS = (4, 8)` |

The five test blocks (*audit points*) are the five families under `warpheld`
(`define_warp_audits.AUDITS`). The four families other than `spline` are the
*unseen* families: the augmentation draws from `spline` only.

## 5. Portfolio

### 5.1 Members

| Group | Member | Implementation | Reads the objective through |
| --- | --- | --- | --- |
| G1 | BIPOP-CMA-ES | `modcma` 1.2.0 (`ModularCMAES`, its own restarts off), BIPOP regimes in `optimizers/build_cmaes.py` | ranks |
| G1 | Adaptive Nelder-Mead | the study's implementation, `optimizers/build_nelder_mead.py` | ranks |
| G2 | SHADE | `modde` 0.0.4 (`ModularDE`), `optimizers/build_differential_evolution.py` | ranks for selection; values for parameter adaptation |
| G2 | L-SHADE | as SHADE, with linear population size reduction | as SHADE |
| G3 | BOBYQA | `nlopt` 2.7.1 (`LN_BOBYQA`), `optimizers/build_nlopt_local.py` | values (quadratic models) |
| G3 | Dual Annealing | `scipy` 1.10.1 components, `optimizers/build_dual_annealing.py` | values (acceptance exp(−Δf/T)) |

The regression experiments predict the first member's superiority over the
second within each group (`define_algorithm_portfolio.FOCUS_PAIRS`). The
selection experiments use all six.

### 5.2 Published settings each member starts from

The *default* configuration of each member, before any tuning
(`define_tunable_portfolio.TUNABLE`, `defaults`):

| Member | Settings | Source |
| --- | --- | --- |
| BIPOP-CMA-ES | σ0 = 0.3 × box width = 3.0; λ0 = 4 + ⌊3 ln d⌋ = 10; no active update. Large regime λ0·2^k at σ0; small regime λ = max(4, ⌊λ0 (λ_large/(2λ0))^(U1²)⌋) at σ0·10^(−2·U2), U1, U2 ~ U(0, 1) drawn independently; the regime that has spent less runs next (Hansen 2009) | `build_cmaes.SIGMA0_FACTOR`, `default_population`, `make_bipop_cmaes` |
| Adaptive Nelder-Mead | Gao–Han coefficients, for d = 10: reflection 1, expansion 1.2, contraction 0.7, shrink 0.9; initial simplex 0.05 × box width | `build_nelder_mead.resolve_coefficients`, `initial_simplex` |
| SHADE | N = 18·d = 180 and memory H = 6 (L-SHADE's initial size and memory), M_F = M_CR = 0.5; current-to-pbest/1/bin with an archive of 2·N; p ~ U(2/N, 0.2) per individual | `define_algorithm_portfolio`, `build_differential_evolution.SHADE_MEMORY_SIZE`, `DE_STRUCTURE` |
| L-SHADE | as SHADE, N reduced linearly in the evaluations from 18·d to `modde`'s minimum, 3; the reduction continues across restarts | `build_differential_evolution.scheduled_population` |
| BOBYQA | NLopt's initial step; `ftol_rel` = 0 | `build_nlopt_local.nlopt_start` |
| Dual Annealing | T0 = 5230, q_v = 2.62, q_a = −5 (SciPy's defaults) | `build_dual_annealing.INITIAL_TEMP`, `VISIT_PARAM`, `ACCEPT_PARAM` |

### 5.3 Restart rule

Every member restarts when its *search scale*, in normalized coordinates
`(x − lb)/(ub − lb)`, falls below `ε_x`; no restart rule reads objective values.
A rank-based member's trajectory is therefore unchanged by `g` (G1); members that
read magnitudes move by design (G2, G3).

| Member | Restarts when | Source |
| --- | --- | --- |
| BIPOP-CMA-ES | σ·√λmax(C) / box width ≤ ε_x | `build_cmaes.search_capability` |
| Adaptive Nelder-Mead | the normalized simplex's L2 diameter ≤ ε_x | `build_nelder_mead.capability_lost` |
| SHADE, L-SHADE | the L2 diameter of population ∪ archive ≤ ε_x/2, so no mutation can move an individual by more than ε_x | `build_differential_evolution.capability_lost` |
| BOBYQA | NLopt stops on `xtol_rel = ε_x`, or raises a round-off or runtime error | `build_nlopt_local.nlopt_start` |
| Dual Annealing | the temperature falls below ε_x·T0: SciPy's re-annealing (a fresh energy state, the schedule started over). Its L-BFGS-B local search stops when its normalized step is ≤ 10⁻⁴ | `build_dual_annealing._drive`, `LOCAL_STEP_TOLERANCE` |

* `ε_x` = 10⁻⁴ by default (`define_optimizer_base.EPS_X`); its grid is
  {10⁻⁸, 10⁻⁶, 10⁻⁴} for every member (`define_tunable_portfolio.EPS_X_GRID`).
* A restart starts from a point drawn uniformly from the box.
* Every point a member evaluates is projected into the box.

### 5.4 Runs, budget and trials

| Setting | Value | Source |
| --- | --- | --- |
| Search budget | 1000·d = 10,000 evaluations per run, shared across restarts; a run must spend exactly its budget | `ExperimentContext.search_budget`, `benchmark_algorithms._solve` |
| Single best solver's budget | (1000 + 50)·d = 10,500, the selection's search budget plus its feature sample | `ExperimentContext.total_budget` |
| Trials | 31 per member, problem and view | `preset.num_trials` |
| Performance | `best_y`, the best objective value reached within the budget, read from the untransformed problem in both views (a strictly increasing `g` orders two runs the same way) | `benchmark_algorithms._solve` |
| Trial seed | derived from 42, the member's name, the problem and the trial number 1–31; the same in both views, under every binding and in the tunings | `benchmark_algorithms.trial_seed`, `tune_algorithms.PairConditions.seed` |

## 6. Tuning

Both tunings use the 24 functions at instance 11, 31 trials, the benchmark's
trial seeds and Â12 (§7.1).

### 6.1 Pair calibration (`tune_pair`): preanalysis, Experiments 1, 2 and 4

In each pair, the member with mean superiority below 0.5 at the published
settings (at exactly 0.5, the second) is tuned; the other keeps its published
settings (`tune_algorithms.weaker_side`). The tuned member takes the configuration
whose 24 superiorities are closest to `Uniform(0, 1)` in Kolmogorov–Smirnov
distance, which spreads the regression target over its range; ties go to the
earlier configuration in grid order (the published one first). The smallest
distance reachable at the estimator's granularity,
1/(2·31²), is 0.021 (`tune_algorithms.floor_for`).

| Tuned member | Grid (each value × `eps_x` ∈ {1e-8, 1e-6, 1e-4}) | Candidates |
| --- | --- | --- |
| Adaptive Nelder-Mead (G1) | `simplex_factor` ∈ {0.01, 0.025, 0.05, 0.1, 0.25} | 15 |
| SHADE (G2) | `population_factor` ∈ {5, 10, 18, 30, 50}, `f0` ∈ {0.3, …, 0.7}, `cr0` ∈ {0.3, …, 0.7} | 375 |
| Dual Annealing (G3) | `initial_temp` ∈ {500, 1000, 5230, 15000, 40000}, `visit` ∈ {1.5, 2.0, 2.62, 2.85, 2.95}, `accept` ∈ {−5, −15, −50, −150, −500} | 375 |

*Computed* (`experiment2/pair_tuning.csv`):

| Group | Tuned member | Configuration | `params` | D_KS | Saturated (of 24) | At the minimum |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | Adaptive Nelder-Mead | default | `{"eps_x": 0.0001, "simplex_factor": 0.05}` | 0.401 | 13 | 1 |
| G1 | Adaptive Nelder-Mead | selected | `{"eps_x": 0.0001, "simplex_factor": 0.25}` | 0.261 | 7 | 1 |
| G2 | SHADE | default | `{"cr0": 0.5, "eps_x": 0.0001, "f0": 0.5, "p": null, "population_factor": 18.0}` | 0.783 | 17 | 3 |
| G2 | SHADE | selected | `{"cr0": 0.4, "eps_x": 1e-08, "f0": 0.7, "p": null, "population_factor": 5.0}` | 0.166 | 1 | 3 |
| G3 | Dual Annealing | default | `{"accept": -5.0, "eps_x": 0.0001, "initial_temp": 5230.0, "visit": 2.62}` | 0.165 | 3 | 3 |
| G3 | Dual Annealing | selected | `{"accept": -5.0, "eps_x": 1e-08, "initial_temp": 5230.0, "visit": 1.5}` | 0.126 | 3 | 3 |

"At the minimum" counts the candidates tied with the selected one.

### 6.2 Role tuning (`grid_errors`, `tuning_roles`): Experiments 3 and 5

All six members are tuned twice from one grid of 1,230 configurations, run at
both budgets (`resolve_tuning_stages.role_tuning`):

| Member | Grid (each × `eps_x` ∈ {1e-8, 1e-6, 1e-4}) | Candidates |
| --- | --- | --- |
| BIPOP-CMA-ES | `sigma0_factor` ∈ {0.1, 0.2, 0.3, 0.4, 0.5}, `lambda0_factor` ∈ {default, 2, 3, 5, 8} (λ0 = max(4, round(factor·10))) | 75 |
| Adaptive Nelder-Mead | `simplex_factor` as in §6.1 | 15 |
| SHADE, L-SHADE | as SHADE in §6.1 | 375 each |
| BOBYQA | `initial_step_factor` ∈ {NLopt's default, 0.05, 0.1, 0.2, 0.4} × box width | 15 |
| Dual Annealing | as in §6.1 | 375 |

* **Score.** J(c, f): the mean superiority of configuration `c` on function `f`
  against the whole grid, pooled over members, J = E[(K − R)/(K − 1)] with K = 1,230
  configurations, estimated by Â12 against each of the other 1,229
  (`rank_configuration_grid`). The pool is the declared grid, not the portfolio.
* **Selection side, θ\*(1000d)** — objective `oracle`: the binding that maximizes
  mean_f max_member J, solved exactly as an integer program (scipy `milp`/HiGHS,
  `mip_rel_gap = 0`, the objective in units of J's step 1/(2·31²·1,229))
  (`select_portfolio_configurations.select_oracle_optimal`).
* **SBS side, θ\*(1050d)** — objective `generalist`: each member's configuration
  maximizing its own mean_f J (`select_generalists`).
* **Ties.** A penalty of 10⁻³ of J's step, increasing along the grid's order,
  keeps the published configuration when nothing beats it (`TIE_BREAK`).

*Computed* (`experiment3/role_tuning.csv`, `configuration = selected`):

| Member | Selection side θ\*(1000d) | SBS side θ\*(1050d) |
| --- | --- | --- |
| BIPOP-CMA-ES | `{"eps_x": 1e-08, "lambda0_factor": 3.0, "sigma0_factor": 0.4}` | `{"eps_x": 1e-08, "lambda0_factor": 3.0, "sigma0_factor": 0.5}` |
| Adaptive Nelder-Mead | `{"eps_x": 1e-08, "simplex_factor": 0.25}` | `{"eps_x": 1e-08, "simplex_factor": 0.25}` |
| SHADE | `{"cr0": 0.4, "eps_x": 1e-08, "f0": 0.5, "p": null, "population_factor": 5.0}` | `{"cr0": 0.7, "eps_x": 1e-08, "f0": 0.4, "p": null, "population_factor": 5.0}` |
| L-SHADE | `{"cr0": 0.4, "eps_x": 1e-08, "f0": 0.3, "p": null, "population_factor": 5.0}` | `{"cr0": 0.6, "eps_x": 1e-08, "f0": 0.3, "p": null, "population_factor": 5.0}` |
| BOBYQA | `{"eps_x": 0.0001, "initial_step_factor": 0.4}` | `{"eps_x": 1e-06, "initial_step_factor": 0.4}` |
| Dual Annealing | `{"accept": -5.0, "eps_x": 1e-06, "initial_temp": 40000.0, "visit": 2.85}` | `{"accept": -50.0, "eps_x": 1e-06, "initial_temp": 500.0, "visit": 1.5}` |

## 7. Targets, selection and scores

### 7.1 Superiority

Vargha–Delaney's Â12 over all 31 × 31 pairs of trials, a tie counting one half:
the probability that a run of one member ends with a lower `best_y` than a run of
the other (`calculate_superiority.vargha_delaney`). Every superiority of the study
uses it: the regression's target, the mean superiority, the score against the
single best solver and both tunings (`tune_algorithms.ESTIMATOR = "vargha-delaney"`).

### 7.2 Regression target (preanalysis, Experiments 1, 2, 4)

A pair's superiority on each problem, first member over second (§5.1); one model
per pair (`predict_superiority.TARGET`). Score: R² = 1 − SS_res/SS_tot over the 240
problems of a block and deployment, 0 when the target's range is at most
10⁻¹² of its largest magnitude (`predict_superiority.r2_score`).

### 7.3 Selection (Experiments 3, 5)

| Setting | Value | Source |
| --- | --- | --- |
| Target | Each member's mean superiority J̄ over the other five on a problem, under the selection side's binding at 1000·d, from the runs of the view it is read on; one model per member | `calculate_mean_superiority`, `predict_ranking` |
| Choice | The member with the highest prediction; ties go to the earlier member in portfolio order | `select_algorithms` (`np.argmax`) |
| Achieved value | The chosen member's `best_y` at 1000·d, or, if lower, the best untransformed value its 50·d feature sample evaluated: in a deployment, all 31 runs are credited with that deployment's sample | `select_algorithms.assign_trials`, `task.include_best_sampled_point` |
| Single best solver (SBS) | The member with the highest average J̄ over the untransformed training block under the SBS side's binding, run at 1050·d; ties to the earlier member; the same member for both views of the test block | `select_algorithms.single_best_solver`, `rank_single_best_solver` |
| Score Θ (`superiority_vs_sbs`) | Â12 of the selection's achieved values against the SBS's 31 runs; 0.5 is parity. A deployment's value is the mean over the 240 problems | `select_algorithms` |
| Oracle | Per problem and view, the member with the highest J̄ (ties in portfolio order), scored as the selection is; it uses no feature, so run t is credited with the sample of deployment ((t − 1) mod 11) + 1 | `select_algorithms.oracle_choices`, `assign_trials` |

*Computed* (`experiment3/single_best_solver.csv`): the SBS is **BIPOP-CMA-ES**.

| Member | Average J̄ over the 2,400 training problems, SBS side | SBS |
| --- | --- | --- |
| BIPOP-CMA-ES | 0.685 | True |
| Adaptive Nelder-Mead | 0.601 | False |
| SHADE | 0.467 | False |
| L-SHADE | 0.578 | False |
| BOBYQA | 0.448 | False |
| Dual Annealing | 0.221 | False |

## 8. Features

| Setting | Value | Source |
| --- | --- | --- |
| Sample | 50·d = 500 points, Latin hypercube in the unit cube; mapped to the box only to evaluate | `preset.sampling_factor`, `create_sampled_points.latin_hypercube` |
| Samples per problem | Training block: 1 (repetition 1). Test block: 11 (repetitions 1–11), the *deployments* every score is reduced over | `preset.num_feature_reps`, `define_repetitions.REPETITIONS`, `ExperimentContext.deployment_feature_config` |
| Sample seed | `("doe", 42, "bbob", f, i, d, repetition)`; no representation, training condition or family enters it | `create_sampled_points.sample_seed` |
| Calculator input | X: the unit-cube coordinates. y: the objective after the transformation (for `rank`, its average ranks), min–max scaled to [0, 1] per sample. ELA and Deep-ELA read the same input | `create_sampled_points.ela_inputs`, `normalize` |
| ELA | `pflacco` at commit `4689566a` (1.2.2): `ela_distr`, `ela_level`, `ela_meta`, `pca`, `nbc` (`dist_tie_breaker="first"`), `disp`, `ic` (seeded with the sample seed), `fitness_distance` (`misc_features`), in this order; pflacco's timings are dropped | `calculate_features.ELA_CALCULATORS`, `pyproject.toml` |
| Deep-ELA | `large_50d_v1` from the upstream repository at commit `4a5d4024`, on the CPU with one torch thread. Its 12 input columns (10 coordinates and the objective, zero-padded) are permuted at random 10 times and the outputs averaged; torch is seeded with the sample seed | `calculate_deep_ela_features`, `DEEP_ELA_COMMIT` |
| Representations | `value` (the objective), `rank` (its ranks, through the same calculators), `concat` (the two side by side) | `build_feature_matrix.VARIANT_ENCODINGS` |

### 8.1 Columns

A feature non-finite in any block the experiments read (the training block in
both views, the test block untransformed, the five `warpheld` blocks) is dropped
from every block; the result is a fixed list (`define_feature_columns`).

| Set | Columns | Excluded |
| --- | --- | --- |
| ELA | 59: `disp` 16, `ela_level` 9, `ela_meta` 8, `pca` 8, `fitness_distance` 6, `nbc` 5, `ic` 4, `ela_distr` 3 | `ela_meta.quad_simple.cond` (non-finite in every block: a zero denominator, observed for the linear slope f5); `ic.eps_ratio` (non-finite in the `power` family's test block only) |
| Deep-ELA | 48: `deep_ela.large_50d_v1.X0` … `X47` | none |

Predictors per representation: `value` and `rank` 59 (Deep-ELA 48), `concat` 118
(96).

**Experiment 1's `invariant` predictors** — the 36 ELA columns a strictly
increasing transformation provably cannot move, by name
(`classify_feature_invariance.is_structurally_invariant`): every `disp` and
`ela_level` column; `nbc` but for `nbc.nb_fitness.cor`; the four `pca` columns on
the coordinates (`*_x`); `fitness_distance.distance_mean` and `distance_std`;
`ic.m0`. The other 23 are `varying`. Experiment 1's `all` is the 59.

## 9. Learning

### 9.1 The main learner, RealMLP-TD

RealMLP-TD (Holzmüller et al., NeurIPS 2024) at pytabkit's regression defaults,
called `realmlp` in the configuration. Two settings depart from the defaults
(§15.2).

| Setting | Value | Source |
| --- | --- | --- |
| Implementation | pytabkit 1.7.3 `RealMLP_TD_Regressor` on the CPU, one thread, `random_state` the model seed (§9.2) | `build_regression_model.RealMLPTDRegressor` |
| Defaults | pytabkit's `DefaultParams.RealMLP_TD_REG`, unchanged: 3 hidden layers of 256; parametric Mish; neural tangent parametrization, biases initialized by `he+5` with learning-rate factor 0.1; dropout 0.15 and weight decay 0.02 (none on biases), both on `flat_cos` schedules; a front scaling layer (learning-rate factor 6); inputs median-centered, robust-scaled, smooth-clipped and embedded by PBLD (σ = 0.1, 16 and 4 hidden units, learning-rate factor 0.1); target standardized, output clipped to the training targets' range; Adam with β2 = 0.95, learning rate 0.2 on the `coslog4` schedule | `build_regression_model.REALMLP_TD_DEFAULTS`, checked against the installed pytabkit at every fit |
| Epochs, batch | 256 epochs (128 with augmentation, §9.2), batch 256; pytabkit drops each epoch's last, partial batch | `task.model.epochs`, `build_regression_model.REALMLP_TD_BATCH_SIZE` |
| Best epoch | 20% of the training problems held out (pytabkit's `val_fraction`), whole problems, so a problem's augmented copy stays on its side, drawn from a generator seeded with the model seed; the whole schedule runs and the epoch with the lowest validation RMSE is kept, ties to the last | `task.model.validation_fraction`, `_validation_split` |
| Inputs, target | handed over unscaled: RealMLP-TD scales both itself | `build_regression_model.RealMLPTDRegressor` |
| Hyperparameter search | none | `task.model_selection.candidates` |

Rows: 2,400 training problems, 1,920 fitted and 480 held out; with augmentation
4,800 rows, 3,840 and 960.

### 9.2 Seeds, augmentation and updates

| Setting | Value | Source |
| --- | --- | --- |
| Model seeds | 11: 42–52. Each seed's model is fitted on the whole training block and scored on every deployment; no seed is selected | `define_repetitions.MODEL_SEEDS`, `model_for` |
| Augmentation (Experiments 4, 5) | The training block plus one transformed copy of every problem, from the `spline` family in the `warp` domain; a copy's row is the problem's transformed view, with the superiority that view earned | `augment_training_samples` |
| Matched updates | An augmented fit trains for half the epochs, 128. With the last, partial batch of every epoch dropped, the plain condition takes 7 × 256 = 1,792 parameter updates and the augmented one 15 × 128 = 1,920; whole epochs cannot make them equal | `match_update_budget.matched_config` |

### 9.3 How the learners were chosen

* No setting of any learner was chosen by comparing results.
* The choice of an MLP as the main learner and of 100 training instances
  followed an earlier comparison, not part of the paper, of untuned RandomForest,
  XGBoost and an MLP on 10 to 100 training instances, scored on instances 1–10
  untransformed. The untransformed test problems were therefore read in choosing
  the learner and the size of the training block.

## 10. Reporting and statistics

| Setting | Value | Source |
| --- | --- | --- |
| A seed's value | the median over the 11 deployments of R² (regression) or of the 240-problem mean Θ (selection); over a range of families, the mean of its families' values | `define_reporting_scopes` |
| Reported level | the median over the 11 seeds' values, from full precision | `compare_training_runs` |
| Ranges | `unseen-family mean` (knots-sparse, knots-dense, power, piecewise-linear) and `spline only`. RQ3 reads the first; RQ2 the second | `define_reporting_scopes.SCOPES` |
| Spread | for the unseen families, the largest minus the smallest family's 11-seed median | `measure_augmented_regression` |
| Per-problem values (Figs. 5 and 7) | the median over the seeds of each seed's median over the deployments | `compare_selection_runs.problem_levels` |
| Representation comparisons | Wilcoxon signed-rank test, paired by seed, n = 11 | `compare_training_runs.compare_paired_runs` |
| Training-condition comparisons | Mann–Whitney U test, n = 11 + 11 | `compare_training_runs.compare_unpaired_runs` |
| p-values | two-sided exact permutation tests that handle ties and zero differences (midranks; zero differences dropped from Wilcoxon); twice the smaller one-sided p | `compare_training_runs` |
| Ties | two values tie when they differ by less than 10⁻⁹; a difference between 10⁻¹² and 10⁻⁸ stops the run | `TIE_TOLERANCE`, `ROUNDING_BOUND`, `REAL_BOUND` |
| Significance | α = 0.05, Holm's correction within each family of comparisons (§13) | `ALPHA` |
| Smallest p | Wilcoxon: 2/2¹¹ = 0.000977, 0.0029 after Holm over three pairs. Mann–Whitney: 2/C(22, 11) = 2.8 × 10⁻⁶ | |
| Differences | only in the tables of tests; the main tables give levels | |
| Rounding | every report file holds full precision; a number the paper quotes is rounded once | |

Significance is conditional on the drawn maps, the training samples and the
benchmark's trials, which the model seeds do not vary. The difference from the
single best solver (0.5) is reported as a level, not tested.

## 11. Preanalysis measure

| Setting | Value | Source |
| --- | --- | --- |
| Quantities | each ELA (59) and Deep-ELA (48) feature of the `value` representation, and each pair's superiority | `run_preanalysis.REPORTED_VARIANT` |
| Shift | per problem, \|x_post − x_pre\| / σ, σ the population standard deviation (ddof = 0) of x_pre over the 240 test problems of the same deployment | `calculate_delta_z.pre_scale`, `PRIMARY_STANDARDIZATION` |
| Reported value | the median over the 11 deployments of the per-deployment mean over the 240 problems; a quantity with no spread in some deployment (σ ≤ 10⁻⁹·\|mean\| + 10⁻¹²) has none | `measure_distribution_shift`, `SPREAD_RTOL`, `SPREAD_ATOL` |
| Moved | a quantity whose reported value is above 0 | `measure_distribution_shift` |
| Tests | none | |

## 12. Run-time checks

| Check | Rule | Where |
| --- | --- | --- |
| G1 invariance (P1) | G1's superiority is identical in both views on every problem, tolerance 0; otherwise the run stops. Experiment 1 checks the test block, Experiment 4 the training block and the five test blocks | `measure_representation_transfer.invariance_holds` |
| Anchor | the test block's untransformed features equal the training feature unit's within a relative 10⁻⁹ | `run_experiment{1,4,5}.ANCHOR_TOLERANCE`, `assemble_paired_views.check_source_agreement` |
| Order and rank invariance | the transformation leaves every sample's ranking unchanged, and the `rank` features agree between its two views | `calculate_features.features_for_sample` (`task.check_invariance`) |
| Columns | a feature unit that loses a column outside the scale's declaration stops the run | `define_feature_columns.check_feature_unit` |
| Budget | a run spends exactly its budget | `benchmark_algorithms._solve` |

## 13. Per-experiment settings

| | Preanalysis | Experiment 1 | Experiment 2 | Experiment 3 | Experiment 4 | Experiment 5 |
| --- | --- | --- | --- | --- | --- | --- |
| RQ | — | RQ1 | RQ2 | RQ2 | RQ3 | RQ3 |
| Portfolio | three pairs | G1's pair | three pairs | six members | three pairs | six members |
| Parameters | `tune_pair` | `tune_pair` | `tune_pair` | `tuning_roles` | `tune_pair` | `tuning_roles` |
| Budget | 1000·d | 1000·d | 1000·d | 50·d + 1000·d; SBS 1050·d | 1000·d | 50·d + 1000·d; SBS 1050·d |
| Training block | none | 11–110, untransformed | as Exp. 1 | as Exp. 1 | 11–110; augmented also transformed (`spline`/`warp`) | as Exp. 4 |
| Test block | 1–10 | 1–10 | 1–10 | 1–10 | 1–10 | 1–10 |
| Families scored | `spline` | `spline` | `spline` | `spline` | five | five |
| Range reported | — | — | `spline only` | `spline only` | `unseen-family mean` | `unseen-family mean` |
| Feature sets | ELA, Deep-ELA | ELA (`all` 59, `invariant` 36) | ELA, Deep-ELA | ELA, Deep-ELA | ELA, Deep-ELA | ELA, Deep-ELA |
| Representations | `value` | `value` | three | three | three | three |
| Training conditions | — | plain | plain | plain | plain, augmented | plain, augmented |
| Target | — | pair superiority | pair superiority | J̄ per member | pair superiority | J̄ per member |
| Metric | shift (§11) | R², both blocks | R², both blocks | Θ, both blocks | R², both blocks | Θ, both blocks |
| Tests | none | Wilcoxon, `invariant` vs `all` | representations (Wilcoxon) | representations (Wilcoxon) | representations (Wilcoxon); training conditions (Mann–Whitney) | as Exp. 4 |
| Holm family | — | none (one comparison) | the three pairs of representations per group | the three pairs per feature set | representations: three pairs per (group, training condition, range); training conditions: the nine cells per range | representations: three pairs per (training condition, range); training conditions: the three cells per range |
| Reads | — | Exp. 4's `all` cell | Exp. 4's plain cells and data | Exp. 5's plain cells, oracles and data | — | — |

Experiments 2 and 3 read the plain × `spline` cells of Experiments 4 and 5
rather than fit their own. Experiment 1's `all` cell is Experiment 4's plain ×
`value` × G1 cell; its `invariant` cell differs in the predictors only. Every
test is taken on the transformed block.

## 14. Appendix: other learners

`--learner {random-forest, xgboost}` on the entry points of Experiments 1–5 sets
the regression's and the selection's learner and writes to
`experiment<N>_<learner>/`. Everything else is as for the main learner.

| Learner | Settings | Source |
| --- | --- | --- |
| RandomForest | scikit-learn's defaults (the same in 1.2.2 and 1.3.2), written out: 100 trees, `squared_error`, no depth limit, `max_features = 1.0`, `min_samples_split = 2`, `min_samples_leaf = 1`, bootstrap; one thread | `build_regression_model.FOREST_DEFAULTS` |
| XGBoost | 2.1.4 at XGB-TD (Holzmüller et al. 2024): up to 1,000 rounds, early stopping after 300 on RealMLP-TD's 20% problem-level split (same seed), depth 9, learning rate 0.05, `min_child_weight = 2`, `reg_lambda = 0`, `subsample = 0.7`, `tree_method = hist`, `max_bin = 256`; one thread | `build_regression_model.XGB_TD_DEFAULTS` |

Neither scales its inputs or target; augmentation does not change their number
of trees or rounds. Verdicts (`read_learner_verdicts`,
`scripts/report_learner_agreement.py`):

* **RQ1** (ELA): the two shortcut conditions — `all` above `invariant` on the
  untransformed block, below it on the transformed one — and the direction and
  significance of the transformed block's Wilcoxon test.
* **RQ2** (`spline only`, plain): per group (regression) or overall (selection),
  the set of representation pairs significant after Holm with their directions,
  and the representation with the highest transformed-block level.
* **RQ3** (`unseen-family mean`): the set of cells where augmented is
  significantly above plain after Holm.

A learner agrees on an RQ when it reaches the same reading as the main learner
(`learner_agreement/learner_agreement.csv`, where it is named `realmlp`).

## 15. Departures from published and library defaults


### 15.1 Optimizers

| Member | Here | Published or library default |
| --- | --- | --- |
| L-SHADE | smallest population 3, `modde`'s minimum for current-to-pbest/1 with an archive | 4 (Tanabe & Fukunaga 2014) |
| L-SHADE | archive of 2·N_init, fixed (`modde`) | 2.6·N (`r_arc = 2.6`) |
| L-SHADE | p ~ U(2/N, 0.2) per individual (SHADE's rule; `p = None`) | p = 0.11 |
| Dual Annealing | re-annealing at ε_x·T0 with ε_x = 10⁻⁴, tuned over {10⁻⁸, 10⁻⁶, 10⁻⁴} | SciPy `restart_temp_ratio = 2e-5` |
| Dual Annealing | visited points projected onto the box | SciPy wraps them around the box |
| Dual Annealing | L-BFGS-B with `ftol = gtol = 0`, stopped when its normalized step is ≤ 10⁻⁴ | SciPy's L-BFGS-B options |
| BIPOP-CMA-ES | restarts on σ·√λmax(C) / width ≤ ε_x; `modcma`'s restart strategy off and BIPOP's regimes run by the study | CMA-ES's own termination criteria |
| BOBYQA | stops on `xtol_rel = ε_x`; a round-off or runtime error starts the next start | NLopt sets no tolerance by default |

### 15.2 Learners

RealMLP-TD runs pytabkit's `DefaultParams.RealMLP_TD_REG` (pytabkit 1.7.3;
Holzmüller et al., NeurIPS 2024, arXiv:2407.04491v3, Table A.1) except:

| | RealMLP-TD (pytabkit's default) | Here |
| --- | --- | --- |
| Validation rows | a random 20% of the rows (`val_fraction = 0.2`) | 20% of the training problems, whole problems, so that a problem's augmented copy is never on the other side of its own row; drawn from a generator seeded with the model seed |
| Epochs | 256 | 256, and 128 with augmentation, whose matrix is twice as long (§9.2) |

The appendix's XGBoost runs XGB-TD (`DefaultParams.XGB_TD_REG`) except that it
early-stops on RealMLP-TD's problem-level split instead of a random 20% of rows.
RandomForest runs scikit-learn's defaults.

## 16. Determinism and software

* Every seed is `blake2b` (8 bytes) of the unit of work's coordinates, modulo
  2³¹ − 1 (`derive_seeds.derive_seed`), with a domain token per kind of seed, so
  results do not depend on `task.n_jobs` or on the order units are computed in.
* One thread everywhere: `PYTHONHASHSEED=0` and `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `NUMEXPR_NUM_THREADS` = 1, set by the justfile and the image before the
  interpreter starts.
* Versions are locked in `uv.lock`; among them Python 3.11, numpy 1.26.4,
  scipy 1.10.1, scikit-learn 1.3.2, pytabkit 1.7.3, torch 2.5.1 (CPU), xgboost
  2.1.4, ioh 0.3.17, modcma 1.2.0, modde 0.0.4, nlopt 2.7.1, pflacco at
  `4689566a`. pflacco declares `numpy~=1.24.3` and `scikit-learn~=1.2.2`;
  `[tool.uv] override-dependencies` replaces them so pytabkit can be installed.
* The reported results come from one run from an empty `outputs/` in the
  container.
* A stage unit's fingerprint covers its settings and inputs, not the source
  code; `sample_monotone_transform.WARP_ID` and `PORTFOLIO_REVISION` stand for
  the code of the transformation and the portfolio.

## 17. Appendix: LOFO against the Rubber Ruler Test

Whether the conditions that score higher under the Rubber Ruler Test (RRT,
Experiment 4's transformed test blocks) also score higher under
leave-one-function-out (LOFO). Entry point `just lofo`, report root
`appendix_lofo/`. Everything not listed is as in Experiment 4.

| Setting | Value | Source |
| --- | --- | --- |
| Condition (a point) | representation (`value`, `rank`, `concat`) × training condition (plain, augmented) × feature set (ELA, Deep-ELA): 12 | `run_appendix_lofo` |
| Task | each pair's superiority regression, G1, G2, G3 | `define_algorithm_portfolio.FOCUS_PAIRS` |
| Folds | 24, one per BBOB function | `measure_lofo_regression.FUNCTIONS` |
| Fold training rows | the training block (instances 11–110) of the other 23 functions; augmented: also their transformed copies (`spline`/`warp`), built as Experiment 4 builds them | `LofoRegressionCell.matrices`, `measure_augmented_regression.training_matrix` |
| Fold test rows | instances 1–10 of the held-out function, untransformed, every deployment | `LofoRegressionCell.matrices` |
| Model seeds | each seed fits every fold with that seed | `predict_fold` |
| Score | R² over the 240 out-of-fold predictions of one seed and deployment, pooled over the 24 folds; never per fold | `measure_lofo_regression.pooled_scores` |
| A seed's LOFO value | the median over the 11 deployments | `summarize_lofo_levels.lofo_runs` |
| A seed's RRT value | Experiment 4's unseen-family mean, from Experiment 4's units (not refitted) | `summarize_lofo_levels.rrt_runs` |
| Level and spread | median over the 11 seeds; first and third quartiles (linear interpolation) | `summarize_lofo_levels.seed_spread` |
| Spread of the predictions | per seed and deployment, the population standard deviation of the 240 out-of-fold predictions (and of the targets); median over the deployments, then over the seeds | `summarize_lofo_levels.prediction_runs` |
| Function-level rank agreement | per seed and deployment, Spearman's rho over the 24 functions between each held-out function's mean target and mean prediction; reduced as above | `summarize_lofo_levels.prediction_runs` |
| Spread against R² | Spearman's rho over the conditions between the predictions' standard deviation and the LOFO R², per group (6 conditions); over the three groups together (18), on that deviation divided by the targets' | `summarize_lofo_levels.spread_correlations` |
| Function-level rho per group | the median, minimum and maximum over the group's 6 conditions of their function-level rank agreement | `summarize_lofo_levels.function_rho_by_group` |
| RRT against LOFO | Spearman's rho over a group's 6 conditions between their RRT and LOFO levels | `summarize_lofo_levels.rrt_correlations` |
| Figure | one row of three panels (G1, G2, G3) per feature set, each a separate file, on axis ranges shared by the two (the extreme quartiles plus 5% of the range on each side); a point is a condition at its two levels, its bars the quartiles on both axes; marker: representation; filled: augmented, open: plain; the panel title carries the rho | `create_figures.draw_lofo_figure` |
| Tests | none | |

Only the regression is measured. The pair tuning, part of the portfolio's
definition, used instance 11 of every function, the held-out one included; no
regression is fitted on the held-out function.

Reports (Deep-ELA with `_deep-ela`): `lofo_regression_levels.csv` (LOFO and RRT
levels with quartiles), `lofo_regression_summary.csv`,
`lofo_spread_correlations.csv`, `lofo_function_rho.csv`,
`lofo_rrt_correlations.csv` and `figures/appendix_lofo_against_rrt.{pdf,png,svg}`.

## 18. Appendix: bootstrap intervals

Percentile intervals of the levels RQ1 to RQ3 compare, and of their
differences. Entry point `just bootstrap`, report root `appendix_bootstrap/`.
Nothing is fitted that the experiments did not fit: the regression's
per-problem predictions come from refitting Experiment 4's (and Experiment 1's
`invariant`) cells with their own matrices, configuration and seeds, and a
refit is kept only if it reproduces every stored R² of its cell bit for bit.

| Setting | Value | Source |
| --- | --- | --- |
| Per-problem predictions | stage `regression_predictions`, one unit per regression cell, named by that cell's unit | `measure_regression_predictions`, `resolve_prediction_cells` |
| Per-problem Θ | Experiment 5's selection units | `resolve_selection_cells.resolve_family_selection` |
| Resampled units | main version: model seeds (11), deployments (11), functions (24) and, within each drawn function, its test instances (10); each with replacement and independently. Second version: the same but every function kept once, only its instances resampled | `bootstrap_reported_levels.draw_samples`, `FUNCTIONS_AND_INSTANCES`, `INSTANCES_WITHIN_FUNCTIONS` |
| Samples | 10,000 per version; seed derived from 42 and the domain `bootstrap` (main) or `bootstrap-within-functions` | `N_SAMPLES`, `SEED_DOMAINS` |
| A deployment's value | R² over the drawn problems, a problem counted as often as drawn (`r2_score`'s definition, its zero-spread rule included), or the mean Θ over them | `weighted_r2`, `weighted_mean` |
| Level | the median over the drawn deployments, the mean over the range's families, the median over the drawn seeds, as in §10 | `seed_values`, `level` |
| Differences | within one sample; representations, predictor sets and the two views of one cell share the drawn seeds, training conditions draw them apart (as the Wilcoxon and Mann–Whitney tests pair them) | `define_bootstrap_rows.Comparison.paired` |
| Interval | the 2.5th and 97.5th percentiles (linear interpolation) | `percentile_interval` |
| Point estimate | the data itself; the run stops if it is more than 10⁻¹² from the level the experiments' report functions give | `run_appendix_bootstrap.POINT_TOLERANCE` |
| Rows | RQ1: invariant − all (G1, transformed, ELA); untransformed − transformed for all and for invariant; all − invariant (untransformed). RQ2 also: untransformed − transformed for G1 and G2 in every representation. RQ2: rank − value and rank − concat (transformed) and rank − value (untransformed), each group; rank − value and rank − concat of Θ. RQ3: augmented − plain per representation and group on the unseen-family mean and on the untransformed block, and of Θ; after augmentation, concat − rank, concat − value and rank − value of R² and Θ | `define_bootstrap_rows.comparisons` |

Reports (Deep-ELA with `_deep-ela`): `bootstrap_intervals.csv`.

## 19. Appendix: checks of the paper's statements

Entry point `just checks`, report root `appendix_checks/`. Reads the same
units as §18.

| Report | What | Source |
| --- | --- | --- |
| `untransformed_selection_levels.csv`, `untransformed_selection_tests.csv` | Experiment 5's Θ on the untransformed test block, plain and augmented; Mann–Whitney as in §10, Holm over the three representations | `run_appendix_checks.selection_on_untransformed` |
| `g1_prediction_change_levels.csv`, `g1_prediction_change_tests.csv` | G1's mean over a deployment's 240 problems of \|Ĵ(p̃) − Ĵ(p)\| (same model, same deployment), reduced as R² is; Mann–Whitney augmented against plain for `value` and `concat`, Holm over the two | `measure_prediction_change.prediction_change` |
| `target_spread.csv` | each pair's target on the test block: mean and variance (over n), per family and view | `describe_target_spread.target_spread` |
| `regression_error_levels.csv` | every cell's R² beside its MSE (reduced as R² is) and the target's variance, per range and view | `measure_prediction_change.squared_errors` |
| `g3_member_superiority.csv` | BOBYQA's and Dual Annealing's mean superiority over each other (regression binding) and over the portfolio (selection binding), per family and view | `describe_target_spread.member_means` |
| `regression_family_levels.csv`, `selection_family_levels.csv` | every cell's level on each family and view | `measure_augmented_regression.seed_levels`, `compare_selection_runs.view_levels` |
| `feature_exclusion_rules.csv`, `nonfinite_observations.csv` | the columns the non-finite rule of §8.1 drops when it reads every block, and when it reads the training block alone (both views, instances 11–110), from the feature shards' manifests | `find_nonfinite_features` |
| `g1_trajectory_identity.csv` | per binding, block and G1 member, the runs whose `best_y`, evaluations, restarts or seed differ between the views | `compare_view_performance.trajectory_identity` |

## 20. Appendix: further test blocks of the control family

Entry point `just blocks`, report root `appendix_blocks/`. Each block is the
test block (instances 1–10, 240 problems) under one more transformation of the
control family (`spline`, `warpheld`); every member runs on it under the
regression, selection and SBS bindings, and its features are computed, as for
the five families.

| Setting | Value | Source |
| --- | --- | --- |
| Blocks | `spline-alpha0`: α = 0, the identity, so only the rescaling z = (y − y\*)/(ŷmax − y\*) remains. `spline-draw1` … `spline-draw4`: α = 1 and draws 1–4, which enter only the curve's seed (draw 0 is the paper's block) | `define_warp_audits.IDENTITY_CONTROL`, `REDRAWS`, `WarpConfig.alpha`, `WarpConfig.draw` |
| Models | not stored, so every regression cell (main learner, both feature sets, and Experiment 1's `invariant` cell) and every selection cell is fitted again with its own matrices, configuration and seeds and scored on the five families and the five blocks together; the run stops unless every score on the five families is the stored one bit for bit | `run_appendix_blocks.check_reproduced` |
| Target shift | the preanalysis's standardized shift of each pair's superiority | `summarize_block_shift.superiority_shift` |
| G3's members | BOBYQA's and Dual Annealing's mean superiority over each other | `describe_target_spread.member_means` |
| G1's runs | per binding, the runs whose `best_y`, evaluations, restarts or seed differ between the views | `compare_view_performance.trajectory_identity` |
| Levels | per block, as for one family: the median over the deployments, then over the seeds | `measure_augmented_regression.seed_levels`, `compare_selection_runs.view_levels` |
| Spread over the draws | minimum, median and maximum of each level over draws 0–4 (`spline` and `spline-draw1` … `spline-draw4`) | `summarize_draw_spread.draw_spread` |

Reports (Deep-ELA with `_deep-ela`): `block_regression_levels.csv`,
`block_selection_levels.csv`, `block_regression_draw_spread.csv`,
`block_selection_draw_spread.csv`, `block_target_shift.csv`, `block_g3_members.csv`,
`block_g1_identity.csv`.

## 21. Appendix: plain-4800, the plain condition at the augmented size

Entry point `just doubled`, report root `appendix_doubled/`. `plain-4800` fits on
instances 11–210, all untransformed: the paper's training block and 100 further
instances per function (2,400 more problems), on which every member runs under the
pair binding and the selection binding, and whose features are computed, as for the
training block. The cells are scored on the five families' test blocks and set beside
the stored `plain` and `augmented` cells.

| Setting | Value | Source |
| --- | --- | --- |
| Instances | training 11–210 (4,800 problems), test 1–10 | `resolve_further_training.PLAIN_4800_INSTANCES` |
| Features | the training feature unit over instances 1–210; the run stops unless its rows on 1–110 are the paper's unit's, bit for bit | `resolve_further_features`, `check_shared_features` |
| Cells | the plain regression cells (main learner, both feature sets) and the plain selection cells, with the further rows added; the run stops unless their matrices on 11–110 are the paper's | `plain_4800_regression_data`, `plain_4800_selection_data`, `check_same_matrix` |
| Updates | the augmented condition's budget: the epochs halved, as for 4,800 augmented rows | `plain_4800_model` |
| SBS | the one named on instances 11–110 (BIPOP-CMA-ES), so Θ has the same baseline in all three conditions; the SBS is not run on 111–210 | `prepare_selection` (`SPLIT.train_instances`) |
| Tests | `augmented` against `plain-4800`, Mann–Whitney U over the 11 seeds, Holm within a feature set and range, on the untransformed test block and over each range of families | `compare_training_conditions`, `compare_selection_trainings` |

Reports (Deep-ELA with `_deep-ela`): `plain4800_regression_levels.csv` (each family
and view), `plain4800_regression_scopes.csv` (the untransformed block and each range),
`plain4800_regression_tests.csv`, and the same three for `selection`.
