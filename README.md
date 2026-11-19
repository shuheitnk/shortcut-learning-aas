# shortcut-learning-aas

Code for the paper "Shortcut Learning in Feature-Based Algorithm Selection"
([arXiv:2611.20440](https://arxiv.org/abs/2611.20440)).

Experiment code for a study of shortcut learning in landscape-aware algorithm
selection: whether a selector trained on exploratory landscape analysis (ELA)
features relies on objective-value magnitudes that a strictly increasing
transformation changes but comparison-based optimizers ignore.

Settings: [`docs/SETTINGS.md`](docs/SETTINGS.md). Source of every number in the
paper: [`docs/PAPER_SOURCES.md`](docs/PAPER_SOURCES.md).

## Environment

The environment is the container defined by [`Dockerfile`](Dockerfile) (Python
3.11, the locked dependencies, `just`). Only Docker is needed on the host. In
VS Code, open the folder and choose **Reopen in Container**; without VS Code:

```bash
docker build --platform=linux/amd64 -t shortcut-learning-aas .
docker run --rm -it --platform=linux/amd64 \
  -v "$PWD:/workspaces/shortcut-learning-aas" -w /workspaces/shortcut-learning-aas shortcut-learning-aas bash -l
just sync              # install the locked dependencies
just vendor-deep-ela   # fetch Deep-ELA and its weights (~90 MB)
just check             # lint, types and the fast tests
```

The image is about 11 GB. It is `linux/amd64` only (`nlopt` ships x86-64 wheels
only), so on ARM machines it runs under emulation.

## Reproducing the paper

```bash
just reproduce                      # every experiment and the learner appendix
just reproduce task.n_jobs=26       # ... on 26 worker processes
just lofo task.n_jobs=26            # the LOFO appendix (after `just reproduce`)
just bootstrap task.n_jobs=26       # the bootstrap appendix (after `just reproduce`)
just checks task.n_jobs=26          # the checks of the paper's statements (after it)
just blocks task.n_jobs=26          # the further test blocks (after `just reproduce`)
just doubled task.n_jobs=26         # the plain-4800 training condition (after `just reproduce`)
just experiment 3                   # one experiment
just experiments debug              # all six at a tiny scale, to test the code
```

Results go to `outputs/`. Each experiment writes a report root
(`outputs/<experiment>/`) and shares computed stages through
`outputs/_stages/`, so experiments can run alone or in any order, and a stage
already computed is reused. Results are bit-identical on the container's
platform regardless of the number of workers or the order of experiments.

## Figures and tables

Paths are relative to `outputs/`. Deep-ELA versions carry the suffix
`_deep-ela`. File names keep the code's internal figure numbers (`fig13_…`); the
first column gives the paper's. The experiments map to the paper's sections:
the preanalysis to 6.1, Experiment 1 to 6.2 (RQ1), Experiments 2 and 3 to 6.3
(RQ2, regression and selection), Experiments 4 and 5 to 6.4 (RQ3, regression
and selection).

| Paper | Command | Files |
| --- | --- | --- |
| Fig. 3; Fig. S3 (Deep-ELA) | `just preanalysis` | `preanalysis/figures/fig13_distribution_shift{,_deep-ela,_algorithm}.{png,svg}` |
| Fig. 1's numbers and Section 6.2 | `just experiment 1`, then `just report 1 predictor_comparison` | `experiment1/figures/fig14_shortcut_hero.{png,svg}` |
| Fig. 4; Fig. S4 (Deep-ELA) | `just experiment 2` | `experiment2/figures/fig15_representation_{original,transformed}_v2{,_deep-ela}.{png,svg}` |
| Fig. 5; Fig. S6, left (Deep-ELA) | `just experiment 3` | `experiment3/figures/fig16_selection_split_pooled{,_deep-ela}.{pdf,png,svg}`, `experiment3/selection_levels{,_deep-ela}.csv` |
| Fig. 6; Fig. S5 (Deep-ELA) | `just experiment 4` | `experiment4/figures/fig27_augmentation_regression_{plain,augmented}{,_deep-ela}.{png,svg}` |
| Fig. 7; Fig. S6, right (Deep-ELA) | `just experiment 5` | `experiment5/figures/fig28_augmentation_selection_split{,_deep-ela}.{pdf,png,svg}`, `experiment5/augmentation_gain.csv` |
| Fig. S1 (Appendix B) | `just warp` | `appendix_warp/figures/fig18_monotone_warp.{png,svg}` |
| Fig. S2 (Appendix C) | `just experiment 2` | `experiment2/figures/fig5_tuning_histogram.{png,svg}`, `experiment2/superiority_calibration{,_deciles}.csv` |
| Appendix F (learners' verdicts); Fig. S7 | `just learners full` | `learner_agreement/learner_agreement.csv`; `experiment1_{random-forest,xgboost}/figures/fig14_shortcut_hero.{png,svg}` |
| Appendix G, Table S5 (LOFO) | `just lofo` | `appendix_lofo/figures/appendix_lofo_against_rrt{,_deep-ela}.{pdf,png,svg}`, `appendix_lofo/lofo_*.csv` |
| Appendix H (bootstrap intervals) | `just bootstrap` | `appendix_bootstrap/bootstrap_intervals{,_deep-ela}.csv` |
| Checks of the paper's statements | `just checks` | `appendix_checks/*.csv` |
| Further test blocks (identity map, redraws) | `just blocks` | `appendix_blocks/block_*.csv` |
| plain-4800 training condition | `just doubled` | `appendix_doubled/plain4800_*.csv` |

Fig. 1 itself is drawn separately from the numbers above.

Numbers quoted in the text are listed, with their file, rows and column, in
[`docs/PAPER_SOURCES.md`](docs/PAPER_SOURCES.md); `just check-numbers` checks
them against `outputs/`.

## Computation time

Measured in the container on 26 worker processes, CPU only, from an empty
`outputs/`, in the order of `just reproduce`. A shared stage is computed by the
first experiment that needs it.

| Step | Time |
| --- | --- |
| Preanalysis | 4 h 44 min |
| Experiment 1 | 5 h 36 min |
| Experiment 2 | 13 min |
| Experiment 3 (includes tuning the six-member portfolio) | 23 h 28 min |
| Experiment 4 | 3 h 44 min |
| Experiment 5 | 4 h 32 min |
| Learner appendix (RandomForest, XGBoost) | 2 h 43 min |
| Total of `just reproduce` | 45 h 1 min |
| LOFO appendix (`just lofo`, after `just reproduce`) | 8 h 41 min |
| Bootstrap appendix (`just bootstrap`, after `just reproduce`; refits 37 cells) | 28 min |
| Checks appendix (`just checks`, after `just bootstrap`) | 1 min |
| Further blocks appendix (`just blocks`, after `just reproduce`; runs the portfolio on 5 blocks, refits every cell) | 9.7 h |
| plain-4800 appendix (`just doubled`, after `just reproduce`; runs the portfolio on instances 111–210, refits the plain cells) | 6.1 h |

About 8 GB of memory is enough; no GPU is used. Long stages checkpoint, so an
interrupted run resumes where it stopped.
