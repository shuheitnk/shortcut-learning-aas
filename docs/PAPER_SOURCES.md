# Sources of the paper's numbers

Every number the paper quotes from a result, as rows of the report files. Figures
and the files they are written to are listed in the README; settings are in
`docs/SETTINGS.md`.

## 1. Reading the tables

* Paths are relative to `outputs/`, the output root of `just reproduce`.
* Each table is a set of rows of one report file: the first columns select a
  row, the others are the file's values there. The heading names the file and
  any selection shared by every row.
* Report files hold full precision; a number is rounded once, from it: levels to
  three decimals, p-values to three significant digits, counts whole.
* ELA and Deep-ELA are never pooled; a Deep-ELA file carries the suffix
  `_deep-ela` unless it has a `feature_set` column.

## 2. Numbers in the text

### 2.1 Preanalysis: the shift (Fig. 3)

#### 2.1.1 Pairs and the most-shifted Deep-ELA coordinate (`preanalysis/metrics/shift_summary_bbob_10d.parquet`, `statistic = mean`, `standardization = population`)

| quantity | mean shift | raw change | share of problems moved |
| --- | --- | --- | --- |
| G1: BIPOP-CMA-ES / Adaptive Nelder-Mead | 0.000 | 0 | 0.000 |
| G2: SHADE / L-SHADE | 0.249 | 0.0642 | 0.912 |
| G3: BOBYQA / Dual Annealing | 0.802 | 0.277 | 0.938 |
| deep_ela.large_50d_v1.X2 | 1.406 | 0.00011 | 0.858 |

#### 2.1.2 Features (the same file and rows, `quantity_kind = feature`, per `feature_set`; statistics over the features)

| Statistic | Value |
| --- | --- |
| ELA: features | 59 |
| ELA: features that moved (shift above 0) | 23 |
| ELA: features that did not move | 36 |
| ELA: first quartile shift of the moved | 0.245 |
| ELA: median shift of the moved | 0.324 |
| ELA: third quartile shift of the moved | 0.592 |
| ELA: largest shift of the moved | 0.885 |
| ELA: feature with the largest shift | ic.eps_s |
| Deep-ELA: features | 48 |
| Deep-ELA: features that moved (shift above 0) | 48 |
| Deep-ELA: features that did not move | 0 |
| Deep-ELA: first quartile shift of the moved | 0.573 |
| Deep-ELA: median shift of the moved | 0.620 |
| Deep-ELA: third quartile shift of the moved | 0.719 |
| Deep-ELA: largest shift of the moved | 1.406 |
| Deep-ELA: feature with the largest shift | deep_ela.large_50d_v1.X2 |

### 2.2 The transformation's strength, max |g(z) − z|

#### 2.2.1 Test block, `spline`/`warpheld` (`preanalysis/warp_strength.csv`, one row per problem)

| Statistic | Value |
| --- | --- |
| smallest over the 240 problems | 0.047 |
| first quartile over the 240 problems | 0.158 |
| median over the 240 problems | 0.230 |
| third quartile over the 240 problems | 0.288 |
| largest over the 240 problems | 0.644 |

#### 2.2.2 The augmentation's maps, `spline`/`warp` (`experiment4/warp_strength.csv`, one row per problem)

| Statistic | Value |
| --- | --- |
| smallest over the 2,400 problems | 0.023 |
| first quartile over the 2,400 problems | 0.153 |
| median over the 2,400 problems | 0.214 |
| third quartile over the 2,400 problems | 0.295 |
| largest over the 2,400 problems | 0.668 |

### 2.3 Premises of RQ1

#### 2.3.1 Test block (`experiment1/premises.csv`, `check = target`, `block = spline/warpheld`; the anchor, `check = anchor`)

| group | problems | superiority differs | largest difference |
| --- | --- | --- | --- |
| G1 | 240 | 0 | 0.000 |
| G2 | 240 | 219 | 0.359 |
| G3 | 240 | 225 | 1.000 |

| Statistic | Value |
| --- | --- |
| untransformed feature values equal to the training block's | 14400 |

#### 2.3.2 Training block and the five test blocks (`experiment4/premises.csv`, `check = target`)

| block | group | problems | superiority differs | largest difference |
| --- | --- | --- | --- | --- |
| training block | G1 | 2400 | 0 | 0.000 |
| training block | G2 | 2400 | 2202 | 0.306 |
| training block | G3 | 2400 | 2269 | 1.000 |
| spline/warpheld | G1 | 240 | 0 | 0.000 |
| spline/warpheld | G2 | 240 | 219 | 0.359 |
| spline/warpheld | G3 | 240 | 225 | 1.000 |
| knots-sparse/warpheld | G1 | 240 | 0 | 0.000 |
| knots-sparse/warpheld | G2 | 240 | 221 | 0.258 |
| knots-sparse/warpheld | G3 | 240 | 228 | 1.000 |
| knots-dense/warpheld | G1 | 240 | 0 | 0.000 |
| knots-dense/warpheld | G2 | 240 | 220 | 0.308 |
| knots-dense/warpheld | G3 | 240 | 228 | 1.000 |
| power/warpheld | G1 | 240 | 0 | 0.000 |
| power/warpheld | G2 | 240 | 221 | 0.264 |
| power/warpheld | G3 | 240 | 230 | 1.000 |
| piecewise-linear/warpheld | G1 | 240 | 0 | 0.000 |
| piecewise-linear/warpheld | G2 | 240 | 214 | 0.234 |
| piecewise-linear/warpheld | G3 | 240 | 229 | 1.000 |

| Statistic | Value |
| --- | --- |
| G1: most problems whose superiority differs, over the six blocks | 0 |
| G1: fewest problems compared, over the six blocks | 240 |

### 2.4 RQ1: shortcut learning (Section 6.2)

#### 2.4.1 Levels (`experiment1/predictor_levels.csv`)

| predictors | columns | R² untransformed | R² transformed |
| --- | --- | --- | --- |
| all | 59 | 0.852 | 0.233 |
| invariant | 36 | 0.452 | 0.452 |

#### 2.4.2 Test on the transformed block (`experiment1/predictor_tests.csv`)

| left | right | seeds | higher | p | median paired difference | significant |
| --- | --- | --- | --- | --- | --- | --- |
| invariant | all | 11 | invariant | 0.000977 | 0.198 | True |

### 2.5 RQ2: regression (Fig. 4)

Experiment 2's cells are Experiment 4's plain cells on the `spline` family.

#### 2.5.1 ELA levels (`experiment4/group_regression_levels.csv`, `scope = spline only`, `training = plain`)

| group | feature_variant | untransformed | transformed |
| --- | --- | --- | --- |
| G1 | value | 0.852 | 0.233 |
| G1 | rank | 0.661 | 0.661 |
| G1 | concat | 0.849 | 0.255 |
| G2 | value | 0.770 | 0.196 |
| G2 | rank | 0.621 | 0.607 |
| G2 | concat | 0.759 | 0.184 |
| G3 | value | 0.784 | -0.798 |
| G3 | rank | 0.635 | -0.949 |
| G3 | concat | 0.782 | -0.804 |

#### 2.5.2 Deep-ELA levels (`experiment4/group_regression_levels_deep-ela.csv`, the same rows)

| group | feature_variant | untransformed | transformed |
| --- | --- | --- | --- |
| G1 | value | 0.845 | 0.131 |
| G1 | rank | 0.615 | 0.615 |
| G1 | concat | 0.834 | 0.135 |
| G2 | value | 0.794 | -0.116 |
| G2 | rank | 0.584 | 0.570 |
| G2 | concat | 0.798 | -0.023 |
| G3 | value | 0.825 | -0.716 |
| G3 | rank | 0.620 | -0.920 |
| G3 | concat | 0.820 | -0.664 |

#### 2.5.3 ELA representation tests (`experiment4/representation_tests.csv`, `scope = spline only`, `training = plain`)

| group | left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | concat | rank | rank | -0.407 | 0.00293 | True |
| G1 | concat | value | concat | 0.013 | 0.413 | False |
| G1 | rank | value | rank | 0.423 | 0.00293 | True |
| G2 | concat | rank | rank | -0.426 | 0.00293 | True |
| G2 | concat | value | value | -0.012 | 0.52 | False |
| G2 | rank | value | rank | 0.374 | 0.00293 | True |
| G3 | concat | rank | concat | 0.166 | 0.084 | False |
| G3 | concat | value | value | -0.022 | 0.966 | False |
| G3 | rank | value | value | -0.163 | 0.0146 | True |

#### 2.5.4 Deep-ELA representation tests (`experiment4/representation_tests_deep-ela.csv`, the same rows)

| group | left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | concat | rank | rank | -0.480 | 0.00293 | True |
| G1 | concat | value | concat | 0.035 | 0.413 | False |
| G1 | rank | value | rank | 0.458 | 0.00293 | True |
| G2 | concat | rank | rank | -0.586 | 0.00293 | True |
| G2 | concat | value | concat | 0.095 | 0.00293 | True |
| G2 | rank | value | rank | 0.706 | 0.00293 | True |
| G3 | concat | rank | concat | 0.271 | 0.00293 | True |
| G3 | concat | value | concat | 0.043 | 0.413 | False |
| G3 | rank | value | value | -0.258 | 0.00977 | True |

### 2.6 RQ2: selection (Fig. 5)

#### 2.6.1 ELA levels (`experiment3/selection_levels.csv`)

| representation | untransformed | transformed | oracle, untransformed | oracle, transformed |
| --- | --- | --- | --- | --- |
| value | 0.704 | 0.598 | 0.756 | 0.753 |
| rank | 0.646 | 0.620 | 0.756 | 0.753 |
| concat | 0.704 | 0.600 | 0.756 | 0.753 |

#### 2.6.2 Deep-ELA levels (`experiment3/selection_levels_deep-ela.csv`)

| representation | untransformed | transformed | oracle, untransformed | oracle, transformed |
| --- | --- | --- | --- | --- |
| value | 0.707 | 0.560 | 0.756 | 0.753 |
| rank | 0.643 | 0.612 | 0.756 | 0.753 |
| concat | 0.708 | 0.566 | 0.756 | 0.753 |

#### 2.6.3 ELA representation tests on the transformed block (`experiment3/representation_tests.csv`)

| left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- |
| concat | rank | rank | -0.022 | 0.0293 | True |
| concat | value | concat | 0.001 | 0.413 | False |
| rank | value | rank | 0.026 | 0.00293 | True |

#### 2.6.4 Deep-ELA representation tests on the transformed block (`experiment3/representation_tests_deep-ela.csv`)

| left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- |
| concat | rank | rank | -0.049 | 0.00293 | True |
| concat | value | concat | 0.004 | 0.278 | False |
| rank | value | rank | 0.052 | 0.00293 | True |

### 2.7 RQ3: regression (Fig. 6)

#### 2.7.1 ELA levels (`experiment4/group_regression_levels.csv`, `scope = unseen-family mean`)

The spread is the range of the four families' 11-seed medians; on the
untransformed block it is 0 by construction.

| group | feature_variant | plain, transformed | augmented, transformed | plain, transformed spread | augmented, transformed spread | plain, untransformed | augmented, untransformed | plain, untransformed spread | augmented, untransformed spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | value | 0.245 | 0.582 | 0.161 | 0.062 | 0.852 | 0.746 | 0.000 | 0.000 |
| G1 | rank | 0.661 | 0.655 | 0.000 | 0.000 | 0.661 | 0.655 | 0.000 | 0.000 |
| G1 | concat | 0.262 | 0.686 | 0.187 | 0.055 | 0.849 | 0.764 | 0.000 | 0.000 |
| G2 | value | 0.252 | 0.525 | 0.190 | 0.132 | 0.770 | 0.684 | 0.000 | 0.000 |
| G2 | rank | 0.613 | 0.616 | 0.019 | 0.016 | 0.621 | 0.632 | 0.000 | 0.000 |
| G2 | concat | 0.280 | 0.613 | 0.158 | 0.118 | 0.759 | 0.703 | 0.000 | 0.000 |
| G3 | value | -0.814 | 0.157 | 0.281 | 0.416 | 0.784 | 0.710 | 0.000 | 0.000 |
| G3 | rank | -1.055 | 0.115 | 0.391 | 0.213 | 0.635 | 0.328 | 0.000 | 0.000 |
| G3 | concat | -0.832 | 0.133 | 0.290 | 0.436 | 0.782 | 0.718 | 0.000 | 0.000 |

#### 2.7.2 Deep-ELA levels (`experiment4/group_regression_levels_deep-ela.csv`, the same rows)

| group | feature_variant | plain, transformed | augmented, transformed | plain, transformed spread | augmented, transformed spread | plain, untransformed | augmented, untransformed | plain, untransformed spread | augmented, untransformed spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | value | 0.091 | 0.441 | 0.254 | 0.230 | 0.845 | 0.817 | 0.000 | 0.000 |
| G1 | rank | 0.615 | 0.612 | 0.000 | 0.000 | 0.615 | 0.612 | 0.000 | 0.000 |
| G1 | concat | 0.187 | 0.613 | 0.275 | 0.106 | 0.834 | 0.792 | 0.000 | 0.000 |
| G2 | value | 0.036 | 0.421 | 0.306 | 0.192 | 0.794 | 0.762 | 0.000 | 0.000 |
| G2 | rank | 0.568 | 0.578 | 0.032 | 0.021 | 0.584 | 0.592 | 0.000 | 0.000 |
| G2 | concat | 0.157 | 0.585 | 0.161 | 0.107 | 0.798 | 0.737 | 0.000 | 0.000 |
| G3 | value | -0.800 | 0.123 | 0.357 | 0.147 | 0.825 | 0.727 | 0.000 | 0.000 |
| G3 | rank | -0.992 | 0.123 | 0.347 | 0.225 | 0.620 | 0.328 | 0.000 | 0.000 |
| G3 | concat | -0.743 | 0.197 | 0.237 | 0.221 | 0.820 | 0.703 | 0.000 | 0.000 |

#### 2.7.3 ELA representation tests (`experiment4/representation_tests.csv`, `scope = unseen-family mean`)

| training | group | left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plain | G1 | concat | rank | rank | -0.396 | 0.00293 | True |
| plain | G1 | concat | value | concat | 0.020 | 0.7 | False |
| plain | G1 | rank | value | rank | 0.417 | 0.00293 | True |
| plain | G2 | concat | rank | rank | -0.325 | 0.00293 | True |
| plain | G2 | concat | value | concat | 0.007 | 0.577 | False |
| plain | G2 | rank | value | rank | 0.334 | 0.00293 | True |
| plain | G3 | concat | rank | concat | 0.227 | 0.0645 | False |
| plain | G3 | concat | value | value | -0.014 | 0.52 | False |
| plain | G3 | rank | value | value | -0.221 | 0.0146 | True |
| augmented | G1 | concat | rank | concat | 0.036 | 0.0186 | True |
| augmented | G1 | concat | value | concat | 0.100 | 0.00293 | True |
| augmented | G1 | rank | value | rank | 0.073 | 0.00293 | True |
| augmented | G2 | concat | rank | concat | 0.005 | 0.966 | False |
| augmented | G2 | concat | value | concat | 0.090 | 0.00293 | True |
| augmented | G2 | rank | value | rank | 0.097 | 0.00293 | True |
| augmented | G3 | concat | rank | concat | 0.035 | 0.0967 | False |
| augmented | G3 | concat | value | concat | 0.034 | 0.638 | False |
| augmented | G3 | rank | value | value | -0.029 | 0.107 | False |

#### 2.7.4 Deep-ELA representation tests (`experiment4/representation_tests_deep-ela.csv`, the same rows)

| training | group | left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plain | G1 | concat | rank | rank | -0.427 | 0.00293 | True |
| plain | G1 | concat | value | concat | 0.085 | 0.00293 | True |
| plain | G1 | rank | value | rank | 0.524 | 0.00293 | True |
| plain | G2 | concat | rank | rank | -0.405 | 0.00293 | True |
| plain | G2 | concat | value | concat | 0.121 | 0.00293 | True |
| plain | G2 | rank | value | rank | 0.523 | 0.00293 | True |
| plain | G3 | concat | rank | concat | 0.249 | 0.00293 | True |
| plain | G3 | concat | value | concat | 0.009 | 0.175 | False |
| plain | G3 | rank | value | value | -0.238 | 0.00391 | True |
| augmented | G1 | concat | rank | concat | 0.003 | 0.638 | False |
| augmented | G1 | concat | value | concat | 0.176 | 0.00293 | True |
| augmented | G1 | rank | value | rank | 0.173 | 0.00293 | True |
| augmented | G2 | concat | rank | concat | 0.008 | 0.365 | False |
| augmented | G2 | concat | value | concat | 0.162 | 0.00293 | True |
| augmented | G2 | rank | value | rank | 0.155 | 0.00293 | True |
| augmented | G3 | concat | rank | concat | 0.057 | 0.00293 | True |
| augmented | G3 | concat | value | concat | 0.068 | 0.00391 | True |
| augmented | G3 | rank | value | rank | 0.006 | 0.52 | False |

#### 2.7.5 ELA training-condition tests (`experiment4/training_tests.csv`, `scope = unseen-family mean`)

| group | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | value | augmented | 0.337 | 2.84e-06 | 2.55e-05 | True |
| G1 | rank | plain | -0.006 | 0.748 | 0.956 | False |
| G1 | concat | augmented | 0.425 | 2.84e-06 | 2.55e-05 | True |
| G2 | value | augmented | 0.273 | 2.84e-06 | 2.55e-05 | True |
| G2 | rank | augmented | 0.003 | 0.478 | 0.956 | False |
| G2 | concat | augmented | 0.333 | 2.84e-06 | 2.55e-05 | True |
| G3 | value | augmented | 0.970 | 2.84e-06 | 2.55e-05 | True |
| G3 | rank | augmented | 1.170 | 2.84e-06 | 2.55e-05 | True |
| G3 | concat | augmented | 0.965 | 2.84e-06 | 2.55e-05 | True |

#### 2.7.6 Deep-ELA training-condition tests (`experiment4/training_tests_deep-ela.csv`, the same rows)

| group | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | value | augmented | 0.350 | 2.84e-06 | 2.55e-05 | True |
| G1 | rank | plain | -0.003 | 0.847 | 0.877 | False |
| G1 | concat | augmented | 0.426 | 2.84e-06 | 2.55e-05 | True |
| G2 | value | augmented | 0.385 | 2.84e-06 | 2.55e-05 | True |
| G2 | rank | augmented | 0.010 | 0.438 | 0.877 | False |
| G2 | concat | augmented | 0.428 | 2.84e-06 | 2.55e-05 | True |
| G3 | value | augmented | 0.923 | 2.84e-06 | 2.55e-05 | True |
| G3 | rank | augmented | 1.115 | 2.84e-06 | 2.55e-05 | True |
| G3 | concat | augmented | 0.940 | 2.84e-06 | 2.55e-05 | True |

#### 2.7.7 Counts over the training-condition tests (the two files above, `scope = unseen-family mean`)

| Statistic | Value |
| --- | --- |
| ELA: cells where augmented is significantly higher | 7 |
| ELA: cells with no significant difference | 2 |
| Deep-ELA: cells where augmented is significantly higher | 7 |
| Deep-ELA: cells with no significant difference | 2 |

### 2.8 RQ3: selection (Fig. 7)

#### 2.8.1 Levels (`experiment5/augmentation_gain.csv`)

The `spline only`, plain rows are Experiment 3's transformed-block levels (§2.6).

| scope | feature_set | representation | plain | augmented | oracle |
| --- | --- | --- | --- | --- | --- |
| unseen-family mean | ela | value | 0.583 | 0.642 | 0.753 |
| unseen-family mean | ela | rank | 0.619 | 0.646 | 0.753 |
| unseen-family mean | ela | concat | 0.598 | 0.652 | 0.753 |
| unseen-family mean | deep-ela | value | 0.543 | 0.606 | 0.753 |
| unseen-family mean | deep-ela | rank | 0.611 | 0.642 | 0.753 |
| unseen-family mean | deep-ela | concat | 0.561 | 0.645 | 0.753 |
| spline only | ela | value | 0.598 | 0.650 | 0.753 |
| spline only | ela | rank | 0.620 | 0.647 | 0.753 |
| spline only | ela | concat | 0.600 | 0.654 | 0.753 |
| spline only | deep-ela | value | 0.560 | 0.609 | 0.753 |
| spline only | deep-ela | rank | 0.612 | 0.644 | 0.753 |
| spline only | deep-ela | concat | 0.566 | 0.654 | 0.753 |

#### 2.8.2 ELA representation tests (`experiment5/representation_tests.csv`, `scope = unseen-family mean`)

| training | left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| plain | concat | rank | rank | -0.022 | 0.00293 | True |
| plain | concat | value | concat | 0.015 | 0.00293 | True |
| plain | rank | value | rank | 0.036 | 0.00293 | True |
| augmented | concat | rank | concat | 0.008 | 0.00293 | True |
| augmented | concat | value | concat | 0.013 | 0.00293 | True |
| augmented | rank | value | rank | 0.005 | 0.00293 | True |

#### 2.8.3 Deep-ELA representation tests (`experiment5/representation_tests_deep-ela.csv`, the same rows)

| training | left | right | higher | median paired difference | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| plain | concat | rank | rank | -0.053 | 0.00293 | True |
| plain | concat | value | concat | 0.020 | 0.00293 | True |
| plain | rank | value | rank | 0.069 | 0.00293 | True |
| augmented | concat | rank | concat | 0.007 | 0.00684 | True |
| augmented | concat | value | concat | 0.040 | 0.00293 | True |
| augmented | rank | value | rank | 0.035 | 0.00293 | True |

#### 2.8.4 ELA training-condition tests (`experiment5/training_tests.csv`, `scope = unseen-family mean`)

| feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- |
| value | augmented | 0.058 | 2.84e-06 | 8.51e-06 | True |
| rank | augmented | 0.027 | 2.84e-06 | 8.51e-06 | True |
| concat | augmented | 0.054 | 2.84e-06 | 8.51e-06 | True |

#### 2.8.5 Deep-ELA training-condition tests (`experiment5/training_tests_deep-ela.csv`, the same rows)

| feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- |
| value | augmented | 0.063 | 2.84e-06 | 8.51e-06 | True |
| rank | augmented | 0.030 | 0.00019 | 0.00019 | True |
| concat | augmented | 0.084 | 2.84e-06 | 8.51e-06 | True |

### 2.9 Calibration of the pairs (Fig. S2)

#### 2.9.1 Summary (`experiment2/superiority_calibration.csv`)

The floor is the smallest KS distance that many problems can reach at the
estimator's granularity (`docs/SETTINGS.md` §6.1).

| block | group | problems | D_KS | floor | mean | variance | saturated | informative | informative problems |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tuning | G1 | 24 | 0.261 | 0.021 | 0.442 | 0.150 | 0.292 | 0.375 | 9 |
| tuning | G2 | 24 | 0.166 | 0.021 | 0.559 | 0.073 | 0.042 | 0.667 | 16 |
| tuning | G3 | 24 | 0.126 | 0.021 | 0.514 | 0.108 | 0.125 | 0.458 | 11 |
| training | G1 | 2400 | 0.212 | 0.00047 | 0.460 | 0.155 | 0.280 | 0.329 | 790 |
| training | G2 | 2400 | 0.171 | 0.00047 | 0.577 | 0.066 | 0.040 | 0.683 | 1639 |
| training | G3 | 2400 | 0.115 | 0.00047 | 0.523 | 0.118 | 0.133 | 0.437 | 1049 |

#### 2.9.2 Deciles (`experiment2/superiority_calibration_deciles.csv`, problems per tenth of [0, 1]; the last includes 1)

| block | group | [0.0, 0.1) | [0.1, 0.2) | [0.2, 0.3) | [0.3, 0.4) | [0.4, 0.5) | [0.5, 0.6) | [0.6, 0.7) | [0.7, 0.8) | [0.8, 0.9) | [0.9, 1.0] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tuning | G1 | 7 | 1 | 5 | 1 | 1 | 0 | 2 | 0 | 0 | 7 |
| tuning | G2 | 0 | 2 | 4 | 1 | 4 | 3 | 2 | 2 | 2 | 4 |
| tuning | G3 | 5 | 1 | 0 | 3 | 3 | 2 | 2 | 1 | 3 | 4 |
| training | G1 | 683 | 175 | 230 | 218 | 136 | 68 | 71 | 67 | 62 | 690 |
| training | G2 | 11 | 65 | 263 | 346 | 382 | 435 | 131 | 82 | 219 | 466 |
| training | G3 | 435 | 182 | 134 | 170 | 155 | 261 | 174 | 155 | 242 | 492 |

### 2.10 Appendix: other learners

#### 2.10.1 Verdicts (`learner_agreement/learner_agreement.csv`)

| learner | feature_set | RQ1 | RQ2 | RQ3 |
| --- | --- | --- | --- | --- |
| realmlp | ela | regression: reference | regression: reference / selection: reference | regression: reference / selection: reference |
| realmlp | deep-ela | — | regression: reference / selection: reference | regression: reference / selection: reference |
| random-forest | ela | regression: agree | regression: disagree (G1 concat vs value: value>concat (RealMLP: not significant); G3 concat vs rank: concat>rank (RealMLP: not significant)) / selection: disagree (concat vs value: concat>value (RealMLP: not significant)) | regression: agree / selection: agree |
| random-forest | deep-ela | — | regression: disagree (G1 concat vs value: concat>value (RealMLP: not significant)) / selection: disagree (concat vs value: concat>value (RealMLP: not significant)) | regression: disagree (G2 x rank: augmented above (RealMLP: not above)) / selection: agree |
| xgboost | ela | regression: agree | regression: disagree (G3 concat vs rank: concat>rank (RealMLP: not significant); G3 concat vs value: concat>value (RealMLP: not significant); G3 rank vs value: not significant (RealMLP: value>rank); G3 highest: concat (RealMLP: value)) / selection: disagree (concat vs value: concat>value (RealMLP: not significant)) | regression: agree / selection: agree |
| xgboost | deep-ela | — | regression: disagree (G1 concat vs value: concat>value (RealMLP: not significant)) / selection: disagree (concat vs value: concat>value (RealMLP: not significant)) | regression: agree / selection: agree |

RandomForest's and XGBoost's levels and tests (2.10.2 on) are written by the
same report functions as the main learner's; each learner's full tables,
untransformed blocks included, are in its own roots
(`experiment<N>_<learner>/`).

#### 2.10.2 RandomForest, ELA: levels (`learner_agreement/learner_levels.csv`, `learner = random-forest`, `feature_set = ela`)

| predictors | untransformed | transformed |
| --- | --- | --- |
| all | 0.834 | 0.174 |
| invariant | 0.429 | 0.429 |

| question | part | group | feature_variant | plain | augmented |
| --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | value | 0.174 | 0.525 |
| RQ2 | regression | G1 | rank | 0.621 | 0.623 |
| RQ2 | regression | G1 | concat | 0.160 | 0.649 |
| RQ2 | regression | G2 | value | 0.099 | 0.492 |
| RQ2 | regression | G2 | rank | 0.567 | 0.564 |
| RQ2 | regression | G2 | concat | 0.093 | 0.583 |
| RQ2 | regression | G3 | value | -0.735 | 0.364 |
| RQ2 | regression | G3 | rank | -0.767 | 0.212 |
| RQ2 | regression | G3 | concat | -0.748 | 0.425 |
| RQ3 | regression | G1 | value | 0.227 | 0.500 |
| RQ3 | regression | G1 | rank | 0.621 | 0.623 |
| RQ3 | regression | G1 | concat | 0.235 | 0.632 |
| RQ3 | regression | G2 | value | 0.191 | 0.475 |
| RQ3 | regression | G2 | rank | 0.586 | 0.581 |
| RQ3 | regression | G2 | concat | 0.183 | 0.579 |
| RQ3 | regression | G3 | value | -0.630 | 0.255 |
| RQ3 | regression | G3 | rank | -0.835 | 0.156 |
| RQ3 | regression | G3 | concat | -0.624 | 0.295 |
| RQ2 | selection | — | value | 0.578 | 0.631 |
| RQ2 | selection | — | rank | 0.599 | 0.632 |
| RQ2 | selection | — | concat | 0.586 | 0.645 |
| RQ3 | selection | — | value | 0.575 | 0.619 |
| RQ3 | selection | — | rank | 0.599 | 0.630 |
| RQ3 | selection | — | concat | 0.579 | 0.634 |

#### 2.10.3 RandomForest, ELA: tests (`learner_agreement/learner_tests.csv`, the same rows)

| question | left | right | higher | median difference | p |
| --- | --- | --- | --- | --- | --- |
| RQ1 | invariant | all | invariant | 0.256 | 0.000977 |

| question | part | group | feature_variant | left | right | higher | median difference | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | — | concat | rank | rank | -0.457 | 0.000977 | 0.00293 |
| RQ2 | regression | G1 | — | concat | value | value | -0.007 | 0.0244 | 0.0244 |
| RQ2 | regression | G1 | — | rank | value | rank | 0.446 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | rank | rank | -0.471 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | value | value | -0.006 | 0.175 | 0.175 |
| RQ2 | regression | G2 | — | rank | value | rank | 0.469 | 0.000977 | 0.00293 |
| RQ2 | regression | G3 | — | concat | rank | concat | 0.019 | 0.00195 | 0.00391 |
| RQ2 | regression | G3 | — | concat | value | value | -0.011 | 0.0674 | 0.0674 |
| RQ2 | regression | G3 | — | rank | value | value | -0.029 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | rank | rank | -0.015 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | value | concat | 0.008 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | rank | value | rank | 0.020 | 0.000977 | 0.00293 |
| RQ3 | regression | G1 | value | — | — | augmented | 0.273 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G1 | rank | — | — | augmented | 0.003 | 0.133 | 0.233 |
| RQ3 | regression | G1 | concat | — | — | augmented | 0.397 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | value | — | — | augmented | 0.285 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | rank | — | — | plain | -0.005 | 0.116 | 0.233 |
| RQ3 | regression | G2 | concat | — | — | augmented | 0.397 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | value | — | — | augmented | 0.885 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | rank | — | — | augmented | 0.991 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | concat | — | — | augmented | 0.920 | 2.84e-06 | 2.55e-05 |
| RQ3 | selection | — | value | — | — | augmented | 0.044 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | rank | — | — | augmented | 0.031 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | concat | — | — | augmented | 0.055 | 2.84e-06 | 8.51e-06 |

#### 2.10.4 RandomForest, Deep-ELA: levels (`learner_agreement/learner_levels.csv`, `learner = random-forest`, `feature_set = deep-ela`)

| question | part | group | feature_variant | plain | augmented |
| --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | value | 0.195 | 0.457 |
| RQ2 | regression | G1 | rank | 0.575 | 0.573 |
| RQ2 | regression | G1 | concat | 0.252 | 0.592 |
| RQ2 | regression | G2 | value | 0.026 | 0.406 |
| RQ2 | regression | G2 | rank | 0.578 | 0.587 |
| RQ2 | regression | G2 | concat | 0.084 | 0.572 |
| RQ2 | regression | G3 | value | -0.609 | 0.327 |
| RQ2 | regression | G3 | rank | -0.790 | 0.186 |
| RQ2 | regression | G3 | concat | -0.603 | 0.416 |
| RQ3 | regression | G1 | value | 0.201 | 0.448 |
| RQ3 | regression | G1 | rank | 0.575 | 0.573 |
| RQ3 | regression | G1 | concat | 0.276 | 0.582 |
| RQ3 | regression | G2 | value | 0.199 | 0.411 |
| RQ3 | regression | G2 | rank | 0.580 | 0.587 |
| RQ3 | regression | G2 | concat | 0.254 | 0.567 |
| RQ3 | regression | G3 | value | -0.709 | 0.156 |
| RQ3 | regression | G3 | rank | -0.849 | 0.134 |
| RQ3 | regression | G3 | concat | -0.681 | 0.235 |
| RQ2 | selection | — | value | 0.548 | 0.586 |
| RQ2 | selection | — | rank | 0.593 | 0.627 |
| RQ2 | selection | — | concat | 0.554 | 0.627 |
| RQ3 | selection | — | value | 0.535 | 0.594 |
| RQ3 | selection | — | rank | 0.593 | 0.626 |
| RQ3 | selection | — | concat | 0.544 | 0.623 |

#### 2.10.5 RandomForest, Deep-ELA: tests (`learner_agreement/learner_tests.csv`, the same rows)

| question | part | group | feature_variant | left | right | higher | median difference | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | — | concat | rank | rank | -0.325 | 0.000977 | 0.00293 |
| RQ2 | regression | G1 | — | concat | value | concat | 0.057 | 0.000977 | 0.00293 |
| RQ2 | regression | G1 | — | rank | value | rank | 0.381 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | rank | rank | -0.494 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | value | concat | 0.067 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | rank | value | rank | 0.554 | 0.000977 | 0.00293 |
| RQ2 | regression | G3 | — | concat | rank | concat | 0.198 | 0.000977 | 0.00293 |
| RQ2 | regression | G3 | — | concat | value | concat | 0.009 | 0.147 | 0.147 |
| RQ2 | regression | G3 | — | rank | value | value | -0.181 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | rank | rank | -0.040 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | value | concat | 0.007 | 0.00195 | 0.00293 |
| RQ2 | selection | — | — | rank | value | rank | 0.048 | 0.000977 | 0.00293 |
| RQ3 | regression | G1 | value | — | — | augmented | 0.248 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G1 | rank | — | — | plain | -0.002 | 0.652 | 0.652 |
| RQ3 | regression | G1 | concat | — | — | augmented | 0.306 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | value | — | — | augmented | 0.212 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | rank | — | — | augmented | 0.007 | 8.51e-05 | 0.00017 |
| RQ3 | regression | G2 | concat | — | — | augmented | 0.313 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | value | — | — | augmented | 0.865 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | rank | — | — | augmented | 0.983 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | concat | — | — | augmented | 0.917 | 2.84e-06 | 2.55e-05 |
| RQ3 | selection | — | value | — | — | augmented | 0.060 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | rank | — | — | augmented | 0.033 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | concat | — | — | augmented | 0.079 | 2.84e-06 | 8.51e-06 |

#### 2.10.6 XGBoost, ELA: levels (`learner_agreement/learner_levels.csv`, `learner = xgboost`, `feature_set = ela`)

| predictors | untransformed | transformed |
| --- | --- | --- |
| all | 0.831 | 0.167 |
| invariant | 0.410 | 0.410 |

| question | part | group | feature_variant | plain | augmented |
| --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | value | 0.167 | 0.535 |
| RQ2 | regression | G1 | rank | 0.612 | 0.605 |
| RQ2 | regression | G1 | concat | 0.179 | 0.639 |
| RQ2 | regression | G2 | value | 0.082 | 0.501 |
| RQ2 | regression | G2 | rank | 0.571 | 0.559 |
| RQ2 | regression | G2 | concat | 0.091 | 0.581 |
| RQ2 | regression | G3 | value | -0.756 | 0.371 |
| RQ2 | regression | G3 | rank | -0.794 | 0.172 |
| RQ2 | regression | G3 | concat | -0.739 | 0.422 |
| RQ3 | regression | G1 | value | 0.250 | 0.520 |
| RQ3 | regression | G1 | rank | 0.612 | 0.605 |
| RQ3 | regression | G1 | concat | 0.255 | 0.621 |
| RQ3 | regression | G2 | value | 0.182 | 0.486 |
| RQ3 | regression | G2 | rank | 0.584 | 0.570 |
| RQ3 | regression | G2 | concat | 0.191 | 0.574 |
| RQ3 | regression | G3 | value | -0.655 | 0.216 |
| RQ3 | regression | G3 | rank | -0.869 | 0.113 |
| RQ3 | regression | G3 | concat | -0.621 | 0.257 |
| RQ2 | selection | — | value | 0.582 | 0.631 |
| RQ2 | selection | — | rank | 0.607 | 0.629 |
| RQ2 | selection | — | concat | 0.584 | 0.643 |
| RQ3 | selection | — | value | 0.583 | 0.622 |
| RQ3 | selection | — | rank | 0.607 | 0.628 |
| RQ3 | selection | — | concat | 0.584 | 0.638 |

#### 2.10.7 XGBoost, ELA: tests (`learner_agreement/learner_tests.csv`, the same rows)

| question | left | right | higher | median difference | p |
| --- | --- | --- | --- | --- | --- |
| RQ1 | invariant | all | invariant | 0.242 | 0.000977 |

| question | part | group | feature_variant | left | right | higher | median difference | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | — | concat | rank | rank | -0.436 | 0.000977 | 0.00293 |
| RQ2 | regression | G1 | — | concat | value | concat | 0.006 | 0.831 | 0.831 |
| RQ2 | regression | G1 | — | rank | value | rank | 0.438 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | rank | rank | -0.482 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | value | concat | 0.010 | 0.123 | 0.123 |
| RQ2 | regression | G2 | — | rank | value | rank | 0.482 | 0.000977 | 0.00293 |
| RQ2 | regression | G3 | — | concat | rank | concat | 0.056 | 0.00488 | 0.0146 |
| RQ2 | regression | G3 | — | concat | value | concat | 0.020 | 0.0244 | 0.0488 |
| RQ2 | regression | G3 | — | rank | value | value | -0.036 | 0.083 | 0.083 |
| RQ2 | selection | — | — | concat | rank | rank | -0.022 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | value | concat | 0.003 | 0.042 | 0.042 |
| RQ2 | selection | — | — | rank | value | rank | 0.027 | 0.000977 | 0.00293 |
| RQ3 | regression | G1 | value | — | — | augmented | 0.270 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G1 | rank | — | — | plain | -0.006 | 0.101 | 0.13 |
| RQ3 | regression | G1 | concat | — | — | augmented | 0.366 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | value | — | — | augmented | 0.303 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | rank | — | — | plain | -0.014 | 0.0652 | 0.13 |
| RQ3 | regression | G2 | concat | — | — | augmented | 0.383 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | value | — | — | augmented | 0.871 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | rank | — | — | augmented | 0.982 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | concat | — | — | augmented | 0.877 | 2.84e-06 | 2.55e-05 |
| RQ3 | selection | — | value | — | — | augmented | 0.039 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | rank | — | — | augmented | 0.021 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | concat | — | — | augmented | 0.054 | 2.84e-06 | 8.51e-06 |

#### 2.10.8 XGBoost, Deep-ELA: levels (`learner_agreement/learner_levels.csv`, `learner = xgboost`, `feature_set = deep-ela`)

| question | part | group | feature_variant | plain | augmented |
| --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | value | 0.176 | 0.444 |
| RQ2 | regression | G1 | rank | 0.569 | 0.569 |
| RQ2 | regression | G1 | concat | 0.219 | 0.599 |
| RQ2 | regression | G2 | value | -0.020 | 0.403 |
| RQ2 | regression | G2 | rank | 0.570 | 0.573 |
| RQ2 | regression | G2 | concat | 0.011 | 0.583 |
| RQ2 | regression | G3 | value | -0.664 | 0.319 |
| RQ2 | regression | G3 | rank | -0.821 | 0.160 |
| RQ2 | regression | G3 | concat | -0.649 | 0.434 |
| RQ3 | regression | G1 | value | 0.151 | 0.440 |
| RQ3 | regression | G1 | rank | 0.569 | 0.569 |
| RQ3 | regression | G1 | concat | 0.232 | 0.585 |
| RQ3 | regression | G2 | value | 0.171 | 0.413 |
| RQ3 | regression | G2 | rank | 0.572 | 0.574 |
| RQ3 | regression | G2 | concat | 0.216 | 0.576 |
| RQ3 | regression | G3 | value | -0.804 | 0.136 |
| RQ3 | regression | G3 | rank | -0.887 | 0.109 |
| RQ3 | regression | G3 | concat | -0.769 | 0.226 |
| RQ2 | selection | — | value | 0.545 | 0.596 |
| RQ2 | selection | — | rank | 0.600 | 0.627 |
| RQ2 | selection | — | concat | 0.553 | 0.631 |
| RQ3 | selection | — | value | 0.531 | 0.597 |
| RQ3 | selection | — | rank | 0.600 | 0.624 |
| RQ3 | selection | — | concat | 0.545 | 0.627 |

#### 2.10.9 XGBoost, Deep-ELA: tests (`learner_agreement/learner_tests.csv`, the same rows)

| question | part | group | feature_variant | left | right | higher | median difference | p | p (Holm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | — | concat | rank | rank | -0.342 | 0.000977 | 0.00293 |
| RQ2 | regression | G1 | — | concat | value | concat | 0.066 | 0.000977 | 0.00293 |
| RQ2 | regression | G1 | — | rank | value | rank | 0.391 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | rank | rank | -0.555 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | concat | value | concat | 0.044 | 0.000977 | 0.00293 |
| RQ2 | regression | G2 | — | rank | value | rank | 0.594 | 0.000977 | 0.00293 |
| RQ2 | regression | G3 | — | concat | rank | concat | 0.176 | 0.000977 | 0.00293 |
| RQ2 | regression | G3 | — | concat | value | concat | 0.013 | 0.147 | 0.147 |
| RQ2 | regression | G3 | — | rank | value | value | -0.166 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | rank | rank | -0.045 | 0.000977 | 0.00293 |
| RQ2 | selection | — | — | concat | value | concat | 0.010 | 0.00293 | 0.00293 |
| RQ2 | selection | — | — | rank | value | rank | 0.055 | 0.000977 | 0.00293 |
| RQ3 | regression | G1 | value | — | — | augmented | 0.289 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G1 | rank | — | — | plain | 0.000 | 0.797 | 1 |
| RQ3 | regression | G1 | concat | — | — | augmented | 0.353 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | value | — | — | augmented | 0.242 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G2 | rank | — | — | augmented | 0.002 | 0.652 | 1 |
| RQ3 | regression | G2 | concat | — | — | augmented | 0.360 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | value | — | — | augmented | 0.940 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | rank | — | — | augmented | 0.997 | 2.84e-06 | 2.55e-05 |
| RQ3 | regression | G3 | concat | — | — | augmented | 0.995 | 2.84e-06 | 2.55e-05 |
| RQ3 | selection | — | value | — | — | augmented | 0.066 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | rank | — | — | augmented | 0.024 | 2.84e-06 | 8.51e-06 |
| RQ3 | selection | — | concat | — | — | augmented | 0.082 | 2.84e-06 | 8.51e-06 |

### 2.11 Appendix H: bootstrap intervals

Levels and their difference (left − right), with the 2.5th and 97.5th
percentiles of the difference over 10,000 samples (`docs/SETTINGS.md` §18).
The main version resamples the model seeds, the deployments, the functions and
each drawn function's instances; the second keeps every function once and
resamples only the instances within it. `view` is the left side's view: `post`
the transformed test block (`spline only` for RQ1 and RQ2, the unseen-family
mean for RQ3), `pre` the untransformed one; a `pre - post` row subtracts the
same cell's transformed level. `Mixed` is the selection over the six members;
its predictor set is written —.

#### 2.11.1 ELA, functions and instances (`appendix_bootstrap/bootstrap_intervals.csv`, `resampling = functions and instances`)

| question | part | group | view | comparison | left_variant | left_predictors | left level | right level | difference | 2.5% | 97.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1 | regression | G1 | post | invariant - all | value | invariant | 0.452 | 0.233 | 0.219 | -0.100 | 0.621 |
| RQ1 | regression | G1 | pre | pre - post | value | all | 0.852 | 0.233 | 0.619 | 0.352 | 1.085 |
| RQ1 | regression | G1 | pre | pre - post | value | invariant | 0.452 | 0.452 | 0.000 | 0.000 | 0.000 |
| RQ1 | regression | G1 | pre | all - invariant | value | all | 0.852 | 0.452 | 0.400 | 0.276 | 0.617 |
| RQ2 | regression | G1 | post | rank - value | rank | all | 0.661 | 0.233 | 0.428 | 0.144 | 0.869 |
| RQ2 | regression | G1 | post | rank - concat | rank | all | 0.661 | 0.255 | 0.407 | 0.133 | 0.848 |
| RQ2 | regression | G1 | pre | rank - value | rank | all | 0.661 | 0.852 | -0.190 | -0.326 | -0.102 |
| RQ2 | regression | G2 | post | rank - value | rank | all | 0.607 | 0.196 | 0.411 | 0.142 | 0.704 |
| RQ2 | regression | G2 | post | rank - concat | rank | all | 0.607 | 0.184 | 0.423 | 0.135 | 0.748 |
| RQ2 | regression | G2 | pre | rank - value | rank | all | 0.621 | 0.770 | -0.148 | -0.283 | -0.062 |
| RQ2 | regression | G3 | post | rank - value | rank | all | -0.949 | -0.798 | -0.151 | -0.575 | 0.295 |
| RQ2 | regression | G3 | post | rank - concat | rank | all | -0.949 | -0.804 | -0.145 | -0.558 | 0.333 |
| RQ2 | regression | G3 | pre | rank - value | rank | all | 0.635 | 0.784 | -0.149 | -0.319 | -0.056 |
| RQ2 | regression | G1 | pre | pre - post | value | all | 0.852 | 0.233 | 0.619 | 0.352 | 1.085 |
| RQ2 | regression | G1 | pre | pre - post | rank | all | 0.661 | 0.661 | 0.000 | 0.000 | 0.000 |
| RQ2 | regression | G1 | pre | pre - post | concat | all | 0.849 | 0.255 | 0.595 | 0.340 | 1.059 |
| RQ2 | regression | G2 | pre | pre - post | value | all | 0.770 | 0.196 | 0.574 | 0.340 | 0.895 |
| RQ2 | regression | G2 | pre | pre - post | rank | all | 0.621 | 0.607 | 0.015 | -0.042 | 0.088 |
| RQ2 | regression | G2 | pre | pre - post | concat | all | 0.759 | 0.184 | 0.575 | 0.331 | 0.935 |
| RQ2 | selection | Mixed | post | rank - value | rank | — | 0.620 | 0.598 | 0.021 | -0.044 | 0.086 |
| RQ2 | selection | Mixed | post | rank - concat | rank | — | 0.620 | 0.600 | 0.020 | -0.047 | 0.082 |
| RQ3 | regression | G1 | post | augmented - plain | value | all | 0.582 | 0.245 | 0.337 | 0.154 | 0.648 |
| RQ3 | regression | G1 | post | augmented - plain | rank | all | 0.655 | 0.661 | -0.006 | -0.056 | 0.053 |
| RQ3 | regression | G1 | post | augmented - plain | concat | all | 0.686 | 0.262 | 0.425 | 0.240 | 0.759 |
| RQ3 | regression | G2 | post | augmented - plain | value | all | 0.525 | 0.252 | 0.273 | 0.090 | 0.453 |
| RQ3 | regression | G2 | post | augmented - plain | rank | all | 0.616 | 0.613 | 0.003 | -0.045 | 0.099 |
| RQ3 | regression | G2 | post | augmented - plain | concat | all | 0.613 | 0.280 | 0.333 | 0.166 | 0.544 |
| RQ3 | regression | G3 | post | augmented - plain | value | all | 0.157 | -0.814 | 0.970 | 0.368 | 2.269 |
| RQ3 | regression | G3 | post | augmented - plain | rank | all | 0.115 | -1.055 | 1.170 | 0.438 | 2.582 |
| RQ3 | regression | G3 | post | augmented - plain | concat | all | 0.133 | -0.832 | 0.965 | 0.391 | 2.310 |
| RQ3 | regression | G1 | pre | augmented - plain | value | all | 0.746 | 0.852 | -0.106 | -0.193 | -0.033 |
| RQ3 | regression | G1 | pre | augmented - plain | rank | all | 0.655 | 0.661 | -0.006 | -0.056 | 0.053 |
| RQ3 | regression | G1 | pre | augmented - plain | concat | all | 0.764 | 0.849 | -0.086 | -0.176 | -0.021 |
| RQ3 | regression | G2 | pre | augmented - plain | value | all | 0.684 | 0.770 | -0.086 | -0.192 | -0.015 |
| RQ3 | regression | G2 | pre | augmented - plain | rank | all | 0.632 | 0.621 | 0.011 | -0.048 | 0.099 |
| RQ3 | regression | G2 | pre | augmented - plain | concat | all | 0.703 | 0.759 | -0.056 | -0.144 | 0.007 |
| RQ3 | regression | G3 | pre | augmented - plain | value | all | 0.710 | 0.784 | -0.074 | -0.162 | -0.014 |
| RQ3 | regression | G3 | pre | augmented - plain | rank | all | 0.328 | 0.635 | -0.307 | -0.557 | -0.085 |
| RQ3 | regression | G3 | pre | augmented - plain | concat | all | 0.718 | 0.782 | -0.064 | -0.139 | -0.008 |
| RQ3 | selection | Mixed | post | augmented - plain | value | — | 0.642 | 0.583 | 0.058 | 0.020 | 0.101 |
| RQ3 | selection | Mixed | post | augmented - plain | rank | — | 0.646 | 0.619 | 0.027 | -0.006 | 0.088 |
| RQ3 | selection | Mixed | post | augmented - plain | concat | — | 0.652 | 0.598 | 0.054 | 0.019 | 0.104 |
| RQ3 | regression | G1 | post | concat - rank | concat | all | 0.686 | 0.655 | 0.031 | -0.043 | 0.109 |
| RQ3 | regression | G1 | post | concat - value | concat | all | 0.686 | 0.582 | 0.104 | 0.033 | 0.196 |
| RQ3 | regression | G1 | post | rank - value | rank | all | 0.655 | 0.582 | 0.073 | -0.019 | 0.182 |
| RQ3 | regression | G2 | post | concat - rank | concat | all | 0.613 | 0.616 | -0.004 | -0.099 | 0.063 |
| RQ3 | regression | G2 | post | concat - value | concat | all | 0.613 | 0.525 | 0.087 | 0.025 | 0.175 |
| RQ3 | regression | G2 | post | rank - value | rank | all | 0.616 | 0.525 | 0.091 | 0.008 | 0.217 |
| RQ3 | regression | G3 | post | concat - rank | concat | all | 0.133 | 0.115 | 0.018 | -0.118 | 0.232 |
| RQ3 | regression | G3 | post | concat - value | concat | all | 0.133 | 0.157 | -0.024 | -0.089 | 0.098 |
| RQ3 | regression | G3 | post | rank - value | rank | all | 0.115 | 0.157 | -0.042 | -0.229 | 0.134 |
| RQ3 | selection | Mixed | post | concat - rank | concat | — | 0.652 | 0.646 | 0.006 | -0.024 | 0.042 |
| RQ3 | selection | Mixed | post | concat - value | concat | — | 0.652 | 0.642 | 0.010 | -0.001 | 0.030 |
| RQ3 | selection | Mixed | post | rank - value | rank | — | 0.646 | 0.642 | 0.004 | -0.034 | 0.043 |

#### 2.11.2 Deep-ELA, functions and instances (`appendix_bootstrap/bootstrap_intervals_deep-ela.csv`, `resampling = functions and instances`)

| question | part | group | view | comparison | left_variant | left_predictors | left level | right level | difference | 2.5% | 97.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | post | rank - value | rank | all | 0.615 | 0.131 | 0.484 | 0.199 | 0.912 |
| RQ2 | regression | G1 | post | rank - concat | rank | all | 0.615 | 0.135 | 0.480 | 0.191 | 0.885 |
| RQ2 | regression | G1 | pre | rank - value | rank | all | 0.615 | 0.845 | -0.230 | -0.411 | -0.122 |
| RQ2 | regression | G2 | post | rank - value | rank | all | 0.570 | -0.116 | 0.686 | 0.341 | 1.170 |
| RQ2 | regression | G2 | post | rank - concat | rank | all | 0.570 | -0.023 | 0.593 | 0.227 | 1.070 |
| RQ2 | regression | G2 | pre | rank - value | rank | all | 0.584 | 0.794 | -0.210 | -0.381 | -0.094 |
| RQ2 | regression | G3 | post | rank - value | rank | all | -0.920 | -0.716 | -0.204 | -0.957 | 0.371 |
| RQ2 | regression | G3 | post | rank - concat | rank | all | -0.920 | -0.664 | -0.256 | -0.905 | 0.195 |
| RQ2 | regression | G3 | pre | rank - value | rank | all | 0.620 | 0.825 | -0.205 | -0.366 | -0.095 |
| RQ2 | regression | G1 | pre | pre - post | value | all | 0.845 | 0.131 | 0.714 | 0.473 | 1.145 |
| RQ2 | regression | G1 | pre | pre - post | rank | all | 0.615 | 0.615 | 0.000 | 0.000 | 0.000 |
| RQ2 | regression | G1 | pre | pre - post | concat | all | 0.834 | 0.135 | 0.698 | 0.445 | 1.125 |
| RQ2 | regression | G2 | pre | pre - post | value | all | 0.794 | -0.116 | 0.910 | 0.646 | 1.377 |
| RQ2 | regression | G2 | pre | pre - post | rank | all | 0.584 | 0.570 | 0.014 | -0.044 | 0.079 |
| RQ2 | regression | G2 | pre | pre - post | concat | all | 0.798 | -0.023 | 0.821 | 0.538 | 1.287 |
| RQ2 | selection | Mixed | post | rank - value | rank | — | 0.612 | 0.560 | 0.052 | -0.034 | 0.128 |
| RQ2 | selection | Mixed | post | rank - concat | rank | — | 0.612 | 0.566 | 0.046 | -0.029 | 0.118 |
| RQ3 | regression | G1 | post | augmented - plain | value | all | 0.441 | 0.091 | 0.350 | 0.165 | 0.646 |
| RQ3 | regression | G1 | post | augmented - plain | rank | all | 0.612 | 0.615 | -0.003 | -0.054 | 0.048 |
| RQ3 | regression | G1 | post | augmented - plain | concat | all | 0.613 | 0.187 | 0.426 | 0.259 | 0.743 |
| RQ3 | regression | G2 | post | augmented - plain | value | all | 0.421 | 0.036 | 0.385 | 0.218 | 0.641 |
| RQ3 | regression | G2 | post | augmented - plain | rank | all | 0.578 | 0.568 | 0.010 | -0.045 | 0.078 |
| RQ3 | regression | G2 | post | augmented - plain | concat | all | 0.585 | 0.157 | 0.428 | 0.221 | 0.735 |
| RQ3 | regression | G3 | post | augmented - plain | value | all | 0.123 | -0.800 | 0.923 | 0.482 | 1.952 |
| RQ3 | regression | G3 | post | augmented - plain | rank | all | 0.123 | -0.992 | 1.115 | 0.475 | 2.459 |
| RQ3 | regression | G3 | post | augmented - plain | concat | all | 0.197 | -0.743 | 0.940 | 0.466 | 2.026 |
| RQ3 | regression | G1 | pre | augmented - plain | value | all | 0.817 | 0.845 | -0.028 | -0.083 | 0.021 |
| RQ3 | regression | G1 | pre | augmented - plain | rank | all | 0.612 | 0.615 | -0.003 | -0.054 | 0.048 |
| RQ3 | regression | G1 | pre | augmented - plain | concat | all | 0.792 | 0.834 | -0.041 | -0.123 | 0.007 |
| RQ3 | regression | G2 | pre | augmented - plain | value | all | 0.762 | 0.794 | -0.032 | -0.110 | 0.019 |
| RQ3 | regression | G2 | pre | augmented - plain | rank | all | 0.592 | 0.584 | 0.008 | -0.048 | 0.078 |
| RQ3 | regression | G2 | pre | augmented - plain | concat | all | 0.737 | 0.798 | -0.061 | -0.147 | 0.003 |
| RQ3 | regression | G3 | pre | augmented - plain | value | all | 0.727 | 0.825 | -0.099 | -0.215 | -0.023 |
| RQ3 | regression | G3 | pre | augmented - plain | rank | all | 0.328 | 0.620 | -0.292 | -0.560 | -0.062 |
| RQ3 | regression | G3 | pre | augmented - plain | concat | all | 0.703 | 0.820 | -0.116 | -0.234 | -0.035 |
| RQ3 | selection | Mixed | post | augmented - plain | value | — | 0.606 | 0.543 | 0.063 | 0.022 | 0.108 |
| RQ3 | selection | Mixed | post | augmented - plain | rank | — | 0.642 | 0.611 | 0.030 | -0.013 | 0.098 |
| RQ3 | selection | Mixed | post | augmented - plain | concat | — | 0.645 | 0.561 | 0.084 | 0.043 | 0.132 |
| RQ3 | regression | G1 | post | concat - rank | concat | all | 0.613 | 0.612 | 0.002 | -0.066 | 0.093 |
| RQ3 | regression | G1 | post | concat - value | concat | all | 0.613 | 0.441 | 0.172 | 0.099 | 0.297 |
| RQ3 | regression | G1 | post | rank - value | rank | all | 0.612 | 0.441 | 0.171 | 0.052 | 0.319 |
| RQ3 | regression | G2 | post | concat - rank | concat | all | 0.585 | 0.578 | 0.006 | -0.080 | 0.101 |
| RQ3 | regression | G2 | post | concat - value | concat | all | 0.585 | 0.421 | 0.164 | 0.083 | 0.281 |
| RQ3 | regression | G2 | post | rank - value | rank | all | 0.578 | 0.421 | 0.157 | 0.029 | 0.314 |
| RQ3 | regression | G3 | post | concat - rank | concat | all | 0.197 | 0.123 | 0.074 | -0.172 | 0.306 |
| RQ3 | regression | G3 | post | concat - value | concat | all | 0.197 | 0.123 | 0.074 | -0.020 | 0.166 |
| RQ3 | regression | G3 | post | rank - value | rank | all | 0.123 | 0.123 | 0.000 | -0.258 | 0.261 |
| RQ3 | selection | Mixed | post | concat - rank | concat | — | 0.645 | 0.642 | 0.004 | -0.026 | 0.041 |
| RQ3 | selection | Mixed | post | concat - value | concat | — | 0.645 | 0.606 | 0.039 | 0.020 | 0.063 |
| RQ3 | selection | Mixed | post | rank - value | rank | — | 0.642 | 0.606 | 0.036 | -0.012 | 0.079 |

#### 2.11.3 ELA, instances within functions (`appendix_bootstrap/bootstrap_intervals.csv`, `resampling = instances within functions`)

| question | part | group | view | comparison | left_variant | left_predictors | left level | right level | difference | 2.5% | 97.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ1 | regression | G1 | post | invariant - all | value | invariant | 0.452 | 0.233 | 0.219 | 0.073 | 0.343 |
| RQ1 | regression | G1 | pre | pre - post | value | all | 0.852 | 0.233 | 0.619 | 0.486 | 0.740 |
| RQ1 | regression | G1 | pre | pre - post | value | invariant | 0.452 | 0.452 | 0.000 | 0.000 | 0.000 |
| RQ1 | regression | G1 | pre | all - invariant | value | all | 0.852 | 0.452 | 0.400 | 0.341 | 0.469 |
| RQ2 | regression | G1 | post | rank - value | rank | all | 0.661 | 0.233 | 0.428 | 0.287 | 0.556 |
| RQ2 | regression | G1 | post | rank - concat | rank | all | 0.661 | 0.255 | 0.407 | 0.256 | 0.559 |
| RQ2 | regression | G1 | pre | rank - value | rank | all | 0.661 | 0.852 | -0.190 | -0.242 | -0.140 |
| RQ2 | regression | G2 | post | rank - value | rank | all | 0.607 | 0.196 | 0.411 | 0.248 | 0.537 |
| RQ2 | regression | G2 | post | rank - concat | rank | all | 0.607 | 0.184 | 0.423 | 0.260 | 0.573 |
| RQ2 | regression | G2 | pre | rank - value | rank | all | 0.621 | 0.770 | -0.148 | -0.215 | -0.093 |
| RQ2 | regression | G3 | post | rank - value | rank | all | -0.949 | -0.798 | -0.151 | -0.312 | 0.005 |
| RQ2 | regression | G3 | post | rank - concat | rank | all | -0.949 | -0.804 | -0.145 | -0.320 | 0.061 |
| RQ2 | regression | G3 | pre | rank - value | rank | all | 0.635 | 0.784 | -0.149 | -0.205 | -0.108 |
| RQ2 | regression | G1 | pre | pre - post | value | all | 0.852 | 0.233 | 0.619 | 0.486 | 0.740 |
| RQ2 | regression | G1 | pre | pre - post | rank | all | 0.661 | 0.661 | 0.000 | 0.000 | 0.000 |
| RQ2 | regression | G1 | pre | pre - post | concat | all | 0.849 | 0.255 | 0.595 | 0.451 | 0.743 |
| RQ2 | regression | G2 | pre | pre - post | value | all | 0.770 | 0.196 | 0.574 | 0.418 | 0.712 |
| RQ2 | regression | G2 | pre | pre - post | rank | all | 0.621 | 0.607 | 0.015 | -0.022 | 0.063 |
| RQ2 | regression | G2 | pre | pre - post | concat | all | 0.759 | 0.184 | 0.575 | 0.427 | 0.736 |
| RQ2 | selection | Mixed | post | rank - value | rank | — | 0.620 | 0.598 | 0.021 | -0.005 | 0.046 |
| RQ2 | selection | Mixed | post | rank - concat | rank | — | 0.620 | 0.600 | 0.020 | -0.008 | 0.042 |
| RQ3 | regression | G1 | post | augmented - plain | value | all | 0.582 | 0.245 | 0.337 | 0.271 | 0.412 |
| RQ3 | regression | G1 | post | augmented - plain | rank | all | 0.655 | 0.661 | -0.006 | -0.044 | 0.037 |
| RQ3 | regression | G1 | post | augmented - plain | concat | all | 0.686 | 0.262 | 0.425 | 0.343 | 0.534 |
| RQ3 | regression | G2 | post | augmented - plain | value | all | 0.525 | 0.252 | 0.273 | 0.191 | 0.320 |
| RQ3 | regression | G2 | post | augmented - plain | rank | all | 0.616 | 0.613 | 0.003 | -0.026 | 0.072 |
| RQ3 | regression | G2 | post | augmented - plain | concat | all | 0.613 | 0.280 | 0.333 | 0.251 | 0.408 |
| RQ3 | regression | G3 | post | augmented - plain | value | all | 0.157 | -0.814 | 0.970 | 0.831 | 1.090 |
| RQ3 | regression | G3 | post | augmented - plain | rank | all | 0.115 | -1.055 | 1.170 | 0.995 | 1.292 |
| RQ3 | regression | G3 | post | augmented - plain | concat | all | 0.133 | -0.832 | 0.965 | 0.849 | 1.111 |
| RQ3 | regression | G1 | pre | augmented - plain | value | all | 0.746 | 0.852 | -0.106 | -0.146 | -0.059 |
| RQ3 | regression | G1 | pre | augmented - plain | rank | all | 0.655 | 0.661 | -0.006 | -0.044 | 0.037 |
| RQ3 | regression | G1 | pre | augmented - plain | concat | all | 0.764 | 0.849 | -0.086 | -0.130 | -0.042 |
| RQ3 | regression | G2 | pre | augmented - plain | value | all | 0.684 | 0.770 | -0.086 | -0.138 | -0.044 |
| RQ3 | regression | G2 | pre | augmented - plain | rank | all | 0.632 | 0.621 | 0.011 | -0.029 | 0.071 |
| RQ3 | regression | G2 | pre | augmented - plain | concat | all | 0.703 | 0.759 | -0.056 | -0.105 | -0.012 |
| RQ3 | regression | G3 | pre | augmented - plain | value | all | 0.710 | 0.784 | -0.074 | -0.120 | -0.037 |
| RQ3 | regression | G3 | pre | augmented - plain | rank | all | 0.328 | 0.635 | -0.307 | -0.354 | -0.259 |
| RQ3 | regression | G3 | pre | augmented - plain | concat | all | 0.718 | 0.782 | -0.064 | -0.108 | -0.030 |
| RQ3 | selection | Mixed | post | augmented - plain | value | — | 0.642 | 0.583 | 0.058 | 0.046 | 0.068 |
| RQ3 | selection | Mixed | post | augmented - plain | rank | — | 0.646 | 0.619 | 0.027 | 0.016 | 0.045 |
| RQ3 | selection | Mixed | post | augmented - plain | concat | — | 0.652 | 0.598 | 0.054 | 0.045 | 0.070 |
| RQ3 | regression | G1 | post | concat - rank | concat | all | 0.686 | 0.655 | 0.031 | -0.014 | 0.077 |
| RQ3 | regression | G1 | post | concat - value | concat | all | 0.686 | 0.582 | 0.104 | 0.057 | 0.148 |
| RQ3 | regression | G1 | post | rank - value | rank | all | 0.655 | 0.582 | 0.073 | 0.014 | 0.131 |
| RQ3 | regression | G2 | post | concat - rank | concat | all | 0.613 | 0.616 | -0.004 | -0.061 | 0.031 |
| RQ3 | regression | G2 | post | concat - value | concat | all | 0.613 | 0.525 | 0.087 | 0.042 | 0.135 |
| RQ3 | regression | G2 | post | rank - value | rank | all | 0.616 | 0.525 | 0.091 | 0.050 | 0.152 |
| RQ3 | regression | G3 | post | concat - rank | concat | all | 0.133 | 0.115 | 0.018 | -0.066 | 0.133 |
| RQ3 | regression | G3 | post | concat - value | concat | all | 0.133 | 0.157 | -0.024 | -0.062 | 0.071 |
| RQ3 | regression | G3 | post | rank - value | rank | all | 0.115 | 0.157 | -0.042 | -0.131 | 0.072 |
| RQ3 | selection | Mixed | post | concat - rank | concat | — | 0.652 | 0.646 | 0.006 | -0.005 | 0.020 |
| RQ3 | selection | Mixed | post | concat - value | concat | — | 0.652 | 0.642 | 0.010 | 0.004 | 0.022 |
| RQ3 | selection | Mixed | post | rank - value | rank | — | 0.646 | 0.642 | 0.004 | -0.008 | 0.019 |

#### 2.11.4 Deep-ELA, instances within functions (`appendix_bootstrap/bootstrap_intervals_deep-ela.csv`, `resampling = instances within functions`)

| question | part | group | view | comparison | left_variant | left_predictors | left level | right level | difference | 2.5% | 97.5% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RQ2 | regression | G1 | post | rank - value | rank | all | 0.615 | 0.131 | 0.484 | 0.335 | 0.620 |
| RQ2 | regression | G1 | post | rank - concat | rank | all | 0.615 | 0.135 | 0.480 | 0.317 | 0.606 |
| RQ2 | regression | G1 | pre | rank - value | rank | all | 0.615 | 0.845 | -0.230 | -0.302 | -0.165 |
| RQ2 | regression | G2 | post | rank - value | rank | all | 0.570 | -0.116 | 0.686 | 0.526 | 0.856 |
| RQ2 | regression | G2 | post | rank - concat | rank | all | 0.570 | -0.023 | 0.593 | 0.433 | 0.758 |
| RQ2 | regression | G2 | pre | rank - value | rank | all | 0.584 | 0.794 | -0.210 | -0.284 | -0.147 |
| RQ2 | regression | G3 | post | rank - value | rank | all | -0.920 | -0.716 | -0.204 | -0.450 | -0.013 |
| RQ2 | regression | G3 | post | rank - concat | rank | all | -0.920 | -0.664 | -0.256 | -0.467 | -0.083 |
| RQ2 | regression | G3 | pre | rank - value | rank | all | 0.620 | 0.825 | -0.205 | -0.249 | -0.147 |
| RQ2 | regression | G1 | pre | pre - post | value | all | 0.845 | 0.131 | 0.714 | 0.573 | 0.844 |
| RQ2 | regression | G1 | pre | pre - post | rank | all | 0.615 | 0.615 | 0.000 | 0.000 | 0.000 |
| RQ2 | regression | G1 | pre | pre - post | concat | all | 0.834 | 0.135 | 0.698 | 0.554 | 0.826 |
| RQ2 | regression | G2 | pre | pre - post | value | all | 0.794 | -0.116 | 0.910 | 0.761 | 1.081 |
| RQ2 | regression | G2 | pre | pre - post | rank | all | 0.584 | 0.570 | 0.014 | -0.025 | 0.056 |
| RQ2 | regression | G2 | pre | pre - post | concat | all | 0.798 | -0.023 | 0.821 | 0.668 | 0.986 |
| RQ2 | selection | Mixed | post | rank - value | rank | — | 0.612 | 0.560 | 0.052 | 0.018 | 0.084 |
| RQ2 | selection | Mixed | post | rank - concat | rank | — | 0.612 | 0.566 | 0.046 | 0.016 | 0.080 |
| RQ3 | regression | G1 | post | augmented - plain | value | all | 0.441 | 0.091 | 0.350 | 0.282 | 0.422 |
| RQ3 | regression | G1 | post | augmented - plain | rank | all | 0.612 | 0.615 | -0.003 | -0.038 | 0.033 |
| RQ3 | regression | G1 | post | augmented - plain | concat | all | 0.613 | 0.187 | 0.426 | 0.367 | 0.506 |
| RQ3 | regression | G2 | post | augmented - plain | value | all | 0.421 | 0.036 | 0.385 | 0.317 | 0.462 |
| RQ3 | regression | G2 | post | augmented - plain | rank | all | 0.578 | 0.568 | 0.010 | -0.025 | 0.053 |
| RQ3 | regression | G2 | post | augmented - plain | concat | all | 0.585 | 0.157 | 0.428 | 0.351 | 0.508 |
| RQ3 | regression | G3 | post | augmented - plain | value | all | 0.123 | -0.800 | 0.923 | 0.810 | 1.061 |
| RQ3 | regression | G3 | post | augmented - plain | rank | all | 0.123 | -0.992 | 1.115 | 1.003 | 1.287 |
| RQ3 | regression | G3 | post | augmented - plain | concat | all | 0.197 | -0.743 | 0.940 | 0.823 | 1.073 |
| RQ3 | regression | G1 | pre | augmented - plain | value | all | 0.817 | 0.845 | -0.028 | -0.065 | 0.008 |
| RQ3 | regression | G1 | pre | augmented - plain | rank | all | 0.612 | 0.615 | -0.003 | -0.038 | 0.033 |
| RQ3 | regression | G1 | pre | augmented - plain | concat | all | 0.792 | 0.834 | -0.041 | -0.094 | -0.006 |
| RQ3 | regression | G2 | pre | augmented - plain | value | all | 0.762 | 0.794 | -0.032 | -0.079 | 0.005 |
| RQ3 | regression | G2 | pre | augmented - plain | rank | all | 0.592 | 0.584 | 0.008 | -0.028 | 0.055 |
| RQ3 | regression | G2 | pre | augmented - plain | concat | all | 0.737 | 0.798 | -0.061 | -0.109 | -0.015 |
| RQ3 | regression | G3 | pre | augmented - plain | value | all | 0.727 | 0.825 | -0.099 | -0.144 | -0.065 |
| RQ3 | regression | G3 | pre | augmented - plain | rank | all | 0.328 | 0.620 | -0.292 | -0.352 | -0.252 |
| RQ3 | regression | G3 | pre | augmented - plain | concat | all | 0.703 | 0.820 | -0.116 | -0.157 | -0.073 |
| RQ3 | selection | Mixed | post | augmented - plain | value | — | 0.606 | 0.543 | 0.063 | 0.048 | 0.077 |
| RQ3 | selection | Mixed | post | augmented - plain | rank | — | 0.642 | 0.611 | 0.030 | 0.011 | 0.042 |
| RQ3 | selection | Mixed | post | augmented - plain | concat | — | 0.645 | 0.561 | 0.084 | 0.070 | 0.101 |
| RQ3 | regression | G1 | post | concat - rank | concat | all | 0.613 | 0.612 | 0.002 | -0.039 | 0.056 |
| RQ3 | regression | G1 | post | concat - value | concat | all | 0.613 | 0.441 | 0.172 | 0.130 | 0.230 |
| RQ3 | regression | G1 | post | rank - value | rank | all | 0.612 | 0.441 | 0.171 | 0.106 | 0.239 |
| RQ3 | regression | G2 | post | concat - rank | concat | all | 0.585 | 0.578 | 0.006 | -0.045 | 0.055 |
| RQ3 | regression | G2 | post | concat - value | concat | all | 0.585 | 0.421 | 0.164 | 0.111 | 0.229 |
| RQ3 | regression | G2 | post | rank - value | rank | all | 0.578 | 0.421 | 0.157 | 0.086 | 0.246 |
| RQ3 | regression | G3 | post | concat - rank | concat | all | 0.197 | 0.123 | 0.074 | -0.025 | 0.154 |
| RQ3 | regression | G3 | post | concat - value | concat | all | 0.197 | 0.123 | 0.074 | 0.017 | 0.120 |
| RQ3 | regression | G3 | post | rank - value | rank | all | 0.123 | 0.123 | 0.000 | -0.089 | 0.098 |
| RQ3 | selection | Mixed | post | concat - rank | concat | — | 0.645 | 0.642 | 0.004 | -0.008 | 0.021 |
| RQ3 | selection | Mixed | post | concat - value | concat | — | 0.645 | 0.606 | 0.039 | 0.029 | 0.051 |
| RQ3 | selection | Mixed | post | rank - value | rank | — | 0.642 | 0.606 | 0.036 | 0.016 | 0.051 |

### 2.12 Experiment 5's selectors on the untransformed test block

Θ, reduced as the paper's levels are; Mann–Whitney augmented against plain,
Holm over the three representations.

#### 2.12.1 ELA levels (`appendix_checks/untransformed_selection_levels.csv`)

| feature_variant | plain | augmented |
| --- | --- | --- |
| value | 0.704 | 0.681 |
| rank | 0.646 | 0.650 |
| concat | 0.704 | 0.683 |

#### 2.12.2 ELA tests (`appendix_checks/untransformed_selection_tests.csv`)

| feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- |
| value | plain | -0.023 | 5.67e-06 | 1.13e-05 | True |
| rank | augmented | 0.004 | 0.0192 | 0.0192 | True |
| concat | plain | -0.022 | 2.84e-06 | 8.51e-06 | True |

#### 2.12.3 Deep-ELA levels (`appendix_checks/untransformed_selection_levels_deep-ela.csv`)

| feature_variant | plain | augmented |
| --- | --- | --- |
| value | 0.707 | 0.703 |
| rank | 0.643 | 0.645 |
| concat | 0.708 | 0.695 |

#### 2.12.4 Deep-ELA tests (`appendix_checks/untransformed_selection_tests_deep-ela.csv`)

| feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- |
| value | plain | -0.004 | 0.0473 | 0.0946 | False |
| rank | augmented | 0.002 | 0.748 | 0.748 | False |
| concat | plain | -0.013 | 2.84e-06 | 8.51e-06 | True |

### 2.13 G1: how far a prediction moves under the transformation

The mean over a deployment's 240 problems of |Ĵ(p̃) − Ĵ(p)|, reduced as R²
is; Mann–Whitney augmented against plain, Holm over `value` and `concat`.

#### 2.13.1 ELA levels (`appendix_checks/g1_prediction_change_levels.csv`)

| scope | feature_variant | plain | augmented |
| --- | --- | --- | --- |
| unseen-family mean | value | 0.189 | 0.086 |
| unseen-family mean | rank | 0.000 | 0.000 |
| unseen-family mean | concat | 0.186 | 0.053 |
| spline only | value | 0.200 | 0.081 |
| spline only | rank | 0.000 | 0.000 |
| spline only | concat | 0.195 | 0.051 |

#### 2.13.2 ELA tests (`appendix_checks/g1_prediction_change_tests.csv`)

| scope | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| unseen-family mean | value | plain | -0.103 | 2.84e-06 | 5.67e-06 | True |
| unseen-family mean | concat | plain | -0.133 | 2.84e-06 | 5.67e-06 | True |
| spline only | value | plain | -0.119 | 2.84e-06 | 5.67e-06 | True |
| spline only | concat | plain | -0.144 | 2.84e-06 | 5.67e-06 | True |

#### 2.13.3 Deep-ELA levels (`appendix_checks/g1_prediction_change_levels_deep-ela.csv`)

| scope | feature_variant | plain | augmented |
| --- | --- | --- | --- |
| unseen-family mean | value | 0.227 | 0.146 |
| unseen-family mean | rank | 0.000 | 0.000 |
| unseen-family mean | concat | 0.205 | 0.083 |
| spline only | value | 0.232 | 0.148 |
| spline only | rank | 0.000 | 0.000 |
| spline only | concat | 0.223 | 0.079 |

#### 2.13.4 Deep-ELA tests (`appendix_checks/g1_prediction_change_tests_deep-ela.csv`)

| scope | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| unseen-family mean | value | plain | -0.082 | 2.84e-06 | 5.67e-06 | True |
| unseen-family mean | concat | plain | -0.122 | 2.84e-06 | 5.67e-06 | True |
| spline only | value | plain | -0.084 | 2.84e-06 | 5.67e-06 | True |
| spline only | concat | plain | -0.144 | 2.84e-06 | 5.67e-06 | True |

### 2.14 The target's spread and the regression's error

#### 2.14.1 The target on the test block (`appendix_checks/target_spread.csv`; variance over n, the R² denominator over n)

| audit | group | mean, pre | variance, pre | mean, post | variance, post |
| --- | --- | --- | --- | --- | --- |
| spline/warpheld | G1 | 0.469 | 0.154 | 0.469 | 0.154 |
| spline/warpheld | G2 | 0.578 | 0.067 | 0.574 | 0.067 |
| spline/warpheld | G3 | 0.513 | 0.119 | 0.687 | 0.097 |
| knots-sparse/warpheld | G1 | 0.469 | 0.154 | 0.469 | 0.154 |
| knots-sparse/warpheld | G2 | 0.578 | 0.067 | 0.577 | 0.067 |
| knots-sparse/warpheld | G3 | 0.513 | 0.119 | 0.687 | 0.094 |
| knots-dense/warpheld | G1 | 0.469 | 0.154 | 0.469 | 0.154 |
| knots-dense/warpheld | G2 | 0.578 | 0.067 | 0.575 | 0.064 |
| knots-dense/warpheld | G3 | 0.513 | 0.119 | 0.677 | 0.088 |
| power/warpheld | G1 | 0.469 | 0.154 | 0.469 | 0.154 |
| power/warpheld | G2 | 0.578 | 0.067 | 0.571 | 0.065 |
| power/warpheld | G3 | 0.513 | 0.119 | 0.725 | 0.084 |
| piecewise-linear/warpheld | G1 | 0.469 | 0.154 | 0.469 | 0.154 |
| piecewise-linear/warpheld | G2 | 0.578 | 0.067 | 0.580 | 0.065 |
| piecewise-linear/warpheld | G3 | 0.513 | 0.119 | 0.691 | 0.084 |

#### 2.14.2 ELA: R², MSE and the target's variance (`appendix_checks/regression_error_levels.csv`)

| scope | transform | group | feature_variant | plain R² | plain MSE | augmented R² | augmented MSE | target variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen-family mean | pre | G1 | value | 0.852 | 0.023 | 0.746 | 0.039 | 0.154 |
| unseen-family mean | pre | G1 | rank | 0.661 | 0.052 | 0.655 | 0.053 | 0.154 |
| unseen-family mean | pre | G1 | concat | 0.849 | 0.023 | 0.764 | 0.036 | 0.154 |
| unseen-family mean | pre | G2 | value | 0.770 | 0.015 | 0.684 | 0.021 | 0.067 |
| unseen-family mean | pre | G2 | rank | 0.621 | 0.025 | 0.632 | 0.025 | 0.067 |
| unseen-family mean | pre | G2 | concat | 0.759 | 0.016 | 0.703 | 0.020 | 0.067 |
| unseen-family mean | pre | G3 | value | 0.784 | 0.026 | 0.710 | 0.035 | 0.119 |
| unseen-family mean | pre | G3 | rank | 0.635 | 0.044 | 0.328 | 0.080 | 0.119 |
| unseen-family mean | pre | G3 | concat | 0.782 | 0.026 | 0.718 | 0.034 | 0.119 |
| unseen-family mean | post | G1 | value | 0.245 | 0.117 | 0.582 | 0.065 | 0.154 |
| unseen-family mean | post | G1 | rank | 0.661 | 0.052 | 0.655 | 0.053 | 0.154 |
| unseen-family mean | post | G1 | concat | 0.262 | 0.114 | 0.686 | 0.048 | 0.154 |
| unseen-family mean | post | G2 | value | 0.252 | 0.049 | 0.525 | 0.031 | 0.065 |
| unseen-family mean | post | G2 | rank | 0.613 | 0.025 | 0.616 | 0.025 | 0.065 |
| unseen-family mean | post | G2 | concat | 0.280 | 0.047 | 0.613 | 0.025 | 0.065 |
| unseen-family mean | post | G3 | value | -0.814 | 0.159 | 0.157 | 0.074 | 0.087 |
| unseen-family mean | post | G3 | rank | -1.055 | 0.179 | 0.115 | 0.077 | 0.087 |
| unseen-family mean | post | G3 | concat | -0.832 | 0.160 | 0.133 | 0.076 | 0.087 |
| spline only | pre | G1 | value | 0.852 | 0.023 | 0.746 | 0.039 | 0.154 |
| spline only | pre | G1 | rank | 0.661 | 0.052 | 0.655 | 0.053 | 0.154 |
| spline only | pre | G1 | concat | 0.849 | 0.023 | 0.764 | 0.036 | 0.154 |
| spline only | pre | G2 | value | 0.770 | 0.015 | 0.684 | 0.021 | 0.067 |
| spline only | pre | G2 | rank | 0.621 | 0.025 | 0.632 | 0.025 | 0.067 |
| spline only | pre | G2 | concat | 0.759 | 0.016 | 0.703 | 0.020 | 0.067 |
| spline only | pre | G3 | value | 0.784 | 0.026 | 0.710 | 0.035 | 0.119 |
| spline only | pre | G3 | rank | 0.635 | 0.044 | 0.328 | 0.080 | 0.119 |
| spline only | pre | G3 | concat | 0.782 | 0.026 | 0.718 | 0.034 | 0.119 |
| spline only | post | G1 | value | 0.233 | 0.118 | 0.610 | 0.060 | 0.154 |
| spline only | post | G1 | rank | 0.661 | 0.052 | 0.655 | 0.053 | 0.154 |
| spline only | post | G1 | concat | 0.255 | 0.115 | 0.691 | 0.048 | 0.154 |
| spline only | post | G2 | value | 0.196 | 0.054 | 0.545 | 0.031 | 0.067 |
| spline only | post | G2 | rank | 0.607 | 0.026 | 0.599 | 0.027 | 0.067 |
| spline only | post | G2 | concat | 0.184 | 0.055 | 0.632 | 0.025 | 0.067 |
| spline only | post | G3 | value | -0.798 | 0.174 | 0.273 | 0.070 | 0.097 |
| spline only | post | G3 | rank | -0.949 | 0.188 | 0.182 | 0.079 | 0.097 |
| spline only | post | G3 | concat | -0.804 | 0.174 | 0.335 | 0.064 | 0.097 |

#### 2.14.3 Deep-ELA (`appendix_checks/regression_error_levels_deep-ela.csv`, the same rows)

| scope | transform | group | feature_variant | plain R² | plain MSE | augmented R² | augmented MSE | target variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unseen-family mean | pre | G1 | value | 0.845 | 0.024 | 0.817 | 0.028 | 0.154 |
| unseen-family mean | pre | G1 | rank | 0.615 | 0.059 | 0.612 | 0.060 | 0.154 |
| unseen-family mean | pre | G1 | concat | 0.834 | 0.026 | 0.792 | 0.032 | 0.154 |
| unseen-family mean | pre | G2 | value | 0.794 | 0.014 | 0.762 | 0.016 | 0.067 |
| unseen-family mean | pre | G2 | rank | 0.584 | 0.028 | 0.592 | 0.027 | 0.067 |
| unseen-family mean | pre | G2 | concat | 0.798 | 0.013 | 0.737 | 0.018 | 0.067 |
| unseen-family mean | pre | G3 | value | 0.825 | 0.021 | 0.727 | 0.033 | 0.119 |
| unseen-family mean | pre | G3 | rank | 0.620 | 0.045 | 0.328 | 0.080 | 0.119 |
| unseen-family mean | pre | G3 | concat | 0.820 | 0.022 | 0.703 | 0.035 | 0.119 |
| unseen-family mean | post | G1 | value | 0.091 | 0.140 | 0.441 | 0.086 | 0.154 |
| unseen-family mean | post | G1 | rank | 0.615 | 0.059 | 0.612 | 0.060 | 0.154 |
| unseen-family mean | post | G1 | concat | 0.187 | 0.125 | 0.613 | 0.060 | 0.154 |
| unseen-family mean | post | G2 | value | 0.036 | 0.063 | 0.421 | 0.038 | 0.065 |
| unseen-family mean | post | G2 | rank | 0.568 | 0.028 | 0.578 | 0.028 | 0.065 |
| unseen-family mean | post | G2 | concat | 0.157 | 0.055 | 0.585 | 0.027 | 0.065 |
| unseen-family mean | post | G3 | value | -0.800 | 0.158 | 0.123 | 0.077 | 0.087 |
| unseen-family mean | post | G3 | rank | -0.992 | 0.174 | 0.123 | 0.076 | 0.087 |
| unseen-family mean | post | G3 | concat | -0.743 | 0.153 | 0.197 | 0.070 | 0.087 |
| spline only | pre | G1 | value | 0.845 | 0.024 | 0.817 | 0.028 | 0.154 |
| spline only | pre | G1 | rank | 0.615 | 0.059 | 0.612 | 0.060 | 0.154 |
| spline only | pre | G1 | concat | 0.834 | 0.026 | 0.792 | 0.032 | 0.154 |
| spline only | pre | G2 | value | 0.794 | 0.014 | 0.762 | 0.016 | 0.067 |
| spline only | pre | G2 | rank | 0.584 | 0.028 | 0.592 | 0.027 | 0.067 |
| spline only | pre | G2 | concat | 0.798 | 0.013 | 0.737 | 0.018 | 0.067 |
| spline only | pre | G3 | value | 0.825 | 0.021 | 0.727 | 0.033 | 0.119 |
| spline only | pre | G3 | rank | 0.620 | 0.045 | 0.328 | 0.080 | 0.119 |
| spline only | pre | G3 | concat | 0.820 | 0.022 | 0.703 | 0.035 | 0.119 |
| spline only | post | G1 | value | 0.131 | 0.134 | 0.450 | 0.085 | 0.154 |
| spline only | post | G1 | rank | 0.615 | 0.059 | 0.612 | 0.060 | 0.154 |
| spline only | post | G1 | concat | 0.135 | 0.133 | 0.639 | 0.056 | 0.154 |
| spline only | post | G2 | value | -0.116 | 0.075 | 0.426 | 0.039 | 0.067 |
| spline only | post | G2 | rank | 0.570 | 0.029 | 0.584 | 0.028 | 0.067 |
| spline only | post | G2 | concat | -0.023 | 0.069 | 0.617 | 0.026 | 0.067 |
| spline only | post | G3 | value | -0.716 | 0.166 | 0.333 | 0.064 | 0.097 |
| spline only | post | G3 | rank | -0.920 | 0.185 | 0.186 | 0.079 | 0.097 |
| spline only | post | G3 | concat | -0.664 | 0.161 | 0.409 | 0.057 | 0.097 |

### 2.15 G3's members

#### 2.15.1 Mean superiority on the test block (`appendix_checks/g3_member_superiority.csv`)

| measure | audit | algorithm | pre | post |
| --- | --- | --- | --- | --- |
| superiority over the other G3 member (regression binding) | spline/warpheld | BOBYQA | 0.513 | 0.687 |
| superiority over the other G3 member (regression binding) | spline/warpheld | Dual Annealing | 0.487 | 0.313 |
| superiority over the other G3 member (regression binding) | knots-sparse/warpheld | BOBYQA | 0.513 | 0.687 |
| superiority over the other G3 member (regression binding) | knots-sparse/warpheld | Dual Annealing | 0.487 | 0.313 |
| superiority over the other G3 member (regression binding) | knots-dense/warpheld | BOBYQA | 0.513 | 0.677 |
| superiority over the other G3 member (regression binding) | knots-dense/warpheld | Dual Annealing | 0.487 | 0.323 |
| superiority over the other G3 member (regression binding) | power/warpheld | BOBYQA | 0.513 | 0.725 |
| superiority over the other G3 member (regression binding) | power/warpheld | Dual Annealing | 0.487 | 0.275 |
| superiority over the other G3 member (regression binding) | piecewise-linear/warpheld | BOBYQA | 0.513 | 0.691 |
| superiority over the other G3 member (regression binding) | piecewise-linear/warpheld | Dual Annealing | 0.487 | 0.309 |
| mean superiority over the portfolio (selection binding) | spline/warpheld | BOBYQA | 0.448 | 0.425 |
| mean superiority over the portfolio (selection binding) | spline/warpheld | Dual Annealing | 0.161 | 0.140 |
| mean superiority over the portfolio (selection binding) | knots-sparse/warpheld | BOBYQA | 0.448 | 0.425 |
| mean superiority over the portfolio (selection binding) | knots-sparse/warpheld | Dual Annealing | 0.161 | 0.146 |
| mean superiority over the portfolio (selection binding) | knots-dense/warpheld | BOBYQA | 0.448 | 0.425 |
| mean superiority over the portfolio (selection binding) | knots-dense/warpheld | Dual Annealing | 0.161 | 0.135 |
| mean superiority over the portfolio (selection binding) | power/warpheld | BOBYQA | 0.448 | 0.413 |
| mean superiority over the portfolio (selection binding) | power/warpheld | Dual Annealing | 0.161 | 0.122 |
| mean superiority over the portfolio (selection binding) | piecewise-linear/warpheld | BOBYQA | 0.448 | 0.437 |
| mean superiority over the portfolio (selection binding) | piecewise-linear/warpheld | Dual Annealing | 0.161 | 0.137 |

### 2.16 Levels on each family

The untransformed block is the same for every family.

#### 2.16.1 ELA regression, R² (`appendix_checks/regression_family_levels.csv`)

| group | feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | value | plain | 0.852 | 0.233 | 0.311 | 0.150 | 0.297 | 0.191 |
| G1 | value | augmented | 0.746 | 0.610 | 0.609 | 0.594 | 0.547 | 0.590 |
| G1 | rank | plain | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 |
| G1 | rank | augmented | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 |
| G1 | concat | plain | 0.849 | 0.255 | 0.341 | 0.154 | 0.311 | 0.213 |
| G1 | concat | augmented | 0.764 | 0.691 | 0.690 | 0.702 | 0.647 | 0.696 |
| G2 | value | plain | 0.770 | 0.196 | 0.369 | 0.266 | 0.179 | 0.258 |
| G2 | value | augmented | 0.684 | 0.545 | 0.541 | 0.561 | 0.429 | 0.554 |
| G2 | rank | plain | 0.621 | 0.607 | 0.609 | 0.605 | 0.606 | 0.624 |
| G2 | rank | augmented | 0.632 | 0.599 | 0.623 | 0.609 | 0.612 | 0.625 |
| G2 | concat | plain | 0.759 | 0.184 | 0.388 | 0.300 | 0.230 | 0.240 |
| G2 | concat | augmented | 0.703 | 0.632 | 0.628 | 0.641 | 0.526 | 0.643 |
| G3 | value | plain | 0.784 | -0.798 | -0.841 | -1.004 | -0.738 | -0.723 |
| G3 | value | augmented | 0.710 | 0.273 | 0.035 | 0.353 | -0.063 | 0.239 |
| G3 | rank | plain | 0.635 | -0.949 | -0.945 | -1.037 | -1.313 | -0.923 |
| G3 | rank | augmented | 0.328 | 0.182 | 0.183 | 0.164 | -0.031 | 0.143 |
| G3 | concat | plain | 0.782 | -0.804 | -0.857 | -0.995 | -0.772 | -0.705 |
| G3 | concat | augmented | 0.718 | 0.335 | -0.011 | 0.366 | -0.070 | 0.224 |

#### 2.16.2 Deep-ELA regression, R² (`appendix_checks/regression_family_levels_deep-ela.csv`)

| group | feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | value | plain | 0.845 | 0.131 | 0.172 | -0.081 | 0.160 | 0.117 |
| G1 | value | augmented | 0.817 | 0.450 | 0.545 | 0.315 | 0.458 | 0.446 |
| G1 | rank | plain | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 |
| G1 | rank | augmented | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 |
| G1 | concat | plain | 0.834 | 0.135 | 0.251 | 0.004 | 0.278 | 0.198 |
| G1 | concat | augmented | 0.792 | 0.639 | 0.642 | 0.557 | 0.614 | 0.663 |
| G2 | value | plain | 0.794 | -0.116 | 0.165 | -0.141 | 0.087 | 0.014 |
| G2 | value | augmented | 0.762 | 0.426 | 0.503 | 0.357 | 0.311 | 0.501 |
| G2 | rank | plain | 0.584 | 0.570 | 0.580 | 0.553 | 0.555 | 0.585 |
| G2 | rank | augmented | 0.592 | 0.584 | 0.584 | 0.566 | 0.567 | 0.587 |
| G2 | concat | plain | 0.798 | -0.023 | 0.269 | 0.109 | 0.131 | 0.148 |
| G2 | concat | augmented | 0.737 | 0.617 | 0.616 | 0.584 | 0.516 | 0.623 |
| G3 | value | plain | 0.825 | -0.716 | -0.837 | -0.961 | -0.808 | -0.604 |
| G3 | value | augmented | 0.727 | 0.333 | 0.105 | 0.218 | 0.085 | 0.071 |
| G3 | rank | plain | 0.620 | -0.920 | -0.906 | -0.996 | -1.206 | -0.859 |
| G3 | rank | augmented | 0.328 | 0.186 | 0.191 | 0.184 | -0.034 | 0.154 |
| G3 | concat | plain | 0.820 | -0.664 | -0.832 | -0.810 | -0.765 | -0.596 |
| G3 | concat | augmented | 0.703 | 0.409 | 0.166 | 0.338 | 0.116 | 0.159 |

#### 2.16.3 ELA selection, Θ (`appendix_checks/selection_family_levels.csv`)

| feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| value | plain | 0.704 | 0.598 | 0.589 | 0.593 | 0.554 | 0.602 |
| value | augmented | 0.681 | 0.650 | 0.647 | 0.647 | 0.625 | 0.647 |
| rank | plain | 0.646 | 0.620 | 0.621 | 0.616 | 0.613 | 0.624 |
| rank | augmented | 0.650 | 0.647 | 0.649 | 0.644 | 0.644 | 0.648 |
| concat | plain | 0.704 | 0.600 | 0.604 | 0.598 | 0.576 | 0.615 |
| concat | augmented | 0.683 | 0.654 | 0.657 | 0.658 | 0.641 | 0.662 |

#### 2.16.4 Deep-ELA selection, Θ (`appendix_checks/selection_family_levels_deep-ela.csv`)

| feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| value | plain | 0.707 | 0.560 | 0.551 | 0.533 | 0.522 | 0.565 |
| value | augmented | 0.703 | 0.609 | 0.623 | 0.587 | 0.585 | 0.630 |
| rank | plain | 0.643 | 0.612 | 0.613 | 0.607 | 0.607 | 0.620 |
| rank | augmented | 0.645 | 0.644 | 0.645 | 0.640 | 0.638 | 0.643 |
| concat | plain | 0.708 | 0.566 | 0.573 | 0.552 | 0.544 | 0.580 |
| concat | augmented | 0.695 | 0.654 | 0.649 | 0.633 | 0.635 | 0.663 |

### 2.17 The non-finite rule read on the training block alone

#### 2.17.1 Columns each reading drops (`appendix_checks/feature_exclusion_rules.csv`)

| rule | excluded |
| --- | --- |
| declared (every block the paper reads) | ela_meta.quad_simple.cond;ic.eps_ratio |
| recomputed over every block the paper reads | ela_meta.quad_simple.cond;ic.eps_ratio |
| training block alone (both views, instances 11-110) | ela_meta.quad_simple.cond |

### 2.18 G1's runs in both views

#### 2.18.1 Runs that differ between the views (`appendix_checks/g1_trajectory_identity.csv`)

| binding | block | algorithm | runs | best_y differs | evaluations differ | restarts differ | any differs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regression | training | BIPOP-CMA-ES | 74400 | 0 | 0 | 0 | 0 |
| regression | training | Adaptive Nelder-Mead | 74400 | 0 | 0 | 0 | 0 |
| regression | spline/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | spline/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | knots-sparse/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | knots-sparse/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | knots-dense/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | knots-dense/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | power/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | power/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | piecewise-linear/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | piecewise-linear/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| selection | training | BIPOP-CMA-ES | 74400 | 96 | 0 | 2 | 96 |
| selection | training | Adaptive Nelder-Mead | 74400 | 899 | 0 | 59 | 933 |
| selection | spline/warpheld | BIPOP-CMA-ES | 7440 | 7 | 0 | 0 | 7 |
| selection | spline/warpheld | Adaptive Nelder-Mead | 7440 | 72 | 0 | 4 | 74 |
| selection | knots-sparse/warpheld | BIPOP-CMA-ES | 7440 | 3 | 0 | 0 | 3 |
| selection | knots-sparse/warpheld | Adaptive Nelder-Mead | 7440 | 62 | 0 | 2 | 63 |
| selection | knots-dense/warpheld | BIPOP-CMA-ES | 7440 | 25 | 0 | 4 | 26 |
| selection | knots-dense/warpheld | Adaptive Nelder-Mead | 7440 | 107 | 0 | 3 | 109 |
| selection | power/warpheld | BIPOP-CMA-ES | 7440 | 37 | 0 | 2 | 37 |
| selection | power/warpheld | Adaptive Nelder-Mead | 7440 | 183 | 0 | 8 | 187 |
| selection | piecewise-linear/warpheld | BIPOP-CMA-ES | 7440 | 6 | 0 | 1 | 7 |
| selection | piecewise-linear/warpheld | Adaptive Nelder-Mead | 7440 | 88 | 0 | 3 | 91 |
| sbs | spline/warpheld | BIPOP-CMA-ES | 7440 | 7 | 0 | 0 | 7 |
| sbs | spline/warpheld | Adaptive Nelder-Mead | 7440 | 69 | 0 | 2 | 70 |
| sbs | knots-sparse/warpheld | BIPOP-CMA-ES | 7440 | 2 | 0 | 0 | 2 |
| sbs | knots-sparse/warpheld | Adaptive Nelder-Mead | 7440 | 59 | 0 | 2 | 60 |
| sbs | knots-dense/warpheld | BIPOP-CMA-ES | 7440 | 25 | 0 | 1 | 25 |
| sbs | knots-dense/warpheld | Adaptive Nelder-Mead | 7440 | 102 | 0 | 9 | 105 |
| sbs | power/warpheld | BIPOP-CMA-ES | 7440 | 34 | 0 | 5 | 36 |
| sbs | power/warpheld | Adaptive Nelder-Mead | 7440 | 179 | 0 | 8 | 182 |
| sbs | piecewise-linear/warpheld | BIPOP-CMA-ES | 7440 | 7 | 0 | 1 | 7 |
| sbs | piecewise-linear/warpheld | Adaptive Nelder-Mead | 7440 | 82 | 0 | 6 | 84 |

### 2.19 Further blocks of the control family: the target and G1's runs

`spline-alpha0` is the identity map, so only the rescaling remains;
`spline-draw1` … `spline-draw4` draw the control family's maps again
(`spline` is the paper's draw).

#### 2.19.1 Standardized shift of each pair's superiority (`appendix_blocks/block_target_shift.csv`)

| audit | quantity_class | moved | standardized shift | raw shift |
| --- | --- | --- | --- | --- |
| spline/warpheld | G1 | 0.000 | 0.000 | 0.000 |
| spline/warpheld | G2 | 0.912 | 0.249 | 0.064 |
| spline/warpheld | G3 | 0.938 | 0.802 | 0.277 |
| spline-alpha0/warpheld | G1 | 0.000 | 0.000 | 0.000 |
| spline-alpha0/warpheld | G2 | 0.471 | 0.085 | 0.022 |
| spline-alpha0/warpheld | G3 | 0.908 | 0.654 | 0.226 |
| spline-draw1/warpheld | G1 | 0.000 | 0.000 | 0.000 |
| spline-draw1/warpheld | G2 | 0.917 | 0.218 | 0.056 |
| spline-draw1/warpheld | G3 | 0.946 | 0.784 | 0.271 |
| spline-draw2/warpheld | G1 | 0.000 | 0.000 | 0.000 |
| spline-draw2/warpheld | G2 | 0.900 | 0.224 | 0.058 |
| spline-draw2/warpheld | G3 | 0.963 | 0.769 | 0.266 |
| spline-draw3/warpheld | G1 | 0.000 | 0.000 | 0.000 |
| spline-draw3/warpheld | G2 | 0.921 | 0.221 | 0.057 |
| spline-draw3/warpheld | G3 | 0.950 | 0.772 | 0.267 |
| spline-draw4/warpheld | G1 | 0.000 | 0.000 | 0.000 |
| spline-draw4/warpheld | G2 | 0.908 | 0.216 | 0.056 |
| spline-draw4/warpheld | G3 | 0.946 | 0.785 | 0.271 |

#### 2.19.2 G3's members: superiority over the other member (regression binding) (`appendix_blocks/block_g3_members.csv`)

| audit | algorithm | pre | post |
| --- | --- | --- | --- |
| spline/warpheld | BOBYQA | 0.513 | 0.687 |
| spline/warpheld | Dual Annealing | 0.487 | 0.313 |
| spline-alpha0/warpheld | BOBYQA | 0.513 | 0.719 |
| spline-alpha0/warpheld | Dual Annealing | 0.487 | 0.281 |
| spline-draw1/warpheld | BOBYQA | 0.513 | 0.681 |
| spline-draw1/warpheld | Dual Annealing | 0.487 | 0.319 |
| spline-draw2/warpheld | BOBYQA | 0.513 | 0.679 |
| spline-draw2/warpheld | Dual Annealing | 0.487 | 0.321 |
| spline-draw3/warpheld | BOBYQA | 0.513 | 0.683 |
| spline-draw3/warpheld | Dual Annealing | 0.487 | 0.317 |
| spline-draw4/warpheld | BOBYQA | 0.513 | 0.677 |
| spline-draw4/warpheld | Dual Annealing | 0.487 | 0.323 |

#### 2.19.3 G1's runs that differ between the views (`appendix_blocks/block_g1_identity.csv`)

| binding | block | algorithm | runs | best_y differs | evaluations differ | restarts differ | any differs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regression | spline-alpha0/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-alpha0/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw1/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw1/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw2/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw2/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw3/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw3/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw4/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| regression | spline-draw4/warpheld | Adaptive Nelder-Mead | 7440 | 0 | 0 | 0 | 0 |
| selection | spline-alpha0/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| selection | spline-alpha0/warpheld | Adaptive Nelder-Mead | 7440 | 25 | 0 | 0 | 25 |
| selection | spline-draw1/warpheld | BIPOP-CMA-ES | 7440 | 6 | 0 | 1 | 7 |
| selection | spline-draw1/warpheld | Adaptive Nelder-Mead | 7440 | 67 | 0 | 3 | 70 |
| selection | spline-draw2/warpheld | BIPOP-CMA-ES | 7440 | 3 | 0 | 0 | 3 |
| selection | spline-draw2/warpheld | Adaptive Nelder-Mead | 7440 | 83 | 0 | 2 | 84 |
| selection | spline-draw3/warpheld | BIPOP-CMA-ES | 7440 | 12 | 0 | 0 | 12 |
| selection | spline-draw3/warpheld | Adaptive Nelder-Mead | 7440 | 71 | 0 | 5 | 75 |
| selection | spline-draw4/warpheld | BIPOP-CMA-ES | 7440 | 5 | 0 | 0 | 5 |
| selection | spline-draw4/warpheld | Adaptive Nelder-Mead | 7440 | 82 | 0 | 1 | 83 |
| sbs | spline-alpha0/warpheld | BIPOP-CMA-ES | 7440 | 0 | 0 | 0 | 0 |
| sbs | spline-alpha0/warpheld | Adaptive Nelder-Mead | 7440 | 23 | 0 | 1 | 23 |
| sbs | spline-draw1/warpheld | BIPOP-CMA-ES | 7440 | 13 | 0 | 1 | 14 |
| sbs | spline-draw1/warpheld | Adaptive Nelder-Mead | 7440 | 69 | 0 | 2 | 71 |
| sbs | spline-draw2/warpheld | BIPOP-CMA-ES | 7440 | 5 | 0 | 1 | 5 |
| sbs | spline-draw2/warpheld | Adaptive Nelder-Mead | 7440 | 80 | 0 | 5 | 82 |
| sbs | spline-draw3/warpheld | BIPOP-CMA-ES | 7440 | 16 | 0 | 1 | 16 |
| sbs | spline-draw3/warpheld | Adaptive Nelder-Mead | 7440 | 69 | 0 | 5 | 72 |
| sbs | spline-draw4/warpheld | BIPOP-CMA-ES | 7440 | 1 | 0 | 0 | 1 |
| sbs | spline-draw4/warpheld | Adaptive Nelder-Mead | 7440 | 75 | 0 | 5 | 79 |

### 2.20 Levels on each further block

The untransformed block is the same for every block.

#### 2.20.1 ELA regression, R² (`appendix_blocks/block_regression_levels.csv`)

| predictors | group | feature_variant | training | untransformed | spline | spline-alpha0 | spline-draw1 | spline-draw2 | spline-draw3 | spline-draw4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | G1 | value | plain | 0.852 | 0.233 | 0.852 | -0.038 | 0.129 | 0.055 | 0.041 |
| all | G1 | value | augmented | 0.746 | 0.610 | 0.746 | 0.545 | 0.598 | 0.594 | 0.547 |
| all | G1 | rank | plain | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 |
| all | G1 | rank | augmented | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 |
| all | G1 | concat | plain | 0.849 | 0.255 | 0.849 | -0.049 | 0.156 | 0.045 | 0.062 |
| all | G1 | concat | augmented | 0.764 | 0.691 | 0.764 | 0.660 | 0.687 | 0.698 | 0.663 |
| all | G2 | value | plain | 0.770 | 0.196 | 0.755 | 0.089 | 0.068 | 0.111 | 0.105 |
| all | G2 | value | augmented | 0.684 | 0.545 | 0.673 | 0.485 | 0.527 | 0.538 | 0.488 |
| all | G2 | rank | plain | 0.621 | 0.607 | 0.603 | 0.629 | 0.622 | 0.619 | 0.623 |
| all | G2 | rank | augmented | 0.632 | 0.599 | 0.620 | 0.626 | 0.633 | 0.623 | 0.629 |
| all | G2 | concat | plain | 0.759 | 0.184 | 0.758 | 0.093 | 0.056 | 0.109 | 0.090 |
| all | G2 | concat | augmented | 0.703 | 0.632 | 0.693 | 0.602 | 0.627 | 0.627 | 0.599 |
| all | G3 | value | plain | 0.784 | -0.798 | -0.723 | -0.769 | -0.874 | -0.735 | -0.736 |
| all | G3 | value | augmented | 0.710 | 0.273 | -0.432 | 0.218 | 0.374 | 0.336 | 0.261 |
| all | G3 | rank | plain | 0.635 | -0.949 | -0.868 | -0.943 | -0.924 | -0.954 | -0.902 |
| all | G3 | rank | augmented | 0.328 | 0.182 | -0.014 | 0.191 | 0.189 | 0.167 | 0.184 |
| all | G3 | concat | plain | 0.782 | -0.804 | -0.737 | -0.759 | -0.878 | -0.699 | -0.766 |
| all | G3 | concat | augmented | 0.718 | 0.335 | -0.512 | 0.280 | 0.393 | 0.337 | 0.317 |
| invariant | G1 | value | plain | 0.452 | 0.452 | 0.452 | 0.452 | 0.452 | 0.452 | 0.452 |

#### 2.20.2 Deep-ELA regression, R² (`appendix_blocks/block_regression_levels_deep-ela.csv`)

| predictors | group | feature_variant | training | untransformed | spline | spline-alpha0 | spline-draw1 | spline-draw2 | spline-draw3 | spline-draw4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | G1 | value | plain | 0.845 | 0.131 | 0.845 | -0.185 | 0.091 | -0.093 | -0.183 |
| all | G1 | value | augmented | 0.817 | 0.450 | 0.817 | 0.400 | 0.491 | 0.435 | 0.397 |
| all | G1 | rank | plain | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 |
| all | G1 | rank | augmented | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 |
| all | G1 | concat | plain | 0.834 | 0.135 | 0.834 | -0.091 | 0.123 | 0.010 | -0.059 |
| all | G1 | concat | augmented | 0.792 | 0.639 | 0.792 | 0.632 | 0.646 | 0.633 | 0.596 |
| all | G2 | value | plain | 0.794 | -0.116 | 0.791 | -0.224 | -0.211 | -0.222 | -0.263 |
| all | G2 | value | augmented | 0.762 | 0.426 | 0.752 | 0.450 | 0.519 | 0.422 | 0.408 |
| all | G2 | rank | plain | 0.584 | 0.570 | 0.575 | 0.585 | 0.590 | 0.565 | 0.583 |
| all | G2 | rank | augmented | 0.592 | 0.584 | 0.581 | 0.595 | 0.604 | 0.575 | 0.582 |
| all | G2 | concat | plain | 0.798 | -0.023 | 0.794 | -0.083 | -0.085 | -0.054 | -0.059 |
| all | G2 | concat | augmented | 0.737 | 0.617 | 0.721 | 0.624 | 0.627 | 0.599 | 0.604 |
| all | G3 | value | plain | 0.825 | -0.716 | -0.635 | -0.707 | -0.892 | -0.782 | -0.576 |
| all | G3 | value | augmented | 0.727 | 0.333 | -0.243 | 0.265 | 0.287 | 0.184 | 0.255 |
| all | G3 | rank | plain | 0.620 | -0.920 | -0.788 | -0.905 | -0.871 | -0.887 | -0.850 |
| all | G3 | rank | augmented | 0.328 | 0.186 | 0.044 | 0.202 | 0.202 | 0.185 | 0.196 |
| all | G3 | concat | plain | 0.820 | -0.664 | -0.681 | -0.650 | -0.826 | -0.725 | -0.593 |
| all | G3 | concat | augmented | 0.703 | 0.409 | -0.311 | 0.352 | 0.365 | 0.307 | 0.385 |

#### 2.20.3 ELA selection, Θ (`appendix_blocks/block_selection_levels.csv`)

| feature_variant | training | untransformed | spline | spline-alpha0 | spline-draw1 | spline-draw2 | spline-draw3 | spline-draw4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| value | plain | 0.704 | 0.598 | 0.704 | 0.574 | 0.575 | 0.589 | 0.564 |
| value | augmented | 0.681 | 0.650 | 0.681 | 0.640 | 0.647 | 0.652 | 0.640 |
| rank | plain | 0.646 | 0.620 | 0.645 | 0.618 | 0.621 | 0.621 | 0.620 |
| rank | augmented | 0.650 | 0.647 | 0.650 | 0.647 | 0.648 | 0.649 | 0.648 |
| concat | plain | 0.704 | 0.600 | 0.704 | 0.577 | 0.586 | 0.600 | 0.570 |
| concat | augmented | 0.683 | 0.654 | 0.683 | 0.652 | 0.656 | 0.662 | 0.656 |

#### 2.20.4 Deep-ELA selection, Θ (`appendix_blocks/block_selection_levels_deep-ela.csv`)

| feature_variant | training | untransformed | spline | spline-alpha0 | spline-draw1 | spline-draw2 | spline-draw3 | spline-draw4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| value | plain | 0.707 | 0.560 | 0.706 | 0.528 | 0.548 | 0.541 | 0.504 |
| value | augmented | 0.703 | 0.609 | 0.703 | 0.608 | 0.625 | 0.612 | 0.601 |
| rank | plain | 0.643 | 0.612 | 0.643 | 0.610 | 0.612 | 0.612 | 0.612 |
| rank | augmented | 0.645 | 0.644 | 0.645 | 0.642 | 0.644 | 0.644 | 0.644 |
| concat | plain | 0.708 | 0.566 | 0.707 | 0.545 | 0.564 | 0.555 | 0.516 |
| concat | augmented | 0.695 | 0.654 | 0.695 | 0.656 | 0.654 | 0.656 | 0.646 |

### 2.21 Spread of the levels over the five draws

Over `spline` and `spline-draw1` … `spline-draw4`, on the transformed view.

#### 2.21.1 ELA regression, R² (`appendix_blocks/block_regression_draw_spread.csv`, `metric = r2_test_post`)

| predictors | group | feature_variant | training | min | median | max |
| --- | --- | --- | --- | --- | --- | --- |
| all | G1 | value | plain | -0.038 | 0.055 | 0.233 |
| all | G1 | value | augmented | 0.545 | 0.594 | 0.610 |
| all | G1 | rank | plain | 0.661 | 0.661 | 0.661 |
| all | G1 | rank | augmented | 0.655 | 0.655 | 0.655 |
| all | G1 | concat | plain | -0.049 | 0.062 | 0.255 |
| all | G1 | concat | augmented | 0.660 | 0.687 | 0.698 |
| all | G2 | value | plain | 0.068 | 0.105 | 0.196 |
| all | G2 | value | augmented | 0.485 | 0.527 | 0.545 |
| all | G2 | rank | plain | 0.607 | 0.622 | 0.629 |
| all | G2 | rank | augmented | 0.599 | 0.626 | 0.633 |
| all | G2 | concat | plain | 0.056 | 0.093 | 0.184 |
| all | G2 | concat | augmented | 0.599 | 0.627 | 0.632 |
| all | G3 | value | plain | -0.874 | -0.769 | -0.735 |
| all | G3 | value | augmented | 0.218 | 0.273 | 0.374 |
| all | G3 | rank | plain | -0.954 | -0.943 | -0.902 |
| all | G3 | rank | augmented | 0.167 | 0.184 | 0.191 |
| all | G3 | concat | plain | -0.878 | -0.766 | -0.699 |
| all | G3 | concat | augmented | 0.280 | 0.335 | 0.393 |
| invariant | G1 | value | plain | 0.452 | 0.452 | 0.452 |

#### 2.21.2 Deep-ELA regression, R² (`appendix_blocks/block_regression_draw_spread_deep-ela.csv`, `metric = r2_test_post`)

| predictors | group | feature_variant | training | min | median | max |
| --- | --- | --- | --- | --- | --- | --- |
| all | G1 | value | plain | -0.185 | -0.093 | 0.131 |
| all | G1 | value | augmented | 0.397 | 0.435 | 0.491 |
| all | G1 | rank | plain | 0.615 | 0.615 | 0.615 |
| all | G1 | rank | augmented | 0.612 | 0.612 | 0.612 |
| all | G1 | concat | plain | -0.091 | 0.010 | 0.135 |
| all | G1 | concat | augmented | 0.596 | 0.633 | 0.646 |
| all | G2 | value | plain | -0.263 | -0.222 | -0.116 |
| all | G2 | value | augmented | 0.408 | 0.426 | 0.519 |
| all | G2 | rank | plain | 0.565 | 0.583 | 0.590 |
| all | G2 | rank | augmented | 0.575 | 0.584 | 0.604 |
| all | G2 | concat | plain | -0.085 | -0.059 | -0.023 |
| all | G2 | concat | augmented | 0.599 | 0.617 | 0.627 |
| all | G3 | value | plain | -0.892 | -0.716 | -0.576 |
| all | G3 | value | augmented | 0.184 | 0.265 | 0.333 |
| all | G3 | rank | plain | -0.920 | -0.887 | -0.850 |
| all | G3 | rank | augmented | 0.185 | 0.196 | 0.202 |
| all | G3 | concat | plain | -0.826 | -0.664 | -0.593 |
| all | G3 | concat | augmented | 0.307 | 0.365 | 0.409 |

#### 2.21.3 ELA selection, Θ (`appendix_blocks/block_selection_draw_spread.csv`, `transform = post`)

| feature_variant | training | min | median | max |
| --- | --- | --- | --- | --- |
| value | plain | 0.564 | 0.575 | 0.598 |
| value | augmented | 0.640 | 0.647 | 0.652 |
| rank | plain | 0.618 | 0.620 | 0.621 |
| rank | augmented | 0.647 | 0.648 | 0.649 |
| concat | plain | 0.570 | 0.586 | 0.600 |
| concat | augmented | 0.652 | 0.656 | 0.662 |

#### 2.21.4 Deep-ELA selection, Θ (`appendix_blocks/block_selection_draw_spread_deep-ela.csv`, `transform = post`)

| feature_variant | training | min | median | max |
| --- | --- | --- | --- | --- |
| value | plain | 0.504 | 0.541 | 0.560 |
| value | augmented | 0.601 | 0.609 | 0.625 |
| rank | plain | 0.610 | 0.612 | 0.612 |
| rank | augmented | 0.642 | 0.644 | 0.644 |
| concat | plain | 0.516 | 0.555 | 0.566 |
| concat | augmented | 0.646 | 0.654 | 0.656 |

### 2.22 plain-4800: levels on each family

`plain-4800` fits on instances 11–210, all untransformed, at the augmented condition's
update budget; `plain` and `augmented` are Experiment 4's and 5's cells. The untransformed
block is the same for every family.

#### 2.22.1 ELA regression, R² (`appendix_doubled/plain4800_regression_levels.csv`)

| group | feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | value | plain | 0.852 | 0.233 | 0.311 | 0.150 | 0.297 | 0.191 |
| G1 | value | plain-4800 | 0.895 | 0.158 | 0.307 | 0.074 | 0.302 | 0.159 |
| G1 | value | augmented | 0.746 | 0.610 | 0.609 | 0.594 | 0.547 | 0.590 |
| G1 | rank | plain | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 | 0.661 |
| G1 | rank | plain-4800 | 0.738 | 0.738 | 0.738 | 0.738 | 0.738 | 0.738 |
| G1 | rank | augmented | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 | 0.655 |
| G1 | concat | plain | 0.849 | 0.255 | 0.341 | 0.154 | 0.311 | 0.213 |
| G1 | concat | plain-4800 | 0.885 | 0.180 | 0.368 | 0.117 | 0.325 | 0.190 |
| G1 | concat | augmented | 0.764 | 0.691 | 0.690 | 0.702 | 0.647 | 0.696 |
| G2 | value | plain | 0.770 | 0.196 | 0.369 | 0.266 | 0.179 | 0.258 |
| G2 | value | plain-4800 | 0.829 | 0.125 | 0.413 | 0.236 | 0.190 | 0.264 |
| G2 | value | augmented | 0.684 | 0.545 | 0.541 | 0.561 | 0.429 | 0.554 |
| G2 | rank | plain | 0.621 | 0.607 | 0.609 | 0.605 | 0.606 | 0.624 |
| G2 | rank | plain-4800 | 0.735 | 0.706 | 0.719 | 0.708 | 0.706 | 0.720 |
| G2 | rank | augmented | 0.632 | 0.599 | 0.623 | 0.609 | 0.612 | 0.625 |
| G2 | concat | plain | 0.759 | 0.184 | 0.388 | 0.300 | 0.230 | 0.240 |
| G2 | concat | plain-4800 | 0.830 | 0.140 | 0.383 | 0.205 | 0.216 | 0.246 |
| G2 | concat | augmented | 0.703 | 0.632 | 0.628 | 0.641 | 0.526 | 0.643 |
| G3 | value | plain | 0.784 | -0.798 | -0.841 | -1.004 | -0.738 | -0.723 |
| G3 | value | plain-4800 | 0.839 | -0.765 | -0.821 | -0.986 | -0.660 | -0.645 |
| G3 | value | augmented | 0.710 | 0.273 | 0.035 | 0.353 | -0.063 | 0.239 |
| G3 | rank | plain | 0.635 | -0.949 | -0.945 | -1.037 | -1.313 | -0.923 |
| G3 | rank | plain-4800 | 0.729 | -0.907 | -0.893 | -0.992 | -1.230 | -0.845 |
| G3 | rank | augmented | 0.328 | 0.182 | 0.183 | 0.164 | -0.031 | 0.143 |
| G3 | concat | plain | 0.782 | -0.804 | -0.857 | -0.995 | -0.772 | -0.705 |
| G3 | concat | plain-4800 | 0.841 | -0.794 | -0.865 | -1.010 | -0.703 | -0.647 |
| G3 | concat | augmented | 0.718 | 0.335 | -0.011 | 0.366 | -0.070 | 0.224 |

#### 2.22.2 ELA selection, Θ (`appendix_doubled/plain4800_selection_levels.csv`)

| feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| value | plain | 0.704 | 0.598 | 0.589 | 0.593 | 0.554 | 0.602 |
| value | plain-4800 | 0.720 | 0.596 | 0.592 | 0.592 | 0.560 | 0.610 |
| value | augmented | 0.681 | 0.650 | 0.647 | 0.647 | 0.625 | 0.647 |
| rank | plain | 0.646 | 0.620 | 0.621 | 0.616 | 0.613 | 0.624 |
| rank | plain-4800 | 0.666 | 0.634 | 0.635 | 0.629 | 0.630 | 0.639 |
| rank | augmented | 0.650 | 0.647 | 0.649 | 0.644 | 0.644 | 0.648 |
| concat | plain | 0.704 | 0.600 | 0.604 | 0.598 | 0.576 | 0.615 |
| concat | plain-4800 | 0.719 | 0.602 | 0.599 | 0.594 | 0.564 | 0.613 |
| concat | augmented | 0.683 | 0.654 | 0.657 | 0.658 | 0.641 | 0.662 |

#### 2.22.3 Deep-ELA regression, R² (`appendix_doubled/plain4800_regression_levels_deep-ela.csv`)

| group | feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G1 | value | plain | 0.845 | 0.131 | 0.172 | -0.081 | 0.160 | 0.117 |
| G1 | value | plain-4800 | 0.875 | 0.115 | 0.125 | -0.087 | 0.158 | 0.077 |
| G1 | value | augmented | 0.817 | 0.450 | 0.545 | 0.315 | 0.458 | 0.446 |
| G1 | rank | plain | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 | 0.615 |
| G1 | rank | plain-4800 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |
| G1 | rank | augmented | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 |
| G1 | concat | plain | 0.834 | 0.135 | 0.251 | 0.004 | 0.278 | 0.198 |
| G1 | concat | plain-4800 | 0.879 | 0.168 | 0.231 | 0.066 | 0.259 | 0.146 |
| G1 | concat | augmented | 0.792 | 0.639 | 0.642 | 0.557 | 0.614 | 0.663 |
| G2 | value | plain | 0.794 | -0.116 | 0.165 | -0.141 | 0.087 | 0.014 |
| G2 | value | plain-4800 | 0.825 | -0.146 | 0.152 | -0.127 | 0.040 | 0.037 |
| G2 | value | augmented | 0.762 | 0.426 | 0.503 | 0.357 | 0.311 | 0.501 |
| G2 | rank | plain | 0.584 | 0.570 | 0.580 | 0.553 | 0.555 | 0.585 |
| G2 | rank | plain-4800 | 0.664 | 0.650 | 0.652 | 0.632 | 0.636 | 0.652 |
| G2 | rank | augmented | 0.592 | 0.584 | 0.584 | 0.566 | 0.567 | 0.587 |
| G2 | concat | plain | 0.798 | -0.023 | 0.269 | 0.109 | 0.131 | 0.148 |
| G2 | concat | plain-4800 | 0.828 | 0.012 | 0.299 | 0.164 | 0.103 | 0.182 |
| G2 | concat | augmented | 0.737 | 0.617 | 0.616 | 0.584 | 0.516 | 0.623 |
| G3 | value | plain | 0.825 | -0.716 | -0.837 | -0.961 | -0.808 | -0.604 |
| G3 | value | plain-4800 | 0.843 | -0.872 | -0.899 | -1.221 | -0.849 | -0.738 |
| G3 | value | augmented | 0.727 | 0.333 | 0.105 | 0.218 | 0.085 | 0.071 |
| G3 | rank | plain | 0.620 | -0.920 | -0.906 | -0.996 | -1.206 | -0.859 |
| G3 | rank | plain-4800 | 0.686 | -0.889 | -0.881 | -0.967 | -1.207 | -0.817 |
| G3 | rank | augmented | 0.328 | 0.186 | 0.191 | 0.184 | -0.034 | 0.154 |
| G3 | concat | plain | 0.820 | -0.664 | -0.832 | -0.810 | -0.765 | -0.596 |
| G3 | concat | plain-4800 | 0.846 | -0.727 | -0.889 | -0.931 | -0.795 | -0.639 |
| G3 | concat | augmented | 0.703 | 0.409 | 0.166 | 0.338 | 0.116 | 0.159 |

#### 2.22.4 Deep-ELA selection, Θ (`appendix_doubled/plain4800_selection_levels_deep-ela.csv`)

| feature_variant | training | untransformed | spline | knots-sparse | knots-dense | power | piecewise-linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| value | plain | 0.707 | 0.560 | 0.551 | 0.533 | 0.522 | 0.565 |
| value | plain-4800 | 0.715 | 0.567 | 0.552 | 0.530 | 0.522 | 0.573 |
| value | augmented | 0.703 | 0.609 | 0.623 | 0.587 | 0.585 | 0.630 |
| rank | plain | 0.643 | 0.612 | 0.613 | 0.607 | 0.607 | 0.620 |
| rank | plain-4800 | 0.657 | 0.619 | 0.621 | 0.615 | 0.616 | 0.629 |
| rank | augmented | 0.645 | 0.644 | 0.645 | 0.640 | 0.638 | 0.643 |
| concat | plain | 0.708 | 0.566 | 0.573 | 0.552 | 0.544 | 0.580 |
| concat | plain-4800 | 0.715 | 0.574 | 0.566 | 0.556 | 0.534 | 0.592 |
| concat | augmented | 0.695 | 0.654 | 0.649 | 0.633 | 0.635 | 0.663 |

### 2.23 plain-4800: ranges and tests

Ranges reduced as the paper's levels are; Mann–Whitney `augmented` against `plain-4800`,
Holm within a feature set and range.

#### 2.23.1 ELA regression, R² (`appendix_doubled/plain4800_regression_scopes.csv`)

| group | feature_variant | training | untransformed test | unseen-family mean | spline only |
| --- | --- | --- | --- | --- | --- |
| G1 | value | plain | 0.852 | 0.245 | 0.233 |
| G1 | value | plain-4800 | 0.895 | 0.229 | 0.158 |
| G1 | value | augmented | 0.746 | 0.582 | 0.610 |
| G1 | rank | plain | 0.661 | 0.661 | 0.661 |
| G1 | rank | plain-4800 | 0.738 | 0.738 | 0.738 |
| G1 | rank | augmented | 0.655 | 0.655 | 0.655 |
| G1 | concat | plain | 0.849 | 0.262 | 0.255 |
| G1 | concat | plain-4800 | 0.885 | 0.245 | 0.180 |
| G1 | concat | augmented | 0.764 | 0.686 | 0.691 |
| G2 | value | plain | 0.770 | 0.252 | 0.196 |
| G2 | value | plain-4800 | 0.829 | 0.266 | 0.125 |
| G2 | value | augmented | 0.684 | 0.525 | 0.545 |
| G2 | rank | plain | 0.621 | 0.613 | 0.607 |
| G2 | rank | plain-4800 | 0.735 | 0.713 | 0.706 |
| G2 | rank | augmented | 0.632 | 0.616 | 0.599 |
| G2 | concat | plain | 0.759 | 0.280 | 0.184 |
| G2 | concat | plain-4800 | 0.830 | 0.269 | 0.140 |
| G2 | concat | augmented | 0.703 | 0.613 | 0.632 |
| G3 | value | plain | 0.784 | -0.814 | -0.798 |
| G3 | value | plain-4800 | 0.839 | -0.775 | -0.765 |
| G3 | value | augmented | 0.710 | 0.157 | 0.273 |
| G3 | rank | plain | 0.635 | -1.055 | -0.949 |
| G3 | rank | plain-4800 | 0.729 | -0.987 | -0.907 |
| G3 | rank | augmented | 0.328 | 0.115 | 0.182 |
| G3 | concat | plain | 0.782 | -0.832 | -0.804 |
| G3 | concat | plain-4800 | 0.841 | -0.799 | -0.794 |
| G3 | concat | augmented | 0.718 | 0.133 | 0.335 |

#### 2.23.2 ELA regression tests (`appendix_doubled/plain4800_regression_tests.csv`)

| scope | group | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| untransformed test | G1 | value | plain-4800 | -0.149 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G1 | rank | plain-4800 | -0.083 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G1 | concat | plain-4800 | -0.121 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G2 | value | plain-4800 | -0.145 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G2 | rank | plain-4800 | -0.102 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G2 | concat | plain-4800 | -0.127 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G3 | value | plain-4800 | -0.129 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G3 | rank | plain-4800 | -0.401 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G3 | concat | plain-4800 | -0.123 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G1 | value | augmented | 0.353 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G1 | rank | plain-4800 | -0.083 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G1 | concat | augmented | 0.441 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G2 | value | augmented | 0.259 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G2 | rank | plain-4800 | -0.097 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G2 | concat | augmented | 0.344 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G3 | value | augmented | 0.931 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G3 | rank | augmented | 1.102 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G3 | concat | augmented | 0.932 | 2.84e-06 | 2.55e-05 | True |
| spline only | G1 | value | augmented | 0.451 | 2.84e-06 | 2.55e-05 | True |
| spline only | G1 | rank | plain-4800 | -0.083 | 2.84e-06 | 2.55e-05 | True |
| spline only | G1 | concat | augmented | 0.512 | 2.84e-06 | 2.55e-05 | True |
| spline only | G2 | value | augmented | 0.420 | 2.84e-06 | 2.55e-05 | True |
| spline only | G2 | rank | plain-4800 | -0.107 | 2.84e-06 | 2.55e-05 | True |
| spline only | G2 | concat | augmented | 0.492 | 2.84e-06 | 2.55e-05 | True |
| spline only | G3 | value | augmented | 1.038 | 2.84e-06 | 2.55e-05 | True |
| spline only | G3 | rank | augmented | 1.088 | 2.84e-06 | 2.55e-05 | True |
| spline only | G3 | concat | augmented | 1.129 | 2.84e-06 | 2.55e-05 | True |

#### 2.23.3 ELA selection, Θ (`appendix_doubled/plain4800_selection_scopes.csv`)

| feature_variant | training | untransformed test | unseen-family mean | spline only |
| --- | --- | --- | --- | --- |
| value | plain | 0.704 | 0.583 | 0.598 |
| value | plain-4800 | 0.720 | 0.589 | 0.596 |
| value | augmented | 0.681 | 0.642 | 0.650 |
| rank | plain | 0.646 | 0.619 | 0.620 |
| rank | plain-4800 | 0.666 | 0.633 | 0.634 |
| rank | augmented | 0.650 | 0.646 | 0.647 |
| concat | plain | 0.704 | 0.598 | 0.600 |
| concat | plain-4800 | 0.719 | 0.594 | 0.602 |
| concat | augmented | 0.683 | 0.652 | 0.654 |

#### 2.23.4 ELA selection tests (`appendix_doubled/plain4800_selection_tests.csv`)

| scope | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| untransformed test | value | plain-4800 | -0.039 | 2.84e-06 | 8.51e-06 | True |
| untransformed test | rank | plain-4800 | -0.016 | 2.84e-06 | 8.51e-06 | True |
| untransformed test | concat | plain-4800 | -0.036 | 2.84e-06 | 8.51e-06 | True |
| unseen-family mean | value | augmented | 0.053 | 2.84e-06 | 8.51e-06 | True |
| unseen-family mean | rank | augmented | 0.013 | 3.4e-05 | 3.4e-05 | True |
| unseen-family mean | concat | augmented | 0.058 | 2.84e-06 | 8.51e-06 | True |
| spline only | value | augmented | 0.054 | 2.84e-06 | 8.51e-06 | True |
| spline only | rank | augmented | 0.014 | 3.4e-05 | 3.4e-05 | True |
| spline only | concat | augmented | 0.051 | 2.84e-06 | 8.51e-06 | True |

#### 2.23.5 Deep-ELA regression, R² (`appendix_doubled/plain4800_regression_scopes_deep-ela.csv`)

| group | feature_variant | training | untransformed test | unseen-family mean | spline only |
| --- | --- | --- | --- | --- | --- |
| G1 | value | plain | 0.845 | 0.091 | 0.131 |
| G1 | value | plain-4800 | 0.875 | 0.062 | 0.115 |
| G1 | value | augmented | 0.817 | 0.441 | 0.450 |
| G1 | rank | plain | 0.615 | 0.615 | 0.615 |
| G1 | rank | plain-4800 | 0.667 | 0.667 | 0.667 |
| G1 | rank | augmented | 0.612 | 0.612 | 0.612 |
| G1 | concat | plain | 0.834 | 0.187 | 0.135 |
| G1 | concat | plain-4800 | 0.879 | 0.178 | 0.168 |
| G1 | concat | augmented | 0.792 | 0.613 | 0.639 |
| G2 | value | plain | 0.794 | 0.036 | -0.116 |
| G2 | value | plain-4800 | 0.825 | 0.022 | -0.146 |
| G2 | value | augmented | 0.762 | 0.421 | 0.426 |
| G2 | rank | plain | 0.584 | 0.568 | 0.570 |
| G2 | rank | plain-4800 | 0.664 | 0.642 | 0.650 |
| G2 | rank | augmented | 0.592 | 0.578 | 0.584 |
| G2 | concat | plain | 0.798 | 0.157 | -0.023 |
| G2 | concat | plain-4800 | 0.828 | 0.184 | 0.012 |
| G2 | concat | augmented | 0.737 | 0.585 | 0.617 |
| G3 | value | plain | 0.825 | -0.800 | -0.716 |
| G3 | value | plain-4800 | 0.843 | -0.923 | -0.872 |
| G3 | value | augmented | 0.727 | 0.123 | 0.333 |
| G3 | rank | plain | 0.620 | -0.992 | -0.920 |
| G3 | rank | plain-4800 | 0.686 | -0.966 | -0.889 |
| G3 | rank | augmented | 0.328 | 0.123 | 0.186 |
| G3 | concat | plain | 0.820 | -0.743 | -0.664 |
| G3 | concat | plain-4800 | 0.846 | -0.811 | -0.727 |
| G3 | concat | augmented | 0.703 | 0.197 | 0.409 |

#### 2.23.6 Deep-ELA regression tests (`appendix_doubled/plain4800_regression_tests_deep-ela.csv`)

| scope | group | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| untransformed test | G1 | value | plain-4800 | -0.058 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G1 | rank | plain-4800 | -0.055 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G1 | concat | plain-4800 | -0.086 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G2 | value | plain-4800 | -0.063 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G2 | rank | plain-4800 | -0.072 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G2 | concat | plain-4800 | -0.090 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G3 | value | plain-4800 | -0.116 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G3 | rank | plain-4800 | -0.358 | 2.84e-06 | 2.55e-05 | True |
| untransformed test | G3 | concat | plain-4800 | -0.143 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G1 | value | augmented | 0.379 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G1 | rank | plain-4800 | -0.055 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G1 | concat | augmented | 0.436 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G2 | value | augmented | 0.399 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G2 | rank | plain-4800 | -0.064 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G2 | concat | augmented | 0.401 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G3 | value | augmented | 1.046 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G3 | rank | augmented | 1.089 | 2.84e-06 | 2.55e-05 | True |
| unseen-family mean | G3 | concat | augmented | 1.008 | 2.84e-06 | 2.55e-05 | True |
| spline only | G1 | value | augmented | 0.335 | 2.84e-06 | 2.55e-05 | True |
| spline only | G1 | rank | plain-4800 | -0.055 | 2.84e-06 | 2.55e-05 | True |
| spline only | G1 | concat | augmented | 0.471 | 2.84e-06 | 2.55e-05 | True |
| spline only | G2 | value | augmented | 0.572 | 2.84e-06 | 2.55e-05 | True |
| spline only | G2 | rank | plain-4800 | -0.066 | 2.84e-06 | 2.55e-05 | True |
| spline only | G2 | concat | augmented | 0.605 | 2.84e-06 | 2.55e-05 | True |
| spline only | G3 | value | augmented | 1.206 | 2.84e-06 | 2.55e-05 | True |
| spline only | G3 | rank | augmented | 1.075 | 2.84e-06 | 2.55e-05 | True |
| spline only | G3 | concat | augmented | 1.136 | 2.84e-06 | 2.55e-05 | True |

#### 2.23.7 Deep-ELA selection, Θ (`appendix_doubled/plain4800_selection_scopes_deep-ela.csv`)

| feature_variant | training | untransformed test | unseen-family mean | spline only |
| --- | --- | --- | --- | --- |
| value | plain | 0.707 | 0.543 | 0.560 |
| value | plain-4800 | 0.715 | 0.544 | 0.567 |
| value | augmented | 0.703 | 0.606 | 0.609 |
| rank | plain | 0.643 | 0.611 | 0.612 |
| rank | plain-4800 | 0.657 | 0.620 | 0.619 |
| rank | augmented | 0.645 | 0.642 | 0.644 |
| concat | plain | 0.708 | 0.561 | 0.566 |
| concat | plain-4800 | 0.715 | 0.564 | 0.574 |
| concat | augmented | 0.695 | 0.645 | 0.654 |

#### 2.23.8 Deep-ELA selection tests (`appendix_doubled/plain4800_selection_tests_deep-ela.csv`)

| scope | feature_variant | higher | difference of medians | p | p (Holm) | significant |
| --- | --- | --- | --- | --- | --- | --- |
| untransformed test | value | plain-4800 | -0.012 | 2.84e-06 | 8.51e-06 | True |
| untransformed test | rank | plain-4800 | -0.012 | 1.13e-05 | 1.13e-05 | True |
| untransformed test | concat | plain-4800 | -0.020 | 2.84e-06 | 8.51e-06 | True |
| unseen-family mean | value | augmented | 0.062 | 2.84e-06 | 8.51e-06 | True |
| unseen-family mean | rank | augmented | 0.022 | 0.0014 | 0.0014 | True |
| unseen-family mean | concat | augmented | 0.082 | 2.84e-06 | 8.51e-06 | True |
| spline only | value | augmented | 0.043 | 2.84e-06 | 8.51e-06 | True |
| spline only | rank | augmented | 0.024 | 0.00104 | 0.00104 | True |
| spline only | concat | augmented | 0.080 | 2.84e-06 | 8.51e-06 | True |

### 2.24 Appendix G: leave-one-function-out levels

The median over the 11 model seeds of the LOFO R² (`lofo_median`); the
upper quartile is `lofo_q3`.

#### 2.24.1 ELA (`appendix_lofo/lofo_regression_levels.csv`)

| feature_variant | training | G1 | G2 | G3 |
| --- | --- | --- | --- | --- |
| value | plain | -0.859 | -0.941 | -0.199 |
| value | augmented | -0.807 | -0.674 | -0.338 |
| rank | plain | -0.702 | -0.650 | -0.321 |
| rank | augmented | -0.633 | -0.634 | -0.374 |
| concat | plain | -0.892 | -0.902 | -0.199 |
| concat | augmented | -0.726 | -0.659 | -0.348 |

#### 2.24.2 Deep-ELA (`appendix_lofo/lofo_regression_levels_deep-ela.csv`)

| feature_variant | training | G1 | G2 | G3 |
| --- | --- | --- | --- | --- |
| value | plain | -0.609 | -0.691 | -0.202 |
| value | augmented | -0.510 | -0.580 | -0.376 |
| rank | plain | -0.694 | -0.650 | -0.381 |
| rank | augmented | -0.718 | -0.669 | -0.490 |
| concat | plain | -0.571 | -0.629 | -0.163 |
| concat | augmented | -0.578 | -0.661 | -0.496 |

#### 2.24.3 Over the 18 conditions of each file

| Statistic | Value |
| --- | --- |
| ELA: smallest median | -0.941 |
| ELA: largest median | -0.199 |
| ELA: largest upper quartile | -0.151 |
| Deep-ELA: smallest median | -0.718 |
| Deep-ELA: largest median | -0.163 |
| Deep-ELA: largest upper quartile | -0.103 |

### 2.25 Single values the main text quotes from the appendices

ELA. Draws: `appendix_blocks/block_regression_levels.csv` (`predictors = all`,
`metric = r2_test_post`); shift: `appendix_blocks/block_target_shift.csv`;
plain-4800: `appendix_doubled/plain4800_regression_scopes.csv`; untransformed
levels: `experiment4/group_regression_levels.csv` (`scope = unseen-family mean`,
`transform = pre`, column `level`). A difference "plain − augmented" is the
plain cell's level minus the augmented cell's, each at full precision, rounded
once.

#### 2.25.1 Values

| Statistic | Value |
| --- | --- |
| G1, value, plain, R² on spline-draw1 (smallest of draws 1–4) | -0.038 |
| G1, value, plain, R² on spline-draw2 (largest of draws 1–4) | 0.129 |
| G3, standardized shift of J on spline-alpha0 | 0.654 |
| G1, rank, plain-4800, R² over the unseen families | 0.738 |
| G2, rank, plain-4800, R² over the unseen families | 0.713 |
| G1, value, plain-4800, R² over the unseen families | 0.229 |
| G2, value, plain-4800, R² over the unseen families | 0.266 |
| G1, concat, plain-4800, R² over the unseen families | 0.245 |
| G2, concat, plain-4800, R² over the unseen families | 0.269 |
| G3, rank, plain, R² on the untransformed test block | 0.635 |
| G3, rank, augmented, R² on the untransformed test block | 0.328 |
| G1, value: plain − augmented, R² on the untransformed block | 0.106 |
| G1, rank: plain − augmented, R² on the untransformed block | 0.006 |
| G1, concat: plain − augmented, R² on the untransformed block | 0.086 |
| G2, value: plain − augmented, R² on the untransformed block | 0.086 |
| G2, rank: plain − augmented, R² on the untransformed block | -0.011 |
| G2, concat: plain − augmented, R² on the untransformed block | 0.056 |
| G3, value: plain − augmented, R² on the untransformed block | 0.074 |
| G3, rank: plain − augmented, R² on the untransformed block | 0.307 |
| G3, concat: plain − augmented, R² on the untransformed block | 0.064 |

## 3. How the numbers are checked

`scripts/check_document_numbers.py` registers every row of §2, and the computed
settings of `docs/SETTINGS.md`, with the file, rows and column each value is
read from.

* `just check-numbers --root <output root>` recomputes every value from the
  files under that root and compares it with this document; a row without a
  registered source fails too. The LOFO, bootstrap, checks, further blocks
  and plain-4800 appendices (2.11 on) write to output roots of their own, read with
  `--also <their root>`.
* `just check-numbers --structure` (run by the tests) checks only that each
  registered row is present, without outputs.
* `just check-numbers --root <output root> --tex <paper>` lists the numbers of a
  LaTeX source that no registered value rounds to.
