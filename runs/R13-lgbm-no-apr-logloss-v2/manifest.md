# Run manifest: R13-lgbm-no-apr-logloss-v2

## Run

- Run id: R13-lgbm-no-apr-logloss-v2
- Date: 2026-11-18
- Family and feature set: lgbm, no_apr

## Question
Does tuning on log loss, with a calibrator fit on about three times the defaults, give a PD
that is better calibrated than v1.1 without losing ranking?

## Status
Registered after the 2024 test window was scored for v1 and v1.1. Development uses data
through 2022-12 only. The 2023 gap year (scored before for v1 and v1.1, never used to choose
anything) is scored once for v2. The 2024 window is not loaded. Results are published
whatever they show.

## Protocol
- Model: monotone LightGBM, no_apr feature set, same search space as R03/R09.
- Search: TPE on CV mean log loss (expanding-quarter-4 folds on 202101-202206), seeds 0-2,
  min_delta 0.0002, window 60, hard cap 400. Best over seeds.
- Final model: fit on 202101-202206 with the chosen parameters.
- Scale check: each fold model and the final model score validation; a fold's out-of-fold
  scores are pooled only if the OLS slope of final logit on fold logit is in [0.9, 1.1] and
  the mean logit shift is at most 0.1.
- Calibrator: candidates log-odds shift, log-odds Platt, isotonic (simplest first). Leave one
  quarter out over the pooled out-of-fold quarters and the two validation quarters; simplest
  method within one paired standard error of the lowest pooled Brier. Refit on all pooled rows.
- Cutoff: break-even PD 0.10 / 0.70.
- Evaluation: 2023 gap year, once, against v1 and v1.1. Brier skill against the training
  default rate, with bootstrap intervals; paired bootstrap of v2 minus v1.1.

## Decision rule
v2 replaces v1.1 as the recommended model if the paired 95% CI of gap-year Brier (v2 minus
v1.1) is entirely below zero and the mean paired AUC difference is at least -0.01. Otherwise
v1.1 stays and v2 is reported.

## Expected
A larger or more flexible model than v1, less compression at the bottom of the scale, and a
higher gap-year Brier skill than v1.1 (+1.3%).

## Reproducibility

- Data hash: 06e6bfa6fe44cede0463b6c2bf7bd2e3df622cf2f8d3f253bd77655d08f046d3
- Environment lock: 45e16d4c7aba6bbc23981e415fee035b5858f8f325822f14fab395c996d58dc4

## Result

### Search

  - seed 0: best log loss 0.13649 after 81 trials (plateau)
  - seed 1: best log loss 0.13683 after 106 trials (plateau)
  - seed 2: best log loss 0.13661 after 69 trials (plateau)
- Trials in total: 256 (plateau); spread of per-seed best across seeds 0.00034
- Final CV log loss 0.13649 (se 0.00242); CV AUC 0.7056 (se 0.0066)
- Validation AUC of the raw final model: 0.7193
- Parameters: `{"n_estimators": 288, "learning_rate": 0.010329509144562514, "num_leaves": 4, "min_child_samples": 306, "reg_lambda": 0.0035283695755218844, "colsample_bytree": 0.9756116580492795, "subsample": 0.7789824665229373, "subsample_freq": 1}`

### Scale check

| fold | quarter | slope | shift | pass |
|---|---|---|---|---|
| 0 | 2021Q3 | 0.894 | 0.10279 | False |
| 1 | 2021Q4 | 0.9314 | 0.058445 | True |
| 2 | 2022Q1 | 0.98741 | 0.0030456 | True |
| 3 | 2022Q2 | 1.0135 | -0.0062273 | True |

### Calibrator selection (leave one quarter out)

| method | brier | log_loss | ece | diff_vs_best | se_vs_best | eligible |
|---|---|---|---|---|---|---|
| logit_intercept | 0.031017 | 0.135 | 0.0037889 | 0 | 0 | True |
| logit_platt | 0.031023 | 0.13504 | 0.0034367 | 5.5504e-06 | 9.9775e-06 | True |
| isotonic | 0.031099 | 0.13551 | 0.0046326 | 8.2024e-05 | 2.94e-05 | False |

- Blocks: ['2021Q4', '2022Q1', '2022Q2', '2022Q3', '2022Q4']; 16844 rows, 550 defaults
- Lowest pooled Brier: logit_intercept; chosen: logit_intercept
- Break-even cutoff: 0.1429; training-window default rate 0.03210
- Validation cutoff table is in-sample for the calibrator (see result.json)

### Gap year 2023

| model | auc | brier | brier_skill | ece | mean_pd | observed | bottom_pred | bottom_obs | top_pred | top_obs | approval | bad_rate_approved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | 0.71242 | 0.024938 | 0.0062383 | 0.0088386 | 0.032004 | 0.025715 | 0.014533 | 0.0037679 | 0.090322 | 0.06184 | 0.99103 | 0.02511 |
| v1.1 | 0.71242 | 0.024778 | 0.012577 | 0.010026 | 0.031991 | 0.025715 | 0.018687 | 0.0037679 | 0.056893 | 0.06184 | 1 | 0.025715 |
| v2 | 0.71857 | 0.024703 | 0.015593 | 0.0069983 | 0.032713 | 0.025715 | 0.0089574 | 0.0037679 | 0.077856 | 0.064103 | 0.99902 | 0.02574 |

### Brier skill, 95% bootstrap interval

| model | brier_skill | ci_lower | ci_upper |
|---|---|---|---|
| v1 | 0.0062383 | -0.0031267 | 0.015328 |
| v1.1 | 0.012577 | 0.0099073 | 0.015357 |
| v2 | 0.015593 | 0.010083 | 0.020976 |

### Paired differences (v2 minus v1.1)

| statistic | point | mean | lower | upper |
|---|---|---|---|---|
| brier_diff (v2 minus v1.1) | -7.5676e-05 | -7.5122e-05 | -0.00015828 | 4.1907e-06 |
| auc_diff (v2 minus v1.1) | 0.0061525 | 0.0061401 | -0.00015024 | 0.012381 |

### Deciles, v1

| bin | n | mean_score | observed_rate |
|---|---|---|---|
| 0 | 1327 | 0.014533 | 0.0037679 |
| 1 | 1326 | 0.014975 | 0.0015083 |
| 2 | 1327 | 0.016427 | 0.009043 |
| 3 | 1325 | 0.019753 | 0.016604 |
| 4 | 1326 | 0.023626 | 0.022624 |
| 5 | 1326 | 0.027427 | 0.027903 |
| 6 | 1326 | 0.031994 | 0.029412 |
| 7 | 1326 | 0.036708 | 0.027903 |
| 8 | 1326 | 0.044293 | 0.056561 |
| 9 | 1326 | 0.090322 | 0.06184 |

### Deciles, v1.1

| bin | n | mean_score | observed_rate |
|---|---|---|---|
| 0 | 1327 | 0.018687 | 0.0037679 |
| 1 | 1326 | 0.019328 | 0.0015083 |
| 2 | 1327 | 0.021266 | 0.009043 |
| 3 | 1325 | 0.025252 | 0.016604 |
| 4 | 1326 | 0.029063 | 0.022624 |
| 5 | 1326 | 0.032332 | 0.027903 |
| 6 | 1326 | 0.035672 | 0.029412 |
| 7 | 1326 | 0.038667 | 0.027903 |
| 8 | 1326 | 0.042759 | 0.056561 |
| 9 | 1326 | 0.056893 | 0.06184 |

### Deciles, v2

| bin | n | mean_score | observed_rate |
|---|---|---|---|
| 0 | 1327 | 0.0089574 | 0.0037679 |
| 1 | 1326 | 0.0098409 | 0.00075415 |
| 2 | 1326 | 0.012076 | 0.010558 |
| 3 | 1326 | 0.016537 | 0.012821 |
| 4 | 1328 | 0.026622 | 0.021837 |
| 5 | 1324 | 0.033013 | 0.021148 |
| 6 | 1326 | 0.041436 | 0.036199 |
| 7 | 1326 | 0.046713 | 0.038462 |
| 8 | 1326 | 0.054104 | 0.047511 |
| 9 | 1326 | 0.077856 | 0.064103 |

### Decision

- Brier difference CI upper bound: 0.000004
- Mean paired AUC difference: 0.00614
- v2_preferred: False
