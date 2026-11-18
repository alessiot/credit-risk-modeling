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
