# Run manifest: R10-calibration-v1.1

## Run

- Run id: R10-calibration-v1.1
- Date: 2026-11-18
- Family and feature set: lgbm, no_apr (frozen v1 base model)

## Question
The v1 calibrator (Platt on the raw probability) loses to a constant on test Brier. Which
calibration shape, chosen only from data mature at a 2024-01 decision date, should v1.1 use,
and how does it score on the gap year and test?

## Status
Registered after the test window was scored for v1 (post-hoc). The base model is the frozen
v1 pipeline, unchanged. The results below are published whatever they show.

## Protocol
- Input: raw scores of the frozen v1 base model on validation 202207-202212.
- Candidates, simplest first: logit_intercept (log-odds shift), logit_platt (intercept and
  slope on log-odds), sigmoid (v1 method), isotonic.
- Selection: two time-ordered folds inside validation (202207-202209, 202210-202212). Fit on
  one, score the other, pool the out-of-fold predictions. Choose the simplest method whose
  pooled Brier is within one paired standard error of the lowest.
- Refit the chosen method on all of validation. Cutoff: break-even PD 0.10 / 0.70.
- Score the 2023 gap year and the 2024 test window once. Brier skill is reported against the
  training-window default rate, the constant available at the decision date.
- Not used: any 2023 or 2024 label for fitting or choosing.

## Expected
Shape fix lowers Brier at the bottom of the scale. Test Brier skill stays near zero because
the 2024 level shift was not knowable at the decision date.

## Reproducibility

- Data hash: 06e6bfa6fe44cede0463b6c2bf7bd2e3df622cf2f8d3f253bd77655d08f046d3
- Environment lock: 45e16d4c7aba6bbc23981e415fee035b5858f8f325822f14fab395c996d58dc4
