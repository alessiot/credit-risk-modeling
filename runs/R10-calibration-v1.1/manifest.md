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

## Result

### Selection (out-of-fold on validation)

| method | brier | log_loss | ece | diff_vs_best | se_vs_best | eligible |
|---|---|---|---|---|---|---|
| logit_intercept | 0.030531 | 0.13515 | 0.01275 | 4.9696e-05 | 0.00014234 | True |
| logit_platt | 0.030523 | 0.1338 | 0.0086002 | 4.1304e-05 | 8.4041e-05 | True |
| sigmoid | 0.030659 | 0.13492 | 0.010069 | 0.00017732 | 8.2428e-05 | False |
| isotonic | 0.030481 | 0.13757 | 0.006383 | 0 | 0 | True |

- Folds: {'A': [202207, 202208, 202209], 'B': [202210, 202211, 202212]}
- Lowest pooled Brier: isotonic
- Chosen: logit_intercept
- Break-even cutoff: 0.1429; training-window default rate 0.03210

### gap

| model | auc | brier | brier_skill | ece | mean_pd | observed | bottom_pred | bottom_obs | top_pred | top_obs | approval | bad_rate_approved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.71242 | 0.024774 | 0.012763 | 0.0097495 | 0.031277 | 0.025715 | 0.018263 | 0.0037679 | 0.055652 | 0.06184 | 1 | 0.025715 |
| v1 | 0.71242 | 0.024938 | 0.0062383 | 0.0088386 | 0.032004 | 0.025715 | 0.014533 | 0.0037679 | 0.090322 | 0.06184 | 0.99103 | 0.02511 |
| v1.1 | 0.71242 | 0.024778 | 0.012577 | 0.010026 | 0.031991 | 0.025715 | 0.018687 | 0.0037679 | 0.056893 | 0.06184 | 1 | 0.025715 |

### test

| model | auc | brier | brier_skill | ece | mean_pd | observed | bottom_pred | bottom_obs | top_pred | top_obs | approval | bad_rate_approved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.68003 | 0.022143 | 0.0073662 | 0.0099379 | 0.03128 | 0.022736 | 0.018246 | 0.001001 | 0.055653 | 0.039039 | 1 | 0.022736 |
| v1 | 0.68003 | 0.02264 | -0.01493 | 0.011911 | 0.032064 | 0.022736 | 0.014522 | 0.001001 | 0.09089 | 0.039039 | 0.99028 | 0.022555 |
| v1.1 | 0.68003 | 0.022155 | 0.0068116 | 0.010349 | 0.031994 | 0.022736 | 0.01867 | 0.001001 | 0.056894 | 0.039039 | 1 | 0.022736 |
