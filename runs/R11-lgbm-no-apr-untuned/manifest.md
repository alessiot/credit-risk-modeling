# Run manifest: R11-lgbm-no-apr-untuned

## Run

- Run id: R11-lgbm-no-apr-untuned
- Date: 2026-11-18
- Family and feature set: lgbm, no_apr
- Training weight: none

## Hypothesis

Reference run after the test pass: untuned LightGBM, monotone constraints on, shows how much of the tuned model's validation result tuning explains. LightGBM defaults apply: n_estimators 100, learning_rate 0.1, num_leaves 31, min_child_samples 20, reg_lambda 0, subsample 1, colsample_bytree 1.

## Pass and kill criteria

- Pass: Reported as a reference; CV and validation only.
- Kill: n/a (reference run, never selectable, test window not read).

## Protocol

- `data_version`: enhanced-v1
- `split_id`: oot-v1
- `cv_scheme`: expanding-quarter-4
- `metric`: roc_auc
- `direction`: max

## Reproducibility

- Data hash: 06e6bfa6fe44cede0463b6c2bf7bd2e3df622cf2f8d3f253bd77655d08f046d3
- Environment lock: 45e16d4c7aba6bbc23981e415fee035b5858f8f325822f14fab395c996d58dc4
- Seeds: 0

## Search

- Search space: none, single fit with fixed parameters {}
- Sampler: none
- Run log: none

## Stopping rule

- Window: n/a
- Minimum delta: n/a
- Hard cap: 1 fit

## Result

- Trials run: 1
- Stop reason: none
- Best value and cv_se: 0.6725, 0.0069 (folds: 0.6648, 0.6618, 0.6709, 0.6923)
- Validation AUC: 0.6822
- Parameters: `{}`
- Leaderboard row: R11-lgbm-no-apr-untuned
