# Run manifest: R00-base-rate

## Run

- Run id: R00-base-rate
- Date: 2026-11-18
- Family and feature set: base_rate, full
- Training weight: none

## Hypothesis

A constant prediction at the training default rate sets the floor (AUC 0.5).

## Pass and kill criteria

- Pass: Runs and scores AUC 0.5.
- Kill: n/a (baseline).

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
- Best value and cv_se: 0.5000, 0.0000 (folds: 0.5000, 0.5000, 0.5000, 0.5000)
- Validation AUC: 0.5000
- Parameters: `{}`
- Leaderboard row: R00-base-rate
