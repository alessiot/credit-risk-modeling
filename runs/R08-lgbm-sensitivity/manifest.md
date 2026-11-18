# Run manifest: R08-lgbm-sensitivity

## Run

- Run id: R08-lgbm-sensitivity
- Date: 2026-11-18
- Family and feature set: lgbm, full
- Training weight: none

## Hypothesis

R03's search stopped on a minimum delta (0.013) coarser than the selection margin; a search with a 0.001 delta shows where LightGBM with APR converges.

## Pass and kill criteria

- Pass: Best CV AUC stable across three seeds within 0.001.
- Kill: n/a (sensitivity check, run after the test pass, never selectable).

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

- Search space: n_estimators [50, 800]; learning_rate log [0.01, 0.2]; num_leaves [2, 32]; min_child_samples log [20, 500]; reg_lambda log [1e-3, 30]; colsample_bytree [0.4, 1]; subsample [0.5, 1] with subsample_freq 1
- Sampler: TPE through Optuna (seed 0)
- Run log: trials.csv

## Stopping rule

- Window: 30 trials
- Minimum delta: 0.001 (fixed, finer than the selection margin); window 60; seeds [0, 1, 2]
- Hard cap: 400 trials

## Result

  - seed 0: best 0.7074 after 93 trials (plateau)
  - seed 1: best 0.7038 after 90 trials (plateau)
  - seed 2: best 0.7079 after 93 trials (plateau)
- Best over seeds: 0.7079 (cv_se 0.0065); spread across seeds 0.0040
- Validation AUC: 0.7195
- Parameters: `{"n_estimators": 221, "learning_rate": 0.01168310320925358, "num_leaves": 4, "min_child_samples": 289, "reg_lambda": 0.08151899460417315, "colsample_bytree": 0.7263007976465055, "subsample": 0.9636644624825333, "subsample_freq": 1}`
