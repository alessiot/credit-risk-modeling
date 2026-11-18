# Run manifest: R09-lgbm-no-apr-sensitivity

## Run

- Run id: R09-lgbm-no-apr-sensitivity
- Date: 2026-11-18
- Family and feature set: lgbm, no_apr
- Training weight: none

## Hypothesis

R05 reused R03's parameters; tuning it directly shows whether the selection holds when both feature sets are searched to convergence.

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

  - seed 0: best 0.7087 after 164 trials (plateau)
  - seed 1: best 0.7070 after 133 trials (plateau)
  - seed 2: best 0.7076 after 81 trials (plateau)
- Best over seeds: 0.7087 (cv_se 0.0067); spread across seeds 0.0017
- Validation AUC: 0.7185
- Parameters: `{"n_estimators": 88, "learning_rate": 0.013465292790408426, "num_leaves": 3, "min_child_samples": 386, "reg_lambda": 0.015409783782431912, "colsample_bytree": 0.6265041407700579, "subsample": 0.7351980582395267, "subsample_freq": 1}`
