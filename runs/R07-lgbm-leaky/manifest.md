# Run manifest: R07-lgbm-leaky

## Run

- Run id: R07-lgbm-leaky
- Date: 2026-11-18
- Family and feature set: lgbm, leaky
- Training weight: none

## Hypothesis

Post-origination columns inflate AUC toward 1; never selectable.

## Pass and kill criteria

- Pass: CV AUC far above R03, showing the inflation.
- Kill: n/a (leakage ablation).

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

- Search space: none, single fit with best parameters of R03-lgbm parameters
- Sampler: none
- Run log: none

## Stopping rule

- Window: n/a
- Minimum delta: n/a
- Hard cap: 1 fit

## Result

- Trials run: 1
- Stop reason: none
- Best value and cv_se: 1.0000, 0.0000 (folds: 1.0000, 1.0000, 1.0000, 1.0000)
- Validation AUC: 1.0000
- Parameters: `{"n_estimators": 64, "learning_rate": 0.01111149590259364, "num_leaves": 18, "min_child_samples": 138, "reg_lambda": 0.0013435846437317542, "colsample_bytree": 0.7901712827020574, "subsample": 0.7497307409391766, "subsample_freq": 1}`
- Leaderboard row: R07-lgbm-leaky
