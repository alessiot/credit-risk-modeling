# Run manifest: R06-lgbm-weighted

## Run

- Run id: R06-lgbm-weighted
- Date: 2026-11-18
- Family and feature set: lgbm, full
- Training weight: sample_weight

## Hypothesis

Training with sample_weight (about 3x on defaults) does not improve ranking.

## Pass and kill criteria

- Pass: CV AUC within one cv_se of R03.
- Kill: CV AUC differs from R03 by more than one cv_se.

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
- Best value and cv_se: 0.7042, 0.0064 (folds: 0.7109, 0.6851, 0.7102, 0.7104)
- Validation AUC: 0.7174
- Parameters: `{"n_estimators": 64, "learning_rate": 0.01111149590259364, "num_leaves": 18, "min_child_samples": 138, "reg_lambda": 0.0013435846437317542, "colsample_bytree": 0.7901712827020574, "subsample": 0.7497307409391766, "subsample_freq": 1}`
- Leaderboard row: R06-lgbm-weighted
