# Run manifest: R03-lgbm

## Run

- Run id: R03-lgbm
- Date: 2026-11-18
- Family and feature set: lgbm, full
- Training weight: none

## Hypothesis

Monotone LightGBM captures threshold effects and interactions beyond R02.

## Pass and kill criteria

- Pass: CV AUC above R02 by more than one cv_se.
- Kill: CV AUC within one cv_se of R02 after a converged search.

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
- Minimum delta: 0.01308 (cv_se of the default configuration, CV AUC 0.6617)
- Hard cap: 200 trials

## Result

- Trials run: 33
- Stop reason: plateau
- Best value and cv_se: 0.7074, 0.0070 (folds: 0.7154, 0.6865, 0.7144, 0.7131)
- Validation AUC: 0.7184
- Parameters: `{"n_estimators": 64, "learning_rate": 0.01111149590259364, "num_leaves": 18, "min_child_samples": 138, "reg_lambda": 0.0013435846437317542, "colsample_bytree": 0.7901712827020574, "subsample": 0.7497307409391766, "subsample_freq": 1}`
- Leaderboard row: R03-lgbm

## Convergence check

- Best-so-far curve: trials.csv (column best_so_far)
- Rerun with a larger cap: window 60, cap 400, minimum delta 0.01308
  - seed 1: best 0.7038 after 61 trials (plateau), log trials_seed1.csv
  - seed 2: best 0.7074 after 61 trials (plateau), log trials_seed2.csv
- Status: converged (recorded best 0.7074)

## Deviations

- 2026-11-18: the convergence status above is withdrawn. The plateau rule used a minimum delta
  of 0.0131 (cv_se of the default configuration), about twice the selection margin it informs,
  so improvements below it did not count and the reruns became fixed 61-trial budgets. Status:
  unconverged. R08-lgbm-sensitivity repeats the search with a minimum delta of 0.001 and three
  seeds (best 0.7079).
