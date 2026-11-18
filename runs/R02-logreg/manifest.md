# Run manifest: R02-logreg

## Run

- Run id: R02-logreg
- Date: 2026-11-18
- Family and feature set: logreg, full
- Training weight: none

## Hypothesis

A binned L2 logistic regression on all application features beats FICO alone.

## Pass and kill criteria

- Pass: CV AUC above R01 by more than one cv_se.
- Kill: CV AUC within one cv_se of R01 after a converged search.

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

- Search space: C log-uniform [1e-3, 10]; n_bins int [3, 20]
- Sampler: TPE through Optuna (seed 0)
- Run log: trials.csv

## Stopping rule

- Window: 30 trials
- Minimum delta: 0.01751 (cv_se of the default configuration, CV AUC 0.6481)
- Hard cap: 150 trials

## Result

- Trials run: 35
- Stop reason: plateau
- Best value and cv_se: 0.6812, 0.0134 (folds: 0.7009, 0.6420, 0.6868, 0.6951)
- Validation AUC: 0.7126
- Parameters: `{"C": 6.34632684458423, "n_bins": 3}`
- Leaderboard row: R02-logreg

## Convergence check

- Best-so-far curve: trials.csv (column best_so_far)
- Rerun with a larger cap: window 60, cap 300, minimum delta 0.01751
  - seed 1: best 0.6812 after 61 trials (plateau), log trials_seed1.csv
  - seed 2: best 0.6812 after 61 trials (plateau), log trials_seed2.csv
- Status: converged (recorded best 0.6812)
