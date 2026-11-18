# Run manifest: R04-logreg-no-apr

## Run

- Run id: R04-logreg-no-apr
- Date: 2026-11-18
- Family and feature set: logreg, no_apr
- Training weight: none

## Hypothesis

APR adds no rank-ordering power (audit: no relation to FICO or default).

## Pass and kill criteria

- Pass: CV AUC within one cv_se of R02.
- Kill: CV AUC below R02 by more than one cv_se.

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

- Search space: none, single fit with best parameters of R02-logreg parameters
- Sampler: none
- Run log: none

## Stopping rule

- Window: n/a
- Minimum delta: n/a
- Hard cap: 1 fit

## Result

- Trials run: 1
- Stop reason: none
- Best value and cv_se: 0.6834, 0.0106 (folds: 0.6947, 0.6521, 0.6885, 0.6982)
- Validation AUC: 0.7134
- Parameters: `{"C": 6.34632684458423, "n_bins": 3}`
- Leaderboard row: R04-logreg-no-apr
