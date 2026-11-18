# Run manifest: R01-fico-only

## Run

- Run id: R01-fico-only
- Date: 2026-11-18
- Family and feature set: logreg, fico_only
- Training weight: none

## Hypothesis

FICO alone, binned, is the single-feature benchmark any model must beat.

## Pass and kill criteria

- Pass: CV AUC clearly above 0.5.
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

- Search space: none, single fit with fixed parameters {"C": 1.0, "n_bins": 10}
- Sampler: none
- Run log: none

## Stopping rule

- Window: n/a
- Minimum delta: n/a
- Hard cap: 1 fit

## Result

- Trials run: 1
- Stop reason: none
- Best value and cv_se: 0.6291, 0.0104 (folds: 0.6490, 0.6251, 0.6409, 0.6016)
- Validation AUC: 0.6479
- Parameters: `{"C": 1.0, "n_bins": 10}`
- Leaderboard row: R01-fico-only
