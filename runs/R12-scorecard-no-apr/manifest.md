# Run manifest: R12-scorecard-no-apr

## Run

- Run id: R12-scorecard-no-apr
- Date: 2026-11-18
- Family and feature set: scorecard, no_apr
- Training weight: none

## Hypothesis

Reference run after the test pass: a supervised-bin scorecard (tree-based bins, one-hot, L2 logistic regression) shows what a conventional scorecard reaches without APR.

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

- Search space: exhaustive grid of 9 configurations, max_bins in [4, 6, 8]; C in [0.01, 0.1, 1.0]; fixed {"min_bin_fraction": 0.05}
- Sampler: none
- Run log: trials.csv

## Stopping rule

- Window: n/a
- Minimum delta: n/a
- Hard cap: 9 configurations (grid)

## Result

- Trials run: 9
- Stop reason: none
- Best value and cv_se: 0.6804, 0.0118 (folds: 0.6846, 0.6459, 0.6981, 0.6930)
- Validation AUC: 0.7079
- Parameters: `{"min_bin_fraction": 0.05, "max_bins": 4, "C": 0.01}`
- Leaderboard row: R12-scorecard-no-apr
