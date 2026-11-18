# Credit risk: 12-month probability of default at origination

## Summary

- **The model ranks applicants out of time.** On 2024 loans the test AUC is 0.680 (95% CI
  0.651-0.708), and the lowest-scored 20% default at 0.20% against 2.27% overall.
- **Its PD is not fit for pricing yet.** The frozen v1 loses to a constant on test Brier (skill
  -1.5%), because its calibrator was fit on the probability instead of the log-odds. A
  post-hoc recalibration (v1.1) reaches +0.7%, no better than the uncalibrated scores.
- **At the assumed economics it declines nobody.** With LGD 60% and margin 10% the break-even
  PD is 14.3%, and the highest v1.1 PD on test is 11.3%. Use the score for ranking: pricing
  tiers, limits and an approval-rate policy, not a break-even decline rule.
- **The scale is compressed at the bottom.** The lowest test decile predicts 1.87% against
  0.10% observed. [v2](#v2-tuned-on-log-loss), tuned on log loss, halves that compression in
  2023 and is better on ranking and Brier there, but worse on mean level and the top decile. It
  misses its pre-registered bar by a margin that reverses under a stratified bootstrap, so
  v1.1 stays the recommended model, narrowly.

The [notebook](notebooks/modeling.ipynb) is the main deliverable (executed, outputs saved).
The [model card](docs/model-card.md) records every decision with its numbers, and
[`reports/leaderboard.csv`](reports/leaderboard.csv) and [`runs/`](runs/) hold the run
history: one manifest per run, written before it ran.

## Results

| | Value |
|---|---|
| Selected model | LightGBM with monotone constraints, 12 application features (APR excluded) |
| Test window | 2024-01 to 2024-09, 9,984 loans, 227 defaults, scored once with the frozen model (later looks disclosed) |
| Test ROC AUC | 0.680 (95% CI 0.651-0.708). Gini 0.36, KS 0.31 |
| Validation AUC, selected model vs FICO only | 0.719 vs 0.648 |
| 2023 gap-year AUC | 0.712 |
| Test Brier skill vs a constant at the training default rate | -1.5% (95% CI -2.5% to -0.5%) for the frozen v1. +0.7% (+0.4% to +1.0%) for v1.1, recalibrated after the test pass |
| Test mean PD vs observed | 3.21% (v1) and 3.20% (v1.1) vs 2.27% |
| Test deciles, predicted vs observed (v1.1) | Top 5.7% vs 3.9%. Bottom four over-predict 2x to 19x |

How the result was reached:

1. **Leakage removed.** `days_past_due_current`, `sample_weight`, payments-to-date over loan
   amount and `months_on_book` are not available at the decision; with them, CV AUC is 1.000.
2. **Placeholder codes become missing values plus flags** (FICO 99999, income -1, inquiries
   99). Missingness itself predicts default.
3. **Out-of-time evaluation with the production gap.** Tune on 2021-01 to 2022-06 (expanding
   quarterly folds), calibrate on 2022-H2, hold out 2023 as the 12-month label gap, test on
   2024.
4. **Selection by a rule written before any model ran.** It takes the simplest model within one
   CV standard error of the best. It chose LightGBM without APR, and later searches confirm APR
   adds nothing.
5. **Recalibration after the test pass (v1.1, run R10).** It was registered before it ran and
   chosen on 2022 data only. The rule picked a one-parameter log-odds shift, which removes v1's
   top-end stretch but cannot change shape. With 214 validation defaults every candidate except
   v1's method was within one standard error, so the rule had no power to prefer a richer one.
   Diagnostics fit on test labels reach 1.0% skill with a level shift and 1.4% with isotonic:
   about half of what remains is the 2024 drop in default rates (cause not established) and
   half is shape.

## v2: tuned on log loss

The AUC-tuned models are barely fitted (v1 is 64 trees at learning rate 0.011; R09, the best AUC
search, is 88 three-leaf trees). AUC cannot see a compressed scale. v2 (run R13) keeps the family, features and folds, and
changes the tuning objective and the calibration data:

- Tuned on CV log loss with the same folds and search space. Three seeds stopped on the plateau
  within 0.00034 of each other (CV se 0.0024), so the objective converged, though the seeds'
  parameters differ. It chose 288 four-leaf trees, about four times v1's boosting budget.
- Calibrator chosen and fit on out-of-fold training scores plus validation (550 defaults
  instead of 214), after a registered check that fold-model scores share the final model's
  scale (3 of 4 folds passed).
- Developed on data through 2022-12, registered before it ran, scored once on 2023. The 2024
  window is dropped on load and never used or scored.

| Gap year 2023 | v1.1 | v2 | v2 minus v1.1 (95% CI) |
|---|---|---|---|
| AUC | 0.712 | 0.719 | +0.006 (-0.0002 to +0.012) |
| Brier skill | +1.3% | +1.6% | Brier -0.000076 (-0.000158 to +0.000004) |
| Bottom decile, predicted vs observed | 1.87% vs 0.38% | 0.90% vs 0.38% | |

The registered rule required the Brier interval to lie entirely below zero. Its upper bound is
+0.000004, so v1.1 stays the recommended model and v2 is reported. The verdict is fragile: the registered script resampled rows without stratifying, while every other comparison in this repository uses a bootstrap stratified by outcome. Under that stratified bootstrap the interval is -0.000140 to -0.000010 (seeds 0-2 agree), entirely below zero, and v2 would pass. The registered result stands because the committed code defines the rule; the gap between v1.1 and v2 is small either way (Brier skill +1.3% vs +1.6%), and neither prices. v2 is worse on mean level
(3.27% vs 3.20%, against 2.57% observed) and on the top decile (7.8% vs 6.4% observed, where
v1.1 is 5.7% vs 6.2%). Next steps: refit on all mature vintages at
deployment, with the calibrator selection rerun on the latest outcomes, and evaluate on a new
out-of-time window. The 2024 window has had three looks and is not scored again.

## Repository layout

```
data/                      input dataset
src/credit_risk/           package: config, data validation, features, splits, models,
                           evaluation, tuning, packaging, pipeline (step functions)
notebooks/modeling.ipynb   analysis notebook (executed)
scripts/run_experiments.py registered experiments: baselines, tuning, ablations, convergence,
                           post-test reference runs (CV and validation only)
scripts/recalibrate.py     registered post-test recalibration of the frozen model (v1.1)
scripts/build_v2.py        registered v2 build (R13); drops the 2024 window on load
scripts/compare_runs.py    compare Metaflow runs
scripts/check.sh           format, lint and tests
flows/credit_risk_flow.py  Metaflow pipeline
flows/sweep.json           candidate configurations for the pipeline sweep
models/pd-lgbm-no-apr-v1/  packaged model (calibrated pipeline + metadata.json), frozen
models/pd-lgbm-no-apr-v1.1/ same base model, recalibrated after the test pass
models/pd-lgbm-no-apr-v2/  v2, tuned on log loss (R13); evaluated on 2023 only
reports/                   leaderboard, test predictions (v1; v1 and v1.1), gap-year predictions
                           (v1, v1.1, v2)
runs/                      one manifest, result and trial log per experiment
docs/model-card.md         model card
docs/pipeline.md           Metaflow pipeline, run comparison, scoring, departures from the brief
tests/                     pytest suite (unit, artifact reproduction, end-to-end flow)
```

## Setup

Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

With pip:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

## Reproduce

```bash
scripts/check.sh                                   # ruff format check, ruff lint, tests
uv run pytest -m slow                              # end-to-end pipeline test (about 20 s)
uv run python scripts/run_experiments.py           # all registered runs, rewrites the leaderboard
uv run python scripts/run_experiments.py --convergence --runs R02 R03
uv run python scripts/run_experiments.py --sensitivity    # converged LightGBM searches (R08, R09)
uv run python scripts/run_experiments.py --references     # untuned LightGBM, scorecard (R11, R12)
uv run python scripts/recalibrate.py                      # v1.1 (R10); refuses if outputs exist
uv run python scripts/build_v2.py                         # v2 (R13); refuses if outputs exist
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/modeling.ipynb
```

Every source of randomness is seeded. Rerunning the experiments and the notebook reproduces the
leaderboard and the test predictions exactly. `tests/test_artifact.py` reloads the saved model
and checks it against `reports/test_predictions.csv`.

## Pipeline, scoring and open questions

- [`docs/pipeline.md`](docs/pipeline.md): the Metaflow pipeline (steps, parameters, parameter
  sweeps, champion and challenger, run comparison), how to score new applications with the
  recommended v1.1 artifact, and where the code departs from the brief.
- [`docs/model-card.md`](docs/model-card.md): every modeling decision with its numbers, and
  the open questions for the data owners with the default used for each.
