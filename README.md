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
  0.10% observed. [Next version](#next-version) says how to fix it.

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

## Next version

The tuned models are barely fitted (R09, the best search, is 88 three-leaf trees at learning
rate 0.0135). Tuning on AUC cannot see a compressed scale, so it keeps choosing such models.

1. Tune on log loss instead of AUC, with the same out-of-time folds.
2. Fit and choose the calibrator on out-of-fold training scores plus validation: 658 defaults
   instead of 214. First check on validation that fold-model scores have the same scale as the
   final model's.
3. At deployment, rerun that selection with 2023 outcomes added. No window is left to test it
   here, so it carries no test claim.
4. Do not score the 2024 window again. It has had three looks; a new version needs a new
   out-of-time window.

## Repository layout

```
data/                      input dataset
src/credit_risk/           package: config, data validation, features, splits, models,
                           evaluation, tuning, packaging, pipeline (step functions)
notebooks/modeling.ipynb   analysis notebook (executed)
scripts/run_experiments.py registered experiments: baselines, tuning, ablations, convergence,
                           post-test reference runs (CV and validation only)
scripts/recalibrate.py     registered post-test recalibration of the frozen model (v1.1)
scripts/compare_runs.py    compare Metaflow runs
scripts/check.sh           format, lint and tests
flows/credit_risk_flow.py  Metaflow pipeline
flows/sweep.json           candidate configurations for the pipeline sweep
models/pd-lgbm-no-apr-v1/  packaged model (calibrated pipeline + metadata.json), frozen
models/pd-lgbm-no-apr-v1.1/ same base model, recalibrated after the test pass
reports/                   leaderboard, test predictions (v1; v1 and v1.1)
runs/                      one manifest, result and trial log per experiment
docs/model-card.md         model card
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
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/modeling.ipynb
```

Every source of randomness is seeded. Rerunning the experiments and the notebook reproduces the
leaderboard and the test predictions exactly. `tests/test_artifact.py` reloads the saved model
and checks it against `reports/test_predictions.csv`.

## Pipeline (Metaflow)

```
start      load and validate the CSV (schema, value domains, uniqueness), build out-of-time splits
train      foreach candidate in the sweep: expanding-window CV on the training window,
           fit, score validation
select     join: champion by the one-standard-error rule, challenger = best other candidate,
           paired bootstrap on validation AUC
calibrate  calibration on validation (default: log-odds shift, as in v1.1); cutoff from LGD
           and margin
evaluate   score the 2023 gap year (and the 2024 test window only with --score_test true);
           per-vintage metrics, calibration, Brier skill, PSI; Metaflow card
end        save the calibrated pipeline and metadata to outputs/models/<run id>/
```

Run it:

```bash
uv run python flows/credit_risk_flow.py run                          # defaults
uv run python flows/credit_risk_flow.py run --calibration isotonic   # change one parameter
uv run python flows/credit_risk_flow.py run --score_test true        # also score the test window
uv run python flows/credit_risk_flow.py run --sweep my_sweep.json --train_end 202203 --val_start 202204
uv run python flows/credit_risk_flow.py card view evaluate           # metrics card of the latest run
```

All parameters: `data_path`, `sweep`, `train_end`, `val_start`, `val_end`, `test_start`,
`test_end`, `calibration` (`logit_intercept`, `logit_platt`, `sigmoid`, `isotonic`), `lgd`,
`margin`, `seed`, `score_test` (default false), `output_dir`. The protocol rejects
inconsistent boundaries (for example, a validation window that does not start the month after
training ends).

**Parameter sweeps.** `flows/sweep.json` lists the candidates (family, feature set,
hyperparameters, training weight). The `train` step fans out over them in parallel, and
`select` applies the selection rule. The default sweep holds the five selectable configurations from
the leaderboard, with hyperparameters tuned on the default training window. Runs do not
re-tune, so a run with another `--train_end` reuses those hyperparameters.

**Champion and challenger.** Every run records the champion and the runner-up and compares
them with a paired bootstrap on validation AUC. To compare pipeline variants (calibration
method, training window, sweep), run the flow with different parameters and compare runs:

```bash
uv run python scripts/compare_runs.py --last 5             # validation and gap-year metrics
uv run python scripts/compare_runs.py --last 5 --with-test # adds test metrics
uv run python scripts/compare_runs.py --run-id <run id>    # per-vintage test table
```

Example with the default run and an isotonic-calibration run (the default table leaves out test
metrics, so variants are compared on validation and the 2023 gap year):

```
          run_id          created            sweep  train_end     calibration    champion  cv_auc  val_auc  gap_auc  gap_ece                        artifact
1795033986547329 2026-11-18 15:33 flows/sweep.json     202206        isotonic lgbm_no_apr  0.7013   0.7195   0.7109   0.0083 outputs/models/1795033986547329
1795033965013512 2026-11-18 15:32 flows/sweep.json     202206 logit_intercept lgbm_no_apr  0.7013   0.7195   0.7124   0.0100 outputs/models/1795033965013512
```

`--with-test` adds test AUC and ECE for runs made with `--score_test true`. Use it to report a
chosen variant, not to choose one: picking a variant by its test score turns the test window
into a tuning set. Neither run above scored the test window. An earlier isotonic run, made
before test scoring became opt-in, did, as a demonstration. It informed no decision.

Each artifact folder holds `model.joblib` (the calibrated pipeline from raw application
columns to PD) and `metadata.json`. The metadata has the candidate, parameters, split protocol,
data SHA-256, cutoff, metrics, library versions, the input columns the model needs (12; APR is
not one), and a hash of the model file that is checked on load. Scoring one application takes
about 2 ms. To score new applications:

```python
from credit_risk.pipeline import score_applications

pd_scores = score_applications(
    "models/pd-lgbm-no-apr-v1", applications_df
)  # checks the input schema
```

## Departures from the brief and the template

- **Logging.** Code reports through `logging`, not `print`. Log levels come from `LOG_LEVEL`.
- **No catch-all error handling.** Invalid data raises specific exceptions (`SchemaError`,
  `DataValidationError`, `ValueError`) with every failure listed. Metaflow's `@retry` covers
  transient failures in the training step only.
- **Warnings are not suppressed.** The test suite treats warnings as errors.
- **One gradient-boosting library.** LightGBM only; XGBoost and CatBoost are not dependencies.
- **No resampling.** `imbalanced-learn` was not needed.

## Open questions for the data owners

1. What is the snapshot date? Are outcomes for 2023-10 to 2024-09 final? They are treated as
   final here.
2. Are `loan_amount`, `term` and `apr` the requested values, or the approved and priced ones?
   Is APR set before or after the risk decision?
3. What does `sample_weight` represent? It encodes the outcome.
4. Are declined applications available for reject inference?
5. What loss-given-default and margin figures should set the cutoff? The values used here
   (LGD 60%, margin 10%) are illustrative.

The dataset generator referenced in the original brief (`scripts/generate_enhanced_data.py`)
was not provided, so this repository treats the CSV as the source of truth.
