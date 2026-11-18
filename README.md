# Credit risk: 12-month probability of default at origination

A probability-of-default (PD) model scored at the application decision, built on 50,000 loans
from vintages 2021-01 to 2024-09. It uses an out-of-time evaluation, has a documented selection
rule, and ships as a Metaflow pipeline that reproduces the analysis and versions the model.

- **Main deliverable:** [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb), executed with
  all outputs saved. It ends with the summary, limitations and recommendations.
- **Decision record:** [`docs/model-card.md`](docs/model-card.md). Every data, split,
  selection, calibration and monitoring decision, with its numbers.
- **Experiment log:** [`reports/leaderboard.csv`](reports/leaderboard.csv). Every run, each
  with a manifest written before it ran, under [`runs/`](runs/).

## Results

| | Value |
|---|---|
| Selected model | LightGBM with monotone constraints, 12 application features (APR excluded) |
| Test window | 2024-01 to 2024-09, 9,984 loans, 227 defaults, scored once |
| Test ROC AUC | 0.680 (95% CI 0.651-0.708). Gini 0.36, KS 0.31 |
| Validation / 2023 gap year AUC | 0.720 / 0.712 |
| FICO-only baseline (validation) | 0.648 |
| Calibration on test | Mean PD 3.21% vs 2.27% observed. Default rates fell in 2023-24 at a constant borrower mix, so the model needs recalibration before use |

What drove the result:

1. **Four columns leak the outcome.** `days_past_due_current`, `total_payments_to_date` and
   `sample_weight` separate defaults perfectly. `months_on_book` is recorded after origination
   and matches no snapshot date. With them, CV AUC is 1.000. All four are excluded, and
   `sample_weight` is tested only as a training weight. That made AUC slightly worse.
2. **Placeholder codes are missing values, and missingness predicts default.** These are
   FICO 99999, income -1 and inquiries 99. Each becomes NaN plus a missing flag.
3. **Out-of-time validation with the production gap.** Tuning runs on 2021-01 to 2022-06 with
   expanding quarterly folds. Calibration and the cutoff use 2022-H2. 2023 is held out as the
   12-month label gap, and 2024 is the test window, used once.
4. **Selection by a rule written before any model ran:** the simplest model within one CV
   standard error of the best.
   - LightGBM beats the binned logistic regression by 0.018-0.026 CV AUC, consistently across
     folds.
   - Dropping APR costs 0.006 CV AUC, inside one standard error. APR may come out of the
     pricing decision, so the rule's choice of the APR-free model is also the safer one.

## Repository layout

```
data/                      input dataset
src/credit_risk/           package: config, data validation, features, splits, models,
                           evaluation, tuning, packaging, pipeline (step functions)
notebooks/modeling.ipynb   analysis notebook (executed)
scripts/run_experiments.py registered experiments: baselines, tuning, ablations, convergence
scripts/compare_runs.py    compare Metaflow runs
scripts/check.sh           format, lint and tests
flows/credit_risk_flow.py  Metaflow pipeline
flows/sweep.json           candidate configurations for the pipeline sweep
models/pd-lgbm-no-apr-v1/  packaged model (calibrated pipeline + metadata.json)
reports/                   leaderboard and test predictions
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
calibrate  Platt or isotonic calibration on validation; cutoff from LGD and margin
evaluate   score the 2023 gap year and the 2024 test window once; per-vintage metrics,
           calibration, PSI; Metaflow card
end        save the calibrated pipeline and metadata to outputs/models/<run id>/
```

Run it:

```bash
uv run python flows/credit_risk_flow.py run                          # defaults
uv run python flows/credit_risk_flow.py run --calibration isotonic   # change one parameter
uv run python flows/credit_risk_flow.py run --sweep my_sweep.json --train_end 202203 --val_start 202204
uv run python flows/credit_risk_flow.py card view evaluate           # metrics card of the latest run
```

All parameters: `data_path`, `sweep`, `train_end`, `val_start`, `val_end`, `test_start`,
`test_end`, `calibration`, `lgd`, `margin`, `seed`, `output_dir`. The protocol rejects
inconsistent boundaries (for example, a validation window that does not start the month after
training ends).

**Parameter sweeps.** `flows/sweep.json` lists the candidates (family, feature set,
hyperparameters, training weight). The `train` step fans out over them in parallel, and
`select` applies the selection rule. The default sweep holds the five tuned configurations from
the leaderboard.

**Champion and challenger.** Every run records the champion and the runner-up and compares
them with a paired bootstrap on validation AUC. To compare pipeline variants (calibration
method, training window, sweep), run the flow with different parameters and compare runs:

```bash
uv run python scripts/compare_runs.py --last 5
uv run python scripts/compare_runs.py --run-id <run id>    # per-vintage test table
```

Example with the default run and an isotonic-calibration run:

```
          run_id          created            sweep  train_end calibration    champion  cv_auc  test_auc      test_auc_ci  test_ece                        artifact
1795009036562874 2026-11-18 08:37 flows/sweep.json     202206    isotonic lgbm_no_apr  0.7013    0.6764 [0.6476, 0.7043]    0.0113 outputs/models/1795009036562874
1795009013944738 2026-11-18 08:36 flows/sweep.json     202206     sigmoid lgbm_no_apr  0.7013    0.6800 [0.6507, 0.7078]    0.0119 outputs/models/1795009013944738
```

Run comparison shows what each variant does. The choice between variants still belongs on
validation data: picking a calibration method by its test score would turn the test window into
a tuning set.

Each artifact folder holds `model.joblib` (the calibrated pipeline from raw application
columns to PD) and `metadata.json`. The metadata has the candidate, parameters, split protocol,
data SHA-256, cutoff, metrics, library versions, and a hash of the model file that is checked
on load. To score new applications:

```python
from credit_risk.pipeline import score_applications
pd_scores = score_applications("models/pd-lgbm-no-apr-v1", applications_df)  # checks the input schema
```

## Departures from the brief and the template

- **Logging.** Code reports through `logging`, not `print`. Log levels come from `LOG_LEVEL`.
- **No catch-all error handling.** Invalid data raises specific exceptions (`SchemaError`,
  `DataValidationError`, `ValueError`) with every failure listed. Metaflow's `@retry` covers
  transient failures in the training step only.
- **Warnings are not suppressed.** The test suite treats warnings as errors.
- **One gradient-boosting library.** LightGBM is used, not LightGBM plus XGBoost plus CatBoost.
  The unused libraries from the original requirements were dropped.
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
