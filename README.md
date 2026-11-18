# Credit risk: 12-month probability of default at origination

A probability-of-default (PD) model scored at the application decision, built on 50,000 loans
from vintages 2021-01 to 2024-09. It uses an out-of-time evaluation, has a documented selection
rule, and ships as a Metaflow pipeline that reproduces the analysis and versions the model.

- [`notebooks/modeling.ipynb`](notebooks/modeling.ipynb) is the main deliverable, executed
  with all outputs saved. It ends with the summary, limitations and recommendations.
- [`docs/model-card.md`](docs/model-card.md) records every data, split, selection,
  calibration and monitoring decision, with its numbers.
- [`reports/leaderboard.csv`](reports/leaderboard.csv) lists every model run. Each run has a
  manifest under [`runs/`](runs/): the script writes it before the run starts and appends the
  result. The post-test recalibration (R10) changes no model, so it has a manifest but no
  leaderboard row.

## Results

| | Value |
|---|---|
| Selected model | LightGBM with monotone constraints, 12 application features (APR excluded) |
| Test window | 2024-01 to 2024-09, 9,984 loans, 227 defaults, scored once with the frozen model (later looks disclosed) |
| Test ROC AUC | 0.680 (95% CI 0.651-0.708). Gini 0.36, KS 0.31 |
| Validation AUC, selected model vs FICO only | 0.719 vs 0.648 |
| 2023 gap-year AUC | 0.712 |
| Lowest-scored 20% of test loans | 0.20% default rate, against 2.27% overall |
| Test Brier skill vs a constant at the training default rate | -1.5% (95% CI -2.5% to -0.5%) for the frozen v1. +0.7% (+0.4% to +1.0%) for v1.1, recalibrated after the test pass |
| Calibration on test (v1) | Mean PD 3.21% vs 2.27% observed. Top decile 9.1% vs 3.9% |

**The model ranks, but its PD does not yet price.** The frozen v1 PD scores worse on test Brier
than predicting the training default rate for every loan. There are two causes:

- **The calibrator (our error).** Platt scaling was fit on the raw probability instead of its
  log-odds, which stretched the top of the scale. The uncalibrated scores beat v1.
- **2024 level shift (drift).** Default rates fell while the measured input distributions did
  not move.

A recalibrated v1.1 was registered after the test pass and chosen on 2022 data only. Its test
figures are a second look at the test window. It removes the top-end stretch. It does not fix
the bottom of the scale (1.87% predicted vs 0.10% observed), which the registered run expected
it to: the one-parameter log-odds shift that the selection rule picked cannot change shape.
Calibrators fit on the test labels themselves, as diagnostics, reach 1.0% skill with a level
shift alone and 1.4% with isotonic. So roughly half of what remains is level and half is shape.

What drove the result:

1. **Four columns are not available at the decision.**
   - `days_past_due_current` and `sample_weight` separate defaults perfectly.
   - `total_payments_to_date` divided by the loan amount does too.
   - `months_on_book` is recorded after origination.
   - With them, CV AUC is 1.000. All four are excluded, and `sample_weight` is tested only as a
     training weight. That made AUC slightly worse.
2. **Placeholder codes are missing values, and missingness predicts default.** These are
   FICO 99999, income -1 and inquiries 99. Each becomes NaN plus a missing flag.
3. **Out-of-time validation with the production gap.** Tuning runs on 2021-01 to 2022-06 with
   expanding quarterly folds. Calibration and the cutoff use 2022-H2. 2023 is held out as the
   12-month label gap, and 2024 is the test window, scored once with the frozen model.
4. **Selection follows a rule written before any model ran.** It picks the simplest model
   within one CV standard error of the best. It chose LightGBM without APR. Finer searches run
   after the test pass confirm that APR adds nothing (0.7087 without, 0.7079 with). They also
   show the frozen configuration is about one standard error behind (R09, the recommended
   next version). Reference runs bracket it. LightGBM with library defaults reaches 0.672 CV
   AUC. A supervised-bin scorecard reaches 0.680 CV and 0.708 validation, a credible
   alternative where a scorecard is required.

Default rates fell from 3.2% (2021-22) to 2.3% (2024) while input distributions stayed stable,
mostly below FICO 650. The cause is not established. Limitations and recommendations close the
notebook.

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
