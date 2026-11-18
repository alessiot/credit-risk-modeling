# Pipeline and engineering notes

How to run the Metaflow pipeline, compare runs and score new applications, and where the code departs from the brief and the template.

## Pipeline (Metaflow)

```
start      load and validate the CSV (schema, value domains, uniqueness), build out-of-time splits
preprocess feature engineering: apply the stateless transformations (placeholders to NaN plus
           missing flags, loan-to-income and payment-to-income ratios) to every split, check
           they are deterministic, and store a per-split feature report (placeholder counts,
           missing rates, engineered-feature ranges)
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
    "models/pd-lgbm-no-apr-v1.1", applications_df
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
- **Feature engineering is a step, but fitting stays in the model.** The `preprocess` step
  applies and reports the stateless transformations. The stateful parts (category encoders,
  bins) are fitted inside each CV fold by the model pipeline in `train`, so they never see
  validation rows, and the saved artifact takes raw application columns.
