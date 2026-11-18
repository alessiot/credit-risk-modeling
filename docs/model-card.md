# Model card: 12-month probability of default at origination

## Decision and costs

- Decision the model drives: approve or decline a loan application, and later the price
  tier. The model returns a calibrated probability of default (PD).
- Prediction point: the application decision, before origination. Nothing observed after the
  decision may be an input.
- Cost of each error type: not given in the brief. For illustration, a declined good loan loses
  the lifetime net margin (assumed 10% of principal), and an approved bad loan loses
  LGD x exposure (assumed LGD 60%). Break-even PD = 0.10 / (0.10 + 0.60) = 14.3%. Cutoffs are
  also shown as a table of approval rate vs bad rate, so the business can choose.
- Constraints: each decline must come with reasons (adverse action notices under ECOA/Reg B),
  so either the model is interpretable or it has per-decision attributions. Batch or online
  scoring with no tight latency limit. Age is a protected basis and is not an input.

## Label and population

- Label definition: `default_12m` = 1 when the loan defaulted within 12 months of origination,
  as provided.
- Outcome horizon: 12 months after origination.
- Who gets a label: every loan in the file (approved, booked loans).
- Who will be scored: all applicants. Rejected applicants have no outcome, so the model is
  trained on approved loans only and inherits that selection (no reject inference).

## Data

- Source and version: `data/credit_risk_data_enhanced.csv`, `data_version` = `enhanced-v1`
  (SHA-256 recorded in the artifact metadata), 50,000 loans, vintages 2021-01 to 2024-09.
- Data decisions (counts out of 50,000):

| Issue | Count | Decision |
|---|---|---|
| `fico_score` = 99999 (missing) | 2,500 | NaN plus missing flag (default rate 4.16% when missing vs 2.78%) |
| `income` = -1 (missing) | 1,500 | NaN plus missing flag (5.93% vs 2.76%) |
| `inquiries_last_6m` = 99 (missing) | 1,000 | NaN plus missing flag (6.10% vs 2.79%) |
| `days_past_due_current`, `total_payments_to_date` | all rows | Dropped: recorded after origination, each separates the label perfectly |
| `sample_weight` | all rows | Not a feature: 0.80-1.20 for every non-default, 2.40-3.60 for every default. Tested only as a training weight (ablation) |
| `months_on_book` | all rows | Dropped: post-origination, and uniform 0-24 in every vintage, so it matches no snapshot date |
| `employment_length` > `age` - 16 | 13,848 | Kept as reported, not capped (a cap would bring age into the model). No signal either way (2.87% vs 2.85%) |
| `apr` clipped at 3.00 | 4,988 | Kept. `apr` shows no relation to FICO or default |
| `debt_to_income` at the 0.01 floor | 33 | Kept |
| Duplicates, NaN, out-of-range values | 0 | None needed |

## Feature availability

| Column | Known at the prediction point | Note |
|---|---|---|
| `fico_score`, `income`, `debt_to_income`, `num_open_trades`, `utilization_rate`, `inquiries_last_6m` | yes | Bureau and application data |
| `loan_amount`, `term` | yes | Assumed to be the requested values. If they are the approved values, the decision sets them |
| `apr` | unknown | Set by pricing. Circular if pricing uses the risk score. Kept as an input, with an ablation without it |
| `channel`, `product_type` | yes | Captured at intake |
| `state` | yes | Input. Geography needs a fair-lending review |
| `employment_length` | yes | Self-reported. Inconsistent with age in 27.7% of rows |
| `age` | yes | Not an input (ECOA protected basis). Used for the disparity check only |
| `vintage`, `origination_date` | yes | Used for the split only, not as features |
| `months_on_book`, `days_past_due_current`, `total_payments_to_date` | no | Post-origination |
| `sample_weight` | no | Derived from the label |

Label maturity: all 50,000 rows carry a label. If the data were cut at 2024-09, the 12-month
window would still be open for vintages 2023-10 to 2024-09 (13,243 rows, 296 defaults). The data
shows no censoring: within those vintages, loans under 12 months on book default at 2.44%
(155/6,352), compared with 2.05% (141/6,891) for loans with 12 or more. The labels appear to be
final, and this analysis treats them as final. The snapshot date is an open question.

The default rate falls from 3.21% (2021-22) to 2.27% (2024). The fall also shows up within
risk strata (FICO < 640: 6.00% in 2021, 3.59% in 2024), so it is a level shift, not a change
in borrower mix. Expect calibration drift in the out-of-time test.

## Split protocol

- `data_version`: `enhanced-v1`
- `split_id`: `oot-v1`: train/CV 202101-202206, validation 202207-202212, gap 2023 (scored once
  as a secondary out-of-time check), test 202401-202409.
- `cv_scheme`: `expanding-quarter-4`: expanding window by calendar quarter inside 202101-202206.
  Fold k trains on quarters 1..k+1 and validates on quarter k+2, for k = 1..4 (4 folds,
  minimum 6 months of training). Minimum 50 positives in each validation fold.
- `metric`: ROC AUC (`roc_auc`), unweighted.
- `direction`: max
- Test set: vintages 202401-202409, scored once after calibration and cutoff are frozen. The
  12-month gap (2023) before it matches production: on a 2024-01 decision date, the latest
  vintages with a full 12-month outcome are from 2022-12.
- Validation set: vintages 202207-202212, used for calibration, cutoff and the top-two
  comparison. There is no gap between CV rows and the validation set, or between CV folds.
  This is a deliberate deviation from a label-horizon gap: rows are independent loans and no
  feature uses another row's label, so overlapping outcome windows cannot leak labels.
  The gap is kept where it makes the test realistic.
- `cv_se` computation: standard deviation of the 4 fold scores divided by sqrt(4). Expanding
  folds share training rows, so this understates the error. Close calls are settled with the
  paired bootstrap on the validation set.
- Segment size threshold: segments with fewer than 1,000 rows or 30 defaults are listed, not
  scored.

## Leakage check

- Verdict: CLEAN for the library code (2026-11-18). No blocking leak. Preprocessing is fit
  inside each fold, and no selectable feature set contains a post-origination, label-derived
  or protected column.
- Findings and fixes (2026-11-18):
  1. CV folds and the validation window have no label-horizon gap. This is deliberate (see
     Split protocol). Shared calendar conditions can make CV and validation AUC slightly
     optimistic for every candidate alike. Only the test AUC, which has the 12-month gap, is
     reported as the expected performance.
  2. The `leaky` feature set (ablation only) could be passed like any other. Fixed:
     `build_model` refuses it unless `allow_leaky=True`, the experiment script marks its row
     `selectable = no`, and selection only considers selectable rows.
  3. The data audit's descriptive statistics used all 50,000 rows, including the test window.
     Nothing was fit or tuned from them. The decisions (placeholder to NaN plus flag,
     monotone directions, fixed category levels) come from domain knowledge, not from
     test-set results.
  4. The no-leakage regression test now permutes the target, weight, post-origination,
     protected and time columns for every selectable feature set and checks that the
     features do not change.
- The experiment script computes folds on the training window only and asserts that the
  latest fold vintage is no later than 202206. Gap and test rows are not loaded during
  tuning or selection.

## Baselines

| Run | CV AUC | Validation AUC | Note |
|---|---|---|---|
| R00-base-rate | 0.5000 | 0.5000 | Constant training default rate |
| R01-fico-only | 0.6291 (se 0.0104) | 0.6479 | Binned FICO plus missing flag, logistic regression |

## Candidates and selection rule

- Stated on: 2026-11-18, before any candidate ran.
- Primary metric: CV mean ROC AUC (unweighted) under `expanding-quarter-4`.
- Hard constraints: per-decision reasons available (coefficients or SHAP); monotone
  relationship to FICO and DTI where the model allows it; scoring a single application in
  under 100 ms on a laptop CPU.
- Rule: among models that meet the constraints, the simplest within one `cv_se` of the best
  (simplest: lower tier, then fewer features).
- Candidates and tier: tier 0 base rate and FICO-only logistic regression; tier 1 L2
  logistic regression on quantile-binned numerics (one-hot bins, a scorecard-style model that
  fits the step-shaped relationships the audit found), missing flags and one-hot categories;
  tier 2 LightGBM with monotone constraints.

## Leaderboard summary

- Leaderboard: `reports/leaderboard.csv` (8 runs, one manifest each under `runs/`).
- Check result: 0 errors, 1 note. The note is `outside-one-se`: the best `cv_mean` belongs to
  R07-lgbm-leaky (1.000). That run is the leakage ablation, marked `selectable = no`, and is
  excluded by design. Among selectable runs, the best is R03-lgbm (0.7074, se 0.0070), and the
  selected R05 (0.7013) is within one standard error of it.
- Selected run: R05-lgbm-no-apr.
- Convergence evidence: both tuned families stopped on the plateau rule (window 30, minimum
  delta = cv_se of the default configuration: 0.0175 for logistic regression, 0.0131 for
  LightGBM). Each was rerun with seeds 1 and 2 and double the window and cap:
  - Logistic regression: 0.6812 on all three seeds.
  - LightGBM: 0.7074, 0.7038 and 0.7074. The spread is 0.0036, below the best run's cv_se of
    0.0070.

  Both count as converged at the resolution of the selection rule. Curves are in
  `runs/R02-logreg/trials*.csv` and `runs/R03-lgbm/trials*.csv`, plotted in notebook section 6.2.
- Ablations, all with R03's parameters:

| Ablation | CV AUC | Difference | Paired per-fold difference |
|---|---|---|---|
| Leakage columns added (R07) | 1.000 | +0.293 | - |
| No APR / payment_to_income, LightGBM (R05) | 0.7013 | -0.0061 | lower on 4/4 folds, se 0.0021 |
| No APR, logistic (R04) | 0.6834 | +0.0022 vs R02 | - |
| `sample_weight` as training weight (R06) | 0.7042 | -0.0032 | lower on 4/4 folds, se 0.0007 |

## Calibration and cutoff

- Calibration method and split: Platt scaling (sigmoid) on the selected model's scores, fit on
  validation 202207-202212 (6,693 loans, 214 defaults). The base model is frozen, not refit.
- Cutoff: break-even PD 0.1429 = margin / (margin + LGD) with assumed margin 10% and LGD 60%.
  It approves 99.0% of validation loans. Cutoff table on calibrated validation scores:

| Approval rate | PD cutoff | Bad rate approved | Bad rate declined | Share of bads declined |
|---|---|---|---|---|
| 80% | 0.0396 | 2.17% | 7.32% | 45.8% |
| 85% | 0.0438 | 2.37% | 7.87% | 36.9% |
| 90% | 0.0513 | 2.67% | 7.91% | 24.8% |
| 95% | 0.0748 | 2.89% | 8.96% | 14.0% |

- Calibration evidence:
  - On validation (in-sample for the calibrator): Brier 0.03054 raw, 0.03059 calibrated; ECE
    0.0120 raw, 0.0086 calibrated.
  - Top validation decile: 9.0% predicted vs 7.9% observed after calibration (5.6% raw). Platt
    scaling over-stretches the top of the scale.
  - On the gap year, the first out-of-sample check: ECE 0.0088, mean PD 3.20% vs 2.57% observed.
- Frozen on: 2026-11-18, before the test window was scored.

## Test result

- Selection: R05-lgbm-no-apr. Challenger R03-lgbm (with APR).
  - Paired bootstrap on validation AUC (2,000 resamples): champion minus challenger +0.0010,
    95% CI -0.0040 to +0.0061.
  - Paired CV folds: challenger ahead by 0.0061 (se 0.0021).
  - The rule chose the simpler model, and the APR dependency on pricing is a further reason.
- `test_score`: ROC AUC 0.6800 on 202401-202409 (9,984 loans, 227 defaults). Gini 0.360,
  KS 0.309, PR AUC 0.037, Brier 0.0226, ECE 0.0119.
- Interval: stratified bootstrap (1,000 resamples), 95% CI 0.651-0.708.
- Compared with the validation score: validation AUC 0.7195 (CI 0.687-0.751), gap year 0.7124
  (CI 0.689-0.737), test 0.680. The test is lower than validation by 0.040 and the intervals
  overlap. Mean PD on test is 3.21% against 2.27% observed (about 41% over), because default
  rates fell in 2023-24 at constant borrower mix.

## Stability

- Metrics by period and segment: notebook sections 8 and 9.
  - Per-vintage predicted vs observed rates for 2023-2024: observed rates move between 1.6% and
    3.3% while mean PD stays at 3.0-3.3%.
- Drift of features and score (PSI):
  - Score against the training window: validation 0.0014, gap 0.0006, test 0.0007.
  - Every numeric input, train vs test: below 0.002 (largest is `employment_length`, 0.0019).
  - The 2024 change is in outcomes, and population measures cannot see it.
- Segments below the size threshold (not scored on test): channel Mail; product Business;
  states CA, FL, GA, IL, NY; FICO bands 650-699 and 700-749.
- Weak segments (test AUC below 0.680 by more than the segment's bootstrap SE):
  - FICO < 650: AUC 0.540 (3,196 loans, 121 defaults).
  - Channel Direct: AUC 0.645 (4,968 loans, 114 defaults), vs Partner 0.727.

## Explanations

- Importance method: exact TreeSHAP from LightGBM (`pred_contrib`), mean absolute contribution
  on validation. Order: FICO, DTI, income, loan-to-income, utilization, inquiries, then small
  contributions.
- Stability across folds: Spearman correlation of importance ranks between the four CV fold
  models is 0.968 or more (median 0.982). The top three features are in the same order on every
  fold.
- Error analysis by segment:
  - The model ranks poorly inside the FICO < 650 band. FICO's contribution is flat below 650
    (a step), so ranking there relies on weaker features.
  - The Direct channel ranks worse than Partner.
- Reasons for an individual decision: yes. The largest positive SHAP contributions give the
  adverse-action reasons, and monotone constraints keep their direction consistent. The
  notebook shows one example.
  - Platt calibration is monotone, so it does not change the ranking of reasons.

## Sensitive attributes

| Attribute | May be an input | Reason | Checked outcome |
|---|---|---|---|
| age | no | ECOA/Reg B protected basis | Test approval rate at cutoff 98.8-99.5% across five age bands; mean PD 3.1-3.3%; observed 2.0-2.6% |
| state | yes | Weak risk signal (CA 2.46% to GA 3.23%) | Needs fair-lending review as a geographic proxy; state segments too small to score on test except Other and TX |
| employment_length | yes | Self-reported; inconsistent with age in 27.7% of rows | Not capped by age, to keep age out of the model |

## Packaging and inference contract

- Artifact: `models/pd-lgbm-no-apr-v1/`, containing:
  - `model.joblib`: the calibrated pipeline.
  - `metadata.json`: candidate, parameters, split protocol, data SHA-256, threshold, metrics,
    library versions (Python 3.12, scikit-learn 1.9.1, LightGBM 4.7.0, pandas 3.1.1, numpy
    2.5.5) and the model file's SHA-256, which is checked on load.
- Pipeline: one object from raw application columns to calibrated PD. It runs placeholder
  cleaning, ratios, categories, monotone LightGBM and Platt calibration.
- Input schema: the 13 application columns in `metadata.json` (`input_columns`).
  `score_applications` raises `SchemaError` on a missing column, and an unseen category is
  logged and treated as missing.
- Environment: `uv.lock`, seed 0 everywhere, data SHA-256 in the metadata.
  `tests/test_artifact.py` reloads the artifact and reproduces the stored test predictions
  (`reports/test_predictions.csv`).
- Refit for deployment: none yet. Recommended: refit on all mature vintages (through 2023) as
  v2, with its own calibration check on the most recent mature vintages.

## Deployment and monitoring

- Mode and latency budget: online scoring at application time. One pipeline call scores an
  application in milliseconds, inside a 100 ms budget. Nightly batch rescoring of the pipeline
  population feeds monitoring.
- Rollout: shadow mode for one to two quarters (score, log, no decisions), then a
  champion-challenger split (for example 90/10) against the current policy, compared on early
  delinquency.
- Rollback: if the challenger's 30+ DPD rate at month 3 exceeds the champion's by more than
  0.5 percentage points on at least 2,000 booked loans, or if a schema or score-distribution
  alert fires, route all traffic back to the previous model version. The previous artifact
  stays deployed alongside.
- Monitors:
  - Input PSI per feature and score PSI against the training window: investigate at 0.1, act
    at 0.25.
  - Missing-flag rates.
  - Unseen-category counts.
  - Early-delinquency rates (30+ DPD at months 3 and 6) by vintage.
  - Predicted vs observed default rate by vintage as 12-month labels mature.
  - AUC on matured vintages.
- Retraining trigger: any of
  - observed/predicted default ratio outside 0.8-1.25 for two consecutive matured quarters
    (recalibrate);
  - matured-vintage AUC below 0.65 (retrain);
  - feature or score PSI above 0.25 (investigate, then retrain).

  At the time of the test, the ratio is 0.71 (2.27% / 3.21%), so recalibration is already due.

## Limitations and open questions

- Known weak segments or periods:
  - FICO < 650 (test AUC 0.54).
  - Direct channel (0.645).
  - 2024 vintages, where the PD level is about 41% too high.
- Open questions, each with the default used here:
  1. Snapshot date and label maturity for 202310-202409. Default: labels treated as final.
  2. Whether `loan_amount`, `term` and `apr` are requested or approved and priced values.
     Default: requested, and APR left out of the selected model.
  3. What `sample_weight` represents. Default: label-derived, never a feature.
  4. Whether declined applications are available. Default: none, so no reject inference.
  5. Real LGD and margin for the cutoff. Default: 60% and 10%, illustrative only.
- Other limitations:
  - Platt calibration over-stretches the top decile. It was kept because it was frozen before
    the test.
  - CV and validation have no label-horizon gap, so they may be slightly optimistic. Only the
    test AUC is quoted as expected performance.
  - The data looks synthetic (independent uniform features, generated post-origination
    columns).
- Reviewer findings: see the notes at the end of this card.
