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
| `days_past_due_current`, `total_payments_to_date` | all rows | Dropped: recorded after origination. DPD separates the label perfectly; payments to date / loan amount does too (the raw column has AUC 0.911) |
| `sample_weight` | all rows | Not a feature: 0.80-1.20 for every non-default, 2.40-3.60 for every default. Tested only as a training weight (ablation) |
| `months_on_book` | all rows | Dropped: post-origination. Uniform 0-24 in every vintage, so it matches no snapshot date; no outcome signal (AUC 0.508) |
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

Label maturity: all 50,000 rows carry a label, and this analysis assumes the labels are final.
If the data were cut at 2024-09, the 12-month window would still be open for vintages 2023-10
to 2024-09 (13,243 rows, 296 defaults). Observed default rates in the test window do not fall
toward the most recent vintages: 202409 has the highest rate, 3.31%. That is consistent with
final labels but does not prove it. `months_on_book` cannot settle the question, because it
matches no snapshot date. The snapshot date is an open question.

The default rate falls from 3.21% (2021-22) to 2.27% (2024), while input distributions stay
stable (PSI below 0.002). The fall is concentrated below FICO 650 (5.85% in 2021, 3.79% in
2024) and above 750 (1.96% to 1.47%). The 650-749 band does not fall. The cause is not
established: economic conditions, servicing changes, or immature labels could each contribute.
Expect calibration drift in the out-of-time test.

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
  3. The data audit and the notebook's EDA used all 50,000 rows, including 2023-24 outcomes:
     default rates by year and within FICO bands, placeholder rates, and decile steps. So the
     2023-24 fall in default rates was known before the test pass. It set one expectation
     (calibration drift) and no modeling choice. The data decisions (placeholder to NaN plus
     flag, monotone directions, fixed category levels) follow domain practice. The test window
     is therefore not pristine.
  4. After the test pass, one pipeline run with isotonic calibration was scored on the test
     window to demonstrate run comparison. It informed no decision. The default run comparison
     now shows validation and gap-year metrics only.
  5. The no-leakage regression test permutes the target, weight, post-origination,
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
- Convergence evidence:
  - Logistic regression (R02) is converged. Seeds 0, 1 and 2 (seeds 1 and 2 with double
    window and cap) all reach 0.6812.
  - LightGBM (R03) is unconverged.
    - Its plateau rule used a minimum delta of 0.0131 (cv_se of the default configuration),
      about twice the selection margin it informs.
    - The recorded run found its best on its last trial, and the reruns stopped at exactly 61
      trials (fixed budgets in effect).
    - The R03 manifest records this as a dated deviation.
  - **Sensitivity searches, after the test pass** (CV only, not selectable). Minimum delta
    0.001, window 60, cap 400, seeds 0-2. All six stopped on the plateau, below the cap.
    - R08 (with APR): seed bests 0.7074, 0.7038, 0.7079; best 0.7079 (cv_se 0.0065).
    - R09 (no APR, tuned directly): 0.7087, 0.7070, 0.7076; best 0.7087 (cv_se 0.0067).
    - Under the rule, the no-APR set has the best CV AUC, so the family choice holds.
    - The frozen R05 configuration (0.7013) is about one cv_se below R09.
- Curves: `runs/*/trials*.csv`, plotted in notebook sections 6.2 and 6.3.
- Ablations, all with R03's parameters:

| Ablation | CV AUC | Difference | Paired per-fold difference |
|---|---|---|---|
| Leakage columns added (R07) | 1.000 | +0.293 | - |
| No APR / payment_to_income, LightGBM (R05) | 0.7013 | -0.0061 | lower on 4/4 folds, se 0.0021; an artifact of reusing R03's parameters: tuned separately, R09 vs R08 is +0.0008 (paired se 0.0017) |
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
  - On the gap year, the first out-of-sample check: ECE 0.0088, mean PD 3.20% vs 2.57%
    observed, top decile 9.0% vs 6.2%.
  - No held-out comparison of raw vs calibrated scores was made before freezing. In hindsight,
    the calibrator shape is a weakness.
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
  (CI 0.689-0.737), test 0.680. Validation lacks a label-horizon gap and is probably optimistic.
  The test interval (0.651-0.708) is the expected range for next-year ranking.
- Level and shape:
  - Mean PD on test is 3.21% against 2.27% observed (ratio 1.41).
  - The top decile predicts 9.1% vs 3.9% observed.
  - Both the 2023-24 fall in default rates and the Platt calibrator's top-end stretch
    contribute.
- Cutoff on test: the break-even cutoff declines 97 loans (1.0%), of which 4 default (4.1%,
  far below break-even). In the gap year it declines 119 loans, 11 default (9.2%). With this
  calibration, the cutoff declines profitable loans and should not be used before
  recalibration.

## Stability

- Metrics by period and segment: notebook sections 8 and 9.
  - Per-vintage predicted vs observed rates for 2023-2024: observed rates move between 1.6% and
    3.3% while mean PD stays at 3.0-3.3%.
- Drift of features and score (PSI):
  - Score against the training window: validation 0.0014, gap 0.0006, test 0.0007.
  - Numeric inputs, train vs test: all below 0.002 (largest is `employment_length`, 0.0019).
  - Categorical inputs: channel 0.0001, product 0.0002, state 0.0011.
  - These cover booked loans and marginal distributions only. Within that scope the
    population did not move, and the 2024 change shows up in outcomes only.
- Segments below the size threshold (not scored on test): channel Mail; product Business;
  states CA, FL, GA, IL, NY; FICO bands 650-699 and 700-749; FICO missing (460 loans, 12
  defaults).
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
| age | no | ECOA/Reg B protected basis | No evidence of rank disparity: Spearman(age, PD) 0.003 on test; approval by band 79.1-81.4% at the 80% policy, 88.1-91.3% at the 90% policy. Predicted/observed by band 1.25-1.60 (37-67 defaults per band, noisy) |
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
- Input schema: the 12 raw application columns of the `no_apr` feature set, stored in
  `metadata.json` (`input_columns`). APR is not required.
  - `score_applications` raises `SchemaError` on a missing column.
  - An unseen category is logged and treated as missing.
- Latency: median 2.1 ms, p99 under 3 ms per single application (200 calls on a laptop CPU; the
  p99 varies by run).
- Environment: `uv.lock`, seed 0 everywhere, data SHA-256 in the metadata.
  `tests/test_artifact.py` reloads the artifact and reproduces the stored test predictions
  (`reports/test_predictions.csv`).
- Refit for deployment: none yet. Recommended v2: R09's configuration refit on all mature
  vintages (through 2023), with recalibration and a fresh out-of-time check on the most recent
  mature vintages.

## Deployment and monitoring

- Mode and latency budget: online scoring at application time. Measured at a median of 2.1 ms
  per application, inside a 100 ms budget. Nightly batch rescoring of the pipeline
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

  At the time of the test, the observed/predicted ratio is 0.71 (2.27% / 3.21%), so
  recalibration is already due.

## Limitations and open questions

- Known weak segments or periods:
  - FICO < 650 (test AUC 0.54).
  - Direct channel (0.645).
  - 2024 vintages, where the mean PD is 41% above the observed rate and the top decile more
    than twice.
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
  - The frozen model is under-tuned relative to the converged search (R05 0.7013 vs R09 0.7087
    CV AUC), because the original LightGBM stopping rule was too coarse.
  - The test window is not pristine: 2023-24 outcome rates were examined in EDA, and an
    isotonic pipeline variant was scored on it after the freeze. Neither changed a decision.
  - CV and validation have no label-horizon gap, so they may be slightly optimistic. Only the
    test AUC is quoted as expected performance.
  - The data looks synthetic (independent uniform features, generated post-origination
    columns).
