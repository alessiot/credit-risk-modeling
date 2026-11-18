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

Filled at stage 5.

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

Filled at stage 6.

## Calibration and cutoff

Filled at stage 7.

## Test result

Filled at stage 7.

## Stability

Filled at stage 8.

## Explanations

Filled at stage 8.

## Sensitive attributes

Filled at stage 8.

## Packaging and inference contract

Filled at stage 9.

## Deployment and monitoring

Filled at stage 10.

## Limitations and open questions

Filled at stage 11.
