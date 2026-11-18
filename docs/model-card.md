# Model card: 12-month probability of default at origination

**Summary.** v1 (R05) is a monotone LightGBM on 12 application features. It ranks out of time
(test AUC 0.680, 95% CI 0.651-0.708), but its PD loses to a constant on test Brier (skill
-1.5%). The post-hoc v1.1 (R10) recalibrates it to +0.7%. At the assumed economics the
break-even cutoff declines nobody, so the score's use is ranking: pricing tiers, limits and an
approval-rate policy. On the 2023 gap year, v2 (R13, tuned on log loss) is better on ranking, Brier, ECE and the bottom decile, and worse on mean level and the top decile
than v1.1 (Brier skill +1.6% vs +1.3%; bottom decile 0.90% vs 1.87% predicted against 0.38%
observed). It misses its pre-registered bar, so v1.1 stays the recommended model; under the
stratified bootstrap used elsewhere in this card, v2 would pass. Run history:
`reports/leaderboard.csv` and `runs/*/manifest.md`.

## Decision, label and data

- Decision: approve or decline at application, then the price tier. Prediction point: the
  application decision; nothing observed after it may be an input.
- Costs: not given. Assumed margin 10% of principal and LGD 60%, so break-even PD = 0.10 /
  0.70 = 14.3%. Cutoffs are also tabulated by approval rate so the business can choose.
- Constraints: declines need reasons (ECOA/Reg B adverse action), so per-decision
  attributions are required. Age is a protected basis and is not an input.
- Label: `default_12m`, default within 12 months of origination, as provided. Only booked loans
  have labels; the model would score all applicants (no reject inference).
- Data: `data/credit_risk_data_enhanced.csv` (`enhanced-v1`, SHA-256 in the artifact
  metadata), 50,000 loans, vintages 2021-01 to 2024-09.

| Issue | Count | Decision |
|---|---|---|
| `fico_score` 99999, `income` -1, `inquiries_last_6m` 99 | 2,500 / 1,500 / 1,000 | NaN plus missing flag; missing rows default at 4.16% / 5.93% / 6.10% vs about 2.8% |
| `days_past_due_current`, `total_payments_to_date`, `months_on_book` | all rows | Dropped: post-origination. DPD and payments / loan amount separate the label perfectly |
| `sample_weight` | all rows | Label-derived (0.80-1.20 for non-defaults, 2.40-3.60 for defaults). Never a feature; tested as a training weight |
| `employment_length` > `age` - 16 | 13,848 | Kept as reported; a cap would bring age into the model |
| `apr` clipped at 3.00 | 4,988 | Kept as data; left out of the selected model (see Selection) |
| Duplicates, NaN, out-of-range values | 0 | None needed |

- Availability: bureau and application fields are known at the decision. `loan_amount` and
  `term` are assumed to be requested values. `apr` is set by pricing, so it may be circular.
  `state` is an input pending fair-lending review. `age` is used only for the disparity check.
- Label maturity: labels are treated as final. If the data were cut at 2024-09, vintages
  2023-10 to 2024-09 (13,243 rows) would be immature. Their default rates do not fall toward
  the latest vintage (202409 is highest, 3.31%), which is consistent with final labels but does
  not prove it.
- Drift: the default rate falls from 3.21% (2021-22) to 2.27% (2024) while input distributions
  stay stable (PSI below 0.002). It falls below FICO 650 and above 750; the 650-749 band does
  not fall. The cause is not established: economic conditions, servicing changes or immature
  labels could each contribute.

## Split protocol and use of held-out windows

- `oot-v1`: train/CV 202101-202206 (`expanding-quarter-4`: fold k trains on quarters 1..k+1,
  validates on quarter k+2; at least 50 defaults per fold), validation 202207-202212
  (calibration, cutoff, champion vs challenger), gap 2023, test 202401-202409. The 2023 gap
  matches production: at a 2024-01 decision, the latest mature vintages are from 2022-12.
- No gap between CV folds or before validation. Rows are independent loans and no feature
  uses another row's label, so this cannot leak labels; it can make CV and validation slightly
  optimistic, because they are closer in time to training than production data would be.
- `cv_se` = std of 4 fold scores / 2. Expanding folds share rows, so it understates the error.
- Only the test AUC is quoted as expected performance. The gap year follows validation with no
  label-horizon gap, so v2's 2023 figures compare models; they do not forecast.
- Looks at the held-out windows, in order:
  1. v1, frozen with its calibrator and cutoff, scored gap and test once.
  2. One pipeline run with isotonic calibration scored test as a demo. It informed nothing.
     Test scoring in the pipeline is now opt-in (`--score_test true`).
  3. R10 (v1.1), registered before it ran, scored gap and test once.
  4. R13 (v2), registered before it ran, scores the gap year once and never loads test.
- EDA used all 50,000 rows, so the 2023-24 fall in default rates was known before the test
  pass. It set one expectation (calibration drift) and no modeling choice.
- Leakage audit: clean. Preprocessing is fit inside each fold; `build_model` refuses the
  leaky ablation feature set unless asked; a regression test permutes the target, weight,
  post-origination, protected and time columns and checks the features do not change.

## Selection

- Rule, stated before any candidate ran: the simplest model within one `cv_se` of the best CV
  ROC AUC (tiers: base rate and FICO-only logistic 0, binned logistic 1, monotone LightGBM 2;
  then fewer features). Constraints: per-decision reasons, monotone in FICO and DTI, under
  100 ms per application.

| Run | Model | CV AUC (se) | Validation AUC | Note |
|---|---|---|---|---|
| R00 | Base rate | 0.500 | 0.500 | |
| R01 | FICO only, binned logistic | 0.629 (0.010) | 0.648 | |
| R02 / R04 | Binned logistic, with / without APR | 0.681 / 0.683 | 0.713 / 0.713 | Search converged (3 seeds reach 0.6812) |
| R03 | LightGBM with APR | 0.707 (0.007) | 0.718 | Best selectable; search unconverged (plateau delta 0.0131 too coarse) |
| **R05** | **LightGBM without APR, R03's parameters** | **0.701 (0.006)** | **0.719** | **Selected (v1): within one se, fewer features** |
| R06 | R03 trained with `sample_weight` | 0.704 | 0.717 | Worse on 4/4 folds; mean PD inflated to 8.7% |
| R07 | R03 plus leaky columns | 1.000 | 1.000 | Leakage ablation, not selectable |
| R08 / R09 | LightGBM fine searches, with / without APR (post-test) | 0.708 / 0.709 | 0.719 / 0.719 | Plateau stop, 3 seeds; APR adds nothing (paired -0.0008, se 0.0017) |
| R11 | LightGBM, library defaults (post-test) | 0.672 | 0.682 | Reference |
| R12 | Supervised-bin scorecard (post-test) | 0.680 | 0.708 | Reference; best point at the grid edge, unconverged |
| R13 | LightGBM without APR tuned on log loss (v2, post-test) | 0.706 (0.007) | 0.719 | CV log loss 0.1365; plateau stop on 3 seeds; see v2 below |

- Champion R05 vs challenger R03 on validation: AUC +0.0010 (paired bootstrap 95% CI -0.0040
  to +0.0061). APR is set by pricing, which is a further reason to leave it out.
- Leaderboard check: 0 errors, 1 note (`outside-one-se`, caused by the non-selectable leaky
  ablation R07).
- Post-test runs are never selectable. R08 and R09 show R05 is about one `cv_se` under-tuned,
  but R09 is itself lightly fitted (88 three-leaf trees at learning rate 0.0135).

## Calibration

- v1: Platt scaling on validation (214 defaults), frozen before the test pass. The pipeline has
  no `decision_function`, so scikit-learn fit the sigmoid to the raw probability (a = -48.5,
  b = 5.10), not its log-odds. That map stretches the top (validation top decile 5.6% raw to
  9.0%, against 7.9% observed) and barely lowers the bottom (1.8% raw to 1.45%, against 0.3%).
  No held-out raw-vs-calibrated comparison was made before freezing.
- v1.1 (R10, post-hoc): same base model; calibrator chosen on validation only by two
  time-ordered folds and the one-SE rule on pooled Brier.

| Method | Out-of-fold Brier | Difference vs best | Paired SE | Eligible |
|---|---|---|---|---|
| Log-odds shift | 0.030531 | 0.000050 | 0.000142 | yes (chosen) |
| Log-odds Platt | 0.030523 | 0.000041 | 0.000084 | yes |
| Sigmoid on probability (v1) | 0.030659 | 0.000177 | 0.000082 | no |
| Isotonic | 0.030481 | 0 | - | yes (lowest) |

- With 214 defaults the rule had no power to prefer a richer shape and took the one-parameter
  shift, which cannot change shape.
- Cutoffs (v1.1, validation): break-even 14.3% approves everyone; the highest v1.1 PD on test
  is 11.3%. At 80% approval the cutoff is PD 0.0403 and the declined 20% default at 7.32%,
  45.8% of all defaults.

## Results

Test window 202401-202409 (9,984 loans, 227 defaults); gap year 2023 (13,261 loans, 341
defaults). Brier skill is against a constant at the training default rate (3.21%).

| | v1 (frozen) | v1.1 (post-hoc) | Raw scores |
|---|---|---|---|
| Test AUC | 0.680 (0.651-0.708) | same | same |
| Test Brier | 0.02264 | 0.02216 | 0.02214 |
| Test Brier skill (95% CI) | -1.5% (-2.5% to -0.5%) | +0.7% (+0.4% to +1.0%) | +0.7% |
| Gap Brier skill | +0.6% | +1.3% | +1.3% |
| Test mean PD (observed 2.27%) | 3.21% | 3.20% | 3.13% |
| Test top decile, predicted vs observed | 9.1% vs 3.9% | 5.7% vs 3.9% | 5.6% vs 3.9% |
| Test bottom decile, predicted vs observed | 1.45% vs 0.10% | 1.87% vs 0.10% | 1.82% vs 0.10% |

- Ranking: validation AUC 0.719 (CI 0.687-0.751), gap 0.712 (CI 0.689-0.737), test 0.680. The
  intervals overlap; with 227 defaults the drop fits sampling noise plus drift. The lowest-scored
  20% of test loans default at 0.20% against 2.27%. Gini 0.36, KS 0.31.
- v1's calibrator caused its loss to the constant: the raw scores do as well as v1.1.
- What remains: calibrators fit on the test labels themselves (diagnostics, not bounds) reach
  1.0% skill with a level shift alone and 1.4% with isotonic. About half of the remaining error
  is the 2024 level and half is shape at the bottom of the scale. R10's registered expectation
  that the bottom would improve was not met.
- v1's break-even cutoff declined 97 test loans, of which 4 defaulted (4.1%, far below
  break-even). v1.1 declines no one.

## v2 (R13): tuned on log loss

Registered and committed before it ran; development used data through 2022-12 only; the gap
year was scored once, and the 2024 window was never used or scored (the data loader reads the
whole file and drops 2024 at once; one whole-file default-rate log line is written before the
drop).

- Search: TPE on CV log loss, same space as R03. All three seeds stopped on the plateau (81,
  106 and 69 trials); best per seed 0.13649 to 0.13683, a spread of 0.00034 against a CV se
  of 0.0024: converged in CV log loss, but the parameters are not identified (seed bests range
  from 4 leaves at learning rate 0.010 to 25 leaves at 0.022, 151 to 288 trees). Chosen: 288
  trees, 4 leaves, learning rate 0.0103. v1 is 64 trees of 18 leaves at 0.011: the same total
  leaf count, shallower trees, and about four times the boosting budget (trees x rate).
- Scale check: folds 2021Q4 to 2022Q2 passed (slope 0.93 to 1.01, shift at most 0.06). Fold
  2021Q3 failed (slope 0.894, shift 0.103), so its out-of-fold rows were left out.
- Calibrator: leave one quarter out over 5 quarters (550 defaults). The log-odds shift had the
  lowest pooled Brier and was chosen; log-odds Platt was within one SE, isotonic was not.

| Gap year 2023 | v1 | v1.1 | v2 |
|---|---|---|---|
| AUC | 0.712 | 0.712 | 0.719 |
| Brier skill (95% CI) | +0.6% (-0.3% to +1.5%) | +1.3% (+1.0% to +1.5%) | +1.6% (+1.0% to +2.1%) |
| ECE | 0.0088 | 0.0100 | 0.0070 |
| Mean PD (observed 2.57%) | 3.20% | 3.20% | 3.27% |
| Bottom decile, predicted vs observed | 1.45% vs 0.38% | 1.87% vs 0.38% | 0.90% vs 0.38% |
| Top decile, predicted vs observed | 9.0% vs 6.2% | 5.7% vs 6.2% | 7.8% vs 6.4% |

- Paired bootstrap, v2 minus v1.1 (2,000 resamples): Brier -0.000076 (95% CI -0.000158 to
  +0.000004), AUC +0.0061 (-0.0002 to +0.0124).
- Decision rule (registered): v2 replaces v1.1 only if the Brier CI is entirely below zero and
  the AUC difference is at least -0.01. The CI upper bound is +0.000004, so the rule keeps
  v1.1. Of the registered expectations, less compression at the bottom was met and higher
  gap-year skill was met on the point estimate only. "A larger or more flexible model" was
  only partly met: more boosting, but not more leaves.
- The verdict is fragile. The registered script resampled rows without stratifying, while every other comparison in this repository uses a bootstrap stratified by outcome. Under that stratified bootstrap the interval is -0.000140 to -0.000010 (seeds 0-2 agree), entirely below zero, and v2 would pass. The registered result stands because the committed code defines the rule; the gap between v1.1 and v2 is small either way (Brier skill +1.3% vs +1.6%), and neither prices. The stratified AUC interval (+0.0001 to +0.0124) also excludes zero.
- Deciles are formed per model, so the bottom and top rows hold different loans. On v1.1's
  own bottom decile v2 predicts 0.91% (observed 0.38%); on v1.1's top decile it predicts 7.65%
  (observed 6.18%), so the comparison holds on identical loans.
- Break-even cutoff on the gap year: v1 declined 119 loans (11 defaults), v1.1 none, v2 13
  loans (0 defaults).
- The level problem remains: v2's mean PD is 3.27% against 2.57% observed in 2023.
- Artifact: `models/pd-lgbm-no-apr-v2/` (same 12 input columns); gap-year PDs of v1, v1.1 and
  v2 in `reports/gap_predictions_v2.csv`. v2 has no test-window number, and none is planned.

## Stability, explanations, sensitive attributes

- PSI against the training window: score 0.0014 (validation), 0.0006 (gap), 0.0007 (test);
  every input below 0.002. These cover marginal distributions of booked loans only; the 2024
  change shows up in outcomes only.
- Weak segments on test: FICO < 650 (AUC 0.540, 121 defaults), Direct channel (0.645 vs
  Partner 0.727). Segments under 1,000 rows or 30 defaults are listed in the notebook, not
  scored.
- Explanations: exact TreeSHAP. Order: FICO, DTI, income, loan-to-income, utilization,
  inquiries. Fold-to-fold Spearman of importance ranks at least 0.968. The largest positive
  contributions give adverse-action reasons; every calibrator here is monotone, so reasons
  do not change.

| Attribute | Input | Checked outcome |
|---|---|---|
| age | no (ECOA protected basis) | Spearman(age, PD) 0.003 on test; approval by age band 79.1-81.4% at the 80% policy, 88.1-91.3% at 90%; predicted/observed by band 1.25-1.60 (37-67 defaults per band) |
| state | yes | Weak signal (CA 2.46% to GA 3.23%); needs fair-lending review as a geographic proxy |
| employment_length | yes | Inconsistent with age in 27.7% of rows; not capped, to keep age out |

## Packaging

- `models/pd-lgbm-no-apr-v1/`, `-v1.1/` and `-v2/`: `model.joblib` (one object
  from the 12 raw `no_apr` columns to PD) and `metadata.json` (candidate, parameters, split
  protocol, data SHA-256, threshold, metrics, library versions, input columns, model-file
  SHA-256 checked on load). APR is not required.
- `score_applications` raises `SchemaError` on a missing column; an unseen category is logged
  and treated as missing. Latency: median 2.1 ms, p99 under 3 ms per application.
- Reproducibility: `uv.lock`, seed 0. `tests/test_artifact.py` reproduces
  `reports/test_predictions.csv` from the v1 artifact; the notebook does the same for v1.1.

## Deployment and monitoring

- Online scoring at application, shadow mode for one to two quarters, then a 90/10
  champion-challenger split compared on early delinquency. Roll back if the challenger's
  month-3 30+ DPD rate exceeds the champion's by more than 0.5 points on at least 2,000
  loans, or on a schema or score-distribution alert.
- Monitors: input and score PSI (investigate at 0.1, act at 0.25), missing-flag rates,
  unseen categories, 30+ DPD at months 3 and 6 by vintage, observed vs predicted default rate
  as labels mature, AUC on matured vintages.
- Triggers: observed/predicted outside 0.8-1.25 for two matured quarters (recalibrate; on
  this test it is 0.71, so it is already due), matured AUC below 0.65 (retrain), PSI above
  0.25 (investigate).
- At deployment, refit on all mature vintages and rerun the calibrator selection with the
  latest mature vintages added. The registered rule keeps v1.1. Choosing v2 instead would rest
  on a single 2023 evaluation whose Brier and AUC intervals both include zero. No window is
  left here to test the refit.

## Limitations and open questions

- The v1 PD loses to a constant on test Brier. v1.1 removes the top-end stretch but not the
  bottom-of-scale miss, and its test figures are a second look at the test window.
- The test window is not pristine: outcome rates were seen in EDA and it has had three looks.
- The data looks synthetic (independent uniform features, generated post-origination columns).
- Open questions, with the default used: snapshot date and label maturity (labels final);
  whether `loan_amount`, `term` and `apr` are requested or priced values (requested, APR left
  out); what `sample_weight` means (label-derived, never a feature); whether declined
  applications exist (none, no reject inference); real LGD and margin (60% and 10%,
  illustrative).
- The brief lists `scripts/generate_enhanced_data.py`, which was not included. It would settle
  the snapshot date, the meaning of `sample_weight` and whether the 2023-24 fall in default rates
  is by design; it has been requested.
