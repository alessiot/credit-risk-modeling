import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from credit_risk.config import FEATURE_SETS, SELECTABLE_FEATURE_SETS
from credit_risk.models import (
    CALIBRATION_METHODS,
    LogitCalibrated,
    MonotoneLGBM,
    PrecalibratedModel,
    ScoreCalibrator,
    SupervisedBinner,
    build_model,
    calibrate,
    cross_validate,
    fit_model,
    positive_scores,
)
from credit_risk.splits import expanding_quarter_folds

LGBM = {"n_estimators": 30, "num_leaves": 7, "min_child_samples": 20}


@pytest.fixture(scope="module")
def xy(raw_sample):
    return raw_sample.drop(columns="default_12m"), raw_sample["default_12m"]


@pytest.mark.parametrize("family", ["base_rate", "logreg", "lgbm", "scorecard"])
@pytest.mark.parametrize("feature_set", list(FEATURE_SETS))
def test_builds_and_fits(xy, family, feature_set):
    feats, y = xy
    model = fit_model(
        family, feature_set, LGBM if family == "lgbm" else None, feats, y, allow_leaky=True
    )
    p = positive_scores(model, feats)
    assert p.shape == (len(feats),) and np.all((p >= 0) & (p <= 1))


def test_unknown_family_and_set():
    with pytest.raises(ValueError, match="family"):
        build_model("svm")
    with pytest.raises(ValueError, match="feature set"):
        build_model("lgbm", "nope")


def test_pipeline_takes_raw_placeholders(xy):
    feats, y = xy
    assert (feats["fico_score"] == 99999).any()
    model = fit_model("lgbm", "full", LGBM, feats, y)
    assert isinstance(model.named_steps["model"], MonotoneLGBM)
    assert model.named_steps["model"].booster_.num_trees() > 0


def test_lgbm_monotone_in_fico(xy):
    feats, y = xy
    model = fit_model("lgbm", "full", {"n_estimators": 100, "num_leaves": 15}, feats, y)
    row = feats[feats["fico_score"] != 99999].iloc[[0]]
    grid = pd.concat([row] * 40, ignore_index=True)
    grid["fico_score"] = np.linspace(300, 850, 40).astype(int)
    p = positive_scores(model, grid)
    assert np.all(np.diff(p) <= 1e-12)


def test_cross_validate(xy, raw_sample):
    feats, y = xy
    folds = expanding_quarter_folds(raw_sample["vintage"], y, n_folds=3, min_positives=1)
    w = raw_sample["sample_weight"]
    res = cross_validate("logreg", "full", {"C": 0.5}, feats, y, folds, sample_weight=w)
    assert len(res.fold_scores) == 3
    unweighted = cross_validate("logreg", "full", {"C": 0.5}, feats, y, folds)
    assert unweighted.fold_scores != res.fold_scores
    assert res.cv_mean == pytest.approx(np.mean(res.fold_scores))
    assert res.cv_se == pytest.approx(np.std(res.fold_scores, ddof=1) / np.sqrt(3))
    val = np.concatenate([v for _, v in folds])
    assert res.oof.iloc[val].notna().all()
    assert res.oof.drop(res.oof.index[val]).isna().all()
    base = cross_validate("base_rate", "full", None, feats, y, folds)
    assert base.fold_scores == [0.5] * 3


def test_cross_validate_records_fold_log_loss(xy, raw_sample):
    feats, y = xy
    folds = expanding_quarter_folds(raw_sample["vintage"], y, n_folds=3, min_positives=1)
    res = cross_validate("logreg", "full", {"C": 0.5}, feats, y, folds)
    expected = []
    for tr, va in folds:
        model = fit_model("logreg", "full", {"C": 0.5}, feats.iloc[tr], y.iloc[tr])
        expected.append(log_loss(y.iloc[va], positive_scores(model, feats.iloc[va]), labels=[0, 1]))
    np.testing.assert_allclose(res.fold_log_loss, expected)
    assert res.ll_mean == pytest.approx(np.mean(expected))
    assert res.ll_se == pytest.approx(np.std(expected, ddof=1) / np.sqrt(3))


def test_calibrate_does_not_refit(xy):
    feats, y = xy
    model = fit_model("logreg", "full", None, feats.iloc[:2000], y.iloc[:2000])
    before = positive_scores(model, feats.iloc[2000:])
    cal = calibrate(model, feats.iloc[2000:], y.iloc[2000:], "sigmoid")
    np.testing.assert_array_equal(positive_scores(model, feats.iloc[2000:]), before)
    assert positive_scores(cal, feats.iloc[2000:]).shape == (1000,)
    cal_iso = calibrate(model, feats.iloc[2000:], y.iloc[2000:], "isotonic")
    assert positive_scores(cal_iso, feats.iloc[2000:]).shape == (1000,)


def test_quantile_binner_ties_and_discrete():
    from credit_risk.models import QuantileBinner  # noqa: PLC0415

    x = np.column_stack([np.repeat([36.0, 60.0], 50), np.arange(100.0), np.zeros(100)])
    b = QuantileBinner(n_bins=10).fit(x)
    out = b.transform(x)
    assert out.shape == (100, 2 + 10 + 1)
    assert out.sum(axis=1).tolist() == [3.0] * 100


def test_quantile_binner_no_empty_bin_discrete():
    from credit_risk.models import QuantileBinner  # noqa: PLC0415

    counts = [500, 900, 800, 700, 600, 500, 400, 300, 200, 100, 50]
    col = np.repeat(np.arange(11.0), counts)[:, None]
    out = QuantileBinner(n_bins=10).fit_transform(col)
    assert (out.sum(axis=0) > 0).all()
    skew = np.repeat([0.0, 1.0, 2.0, 3.0], [5000, 300, 200, 100])[:, None]
    assert (QuantileBinner(n_bins=3).fit_transform(skew).sum(axis=0) > 0).all()


def test_quantile_binner_rejects_nan():
    from credit_risk.models import QuantileBinner  # noqa: PLC0415

    x = np.arange(20.0)[:, None]
    with pytest.raises(ValueError, match="NaN"):
        QuantileBinner().fit(np.array([[np.nan], [1.0]]))
    with pytest.raises(ValueError, match="NaN"):
        QuantileBinner().fit(x).transform(np.array([[np.nan]]))


def test_pipeline_predict(xy):
    feats, y = xy
    model = fit_model("lgbm", "full", LGBM, feats, y)
    assert set(np.unique(model.predict(feats))) <= {0, 1}
    assert 0.0 <= model.score(feats, y) <= 1.0


def test_leaky_guard(xy):
    feats, y = xy
    with pytest.raises(ValueError, match="leaky"):
        build_model("lgbm", "leaky")
    with pytest.raises(ValueError, match="leaky"):
        fit_model("logreg", "leaky", None, feats, y)
    folds = [(np.arange(2000), np.arange(2000, 3000))]
    with pytest.raises(ValueError, match="leaky"):
        cross_validate("logreg", "leaky", None, feats, y, folds)
    assert cross_validate("logreg", "leaky", None, feats, y, folds, allow_leaky=True).cv_mean > 0
    assert "leaky" not in SELECTABLE_FEATURE_SETS
    assert set(SELECTABLE_FEATURE_SETS) < set(FEATURE_SETS)


class _FixedProb:
    """Stand-in fitted model that returns a stored probability per row."""

    def predict_proba(self, X):
        p = np.asarray(X["p"], dtype=float)
        return np.column_stack([1 - p, p])


def _score_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(-3.0, 1.0, n)
    y = (rng.random(n) < expit(z)).astype(int)
    return pd.DataFrame({"p": expit(z)}), pd.Series(y), z


def test_calibration_methods_order():
    assert CALIBRATION_METHODS == ("logit_intercept", "logit_platt", "sigmoid", "isotonic")


def test_calibrate_unknown_method(xy):
    feats, y = xy
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate(_FixedProb(), feats, y, "beta")


def test_logit_intercept_matches_observed_mean():
    X, y, z = _score_frame()
    X["p"] = expit(z - 1.0)  # shifted: the true rate is higher than the scores say
    cal = calibrate(_FixedProb(), X, y, "logit_intercept")
    assert isinstance(cal, LogitCalibrated) and cal.slope_ == 1.0
    p = positive_scores(cal, X)
    assert abs(p.mean() - y.mean()) < 1e-9
    assert cal.intercept_ == pytest.approx(1.0, abs=0.2)


def test_logit_platt_recovers_identity_on_calibrated_logodds():
    X, y, _ = _score_frame(n=60000, seed=1)
    cal = calibrate(_FixedProb(), X, y, "logit_platt")
    assert cal.slope_ == pytest.approx(1.0, abs=0.05)
    assert cal.intercept_ == pytest.approx(0.0, abs=0.1)
    assert set(cal.predict(X)) <= {0, 1} and list(cal.classes_) == [0, 1]


@pytest.mark.parametrize("method", ["logit_intercept", "logit_platt"])
def test_logit_calibrated_pickle_and_no_refit(method):
    X, y, _ = _score_frame()
    base = _FixedProb()
    cal = calibrate(base, X, y, method)
    assert cal.estimator is base
    clone = pickle.loads(pickle.dumps(cal))
    np.testing.assert_array_equal(positive_scores(clone, X), positive_scores(cal, X))


def test_logit_calibrated_clips_extremes_and_rejects_single_class():
    X = pd.DataFrame({"p": [0.0, 1.0, 0.5, 0.2]})
    cal = LogitCalibrated(_FixedProb(), "logit_intercept").fit(X, [0, 1, 0, 1])
    assert np.isfinite(positive_scores(cal, X)).all()
    with pytest.raises(ValueError, match="classes"):
        LogitCalibrated(_FixedProb(), "logit_intercept").fit(X, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="unknown logit method"):
        LogitCalibrated(_FixedProb(), "x").fit(X, [0, 1, 0, 1])
    assert logit(1e-6) < logit(0.5)


def test_score_calibrator_intercept_matches_mean_exactly():
    X, y, z = _score_frame()
    p = expit(z - 1.0)
    cal = ScoreCalibrator("logit_intercept").fit(p, y)
    assert cal.transform(p).mean() == pytest.approx(y.mean(), abs=1e-9)
    ref = calibrate(_FixedProb(), pd.DataFrame({"p": p}), y, "logit_intercept")
    assert cal.intercept_ == pytest.approx(ref.intercept_)


def test_score_calibrator_platt_matches_logit_calibrated():
    X, y, z = _score_frame(seed=2)
    p = expit(0.6 * z - 0.5)
    cal = ScoreCalibrator("logit_platt").fit(p, y)
    ref = calibrate(_FixedProb(), pd.DataFrame({"p": p}), y, "logit_platt")
    assert cal.slope_ == pytest.approx(ref.slope_) and cal.intercept_ == pytest.approx(
        ref.intercept_
    )


def test_score_calibrator_isotonic_monotone_and_bounded():
    X, y, z = _score_frame(seed=3)
    p = expit(z)
    cal = ScoreCalibrator("isotonic").fit(p, y)
    grid = np.linspace(0.0, 1.0, 500)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-15)
    assert out.min() >= 1e-6 and out.max() <= 1 - 1e-6


def test_score_calibrator_rejects_bad_inputs():
    with pytest.raises(ValueError, match="unknown score calibration"):
        ScoreCalibrator("x").fit([0.1, 0.2], [0, 1])
    with pytest.raises(ValueError, match="classes"):
        ScoreCalibrator("isotonic").fit([0.1, 0.2], [0, 0])


class _NoFit:
    """Base model with predict_proba only; a refit would raise AttributeError."""

    def predict_proba(self, X):
        p = np.asarray(X["p"], dtype=float)
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("method", ["logit_intercept", "logit_platt", "isotonic"])
def test_precalibrated_model_equals_calibrator_and_pickles(method):
    X, y, _ = _score_frame()
    cal = ScoreCalibrator(method).fit(X["p"], y)
    model = PrecalibratedModel(_NoFit(), cal)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba[:, 1], cal.transform(X["p"]))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == [0, 1]
    clone = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(clone.predict_proba(X), proba)
    assert set(model.predict(X)) <= {0, 1}


def test_supervised_binner_finds_step_and_handles_nan():
    rng = np.random.default_rng(0)
    x = rng.uniform(300, 850, 5000)
    y = (rng.random(5000) < np.where(x < 650, 0.3, 0.02)).astype(int)
    x[:100] = np.nan
    y[:100] = 1
    b = SupervisedBinner(max_bins=2, min_bin_fraction=0.05).fit(x[:, None], y)
    (thr,) = b.thresholds_
    assert len(thr) == 1 and abs(thr[0] - 650) < 15
    out = b.transform(np.array([[np.nan], [400.0], [800.0], [10000.0], [-5.0]]))
    assert out[:, 0].tolist() == [2.0, 0.0, 1.0, 1.0, 0.0]


def test_supervised_binner_rejects_bad_input():
    x = np.arange(100.0)[:, None]
    y = (x[:, 0] > 50).astype(int)
    with pytest.raises(ValueError, match="needs y"):
        SupervisedBinner().fit(x)
    with pytest.raises(ValueError, match="all NaN"):
        SupervisedBinner().fit(np.full((100, 1), np.nan), y)
    with pytest.raises(ValueError, match="match y"):
        SupervisedBinner().fit(x, y[:50])
    with pytest.raises(ValueError, match="columns"):
        SupervisedBinner().fit(x, y).transform(np.zeros((3, 2)))


def test_scorecard_params_and_leaky_guard(xy):
    feats, y = xy
    model = fit_model(
        "scorecard", "no_apr", {"max_bins": 4, "min_bin_fraction": 0.1, "C": 0.1}, feats, y
    )
    binner = model.named_steps["encode"].named_transformers_["num"].named_steps["bin"]
    assert all(len(t) <= 3 for t in binner.thresholds_)
    with pytest.raises(ValueError, match="leaky"):
        build_model("scorecard", "leaky")
