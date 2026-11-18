import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FEATURE_SETS, SELECTABLE_FEATURE_SETS
from credit_risk.models import (
    MonotoneLGBM,
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


@pytest.mark.parametrize("family", ["base_rate", "logreg", "lgbm"])
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
    assert len(model.named_steps["model"].feature_importances_) == 18
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
