import json

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from credit_risk.config import APPLICATION_COLUMNS, DEFAULT_PROTOCOL, FEATURE_SETS
from credit_risk.data import load_raw
from credit_risk.evaluation import brier_skill_score
from credit_risk.models import CALIBRATION_METHODS, ScoreCalibrator, calibrate, positive_scores
from credit_risk.pipeline import (
    Candidate,
    CandidateResult,
    _one_se_choice,
    calibrate_and_cutoff,
    evaluate_holdout,
    load_sweep,
    package_model,
    score_applications,
    select_calibrator,
    select_calibrator_blocks,
    select_champion,
    split_frames,
    stability_report,
    train_candidate,
)
from credit_risk.schema import SchemaError

LGBM = {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 20}


def _write(tmp_path, items):
    p = tmp_path / "sweep.json"
    p.write_text(json.dumps(items))
    return p


def _item(name="a", family="logreg", feature_set="full"):
    return {"name": name, "family": family, "feature_set": feature_set, "params": {}}


def test_load_sweep_ok(tmp_path):
    cands = load_sweep(_write(tmp_path, [_item("a"), _item("b", "lgbm", "no_apr")]))
    assert [c.name for c in cands] == ["a", "b"] and not cands[0].weighted


@pytest.mark.parametrize(
    "items",
    [
        [_item(family="svm")],
        [_item(feature_set="leaky")],
        [_item("a"), _item("a")],
    ],
)
def test_load_sweep_rejects(tmp_path, items):
    with pytest.raises(ValueError):
        load_sweep(_write(tmp_path, items))


def test_split_frames(data_path):
    frames = split_frames(load_raw(data_path))
    assert set(frames) == {"train", "val", "gap", "test"}
    assert frames["train"]["vintage"].max() <= DEFAULT_PROTOCOL.train_end
    assert frames["test"]["vintage"].min() >= DEFAULT_PROTOCOL.test_start
    assert frames["val"].index.equals(pd.RangeIndex(len(frames["val"])))


def test_split_frames_empty(raw_sample):
    with pytest.raises(ValueError, match="empty"):
        split_frames(raw_sample[raw_sample["vintage"] <= DEFAULT_PROTOCOL.train_end])


def _result(name, family, feature_set, cv_mean, cv_se):
    cand = Candidate(name, family, feature_set, {})
    return CandidateResult(cand, cv_mean, cv_se, [cv_mean], None, np.zeros(1), cv_mean)


def test_select_simpler_within_one_se():
    lgbm = _result("gb", "lgbm", "full", 0.700, 0.005)
    lr = _result("lr", "logreg", "full", 0.696, 0.002)
    champ, chall, why = select_champion([lgbm, lr])
    assert champ is lr and chall is lgbm and "lr" in why


def test_select_best_outside_one_se():
    lgbm = _result("gb", "lgbm", "full", 0.700, 0.005)
    lr = _result("lr", "logreg", "full", 0.690, 0.002)
    champ, chall, _ = select_champion([lr, lgbm])
    assert champ is lgbm and chall is lr


def test_select_tie_break_higher_cv_mean():
    a = _result("a", "logreg", "full", 0.700, 0.01)
    b = _result("b", "logreg", "full", 0.704, 0.01)
    champ, chall, _ = select_champion([a, b])
    assert champ is b and chall is a


def test_select_fewer_features_breaks_tier_tie_and_single():
    a = _result("a", "logreg", "full", 0.70, 0.01)
    b = _result("b", "logreg", "fico_only", 0.695, 0.01)
    champ, chall, _ = select_champion([a, b])
    assert champ is b and chall is a
    only, none, _ = select_champion([a])
    assert only is a and none is None


@pytest.fixture(scope="module")
def synthetic_frames(data_path):
    return split_frames(load_raw(data_path))


@pytest.fixture(scope="module")
def champion(synthetic_frames):
    return train_candidate(Candidate("lr", "logreg", "full", {"C": 1.0}), synthetic_frames)


def test_train_candidate(champion):
    assert 0.5 < champion.val_auc < 1 and len(champion.fold_scores) == 4


def test_calibrate_and_cutoff(champion, synthetic_frames):
    out = calibrate_and_cutoff(champion, synthetic_frames)
    assert out["breakeven_pd"] == pytest.approx(0.1 / 0.7)
    assert len(out["val_scores_calibrated"]) == len(synthetic_frames["val"])
    assert len(out["cutoff_table"]) == 4
    expected = float((out["val_scores_calibrated"] <= out["breakeven_pd"]).mean())
    assert out["approval_rate_at_breakeven"] == expected
    assert out["brier_after"] <= out["brier_before"] + 1e-3


def ev_scores(cal, frames):
    return positive_scores(cal, frames["test"][list(APPLICATION_COLUMNS)])


def test_evaluate_holdout_and_stability(champion, synthetic_frames, caplog):
    cal = calibrate_and_cutoff(champion, synthetic_frames)["calibrated_model"]
    with caplog.at_level("INFO", logger="credit_risk.pipeline"):
        ev = evaluate_holdout(
            cal,
            synthetic_frames["test"],
            "test",
            threshold=0.1428,
            reference_rate=0.03,
            n_boot=50,
        )
    assert sum("scoring holdout test" in m for m in caplog.messages) == 1
    assert ev["auc_ci"][0] <= ev["metrics"]["roc_auc"] <= ev["auc_ci"][1]
    assert ev["brier_skill"] == pytest.approx(
        brier_skill_score(
            synthetic_frames["test"]["default_12m"], ev_scores(cal, synthetic_frames), 0.03
        )
    )
    pv = ev["per_vintage"]
    assert pv["n"].sum() == len(synthetic_frames["test"])
    scores = positive_scores(cal, synthetic_frames["test"][list(APPLICATION_COLUMNS)])
    approved = scores <= 0.1428
    assert ev["approval_rate"] == pytest.approx(approved.mean())
    assert ev["bad_rate_approved"] == pytest.approx(
        synthetic_frames["test"]["default_12m"][approved].mean()
    )
    rep = stability_report(cal, synthetic_frames)
    assert set(rep["score_psi"]) == {"val", "gap", "test"}
    assert rep["feature_psi"]["psi"].is_monotonic_decreasing


def test_evaluate_holdout_requires_reference_rate(champion, synthetic_frames):
    with pytest.raises(TypeError):
        evaluate_holdout(champion.model, synthetic_frames["test"], "t", threshold=0.1)


def test_calibrate_and_cutoff_logit_method(champion, synthetic_frames):
    out = calibrate_and_cutoff(champion, synthetic_frames, method="logit_intercept")
    y = synthetic_frames["val"]["default_12m"]
    assert abs(out["val_scores_calibrated"].mean() - y.mean()) < 1e-9


class _ColumnProb(ClassifierMixin, BaseEstimator):
    """Stand-in fitted model whose probability is stored in the fico_score column."""

    classes_ = np.array([0, 1])

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (np.asarray(X["fico_score"]) > 0.5).astype(int)

    def predict_proba(self, X):
        p = np.asarray(X["fico_score"], dtype=float)
        return np.column_stack([1 - p, p])


def _val_frame(
    slope, shift, n_per_vintage=5000, vintages=(202207, 202208, 202209, 202210, 202211, 202212)
):
    rng = np.random.default_rng(0)
    n = n_per_vintage * len(vintages)
    z = rng.normal(-3.0, 1.5, n)
    y = (rng.random(n) < expit(z)).astype(int)
    frame = pd.DataFrame({c: 0.0 for c in APPLICATION_COLUMNS}, index=range(n))
    frame["fico_score"] = expit(slope * z + shift)
    frame["vintage"] = np.repeat(vintages, n_per_vintage)
    frame["default_12m"] = y
    return frame


def test_select_calibrator_distorted_scores_reject_shift_only():
    out = select_calibrator(_ColumnProb(), _val_frame(slope=0.4, shift=-1.0))
    t = out["table"].set_index("method")
    assert out["fold_vintages"] == {"A": [202207, 202208, 202209], "B": [202210, 202211, 202212]}
    assert list(t.index) == list(CALIBRATION_METHODS)
    assert out["best"] == t["brier"].idxmin()
    assert t.loc[out["best"], "diff_vs_best"] == 0
    assert not t.loc["logit_intercept", "eligible"]
    assert t["eligible"].idxmax() == out["chosen"] and t.loc[out["chosen"], "eligible"]
    assert out["chosen"] != "logit_intercept"


def test_select_calibrator_se_matches_independent_recompute():
    val = _val_frame(slope=0.4, shift=-1.0, n_per_vintage=1500)
    out = select_calibrator(_ColumnProb(), val)
    in_a = val["vintage"].isin(out["fold_vintages"]["A"]).to_numpy()
    y = val["default_12m"].to_numpy()
    X = val[list(APPLICATION_COLUMNS)]
    err = {}
    for m in CALIBRATION_METHODS:
        pooled = np.empty(len(val))
        for fit_mask in (in_a, ~in_a):
            cal = calibrate(_ColumnProb(), X[fit_mask], pd.Series(y[fit_mask]), m)
            pooled[~fit_mask] = positive_scores(cal, X[~fit_mask])
        err[m] = (pooled - y) ** 2
    best = min(err, key=lambda m: err[m].mean())
    assert out["best"] == best
    t = out["table"].set_index("method")
    for m in CALIBRATION_METHODS:
        diff = err[m] - err[best]
        assert t.loc[m, "se_vs_best"] == pytest.approx(diff.std(ddof=1) / np.sqrt(len(y)))
        assert t.loc[m, "brier"] == pytest.approx(err[m].mean())
        assert t.loc[m, "diff_vs_best"] == pytest.approx(err[m].mean() - err[best].mean())


def test_select_calibrator_shift_only_picks_simplest():
    out = select_calibrator(_ColumnProb(), _val_frame(slope=1.0, shift=-0.8))
    assert out["chosen"] == "logit_intercept"


def test_select_calibrator_needs_four_vintages():
    with pytest.raises(ValueError, match="4"):
        select_calibrator(_ColumnProb(), _val_frame(1.0, 0.0, 500, (202207, 202208, 202209)))


def _block_data(slope, shift, n_per_block=3000, blocks=("a", "b", "c", "d")):
    rng = np.random.default_rng(1)
    n = n_per_block * len(blocks)
    z = rng.normal(-3.0, 1.5, n)
    y = (rng.random(n) < expit(z)).astype(int)
    return expit(slope * z + shift), y, np.repeat(blocks, n_per_block)


def test_select_calibrator_blocks_is_leave_one_block_out(monkeypatch):
    p, y, blocks = _block_data(0.5, -0.5, n_per_block=1000)
    fits = []
    real_fit = ScoreCalibrator.fit

    def spy(self, p_fit, y_fit):
        fits.append((self.method, len(p_fit)))
        return real_fit(self, p_fit, y_fit)

    monkeypatch.setattr(ScoreCalibrator, "fit", spy)
    out = select_calibrator_blocks(p, y, blocks)
    assert out["blocks"] == ["a", "b", "c", "d"]
    assert len(fits) == 3 * 4 and all(n == 3000 for _, n in fits)
    monkeypatch.undo()
    t = out["table"].set_index("method")
    for method in ("logit_intercept", "logit_platt", "isotonic"):
        pooled = np.empty(len(y))
        for b in out["blocks"]:
            held = blocks == b
            pooled[held] = ScoreCalibrator(method).fit(p[~held], y[~held]).transform(p[held])
        assert t.loc[method, "brier"] == pytest.approx(np.mean((pooled - y) ** 2))
    assert list(out["table"].columns) == [
        "method",
        "brier",
        "log_loss",
        "ece",
        "diff_vs_best",
        "se_vs_best",
        "eligible",
    ]
    assert out["chosen"] in t.index and out["best"] == t["brier"].idxmin()


def test_select_calibrator_blocks_distorted_scores_reject_shift_only():
    p, y, blocks = _block_data(0.4, -1.0)
    assert select_calibrator_blocks(p, y, blocks)["chosen"] != "logit_intercept"


def test_select_calibrator_blocks_shift_only_picks_simplest():
    p, y, blocks = _block_data(1.0, -0.8)
    assert select_calibrator_blocks(p, y, blocks)["chosen"] == "logit_intercept"


def test_select_calibrator_blocks_rejects_bad_blocks():
    p, y, blocks = _block_data(1.0, 0.0, n_per_block=500, blocks=("a", "b"))
    with pytest.raises(ValueError, match="3 blocks"):
        select_calibrator_blocks(p, y, blocks)
    p, y, blocks = _block_data(1.0, 0.0, n_per_block=500, blocks=("a", "b", "c"))
    y = y.copy()
    y[blocks == "c"] = 0
    with pytest.raises(ValueError, match="both classes"):
        select_calibrator_blocks(p, y, blocks)


def test_one_se_choice_simpler_within_one_se():
    table = pd.DataFrame(
        {
            "method": ["a", "b", "c"],
            "brier": [0.0301, 0.0300, 0.0310],
            "se_vs_best": [0.0002, 0.0, 0.0003],
        }
    )
    out, chosen, best = _one_se_choice(table)
    assert best == "b" and chosen == "a"
    assert out["eligible"].tolist() == [True, True, False]
    assert out["diff_vs_best"].iloc[1] == 0


def test_one_se_choice_none_within_one_se_picks_best():
    table = pd.DataFrame(
        {
            "method": ["a", "b", "c"],
            "brier": [0.0310, 0.0300, 0.0305],
            "se_vs_best": [0.0002, 0.0, 0.0002],
        }
    )
    out, chosen, best = _one_se_choice(table)
    assert best == "b" and chosen == "b"
    assert out["eligible"].tolist() == [False, True, False]


def test_package_round_trip(champion, synthetic_frames, tmp_path):
    cal = calibrate_and_cutoff(champion, synthetic_frames)["calibrated_model"]
    ev = evaluate_holdout(
        cal, synthetic_frames["test"], "test", threshold=0.1428, reference_rate=0.03, n_boot=20
    )
    out = package_model(
        cal,
        tmp_path / "m",
        candidate=champion.candidate,
        metrics={"test": ev},
        data_hash="abc",
        protocol=DEFAULT_PROTOCOL,
        threshold=0.1428,
        extra={"run_id": "1"},
    )
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["input_columns"] == list(FEATURE_SETS[champion.candidate.feature_set])
    assert meta["threshold"] == 0.1428
    assert meta["split_protocol"]["split_id"] == "oot-v1"
    df = synthetic_frames["test"]
    pds = score_applications(out, df)
    assert pds.index.equals(df.index) and pds.between(0, 1).all()
    with pytest.raises(SchemaError):
        score_applications(out, df.drop(columns="fico_score"))


def test_train_lgbm_candidate(synthetic_frames):
    res = train_candidate(Candidate("gb", "lgbm", "full", LGBM, weighted=True), synthetic_frames)
    assert pd.notna(res.cv_mean)
