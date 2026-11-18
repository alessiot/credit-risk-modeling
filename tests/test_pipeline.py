import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import APPLICATION_COLUMNS, DEFAULT_PROTOCOL
from credit_risk.data import load_raw
from credit_risk.models import positive_scores
from credit_risk.pipeline import (
    Candidate,
    CandidateResult,
    calibrate_and_cutoff,
    evaluate_holdout,
    load_sweep,
    package_model,
    score_applications,
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


def test_evaluate_holdout_and_stability(champion, synthetic_frames, caplog):
    cal = calibrate_and_cutoff(champion, synthetic_frames)["calibrated_model"]
    with caplog.at_level("INFO", logger="credit_risk.pipeline"):
        ev = evaluate_holdout(cal, synthetic_frames["test"], "test", threshold=0.1428, n_boot=50)
    assert sum("scoring holdout test" in m for m in caplog.messages) == 1
    assert ev["auc_ci"][0] <= ev["metrics"]["roc_auc"] <= ev["auc_ci"][1]
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


def test_package_round_trip(champion, synthetic_frames, tmp_path):
    cal = calibrate_and_cutoff(champion, synthetic_frames)["calibrated_model"]
    ev = evaluate_holdout(cal, synthetic_frames["test"], "test", threshold=0.1428, n_boot=20)
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
    assert meta["input_columns"] == list(APPLICATION_COLUMNS) and meta["threshold"] == 0.1428
    assert meta["split_protocol"]["split_id"] == "oot-v1"
    df = synthetic_frames["test"]
    pds = score_applications(out, df)
    assert pds.index.equals(df.index) and pds.between(0, 1).all()
    with pytest.raises(SchemaError):
        score_applications(out, df.drop(columns="fico_score"))


def test_train_lgbm_candidate(synthetic_frames):
    res = train_candidate(Candidate("gb", "lgbm", "full", LGBM, weighted=True), synthetic_frames)
    assert pd.notna(res.cv_mean)
