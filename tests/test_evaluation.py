import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from credit_risk.evaluation import (
    binary_metrics,
    bootstrap_ci,
    breakeven_pd,
    brier_skill_score,
    calibration_table,
    cutoff_table,
    expected_calibration_error,
    paired_bootstrap_diff,
    psi,
)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    s = np.clip(0.2 + 0.1 * y + rng.normal(0, 0.05, 2000), 0.001, 0.999)
    return y, s


def test_binary_metrics(data):
    y, s = data
    m = binary_metrics(y, s)
    assert set(m) == {
        "n", "n_pos", "base_rate", "mean_score", "roc_auc", "gini", "ks", "pr_auc", "brier",
        "log_loss",
    }  # fmt: skip
    assert m["n"] == 2000 and m["n_pos"] == y.sum()
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert 0 < m["ks"] <= 1
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))


def test_binary_metrics_perfect_ks_and_clipping():
    m = binary_metrics([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0])
    assert m["ks"] == 1.0
    assert np.isfinite(m["log_loss"])


def test_binary_metrics_errors():
    with pytest.raises(ValueError):
        binary_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.1, np.nan])


def test_bootstrap_ci_deterministic_and_brackets(data):
    y, s = data
    r1 = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=3)
    r2 = bootstrap_ci(y, s, roc_auc_score, n_boot=200, seed=3)
    assert r1 == r2
    p, lo, hi = r1
    assert lo < p < hi
    assert p == pytest.approx(roc_auc_score(y, s))


def test_bootstrap_is_stratified(data):
    y, s = data
    counts = []

    def metric(yy: np.ndarray, ss: np.ndarray) -> float:
        counts.append(int(yy.sum()))
        return 0.0

    bootstrap_ci(y, s, metric, n_boot=20)
    assert set(counts) == {int(y.sum())}


def test_paired_bootstrap_diff(data):
    y, s = data
    rng = np.random.default_rng(5)
    noise = rng.random(len(y))
    res = paired_bootstrap_diff(y, s, noise, roc_auc_score, n_boot=200)
    assert set(res) == {"diff", "lower", "upper", "share_a_better"}
    assert res["diff"] == pytest.approx(roc_auc_score(y, s) - roc_auc_score(y, noise))
    assert res["lower"] > 0 and res["share_a_better"] == 1.0
    same = paired_bootstrap_diff(y, s, s, roc_auc_score, n_boot=50)
    assert same["diff"] == 0 and same["share_a_better"] == 0.0


def test_calibration_table_and_ece(data):
    y, s = data
    t = calibration_table(y, s, n_bins=10)
    assert list(t.columns) == ["bin", "n", "mean_score", "observed_rate"]
    assert len(t) == 10 and t["n"].sum() == len(y)
    gap = (t["mean_score"] - t["observed_rate"]).abs()
    assert expected_calibration_error(y, s) == pytest.approx((gap * t["n"]).sum() / len(y))


def test_calibration_ties_merge():
    y = np.array([0, 1] * 20)
    s = np.array([0.1] * 30 + [0.5] * 10)
    t = calibration_table(y, s, n_bins=10)
    assert 1 <= len(t) < 10 and t["n"].sum() == 40


def test_psi_identity_and_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(psi(x, x)) < 1e-12
    assert psi(x, rng.normal(2, 1, 5000)) > 0.25


def test_psi_nan_bin():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    y = x.copy()
    y[:200] = np.nan
    assert psi(x, y) > 0
    assert abs(psi(y, y)) < 1e-12


def test_cutoff_table(data):
    y, s = data
    t = cutoff_table(y, s, [0.5, 0.8, 1.0])
    assert list(t.columns) == [
        "approval_rate", "threshold", "n_approved", "bad_rate_approved", "n_declined",
        "bad_rate_declined", "share_of_bads_declined",
    ]  # fmt: skip
    assert ((t["n_approved"] + t["n_declined"]) == len(y)).all()
    assert t["n_approved"].iloc[0] == pytest.approx(1000, abs=5)
    assert t["share_of_bads_declined"].iloc[-1] == 0
    assert t["bad_rate_declined"].iloc[0] > t["bad_rate_approved"].iloc[0]
    assert isinstance(t, pd.DataFrame)


def test_breakeven_pd():
    assert breakeven_pd(0.6, 0.2) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        breakeven_pd(0, 0.2)
    with pytest.raises(ValueError):
        breakeven_pd(0.5, -1)


def test_log_loss_clipping():
    m = binary_metrics([0, 1], [1.0, 0.0])
    assert m["log_loss"] == pytest.approx(-math.log(1e-15), rel=1e-3)


def test_labels_must_be_zero_one():
    with pytest.raises(ValueError):
        binary_metrics([1, 2], [0.1, 0.2])


def test_calibration_all_tied():
    y, s = [0, 1, 0, 1], [0.2] * 4
    t = calibration_table(y, s)
    assert len(t) == 1 and t["n"].sum() == 4
    assert expected_calibration_error(y, s) == pytest.approx(0.3)


def test_ece_independent(data):
    y, s = data
    order = np.argsort(s, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, 10):  # 2000 distinct scores -> equal bins of 200
        total += len(chunk) * abs(s[chunk].mean() - y[chunk].mean())
    assert expected_calibration_error(y, s) == pytest.approx(total / len(y))


def test_ks_brute_force():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.3).astype(int)
    s = 0.7 * rng.random(300) + 0.3 * y
    best = max((s[y == 1] > t).mean() - (s[y == 0] > t).mean() for t in np.unique(s))
    assert binary_metrics(y, s)["ks"] == pytest.approx(best)
    assert 0 < best < 1


def test_psi_floor_and_nan_only_in_expected():
    x = np.arange(100.0)
    shifted = np.full(100, 1000.0)  # all mass in the top bin, other bins empty
    val = psi(x, shifted)
    pe = 0.1
    expect = (1 - pe) * math.log(1 / pe) + 9 * ((1e-6 - pe) * math.log(1e-6 / pe))
    assert val == pytest.approx(expect, rel=1e-6)
    e = x.copy()
    e[:10] = np.nan
    v = psi(e, x)
    assert np.isfinite(v) and v > 0


def test_psi_empty_raises():
    with pytest.raises(ValueError):
        psi([], [1.0, 2.0])
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [])


def test_breakeven_nonfinite():
    with pytest.raises(ValueError):
        breakeven_pd(float("nan"), 0.2)
    with pytest.raises(ValueError):
        breakeven_pd(0.5, float("inf"))


def test_brier_skill_score(data):
    y, _ = data
    rate = float(y.mean())
    assert brier_skill_score(y, np.full(len(y), rate), rate) == pytest.approx(0.0, abs=1e-12)
    assert brier_skill_score(y, y.astype(float), rate) == pytest.approx(1.0)
    assert brier_skill_score(y, np.full(len(y), 0.5), rate) < 0
    expected = 1 - np.mean((0.3 - y) ** 2) / np.mean((rate - y) ** 2)
    assert brier_skill_score(y, np.full(len(y), 0.3), rate) == pytest.approx(expected)


@pytest.mark.parametrize("rate", [0.0, 1.0, -0.1, 1.5])
def test_brier_skill_score_rejects_reference_rate(data, rate):
    y, s = data
    with pytest.raises(ValueError, match="reference_rate"):
        brier_skill_score(y, s, rate)
