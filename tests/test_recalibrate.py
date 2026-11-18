import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from credit_risk.config import APPLICATION_COLUMNS, DEFAULT_PROTOCOL, FEATURE_SETS
from credit_risk.data import load_raw
from credit_risk.models import calibrate, fit_model, positive_scores
from credit_risk.packaging import load_artifact
from credit_risk.pipeline import Candidate, package_model, split_frames

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recalibrate.py"
spec = importlib.util.spec_from_file_location("recalibrate", SCRIPT)
recalibrate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recalibrate)


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n")
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    return data, lock, tmp_path / "run"


def test_register_writes_manifest_and_is_idempotent(inputs):
    data, lock, run_dir = inputs
    path = recalibrate.register(run_dir, data, lock, "2026-01-01")
    text = path.read_text()
    assert text.startswith("# Run manifest: R10-calibration-v1.1")
    assert recalibrate.MANIFEST_BODY in text
    assert recalibrate.file_sha256(data) in text and recalibrate.file_sha256(lock) in text
    assert recalibrate.register(run_dir, data, lock, "2026-01-01") == path


def test_register_refuses_different_manifest(inputs):
    data, lock, run_dir = inputs
    recalibrate.register(run_dir, data, lock, "2026-01-01")
    with pytest.raises(FileExistsError, match="different manifest"):
        recalibrate.register(run_dir, data, lock, "2026-01-02")
    lock.write_text("changed lock")
    with pytest.raises(FileExistsError):
        recalibrate.register(run_dir, data, lock, "2026-01-01")


def _synthetic_csv(path: Path, per_month: int = 150, seed: int = 0) -> Path:
    """Write a schema-valid synthetic raw CSV covering train, val, gap and test vintages."""
    rng = np.random.default_rng(seed)
    months = [
        y * 100 + m
        for y in (2021, 2022, 2023, 2024)
        for m in range(1, 13)
        if 202101 <= y * 100 + m <= 202409
    ]
    vintage = np.repeat(months, per_month)
    n = len(vintage)
    fico = rng.integers(500, 800, n)
    p = 1 / (1 + np.exp(3.0 + (fico - 650) / 60.0))
    df = pd.DataFrame(
        {
            "loan_id": [f"L{i:06d}" for i in range(n)],
            "origination_date": [f"{v // 100}-{v % 100:02d}-01" for v in vintage],
            "vintage": vintage,
            "months_on_book": rng.integers(1, 24, n),
            "fico_score": fico,
            "income": rng.integers(20000, 120000, n),
            "debt_to_income": rng.uniform(0.05, 0.6, n).round(3),
            "num_open_trades": rng.integers(1, 15, n),
            "utilization_rate": rng.uniform(0, 1, n).round(4),
            "inquiries_last_6m": rng.integers(0, 6, n),
            "loan_amount": rng.integers(5000, 40000, n),
            "term": rng.choice([36, 60], n),
            "apr": rng.uniform(5, 20, n).round(2),
            "channel": rng.choice(["Direct", "Partner", "Mail"], n),
            "product_type": rng.choice(["Personal", "Auto", "Business"], n),
            "employment_length": rng.integers(0, 30, n),
            "state": rng.choice(["CA", "TX", "Other"], n),
            "age": rng.integers(21, 70, n),
            "days_past_due_current": rng.integers(0, 3, n),
            "total_payments_to_date": rng.uniform(0, 30000, n).round(2),
            "sample_weight": np.ones(n),
            "default_12m": (rng.random(n) < p).astype(int),
        }
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_data(tmp_path):
    return _synthetic_csv(tmp_path / "synthetic.csv")


def _kwargs(tmp_path, data, lock):
    return {
        "data_path": data,
        "lock_path": lock,
        "v1_dir": tmp_path / "v1",
        "out_dir": tmp_path / "out",
        "run_dir": tmp_path / "run",
        "predictions_path": tmp_path / "reports" / "pred.csv",
        "n_boot": 20,
    }


def test_full_mode_refuses_without_manifest(tmp_path, synthetic_data):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    with pytest.raises(FileNotFoundError, match="register"):
        recalibrate.run_full(**_kwargs(tmp_path, synthetic_data, lock))
    assert not (tmp_path / "out").exists()


def test_full_mode_refuses_when_hashes_differ(tmp_path, synthetic_data):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    recalibrate.register(tmp_path / "run", synthetic_data, lock, "2026-01-01")
    lock.write_text("changed")
    with pytest.raises(recalibrate.RegistrationError, match="Environment lock"):
        recalibrate.run_full(**_kwargs(tmp_path, synthetic_data, lock))
    lock.write_text("lock")
    synthetic_data.write_text(synthetic_data.read_text() + "\n")
    with pytest.raises(recalibrate.RegistrationError, match="Data hash"):
        recalibrate.run_full(**_kwargs(tmp_path, synthetic_data, lock))


def test_hash_check_ignores_date(tmp_path, synthetic_data):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    path = recalibrate.register(tmp_path / "run", synthetic_data, lock, "2020-05-05")
    recalibrate.check_registration(path, synthetic_data, lock)


@pytest.mark.parametrize("existing", ["result", "artifact", "predictions"])
def test_full_mode_refuses_existing_outputs_before_loading(tmp_path, synthetic_data, existing):
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    kwargs = _kwargs(tmp_path, synthetic_data, lock)
    recalibrate.register(kwargs["run_dir"], synthetic_data, lock, "2026-01-01")
    if existing == "result":
        (kwargs["run_dir"] / "result.json").write_text("{}")
    elif existing == "artifact":
        kwargs["out_dir"].mkdir()
        (kwargs["out_dir"] / "model.joblib").write_text("x")
    else:
        kwargs["predictions_path"].parent.mkdir()
        kwargs["predictions_path"].write_text("x")
    # The v1 directory does not exist: reaching the load step would raise a different error.
    with pytest.raises(FileExistsError):
        recalibrate.run_full(**kwargs)


def test_frozen_base_rejects_model_without_frozen_estimator():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = np.tile([0, 1], 30)
    plain = CalibratedClassifierCV(LogisticRegression(), cv=2).fit(X, y)
    with pytest.raises(ValueError, match="FrozenEstimator"):
        recalibrate.frozen_base(plain)


def test_full_mode_on_tmp_paths(tmp_path, synthetic_data):
    frames = split_frames(load_raw(synthetic_data))
    train, val = frames["train"], frames["val"]
    assert val["vintage"].nunique() >= 4
    assert all(f["default_12m"].nunique() == 2 for f in frames.values())
    cols = list(APPLICATION_COLUMNS)
    base = fit_model(
        "lgbm",
        "no_apr",
        {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 50},
        train[cols],
        train["default_12m"],
    )
    v1 = calibrate(base, val[cols], val["default_12m"], "sigmoid")
    cand = Candidate("lgbm-small", "lgbm", "no_apr", {})
    v1_dir = package_model(
        v1,
        tmp_path / "v1",
        candidate=cand,
        metrics={},
        data_hash="x",
        protocol=DEFAULT_PROTOCOL,
        threshold=0.1,
    )
    lock = tmp_path / "uv.lock"
    lock.write_text("lock")
    run_dir = tmp_path / "run"
    recalibrate.register(run_dir, synthetic_data, lock, "2026-01-01")
    result = recalibrate.run_full(**_kwargs(tmp_path, synthetic_data, lock))
    assert result["chosen"] in {"logit_intercept", "logit_platt", "sigmoid", "isotonic"}
    model, meta = load_artifact(tmp_path / "out")
    assert meta["extra"]["calibration"] == result["chosen"]
    assert meta["extra"]["v1_model_sha256"] == load_artifact(v1_dir)[1]["model_sha256"]
    assert meta["input_columns"] == list(FEATURE_SETS["no_apr"])
    pred = pd.read_csv(tmp_path / "reports" / "pred.csv")
    test = frames["test"]
    assert list(pred.columns) == ["loan_id", "vintage", "default_12m", "pd_v1", "pd_v1_1"]
    assert len(pred) == len(test)
    np.testing.assert_allclose(pred["pd_v1_1"], positive_scores(model, test[cols]), atol=1e-12)
    np.testing.assert_allclose(pred["pd_v1"], positive_scores(v1, test[cols]), atol=1e-12)
    saved = json.loads((run_dir / "result.json").read_text())
    assert set(saved["evaluations"]["test"]) == {"raw", "v1", "v1.1"}
    text = (run_dir / "manifest.md").read_text()
    assert "## Result" in text and "### test" in text and "brier_skill" in text
