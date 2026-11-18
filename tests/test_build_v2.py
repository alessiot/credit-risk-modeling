import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import APPLICATION_COLUMNS, DEFAULT_PROTOCOL, FEATURE_SETS
from credit_risk.data import load_raw
from credit_risk.models import calibrate, fit_model, positive_scores
from credit_risk.packaging import load_artifact
from credit_risk.pipeline import Candidate, package_model, split_frames

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_v2 = _load("build_v2")
run_experiments = _load("run_experiments")

TINY = {"n_estimators": 15, "num_leaves": 4, "min_child_samples": 40}


def _synthetic_csv(path: Path, per_month: int = 250, seed: int = 0, poison: int = 20) -> Path:
    """Write a schema-valid CSV for 202101-202312 plus poisoned 2024 rows."""
    rng = np.random.default_rng(seed)
    months = [y * 100 + m for y in (2021, 2022, 2023) for m in range(1, 13)]
    vintage = np.repeat(months, per_month)
    vintage = np.concatenate([vintage, np.full(poison, 202405)])
    n = len(vintage)
    fico = rng.integers(500, 800, n)
    p = 1 / (1 + np.exp(1.5 + (fico - 650) / 60.0))
    default = (rng.random(n) < p).astype(int)
    default[-poison:] = 1
    ids = [f"L{i:06d}" for i in range(n - poison)] + [f"POISON{i:03d}" for i in range(poison)]
    df = pd.DataFrame(
        {
            "loan_id": ids,
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
            "default_12m": default,
        }
    )
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_data(tmp_path):
    return _synthetic_csv(tmp_path / "synthetic.csv")


@pytest.fixture
def lock(tmp_path):
    path = tmp_path / "uv.lock"
    path.write_text("lock")
    return path


def _kwargs(tmp_path, data, lock):
    return {
        "data_path": data,
        "lock_path": lock,
        "v1_dir": tmp_path / "v1",
        "v11_dir": tmp_path / "v11",
        "out_dir": tmp_path / "out",
        "run_dir": tmp_path / "run",
        "predictions_path": tmp_path / "reports" / "pred.csv",
        "seeds": (0,),
        "cap": 3,
        "window": 60,
        "n_boot": 30,
    }


def test_register_writes_manifest_and_is_idempotent(tmp_path, synthetic_data, lock):
    path = build_v2.register(tmp_path / "run", synthetic_data, lock, "2026-01-01")
    text = path.read_text()
    assert text.startswith("# Run manifest: R13-lgbm-no-apr-logloss-v2")
    assert build_v2.MANIFEST_BODY in text
    assert build_v2.file_sha256(synthetic_data) in text and build_v2.file_sha256(lock) in text
    assert build_v2.register(tmp_path / "run", synthetic_data, lock, "2026-01-01") == path


def test_register_refuses_different_manifest(tmp_path, synthetic_data, lock):
    build_v2.register(tmp_path / "run", synthetic_data, lock, "2026-01-01")
    with pytest.raises(FileExistsError, match="different manifest"):
        build_v2.register(tmp_path / "run", synthetic_data, lock, "2026-01-02")


def test_full_mode_refuses_without_manifest(tmp_path, synthetic_data, lock):
    with pytest.raises(FileNotFoundError, match="register"):
        build_v2.run_full(**_kwargs(tmp_path, synthetic_data, lock))
    assert not (tmp_path / "out").exists()


def test_full_mode_refuses_when_hashes_differ(tmp_path, synthetic_data, lock):
    build_v2.register(tmp_path / "run", synthetic_data, lock, "2026-01-01")
    lock.write_text("changed")
    with pytest.raises(build_v2.RegistrationError, match="Environment lock"):
        build_v2.run_full(**_kwargs(tmp_path, synthetic_data, lock))
    lock.write_text("lock")
    synthetic_data.write_text(synthetic_data.read_text() + "\n")
    with pytest.raises(build_v2.RegistrationError, match="Data hash"):
        build_v2.run_full(**_kwargs(tmp_path, synthetic_data, lock))


@pytest.mark.parametrize("existing", ["result", "artifact", "predictions"])
def test_full_mode_refuses_existing_outputs_before_loading(
    tmp_path, synthetic_data, lock, existing
):
    kwargs = _kwargs(tmp_path, synthetic_data, lock)
    build_v2.register(kwargs["run_dir"], synthetic_data, lock, "2026-01-01")
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
        build_v2.run_full(**kwargs)


def test_load_pre_test_frames_drops_test_rows(synthetic_data):
    frames = build_v2.load_pre_test_frames(synthetic_data)
    assert set(frames) == {"train", "val", "gap"}
    for frame in frames.values():
        assert frame["vintage"].max() < DEFAULT_PROTOCOL.test_start
        assert not frame["loan_id"].str.startswith("POISON").any()
    assert frames["gap"]["vintage"].between(202301, 202312).all()


def test_explicit_drop_alone_keeps_test_rows_out(synthetic_data, monkeypatch):
    # Relabel every test-window row as train, so only the explicit drop can exclude them.
    real_assign = build_v2.assign_split

    def test_as_train(vintage):
        return real_assign(vintage).replace("test", "train")

    monkeypatch.setattr(build_v2, "assign_split", test_as_train)
    frames = build_v2.load_pre_test_frames(synthetic_data)
    for frame in frames.values():
        assert frame["vintage"].max() < DEFAULT_PROTOCOL.test_start
        assert not frame["loan_id"].str.startswith("POISON").any()
    # Without the drop the poison rows would be in train: confirm the relabel would admit them.
    raw = pd.read_csv(synthetic_data)
    assert (test_as_train(raw["vintage"]) == "train").sum() > (raw["vintage"] <= 202206).sum()


def test_scale_check_and_decision_rules():
    z = np.linspace(-4, -1, 200)
    assert build_v2.check_fold_scale(z, z)["pass"]
    assert not build_v2.check_fold_scale(1.5 * z, z)["pass"]
    assert not build_v2.check_fold_scale(z + 0.5, z)["pass"]
    out = build_v2.check_fold_scale(2.0 * z + 1.0, z)
    assert out["slope"] == pytest.approx(2.0) and out["shift"] == pytest.approx(np.mean(z) + 1.0)
    paired = {"brier_diff": {"upper": -0.001}, "auc_diff": {"mean": -0.02}}
    assert not build_v2.decide(paired)["v2_preferred"]
    paired = {"brier_diff": {"upper": -0.001}, "auc_diff": {"mean": 0.0}}
    assert build_v2.decide(paired)["v2_preferred"]
    paired = {"brier_diff": {"upper": 0.001}, "auc_diff": {"mean": 0.0}}
    assert not build_v2.decide(paired)["v2_preferred"]


def test_paired_bootstrap_matches_independent_recompute():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    a = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.1, 400), 0.01, 0.99)
    b = np.full(400, 0.2) + rng.normal(0, 0.01, 400)
    out = build_v2.paired_bootstrap(
        y, {"a": a, "b": b}, reference_rate=0.2, a="a", b="b", n_boot=50
    )
    assert out["brier_diff"]["point"] == pytest.approx(np.mean((a - y) ** 2 - (b - y) ** 2))
    assert out["skill"]["a"]["lower"] <= out["skill"]["a"]["point"] <= out["skill"]["a"]["upper"]
    assert out["auc_diff"]["point"] > 0 and out["brier_diff"]["upper"] < 0


def _frozen_models(tmp_path, data):
    """Package a v1 and a v1.1 stand-in built on the synthetic training window."""
    frames = split_frames(load_raw(data))
    train, val = frames["train"], frames["val"]
    cols = list(APPLICATION_COLUMNS)
    base = fit_model("lgbm", "no_apr", TINY, train[cols], train["default_12m"])
    cand = Candidate("lgbm-small", "lgbm", "no_apr", {})
    for name, method in (("v1", "sigmoid"), ("v11", "logit_intercept")):
        model = calibrate(base, val[cols], val["default_12m"], method)
        package_model(
            model,
            tmp_path / name,
            candidate=cand,
            metrics={},
            data_hash="x",
            protocol=DEFAULT_PROTOCOL,
            threshold=0.1,
        )


def test_full_run_on_synthetic_data(tmp_path, synthetic_data, lock, monkeypatch):
    # split_frames needs a test window; the helper adds one clean frame through the CSV itself.
    df = pd.read_csv(synthetic_data)
    clean_test = df[df["vintage"] >= 202301].copy()
    clean_test["loan_id"] = "T" + clean_test["loan_id"]
    clean_test["vintage"] = 202401
    clean_test["origination_date"] = "2024-01-01"
    helper_csv = tmp_path / "with_test.csv"
    pd.concat([df, clean_test]).to_csv(helper_csv, index=False)
    _frozen_models(tmp_path, helper_csv)

    seen = {}
    real_eval = build_v2.evaluate_holdout

    def spy_eval(model, frame, name, **kw):
        seen[name] = (frame["vintage"].max(), frame["loan_id"].str.startswith("POISON").any())
        return real_eval(model, frame, name, **kw)

    monkeypatch.setattr(build_v2, "evaluate_holdout", spy_eval)
    real_check = build_v2.check_fold_scale
    calls = []

    def forced(final_logit, fold_logit):
        out = real_check(final_logit, fold_logit)
        calls.append(out)
        if len(calls) == 1:
            out = {**out, "pass": False}
        return out

    monkeypatch.setattr(build_v2, "check_fold_scale", forced)

    kwargs = _kwargs(tmp_path, synthetic_data, lock)
    build_v2.register(kwargs["run_dir"], synthetic_data, lock, "2026-01-01")
    result = build_v2.run_full(**kwargs)

    assert len(calls) == 4
    assert set(seen) == {"gap/v1", "gap/v1.1", "gap/v2"}
    assert all(v[0] <= 202312 and not v[1] for v in seen.values())
    sc = result["scale_check"]
    assert sc["pass"].tolist() == [False, True, True, True]
    blocks = result["calibration"]["blocks"]
    excluded = sc.iloc[0]["quarter"]
    assert excluded not in blocks and "2022Q3" in blocks and "2022Q4" in blocks
    assert len(blocks) == 5

    pred = pd.read_csv(kwargs["predictions_path"])
    assert list(pred.columns) == ["loan_id", "vintage", "default_12m", "pd_v1", "pd_v1_1", "pd_v2"]
    assert not pred["loan_id"].str.startswith("POISON").any() and pred["vintage"].max() <= 202312
    model, meta = load_artifact(kwargs["out_dir"])
    assert meta["model_name"] == "R13-lgbm-no-apr-logloss"
    assert meta["input_columns"] == list(FEATURE_SETS["no_apr"])
    assert meta["extra"]["calibration"] == result["calibration"]["chosen"]
    gap = build_v2.load_pre_test_frames(synthetic_data)["gap"]
    np.testing.assert_allclose(
        pred["pd_v2"], positive_scores(model, gap[list(APPLICATION_COLUMNS)]), atol=1e-12
    )
    saved = json.loads((kwargs["run_dir"] / "result.json").read_text())
    assert set(saved["evaluations"]) == {"v1", "v1.1", "v2"}
    assert isinstance(saved["decision"]["v2_preferred"], bool)
    assert (kwargs["run_dir"] / "trials_seed0.csv").exists()
    text = (kwargs["run_dir"] / "manifest.md").read_text()
    for heading in ("## Result", "### Scale check", "### Calibrator selection", "### Decision"):
        assert heading in text

    with pytest.raises(FileExistsError, match="result already exists"):
        build_v2.run_full(**kwargs)


def test_leaderboard_rebuild_keeps_r13_row(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    with (root / "reports" / "leaderboard.csv").open(newline="") as fh:
        current = {r["run_id"]: r for r in csv.DictReader(fh)}
    r13_id = run_experiments.R13_RUN_ID
    run_dir = tmp_path / "runs" / r13_id
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text((root / "runs" / r13_id / "result.json").read_text())
    board = tmp_path / "reports" / "leaderboard.csv"
    board.parent.mkdir()
    board.write_text((root / "reports" / "leaderboard.csv").read_text())
    monkeypatch.setattr(run_experiments, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run_experiments, "LEADERBOARD", board)
    # Rebuild with no registered results: only the R13 row is regenerated.
    run_experiments.write_leaderboard([])
    with board.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["run_id"] for r in rows] == [r13_id]
    assert rows[0] == current[r13_id]
