import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SWEEP = (
    '[{"name": "lr", "family": "logreg", "feature_set": "full", "params": {"C": 1.0}},'
    ' {"name": "gb", "family": "lgbm", "feature_set": "full",'
    ' "params": {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 100}}]'
)


def _run_flow(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    sweep = tmp_path / "sweep.json"
    sweep.write_text(SWEEP)
    out = tmp_path / "models"
    env = os.environ | {
        "METAFLOW_HOME": str(tmp_path / "mf_home"),
        "METAFLOW_DATASTORE_SYSROOT_LOCAL": str(tmp_path),
        "USERNAME": "pytest",
    }
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "flows" / "credit_risk_flow.py"),
            "run",
            "--sweep",
            str(sweep),
            "--output_dir",
            str(out),
            *extra,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc, out


def _metadata(proc: subprocess.CompletedProcess, out: Path) -> dict:
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "Warning" not in proc.stderr
    assert "Warning" not in proc.stdout
    artifacts = [p for p in out.iterdir() if (p / "model.joblib").exists()]
    assert len(artifacts) == 1
    return json.loads((artifacts[0] / "metadata.json").read_text())


@pytest.mark.slow
def test_flow_end_to_end_does_not_score_test_by_default(tmp_path):
    meta = _metadata(*_run_flow(tmp_path))
    assert meta["metrics"]["test"] is None
    assert meta["metrics"]["gap"]["name"] == "gap"
    assert "brier_skill" in meta["metrics"]["gap"]
    assert meta["extra"]["score_test"] is False


@pytest.mark.slow
def test_flow_scores_test_when_asked(tmp_path):
    meta = _metadata(*_run_flow(tmp_path, "--score_test", "true", "--calibration", "logit_platt"))
    assert meta["metrics"]["test"]["name"] == "test"
    assert meta["extra"]["score_test"] is True
    assert meta["extra"]["calibration"] == "logit_platt"


def test_flow_rejects_unknown_calibration(tmp_path):
    proc, _ = _run_flow(tmp_path, "--calibration", "beta")
    assert proc.returncode != 0
    assert "calibration must be one of" in proc.stderr + proc.stdout


def test_compare_runs_blank_test_columns_when_not_scored():
    code = f"""
import importlib.util, math
from datetime import datetime
from types import SimpleNamespace

path = {str(ROOT / "scripts" / "compare_runs.py")!r}
spec = importlib.util.spec_from_file_location("compare_runs", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cand = SimpleNamespace(name="lr")
data = SimpleNamespace(
    gap_eval={{"metrics": {{"roc_auc": 0.7}}, "ece": 0.01}},
    test_eval=None,
    sweep="flows/sweep.json",
    train_end=202206,
    calibration="sigmoid",
    champion=SimpleNamespace(candidate=cand, cv_mean=0.65, val_auc=0.66),
    artifact_path="models/x",
)
run = SimpleNamespace(id="1", created_at=datetime(2026, 1, 1), data=data)
row = mod.summarize(run, with_test=True)
assert math.isnan(row["test_auc"]) and math.isnan(row["test_ece"]) and row["test_auc_ci"] == ""
"""
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, cwd=ROOT
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
