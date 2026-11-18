import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_flow_end_to_end(tmp_path):
    sweep = tmp_path / "sweep.json"
    sweep.write_text(
        '[{"name": "lr", "family": "logreg", "feature_set": "full", "params": {"C": 1.0}},'
        ' {"name": "gb", "family": "lgbm", "feature_set": "full",'
        ' "params": {"n_estimators": 20, "num_leaves": 4, "min_child_samples": 100}}]'
    )
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
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "Warning" not in proc.stderr
    assert "Warning" not in proc.stdout
    artifacts = [p for p in out.iterdir() if (p / "model.joblib").exists()]
    assert len(artifacts) == 1
    assert (artifacts[0] / "metadata.json").exists()
