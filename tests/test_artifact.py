"""The packaged model reproduces the recorded test predictions."""

from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.config import APPLICATION_COLUMNS
from credit_risk.data import data_sha256, load_raw
from credit_risk.packaging import load_artifact
from credit_risk.pipeline import score_applications

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "pd-lgbm-no-apr-v1"
PREDICTIONS = ROOT / "reports" / "test_predictions.csv"


def test_artifact_reproduces_test_predictions(data_path):
    _, meta = load_artifact(ARTIFACT)
    assert meta["data_sha256"] == data_sha256(data_path)
    df = load_raw(data_path)
    recorded = pd.read_csv(PREDICTIONS)
    rows = df.set_index("loan_id").loc[recorded["loan_id"]].reset_index()
    scored = score_applications(ARTIFACT, rows[list(APPLICATION_COLUMNS)])
    np.testing.assert_allclose(scored.to_numpy(), recorded["pd_calibrated"].to_numpy(), atol=1e-9)


def test_artifact_metadata_matches_selection():
    _, meta = load_artifact(ARTIFACT)
    assert meta["candidate"]["name"] == "R05-lgbm-no-apr"
    assert meta["candidate"]["feature_set"] == "no_apr"
    assert meta["input_columns"] == list(APPLICATION_COLUMNS)
    board = pd.read_csv(ROOT / "reports" / "leaderboard.csv")
    selected = board[board["selected"] == "yes"]
    assert list(selected["run_id"]) == ["R05-lgbm-no-apr"]
    assert abs(selected["test_score"].iloc[0] - meta["metrics"]["test"]["roc_auc"]) < 1e-5
