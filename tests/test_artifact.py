"""The packaged models reproduce their recorded predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FEATURE_SETS
from credit_risk.data import data_sha256, load_raw
from credit_risk.packaging import load_artifact
from credit_risk.pipeline import score_applications

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "pd-lgbm-no-apr-v1"


@pytest.mark.parametrize(
    ("artifact", "predictions", "column"),
    [
        ("pd-lgbm-no-apr-v1", "test_predictions.csv", "pd_calibrated"),
        ("pd-lgbm-no-apr-v1.1", "test_predictions_v1.1.csv", "pd_v1_1"),
        ("pd-lgbm-no-apr-v2", "gap_predictions_v2.csv", "pd_v2"),
    ],
)
def test_artifact_reproduces_recorded_predictions(data_path, artifact, predictions, column):
    _, meta = load_artifact(ROOT / "models" / artifact)
    assert meta["data_sha256"] == data_sha256(data_path)
    df = load_raw(data_path)
    recorded = pd.read_csv(ROOT / "reports" / predictions)
    rows = df.set_index("loan_id").loc[recorded["loan_id"]].reset_index()
    scored = score_applications(ROOT / "models" / artifact, rows[list(FEATURE_SETS["no_apr"])])
    np.testing.assert_allclose(scored.to_numpy(), recorded[column].to_numpy(), atol=1e-9)


def test_artifact_metadata_matches_selection():
    _, meta = load_artifact(ARTIFACT)
    assert meta["candidate"]["name"] == "R05-lgbm-no-apr"
    assert meta["candidate"]["feature_set"] == "no_apr"
    assert meta["input_columns"] == list(FEATURE_SETS["no_apr"])
    assert "apr" not in meta["input_columns"]
    board = pd.read_csv(ROOT / "reports" / "leaderboard.csv")
    selected = board[board["selected"] == "yes"]
    assert list(selected["run_id"]) == ["R05-lgbm-no-apr"]
    assert abs(selected["test_score"].iloc[0] - meta["metrics"]["test"]["roc_auc"]) < 1e-5
