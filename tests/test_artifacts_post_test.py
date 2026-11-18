"""The packaged post-test models reproduce their recorded predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_risk.config import FEATURE_SETS
from credit_risk.data import load_raw
from credit_risk.pipeline import score_applications

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("artifact", "predictions", "column"),
    [
        ("pd-lgbm-no-apr-v1.1", "test_predictions_v1.1.csv", "pd_v1_1"),
        ("pd-lgbm-no-apr-v2", "gap_predictions_v2.csv", "pd_v2"),
    ],
)
def test_artifact_reproduces_recorded_predictions(data_path, artifact, predictions, column):
    df = load_raw(data_path)
    recorded = pd.read_csv(ROOT / "reports" / predictions)
    rows = df.set_index("loan_id").loc[recorded["loan_id"]].reset_index()
    scored = score_applications(ROOT / "models" / artifact, rows[list(FEATURE_SETS["no_apr"])])
    np.testing.assert_allclose(scored.to_numpy(), recorded[column].to_numpy(), atol=1e-9)


def test_perturbed_prediction_is_detected(data_path):
    df = load_raw(data_path)
    recorded = pd.read_csv(ROOT / "reports" / "gap_predictions_v2.csv")
    rows = df.set_index("loan_id").loc[recorded["loan_id"]].reset_index()
    scored = score_applications(
        ROOT / "models" / "pd-lgbm-no-apr-v2", rows[list(FEATURE_SETS["no_apr"])]
    )
    perturbed = recorded["pd_v2"].to_numpy().copy()
    perturbed[0] += 1e-6
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(scored.to_numpy(), perturbed, atol=1e-9)
