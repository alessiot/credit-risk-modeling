from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "credit_risk_data_enhanced.csv"


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """3,000 random rows of the real CSV."""
    return pd.read_csv(DATA).sample(3000, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def data_path() -> Path:
    """Path of the real CSV."""
    return DATA
