import numpy as np
import pandas as pd
import pytest

from credit_risk.config import (
    FEATURE_SETS,
    LEAKAGE_COLUMNS,
    SENSITIVE_COLUMNS,
    TARGET,
    TIME,
)
from credit_risk.features import add_engineered_features, clean_special_values, make_preprocessor


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fico_score": [700, 99999, 650],
            "income": [60000, -1, 0],
            "debt_to_income": [0.2, 0.3, 0.4],
            "num_open_trades": [3, 4, 5],
            "utilization_rate": [0.1, 0.2, 0.3],
            "inquiries_last_6m": [0, 99, 2],
            "loan_amount": [12000, 5000, 7000],
            "term": [36, 60, 36],
            "apr": [12.0, 0.0, 8.0],
            "employment_length": [1, 2, 3],
            "channel": ["Direct", "Mail", "Partner"],
            "product_type": ["Auto", "Personal", "Business"],
            "state": ["CA", "TX", "Other"],
        }
    )


def test_clean_special_values():
    raw = _raw()
    out = clean_special_values(raw)
    assert out["fico_score_missing"].tolist() == [0, 1, 0]
    assert out["income_missing"].tolist() == [0, 1, 0]
    assert out["inquiries_last_6m_missing"].tolist() == [0, 1, 0]
    assert out["fico_score"].isna().tolist() == [False, True, False]
    assert raw["fico_score"].tolist() == [700, 99999, 650]  # input untouched


def test_engineered_values():
    out = add_engineered_features(clean_special_values(_raw()))
    assert out["loan_to_income"].iloc[0] == pytest.approx(0.2)
    r = 0.12 / 12
    pay = 12000 * r / (1 - (1 + r) ** -36)
    assert out["payment_to_income"].iloc[0] == pytest.approx(pay / 5000)
    assert out["payment_to_income"].isna().tolist() == [False, True, True]
    assert out["loan_to_income"].isna().tolist() == [False, True, True]


def test_zero_apr_payment():
    df = clean_special_values(_raw()).assign(income=60000.0)
    out = add_engineered_features(df)
    assert out["payment_to_income"].iloc[1] == pytest.approx(5000 / 60 / 5000)


def test_preprocessor_full_dtypes():
    out = make_preprocessor("full").fit_transform(_raw())
    assert isinstance(out, pd.DataFrame)
    assert out["channel"].dtype.name == "category"
    assert list(out["channel"].cat.categories) == ["Direct", "Partner", "Mail"]
    assert "payment_to_income" in out and "loan_to_income" in out


def test_no_apr_keeps_loan_to_income_only():
    out = make_preprocessor("no_apr").fit_transform(_raw())
    assert "loan_to_income" in out
    assert "payment_to_income" not in out and "apr" not in out


def test_fico_only_columns():
    out = make_preprocessor("fico_only").fit_transform(_raw())
    assert list(out.columns) == ["fico_score", "fico_score_missing"]


def test_unseen_category_nan_and_warns(caplog):
    raw = _raw()
    raw.loc[0, "state"] = "WA"
    with caplog.at_level("WARNING"):
        out = make_preprocessor("full").fit_transform(raw)
    assert pd.isna(out["state"].iloc[0])
    assert "1 unseen values in state" in caplog.text


def test_missing_columns_keyerror():
    with pytest.raises(KeyError, match="apr"):
        make_preprocessor("full").fit_transform(_raw().drop(columns="apr"))


def test_unknown_feature_set():
    with pytest.raises(ValueError, match="unknown feature set"):
        make_preprocessor("nope")


@pytest.mark.parametrize("feature_set", ["full", "no_apr", "fico_only"])
def test_no_leakage_columns_used(raw_sample, feature_set):
    df = raw_sample.copy()
    prep = make_preprocessor(feature_set)
    base = prep.fit_transform(df)
    rng = np.random.default_rng(0)
    shuffled = df.copy()
    for c in (TARGET, TIME, "origination_date", *SENSITIVE_COLUMNS, *LEAKAGE_COLUMNS):
        shuffled[c] = rng.permutation(shuffled[c].to_numpy())
    assert not shuffled[["age", "sample_weight"]].equals(df[["age", "sample_weight"]])
    pd.testing.assert_frame_equal(prep.transform(shuffled), base)
    assert not set(base.columns) & (set(LEAKAGE_COLUMNS) | {TARGET, TIME, "age"})
    assert set(FEATURE_SETS[feature_set]) <= set(base.columns)


def test_leaky_preprocessor_guard():
    with pytest.raises(ValueError, match="leaky"):
        make_preprocessor("leaky")
    make_preprocessor("leaky", allow_leaky=True)
