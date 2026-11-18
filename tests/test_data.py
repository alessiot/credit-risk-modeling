import pytest

from credit_risk.config import RAW_SCHEMA
from credit_risk.data import DataValidationError, data_sha256, load_raw
from credit_risk.packaging import file_sha256
from credit_risk.schema import SchemaError


@pytest.fixture
def good(raw_sample, tmp_path):
    return raw_sample.copy(), tmp_path / "d.csv"


def test_schema_covers_csv(raw_sample):
    assert list(raw_sample.columns) == list(RAW_SCHEMA)


def test_load_ok(good, caplog):
    df, p = good
    df.to_csv(p, index=False)
    with caplog.at_level("INFO"):
        out = load_raw(p)
    assert len(out) == 3000
    assert "default rate" in caplog.text
    assert data_sha256(p) == file_sha256(p)


@pytest.mark.parametrize(
    ("col", "value", "msg"),
    [
        ("loan_id", "LOAN_X", "not unique"),
        ("default_12m", 2, "default_12m"),
        ("vintage", 202001, "origination_date"),
        ("term", 48, "term"),
        ("utilization_rate", 1.5, "utilization_rate"),
        ("debt_to_income", -0.1, "debt_to_income"),
        ("fico_score", 200, "fico_score"),
    ],
)
def test_value_checks(good, col, value, msg):
    df, p = good
    if col == "loan_id":
        df.loc[1, col] = df.loc[0, col]
    else:
        df.loc[0, col] = value
    df.to_csv(p, index=False)
    with pytest.raises(DataValidationError, match=msg):
        load_raw(p)


def test_placeholder_fico_allowed(good):
    df, p = good
    df.loc[0, "fico_score"] = 99999
    df.to_csv(p, index=False)
    load_raw(p)


def test_nan_and_extra_column_reported_together(good):
    df, p = good
    df.loc[0, "apr"] = float("nan")
    df["extra"] = 1
    df.to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="NaN") as exc:
        load_raw(p)
    assert "unexpected columns" in str(exc.value)
    assert isinstance(exc.value, SchemaError)


def test_nan_in_float_column(good):
    df, p = good
    df.loc[0, "apr"] = float("nan")
    df.to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="NaN"):
        load_raw(p)


def test_multiple_failures_one_message(good):
    df, p = good
    df.loc[0, "term"] = 48
    df.loc[1, "utilization_rate"] = 2.0
    df.to_csv(p, index=False)
    with pytest.raises(DataValidationError) as exc:
        load_raw(p)
    assert "term" in str(exc.value) and "utilization_rate" in str(exc.value)


def test_real_csv_loads(data_path):
    assert len(load_raw(data_path)) == 50000
