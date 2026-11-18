import pandas as pd
import pytest

from credit_risk.schema import SchemaError, validate_frame

SCHEMA = {"a": "int", "b": "float", "c": "str", "d": "date"}


def frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"a": [1, 2], "b": [1.5, 2.5], "c": ["x", "y"], "d": ["2024-01-31", "2024-02-01"]}
    )


def test_valid():
    validate_frame(frame(), SCHEMA)
    df = frame().assign(a=[1.0, 2.0])  # float column where int is required fails
    with pytest.raises(SchemaError):
        validate_frame(df, SCHEMA)


def test_float_accepts_int_and_string_dtype_and_datetime():
    df = frame().assign(b=[1, 2], c=pd.array(["x", "y"], dtype="string"))
    df["d"] = pd.to_datetime(df["d"])
    validate_frame(df, SCHEMA)


def test_lists_everything_in_one_message():
    df = frame().drop(columns=["b"]).assign(a=["1", "2"], d=["bad", "2024-01-01"], extra=1)
    with pytest.raises(SchemaError) as exc:
        validate_frame(df, SCHEMA, allow_extra=False)
    msg = str(exc.value)
    for token in ("b", "a (expected int", "d (expected date", "extra"):
        assert token in msg
    assert isinstance(exc.value, ValueError)


def test_extra_allowed_by_default():
    validate_frame(frame().assign(extra=1), SCHEMA)
    with pytest.raises(SchemaError, match="extra"):
        validate_frame(frame().assign(extra=1), SCHEMA, allow_extra=False)
