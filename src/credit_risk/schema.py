"""Frame schema validation."""

import logging
from collections.abc import Mapping

import pandas as pd
from pandas.api import types as pdt

logger = logging.getLogger(__name__)

_DATE_FORMAT = "%Y-%m-%d"
_KINDS = ("int", "float", "str", "date")


class SchemaError(ValueError):
    """Raised when a frame does not match its schema."""


def _kind_ok(series: pd.Series, kind: str) -> bool:
    """Check whether a column matches a schema kind.

    Args:
        series: Column to check.
        kind: One of "int", "float", "str", "date".

    Returns:
        True if the column is of that kind.
    """
    if kind == "int":
        return bool(pdt.is_integer_dtype(series))
    if kind == "float":
        return bool(pdt.is_float_dtype(series) or pdt.is_integer_dtype(series))
    if kind == "str":
        return bool(pdt.is_string_dtype(series) and not pdt.is_bool_dtype(series))
    if pdt.is_datetime64_any_dtype(series):
        return True
    if not pdt.is_string_dtype(series):
        return False
    try:
        pd.to_datetime(series, format=_DATE_FORMAT, errors="raise")
    except ValueError:
        return False
    return True


def validate_frame(
    df: pd.DataFrame, schema: Mapping[str, str], *, allow_extra: bool = True
) -> None:
    """Validate column presence and kinds.

    Args:
        df: Frame to check.
        schema: Mapping of column name to kind ("int", "float", "str", "date").
        allow_extra: If False, columns not in the schema are reported.

    Raises:
        SchemaError: Listing every missing, wrong-kind and (optionally) unexpected column.
    """
    unknown = {c: k for c, k in schema.items() if k not in _KINDS}
    if unknown:
        raise SchemaError(f"unknown schema kinds: {unknown}")
    missing = [c for c in schema if c not in df.columns]
    wrong = [
        f"{c} (expected {k}, got {df[c].dtype})"
        for c, k in schema.items()
        if c in df.columns and not _kind_ok(df[c], k)
    ]
    extra = [] if allow_extra else [c for c in df.columns if c not in schema]
    problems = []
    if missing:
        problems.append(f"missing columns: {missing}")
    if wrong:
        problems.append(f"wrong kind: {wrong}")
    if extra:
        problems.append(f"unexpected columns: {extra}")
    if problems:
        raise SchemaError("; ".join(problems))
