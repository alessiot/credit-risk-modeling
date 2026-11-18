"""Load and validate the raw loan CSV."""

import logging
from pathlib import Path

import pandas as pd

from credit_risk.config import ID, RAW_SCHEMA, SPECIAL_VALUES, TARGET, TIME
from credit_risk.packaging import file_sha256
from credit_risk.schema import SchemaError, validate_frame

logger = logging.getLogger(__name__)


class DataValidationError(SchemaError):
    """Raised when the raw data violates a schema or value check."""


def _nan_problem(df: pd.DataFrame) -> list[str]:
    """Describe columns containing NaN.

    Args:
        df: Raw frame.

    Returns:
        A one-element list with the message, or an empty list.
    """
    counts = df.isna().sum()
    counts = counts[counts > 0]
    if counts.empty:
        return []
    return [f"NaN values (placeholders are the only allowed missing encoding): {counts.to_dict()}"]


def _value_problems(df: pd.DataFrame) -> list[str]:
    """Check value-level invariants of a schema-valid frame.

    Args:
        df: Raw frame that passed schema validation.

    Returns:
        One message per violated check.
    """
    problems = _nan_problem(df)
    n_dup = int(df[ID].duplicated().sum())
    if n_dup:
        problems.append(f"{ID} not unique: {n_dup} duplicated rows")
    n_bad = int((~df[TARGET].isin([0, 1])).sum())
    if n_bad:
        problems.append(f"{TARGET} not in {{0, 1}}: {n_bad} rows")
    dates = pd.to_datetime(df["origination_date"], format="%Y-%m-%d")
    n_bad = int((dates.dt.year * 100 + dates.dt.month != df[TIME]).sum())
    if n_bad:
        problems.append(f"{TIME} differs from origination_date YYYYMM: {n_bad} rows")
    n_bad = int((~df["term"].isin([36, 60])).sum())
    if n_bad:
        problems.append(f"term not in {{36, 60}}: {n_bad} rows")
    n_bad = int((~df["utilization_rate"].between(0, 1)).sum())
    if n_bad:
        problems.append(f"utilization_rate outside [0, 1]: {n_bad} rows")
    n_bad = int((df["debt_to_income"] < 0).sum())
    if n_bad:
        problems.append(f"debt_to_income < 0: {n_bad} rows")
    fico = df["fico_score"]
    n_bad = int((~(fico.between(300, 850) | (fico == SPECIAL_VALUES["fico_score"]))).sum())
    if n_bad:
        problems.append(f"fico_score outside [300, 850] and not the placeholder: {n_bad} rows")
    return problems


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV and validate it.

    Args:
        path: CSV file.

    Returns:
        The raw frame, unchanged (placeholders included).

    Raises:
        DataValidationError: One message listing every failed check.
    """
    df = pd.read_csv(path)
    try:
        validate_frame(df, RAW_SCHEMA, allow_extra=False)
    except SchemaError as exc:
        # Columns may be absent or mistyped (e.g. ints turned float by NaN), so value checks
        # are skipped; report the schema failure together with any NaN counts.
        raise DataValidationError("; ".join([str(exc), *_nan_problem(df)])) from exc
    problems = _value_problems(df)
    if problems:
        raise DataValidationError("; ".join(problems))
    logger.info(
        "loaded %d rows from %s; vintages %d-%d; default rate %.4f",
        len(df),
        path,
        df[TIME].min(),
        df[TIME].max(),
        df[TARGET].mean(),
    )
    return df


def data_sha256(path: Path) -> str:
    """Hash the data file.

    Args:
        path: CSV file.

    Returns:
        SHA-256 hex digest.
    """
    return file_sha256(path)
