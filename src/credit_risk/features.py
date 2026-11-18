"""Stateless cleaning and feature engineering on raw application columns."""

import logging

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from credit_risk.config import CATEGORY_LEVELS, FEATURE_SETS, SPECIAL_VALUES

logger = logging.getLogger(__name__)

_PAYMENT_INPUTS = ("loan_amount", "income", "apr", "term")


def clean_special_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace placeholder codes with NaN and add missing flags.

    Args:
        df: Frame with raw columns.

    Returns:
        Copy with `<col>_missing` (0/1 int) added and placeholders set to NaN, for each
        SPECIAL_VALUES column present.
    """
    out = df.copy()
    for col, placeholder in SPECIAL_VALUES.items():
        if col not in out.columns:
            continue
        is_placeholder = out[col] == placeholder
        out[f"{col}_missing"] = is_placeholder.astype(int)
        out[col] = out[col].astype(float).mask(is_placeholder)
    return out


def _monthly_payment(amount: pd.Series, apr_pct: pd.Series, term: pd.Series) -> pd.Series:
    """Compute the amortized monthly payment, safe for zero APR.

    Args:
        amount: Loan amount.
        apr_pct: Annual percentage rate in percent.
        term: Term in months.

    Returns:
        Monthly payment.
    """
    r = apr_pct / 100.0 / 12.0
    amount = amount.astype(float)
    term = term.astype(float)
    zero = r == 0
    safe_r = r.mask(zero, 1.0)
    annuity = amount * safe_r / (1.0 - (1.0 + safe_r) ** -term)
    return annuity.mask(zero, amount / term)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add loan_to_income and payment_to_income from the row's own columns.

    Args:
        df: Frame with loan_amount, income, apr and term (NaN income allowed).

    Returns:
        Copy with the two ratios added; NaN where income is NaN or <= 0.
    """
    out = df.copy()
    income = out["income"].astype(float)
    income = income.where(income > 0)
    out["loan_to_income"] = out["loan_amount"] / income
    out["payment_to_income"] = _monthly_payment(out["loan_amount"], out["apr"], out["term"]) / (
        income / 12.0
    )
    return out


def _has(columns: tuple[str, ...], needed: tuple[str, ...]) -> bool:
    """Check that all `needed` are in `columns`.

    Args:
        columns: Available columns.
        needed: Required columns.

    Returns:
        True if all are present.
    """
    return all(c in columns for c in needed)


def output_columns(feature_set: str) -> list[str]:
    """List the columns `make_preprocessor(feature_set)` produces, in order.

    Args:
        feature_set: Key of FEATURE_SETS.

    Returns:
        Column names.

    Raises:
        ValueError: For an unknown feature set.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    cols = FEATURE_SETS[feature_set]
    out = list(cols) + [f"{c}_missing" for c in SPECIAL_VALUES if c in cols]
    if _has(cols, ("loan_amount", "income")):
        out.append("loan_to_income")
    if _has(cols, _PAYMENT_INPUTS):
        out.append("payment_to_income")
    return out


def _prepare(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Select, clean and engineer features for a feature set.

    Args:
        X: Frame containing at least the set's raw columns.
        feature_set: Key of FEATURE_SETS.

    Returns:
        Frame with columns `output_columns(feature_set)`; categoricals have fixed levels.

    Raises:
        KeyError: Naming the missing input columns.
    """
    cols = FEATURE_SETS[feature_set]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"missing input columns: {missing}")
    df = clean_special_values(X[list(cols)])
    if _has(cols, ("loan_amount", "income")):
        if _has(cols, _PAYMENT_INPUTS):
            df = add_engineered_features(df)
        else:
            income = df["income"].where(df["income"] > 0)
            df["loan_to_income"] = df["loan_amount"] / income
    for col, levels in CATEGORY_LEVELS.items():
        if col not in df.columns:
            continue
        unseen = df[col].notna() & ~df[col].isin(levels)
        if unseen.any():
            logger.warning("%d unseen values in %s set to NaN", int(unseen.sum()), col)
        df[col] = pd.Categorical(df[col].mask(unseen), categories=list(levels))
    return df[output_columns(feature_set)]


def make_preprocessor(feature_set: str, *, allow_leaky: bool = False) -> FunctionTransformer:
    """Build the stateless preprocessing step for a feature set.

    Args:
        feature_set: Key of FEATURE_SETS.
        allow_leaky: Permit the "leaky" ablation set (never for a real model).

    Returns:
        FunctionTransformer mapping raw columns to the cleaned, engineered DataFrame.

    Raises:
        ValueError: For an unknown feature set, or "leaky" without `allow_leaky`.
    """
    output_columns(feature_set)
    if feature_set == "leaky" and not allow_leaky:
        raise ValueError('feature set "leaky" is an ablation only; pass allow_leaky=True')
    return FunctionTransformer(
        _prepare, kw_args={"feature_set": feature_set}, validate=False, check_inverse=False
    )
