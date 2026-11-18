"""Constants, column roles, feature sets and the split protocol."""

from dataclasses import dataclass

TARGET = "default_12m"
WEIGHT = "sample_weight"
TIME = "vintage"
ID = "loan_id"

SPECIAL_VALUES: dict[str, int] = {"fico_score": 99999, "income": -1, "inquiries_last_6m": 99}
LEAKAGE_COLUMNS = (
    "days_past_due_current",
    "total_payments_to_date",
    "months_on_book",
    "sample_weight",
)
SENSITIVE_COLUMNS = ("age",)

NUMERIC_INPUTS = (
    "fico_score",
    "income",
    "debt_to_income",
    "num_open_trades",
    "utilization_rate",
    "inquiries_last_6m",
    "loan_amount",
    "term",
    "apr",
    "employment_length",
)
CATEGORICAL_INPUTS = ("channel", "product_type", "state")
APPLICATION_COLUMNS = NUMERIC_INPUTS + CATEGORICAL_INPUTS

CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "channel": ("Direct", "Partner", "Mail"),
    "product_type": ("Personal", "Debt_Consolidation", "Auto", "Home_Improvement", "Business"),
    "state": ("CA", "TX", "FL", "NY", "GA", "IL", "Other"),
}

RAW_SCHEMA: dict[str, str] = {
    "loan_id": "str",
    "origination_date": "date",
    "vintage": "int",
    "months_on_book": "int",
    "fico_score": "int",
    "income": "int",
    "debt_to_income": "float",
    "num_open_trades": "int",
    "utilization_rate": "float",
    "inquiries_last_6m": "int",
    "loan_amount": "int",
    "term": "int",
    "apr": "float",
    "channel": "str",
    "product_type": "str",
    "employment_length": "int",
    "state": "str",
    "age": "int",
    "days_past_due_current": "int",
    "total_payments_to_date": "float",
    "sample_weight": "float",
    "default_12m": "int",
}

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "full": APPLICATION_COLUMNS,
    "no_apr": tuple(c for c in APPLICATION_COLUMNS if c != "apr"),
    "fico_only": ("fico_score",),
    # Ablation only, to measure leakage inflation; never selectable.
    "leaky": APPLICATION_COLUMNS
    + ("days_past_due_current", "total_payments_to_date", "months_on_book"),
}
SELECTABLE_FEATURE_SETS = ("full", "no_apr", "fico_only")


@dataclass(frozen=True)
class SplitProtocol:
    """Out-of-time split boundaries as YYYYMM vintages (inclusive).

    Attributes:
        train_end: Last training vintage.
        val_start: First validation vintage.
        val_end: Last validation vintage.
        test_start: First test vintage; vintages between val_end and it are "gap".
        test_end: Last test vintage.
        split_id: Identifier recorded with results.
    """

    train_end: int = 202206
    val_start: int = 202207
    val_end: int = 202212
    test_start: int = 202401
    test_end: int = 202409
    split_id: str = "oot-v1"


DEFAULT_PROTOCOL = SplitProtocol()
