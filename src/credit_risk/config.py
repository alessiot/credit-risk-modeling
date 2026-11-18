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

    def __post_init__(self) -> None:
        """Validate that the boundaries are YYYYMM months in a consistent order.

        Raises:
            ValueError: If a boundary is not a valid YYYYMM, `val_start` is not the month right
                after `train_end`, or the order val_start <= val_end < test_start <= test_end
                is violated.
        """
        bounds = (self.train_end, self.val_start, self.val_end, self.test_start, self.test_end)
        if any(not 1 <= b % 100 <= 12 or b // 100 < 1 for b in bounds):
            raise ValueError(f"split boundaries must be valid YYYYMM: {bounds}")
        year, month = divmod(self.train_end, 100)
        next_month = (year + 1) * 100 + 1 if month == 12 else self.train_end + 1
        if self.val_start != next_month:
            raise ValueError(
                f"val_start {self.val_start} must be the month after train_end {self.train_end}"
                f" ({next_month})"
            )
        if not self.val_start <= self.val_end < self.test_start <= self.test_end:
            raise ValueError(f"boundaries out of order: {bounds}")


DEFAULT_PROTOCOL = SplitProtocol()
