"""Model pipelines, cross-validation and calibration."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from credit_risk.config import CATEGORICAL_INPUTS, FEATURE_SETS
from credit_risk.features import make_preprocessor, output_columns

logger = logging.getLogger(__name__)

FAMILIES = ("base_rate", "logreg", "lgbm")

MONOTONE_DIRECTIONS: dict[str, int] = {
    "fico_score": -1,
    "income": -1,
    "debt_to_income": 1,
    "utilization_rate": 1,
    "inquiries_last_6m": 1,
    "payment_to_income": 1,
    "loan_to_income": 1,
}


class MonotoneLGBM(ClassifierMixin, BaseEstimator):
    """LightGBM classifier whose monotone constraints are built from column names at fit time.

    Attributes:
        params: Keyword arguments for LGBMClassifier.
        random_state: Seed.
    """

    def __init__(self, params: Mapping[str, Any] | None = None, random_state: int = 0) -> None:
        """Store hyperparameters.

        Args:
            params: Keyword arguments for LGBMClassifier.
            random_state: Seed.
        """
        self.params = params
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: object,
        sample_weight: np.ndarray | None = None,
    ) -> "MonotoneLGBM":
        """Fit on a DataFrame, constraining columns listed in MONOTONE_DIRECTIONS.

        Args:
            X: Feature frame (category dtype is consumed natively).
            y: Binary target.
            sample_weight: Optional row weights.

        Returns:
            self.
        """
        constraints = [MONOTONE_DIRECTIONS.get(c, 0) for c in X.columns]
        self.model_ = lgb.LGBMClassifier(
            random_state=self.random_state,
            n_jobs=1,
            verbose=-1,
            monotone_constraints=constraints,
            **dict(self.params or {}),
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature frame.

        Returns:
            Array of shape (n, 2).
        """
        return self.model_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature frame.

        Returns:
            Array of labels from `classes_`.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Split-count importances of the fitted model."""
        return self.model_.feature_importances_

    @property
    def booster_(self) -> lgb.Booster:
        """Underlying fitted booster."""
        return self.model_.booster_


class QuantileBinner(TransformerMixin, BaseEstimator):
    """One-hot quantile binning that adapts to ties and low-cardinality columns.

    A column with at most `n_bins` distinct values gets one bin per value. Otherwise bin edges
    are the distinct quantiles strictly inside the column range, so no bin is empty on the fit
    data (KBinsDiscretizer warns about empty bins).

    Attributes:
        n_bins: Maximum number of bins per column.
    """

    def __init__(self, n_bins: int = 10) -> None:
        """Store the bin count.

        Args:
            n_bins: Maximum number of bins per column.
        """
        self.n_bins = n_bins

    def fit(self, X: np.ndarray, y: object = None) -> "QuantileBinner":
        """Learn bin edges per column.

        Args:
            X: Array without NaN.
            y: Ignored.

        Returns:
            self.
        """
        X = np.asarray(X, dtype=float)
        if np.isnan(X).any():
            raise ValueError("QuantileBinner does not accept NaN; impute first")
        self.edges_: list[np.ndarray] = []
        for col in X.T:
            values = np.unique(col)
            if len(values) <= self.n_bins:
                edges = (values[:-1] + values[1:]) / 2.0
            else:
                probs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
                edges = np.unique(np.quantile(col, probs, method="averaged_inverted_cdf"))
                # A value equal to an edge goes to the upper bin: edges <= min leave bin 0
                # empty, edges > max leave the last bin empty.
                edges = edges[(edges > values[0]) & (edges <= values[-1])]
            self.edges_.append(edges)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """One-hot encode each column by bin.

        Args:
            X: Array without NaN.

        Returns:
            Dense indicator matrix.
        """
        X = np.asarray(X, dtype=float)
        if np.isnan(X).any():
            raise ValueError("QuantileBinner does not accept NaN; impute first")
        blocks = []
        for col, edges in zip(X.T, self.edges_, strict=True):
            idx = np.searchsorted(edges, col, side="right")
            blocks.append(np.eye(len(edges) + 1)[idx])
        return np.hstack(blocks)


def _logreg_encoder(feature_set: str, n_bins: int) -> ColumnTransformer:
    """Build the binning and one-hot encoder for the logistic regression.

    Args:
        feature_set: Key of FEATURE_SETS.
        n_bins: Quantile bins per numeric column.

    Returns:
        ColumnTransformer over the preprocessor's output columns.
    """
    cols = output_columns(feature_set)
    flags = [c for c in cols if c.endswith("_missing")]
    cats = [c for c in cols if c in CATEGORICAL_INPUTS]
    nums = [c for c in cols if c not in flags and c not in cats]
    parts: list[tuple[str, Any, list[str]]] = [
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("bin", QuantileBinner(n_bins=n_bins)),
                ]
            ),
            nums,
        )
    ]
    if flags:
        parts.append(("flags", "passthrough", flags))
    if cats:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cats,
            )
        )
    return ColumnTransformer(parts, sparse_threshold=0.0)


def build_model(
    family: str,
    feature_set: str = "full",
    params: Mapping[str, Any] | None = None,
    seed: int = 0,
    *,
    allow_leaky: bool = False,
) -> Pipeline:
    """Build an unfitted pipeline from raw application columns to PD.

    Args:
        family: One of "base_rate", "logreg", "lgbm".
        feature_set: Key of FEATURE_SETS.
        params: Family hyperparameters (logreg: C, n_bins; lgbm: LGBMClassifier kwargs).
        seed: Random seed.
        allow_leaky: Permit the "leaky" ablation feature set.

    Returns:
        Pipeline whose final step is named "model".

    Raises:
        ValueError: For an unknown family or feature set, or "leaky" without `allow_leaky`.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {list(FAMILIES)}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    params = dict(params or {})
    steps: list[tuple[str, Any]] = [
        ("prep", make_preprocessor(feature_set, allow_leaky=allow_leaky))
    ]
    if family == "base_rate":
        steps.append(("model", DummyClassifier(strategy="prior")))
    elif family == "logreg":
        steps.append(("encode", _logreg_encoder(feature_set, params.get("n_bins", 10))))
        steps.append(
            ("model", LogisticRegression(C=params.get("C", 1.0), max_iter=2000, random_state=seed))
        )
    else:
        steps.append(("model", MonotoneLGBM(params=params, random_state=seed)))
    return Pipeline(steps)


@dataclass
class CVResult:
    """Cross-validation outcome.

    Attributes:
        fold_scores: ROC AUC per fold.
        cv_mean: Mean of fold_scores.
        cv_se: Standard error, std(ddof=1) / sqrt(n_folds).
        oof: Out-of-fold scores; NaN for rows never in a validation fold.
    """

    fold_scores: list[float]
    cv_mean: float
    cv_se: float
    oof: pd.Series


def _fit_pipeline(
    model: Pipeline,
    X: pd.DataFrame,
    y: object,
    sample_weight: np.ndarray | None,
) -> Pipeline:
    """Fit a pipeline, passing weights to the final estimator only.

    Args:
        model: Unfitted pipeline.
        X: Raw features.
        y: Target.
        sample_weight: Optional weights.

    Returns:
        The fitted pipeline.
    """
    if sample_weight is None:
        return model.fit(X, y)
    return model.fit(X, y, model__sample_weight=sample_weight)


def cross_validate(
    family: str,
    feature_set: str,
    params: Mapping[str, Any] | None,
    X: pd.DataFrame,
    y: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    sample_weight: pd.Series | np.ndarray | None = None,
    seed: int = 0,
    allow_leaky: bool = False,
) -> CVResult:
    """Cross-validate over given folds, fitting a fresh pipeline per fold.

    Args:
        family: Model family.
        feature_set: Key of FEATURE_SETS.
        params: Family hyperparameters.
        X: Raw features.
        y: Target aligned to X.
        folds: (train_idx, val_idx) positional indices.
        sample_weight: Optional weights aligned to X; used from training rows only.
        seed: Random seed.
        allow_leaky: Permit the "leaky" ablation feature set.

    Returns:
        CVResult with per-fold ROC AUC and out-of-fold scores.
    """
    w = None if sample_weight is None else np.asarray(sample_weight)
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    scores: list[float] = []
    for i, (tr, va) in enumerate(folds):
        model = build_model(family, feature_set, params, seed, allow_leaky=allow_leaky)
        _fit_pipeline(model, X.iloc[tr], y.iloc[tr], None if w is None else w[tr])
        pred = positive_scores(model, X.iloc[va])
        oof.iloc[va] = pred
        scores.append(float(roc_auc_score(y.iloc[va], pred)))
        logger.info("%s/%s fold %d AUC %.4f", family, feature_set, i, scores[-1])
    arr = np.asarray(scores)
    se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else float("nan")
    return CVResult(scores, float(arr.mean()), se, oof)


def fit_model(
    family: str,
    feature_set: str,
    params: Mapping[str, Any] | None,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    sample_weight: pd.Series | np.ndarray | None = None,
    seed: int = 0,
    allow_leaky: bool = False,
) -> Pipeline:
    """Build and fit a pipeline.

    Args:
        family: Model family.
        feature_set: Key of FEATURE_SETS.
        params: Family hyperparameters.
        X: Raw features.
        y: Target.
        sample_weight: Optional weights passed to the final estimator.
        seed: Random seed.
        allow_leaky: Permit the "leaky" ablation feature set.

    Returns:
        Fitted pipeline.
    """
    w = None if sample_weight is None else np.asarray(sample_weight)
    return _fit_pipeline(
        build_model(family, feature_set, params, seed, allow_leaky=allow_leaky), X, y, w
    )


def calibrate(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: Literal["isotonic", "sigmoid"],
) -> CalibratedClassifierCV:
    """Calibrate a fitted model on validation data without refitting it.

    Args:
        model: Fitted pipeline.
        X_val: Raw validation features.
        y_val: Validation target.
        method: Calibration method.

    Returns:
        Fitted CalibratedClassifierCV wrapping a FrozenEstimator.
    """
    return CalibratedClassifierCV(FrozenEstimator(model), method=method).fit(X_val, y_val)


def positive_scores(model: object, X: pd.DataFrame) -> np.ndarray:
    """Return the positive-class probability.

    Args:
        model: Fitted model with predict_proba.
        X: Raw features.

    Returns:
        PD array of shape (n,).
    """
    return model.predict_proba(X)[:, 1]
