"""Model pipelines, cross-validation and calibration."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from credit_risk.config import CATEGORICAL_INPUTS, FEATURE_SETS
from credit_risk.features import make_preprocessor, output_columns

logger = logging.getLogger(__name__)

FAMILIES = ("base_rate", "logreg", "lgbm", "scorecard")
CALIBRATION_METHODS = ("logit_intercept", "logit_platt", "sigmoid", "isotonic")
_LOGIT_CLIP = 1e-6

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


class SupervisedBinner(TransformerMixin, BaseEstimator):
    """Bin each numeric column at the split points of a shallow decision tree on the target.

    NaN gets its own bin, so missing placeholders keep their own default rate.

    Attributes:
        max_bins: Maximum number of bins per column (leaves of the tree).
        min_bin_fraction: Minimum share of a column's non-NaN rows in each bin.
    """

    def __init__(self, max_bins: int = 6, min_bin_fraction: float = 0.05) -> None:
        """Store the binning parameters.

        Args:
            max_bins: Maximum number of bins per column.
            min_bin_fraction: Minimum share of non-NaN rows in each bin.
        """
        self.max_bins = max_bins
        self.min_bin_fraction = min_bin_fraction

    def fit(self, X: object, y: object = None) -> "SupervisedBinner":
        """Learn the thresholds per column.

        Args:
            X: 2-d array-like; NaN allowed.
            y: Binary target.

        Returns:
            self.

        Raises:
            ValueError: If `y` is None, its length differs from `X`, or a column is all NaN.
        """
        if y is None:
            raise ValueError("SupervisedBinner needs y")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(y) != len(X):
            raise ValueError(f"X must be 2-d and match y: {X.shape} vs {y.shape}")
        self.thresholds_: list[np.ndarray] = []
        for j, col in enumerate(X.T):
            ok = ~np.isnan(col)
            if not ok.any():
                raise ValueError(f"column {j} is all NaN")
            tree = DecisionTreeClassifier(
                max_leaf_nodes=self.max_bins,
                min_samples_leaf=int(np.ceil(self.min_bin_fraction * ok.sum())),
                random_state=0,
            ).fit(col[ok].reshape(-1, 1), y[ok])
            split = tree.tree_.feature >= 0
            self.thresholds_.append(np.sort(tree.tree_.threshold[split]))
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: object) -> np.ndarray:
        """Map values to integer bin ids.

        Args:
            X: 2-d array-like with the fitted number of columns.

        Returns:
            Float array of bin ids; NaN maps to `len(thresholds) + 1` of its column.

        Raises:
            ValueError: If the column count differs from the fit.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.thresholds_):
            raise ValueError(f"expected {len(self.thresholds_)} columns, got shape {X.shape}")
        out = np.empty(X.shape)
        for j, (col, thr) in enumerate(zip(X.T, self.thresholds_, strict=True)):
            ids = np.searchsorted(thr, col, side="left").astype(float)
            out[:, j] = np.where(np.isnan(col), len(thr) + 1, ids)
        return out


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


def _scorecard_encoder(
    feature_set: str, max_bins: int, min_bin_fraction: float
) -> ColumnTransformer:
    """Build the supervised binning and one-hot encoder for the scorecard.

    Args:
        feature_set: Key of FEATURE_SETS.
        max_bins: Maximum bins per numeric column.
        min_bin_fraction: Minimum share of non-NaN rows per bin.

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
                    ("bin", SupervisedBinner(max_bins=max_bins, min_bin_fraction=min_bin_fraction)),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
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
        family: One of "base_rate", "logreg", "lgbm", "scorecard".
        feature_set: Key of FEATURE_SETS.
        params: Family hyperparameters (logreg: C, n_bins; lgbm: LGBMClassifier kwargs;
            scorecard: C, max_bins, min_bin_fraction).
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
    elif family == "scorecard":
        steps.append(
            (
                "encode",
                _scorecard_encoder(
                    feature_set, params.get("max_bins", 6), params.get("min_bin_fraction", 0.05)
                ),
            )
        )
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
        fold_log_loss: Log loss per fold.
        ll_mean: Mean of fold_log_loss.
        ll_se: Standard error of fold_log_loss, std(ddof=1) / sqrt(n_folds).
    """

    fold_scores: list[float]
    cv_mean: float
    cv_se: float
    oof: pd.Series
    fold_log_loss: list[float]
    ll_mean: float
    ll_se: float


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
        CVResult with per-fold ROC AUC and log loss and out-of-fold scores.
    """
    w = None if sample_weight is None else np.asarray(sample_weight)
    oof = pd.Series(np.nan, index=X.index, dtype=float)
    scores: list[float] = []
    losses: list[float] = []
    for i, (tr, va) in enumerate(folds):
        model = build_model(family, feature_set, params, seed, allow_leaky=allow_leaky)
        _fit_pipeline(model, X.iloc[tr], y.iloc[tr], None if w is None else w[tr])
        pred = positive_scores(model, X.iloc[va])
        oof.iloc[va] = pred
        scores.append(float(roc_auc_score(y.iloc[va], pred)))
        losses.append(float(log_loss(y.iloc[va], pred, labels=[0, 1])))
        logger.info(
            "%s/%s fold %d AUC %.4f log loss %.4f", family, feature_set, i, scores[-1], losses[-1]
        )
    arr, ll = np.asarray(scores), np.asarray(losses)
    n = len(arr)
    se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    ll_se = float(ll.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return CVResult(scores, float(arr.mean()), se, oof, losses, float(ll.mean()), ll_se)


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


def _check_binary_target(y: object) -> np.ndarray:
    """Return `y` as an array after checking that it holds exactly the classes 0 and 1.

    Args:
        y: Binary target.

    Returns:
        The target as a numpy array.

    Raises:
        ValueError: If `y` does not contain both classes 0 and 1 and nothing else.
    """
    arr = np.asarray(y)
    if set(np.unique(arr).tolist()) != {0, 1}:
        raise ValueError("y must contain exactly the classes 0 and 1")
    return arr


def _fit_logit_shift(z: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """Fit the log-odds recalibration map z -> slope * z + intercept.

    The intercept method solves sum(y - sigmoid(z + c)) = 0, the maximum-likelihood shift, with
    the slope fixed at 1; the Platt method fits shift and slope by unpenalized logistic
    regression.

    Args:
        z: Clipped log-odds of the base probability.
        y: Binary target with both classes.
        method: "logit_intercept" or "logit_platt".

    Returns:
        Tuple (slope, intercept).

    Raises:
        ValueError: For an unknown method.
    """
    if method == "logit_intercept":
        c = brentq(lambda c: float(np.sum(y - expit(z + c))), -40.0, 40.0, xtol=1e-12)
        return 1.0, float(c)
    if method == "logit_platt":
        lr = LogisticRegression(C=np.inf, max_iter=2000).fit(z.reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    raise ValueError(f"unknown logit method {method!r}")


class LogitCalibrated(ClassifierMixin, BaseEstimator):
    """Recalibrate a fitted model's probability on the log-odds scale without refitting it.

    Attributes:
        estimator: Fitted model with `predict_proba`; never refitted.
        method: "logit_intercept" (shift only, slope fixed at 1) or "logit_platt" (shift and
            slope).
    """

    def __init__(self, estimator: object, method: str = "logit_intercept") -> None:
        """Store the base model and the method.

        Args:
            estimator: Fitted model with `predict_proba`.
            method: "logit_intercept" or "logit_platt".
        """
        self.estimator = estimator
        self.method = method

    def _logit(self, X: pd.DataFrame) -> np.ndarray:
        """Return the clipped log-odds of the base model's positive-class probability.

        Args:
            X: Features for the base model.

        Returns:
            Array of shape (n,).
        """
        p = self.estimator.predict_proba(X)[:, 1]
        return logit(np.clip(p, _LOGIT_CLIP, 1 - _LOGIT_CLIP))

    def fit(self, X: pd.DataFrame, y: object) -> "LogitCalibrated":
        """Fit the shift (and slope) on `X`, `y`.

        The intercept method solves sum(y - sigmoid(z + c)) = 0, the maximum-likelihood shift.

        Args:
            X: Features for the base model.
            y: Binary target.

        Returns:
            self.

        Raises:
            ValueError: For an unknown method or a single-class target.
        """
        if self.method not in ("logit_intercept", "logit_platt"):
            raise ValueError(f"unknown logit method {self.method!r}")
        y = _check_binary_target(y)
        self.slope_, self.intercept_ = _fit_logit_shift(self._logit(X), y, self.method)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated class probabilities.

        Args:
            X: Features for the base model.

        Returns:
            Array of shape (n, 2).
        """
        p = expit(self.slope_ * self._logit(X) + self.intercept_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features for the base model.

        Returns:
            Array of labels from `classes_`.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class ScoreCalibrator(BaseEstimator):
    """Recalibrate raw probabilities, fit on (probability, label) pairs.

    Attributes:
        method: "logit_intercept" (shift on the log-odds scale), "logit_platt" (shift and slope)
            or "isotonic" (monotone step function, output clipped to [1e-6, 1 - 1e-6]).
    """

    def __init__(self, method: str = "logit_intercept") -> None:
        """Store the method.

        Args:
            method: "logit_intercept", "logit_platt" or "isotonic".
        """
        self.method = method

    def fit(self, p: object, y: object) -> "ScoreCalibrator":
        """Fit the recalibration map.

        Args:
            p: Raw positive-class probabilities.
            y: Binary target aligned to `p`.

        Returns:
            self.

        Raises:
            ValueError: For an unknown method or a target without both classes.
        """
        if self.method not in ("logit_intercept", "logit_platt", "isotonic"):
            raise ValueError(f"unknown score calibration method {self.method!r}")
        p = np.asarray(p, dtype=float)
        y = _check_binary_target(y)
        if self.method == "isotonic":
            self.isotonic_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
                p, y
            )
        else:
            z = logit(np.clip(p, _LOGIT_CLIP, 1 - _LOGIT_CLIP))
            self.slope_, self.intercept_ = _fit_logit_shift(z, y, self.method)
        return self

    def transform(self, p: object) -> np.ndarray:
        """Map raw probabilities to calibrated ones.

        Args:
            p: Raw positive-class probabilities.

        Returns:
            Calibrated probabilities of shape (n,).
        """
        p = np.asarray(p, dtype=float)
        if self.method == "isotonic":
            return np.clip(self.isotonic_.predict(p), _LOGIT_CLIP, 1 - _LOGIT_CLIP)
        z = logit(np.clip(p, _LOGIT_CLIP, 1 - _LOGIT_CLIP))
        return expit(self.slope_ * z + self.intercept_)


class PrecalibratedModel(ClassifierMixin, BaseEstimator):
    """A fitted base model followed by a fitted score calibrator; neither is ever refitted.

    Attributes:
        estimator: Fitted model with `predict_proba`.
        calibrator: Fitted ScoreCalibrator.
    """

    def __init__(self, estimator: object, calibrator: ScoreCalibrator) -> None:
        """Store the fitted parts.

        Args:
            estimator: Fitted model with `predict_proba`.
            calibrator: Fitted ScoreCalibrator.
        """
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict calibrated class probabilities.

        Args:
            X: Features for the base model.

        Returns:
            Array of shape (n, 2).
        """
        p = self.calibrator.transform(self.estimator.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Features for the base model.

        Returns:
            Array of labels from `classes_`.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def calibrate(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: Literal["logit_intercept", "logit_platt", "sigmoid", "isotonic"],
) -> CalibratedClassifierCV | LogitCalibrated:
    """Calibrate a fitted model on validation data without refitting it.

    Args:
        model: Fitted pipeline.
        X_val: Raw validation features.
        y_val: Validation target.
        method: One of `CALIBRATION_METHODS`.

    Returns:
        Fitted LogitCalibrated for the logit methods, otherwise a fitted CalibratedClassifierCV
        wrapping a FrozenEstimator.

    Raises:
        ValueError: For an unknown method.
    """
    if method not in CALIBRATION_METHODS:
        raise ValueError(
            f"unknown calibration method {method!r}; choose from {CALIBRATION_METHODS}"
        )
    if method in ("logit_intercept", "logit_platt"):
        return LogitCalibrated(model, method).fit(X_val, y_val)
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
