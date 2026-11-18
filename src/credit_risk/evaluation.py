"""Metrics, bootstrap intervals, PSI, calibration and cutoff tables."""

import logging
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

Metric = Callable[[np.ndarray, np.ndarray], float]

_PSI_FLOOR = 1e-6


def _check_inputs(y_true: object, y_score: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert labels and scores to arrays.

    Args:
        y_true: Binary labels.
        y_score: Predicted scores, same length as `y_true`.

    Returns:
        Tuple of the label array (int) and the score array (float).

    Raises:
        ValueError: If lengths differ, scores contain NaN, or labels have a single class.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError(f"y_true and y_score must be 1-d and equal length: {y.shape} vs {s.shape}")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("y_true must contain exactly the classes 0 and 1")
    return y.astype(int), s


def binary_metrics(y_true: object, y_score: object) -> dict[str, float]:
    """Compute discrimination, calibration and loss metrics for a binary score.

    Args:
        y_true: Binary labels.
        y_score: Predicted probabilities or scores.

    Returns:
        Dict with n, n_pos, base_rate, mean_score, roc_auc, gini, ks, pr_auc, brier, log_loss.

    Raises:
        ValueError: On a single class, length mismatch or NaN scores.
    """
    y, s = _check_inputs(y_true, y_score)
    auc = float(roc_auc_score(y, s))
    fpr, tpr, _ = roc_curve(y, s)
    clipped = np.clip(s, 1e-15, 1 - 1e-15)
    return {
        "n": float(len(y)),
        "n_pos": float(y.sum()),
        "base_rate": float(y.mean()),
        "mean_score": float(s.mean()),
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, s)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
    }


def _resample_idx(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a stratified bootstrap index sample.

    Args:
        pos: Indices of positives.
        neg: Indices of negatives.
        rng: Random generator.

    Returns:
        Concatenated resampled indices with the original class counts.
    """
    return np.concatenate(
        [rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
    )


def bootstrap_ci(
    y_true: object,
    y_score: object,
    metric: Metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Stratified percentile bootstrap interval for a metric.

    Args:
        y_true: Binary labels.
        y_score: Scores.
        metric: Callable `(y_true, y_score) -> float`.
        n_boot: Number of resamples.
        alpha: Total tail mass; interval is [alpha/2, 1 - alpha/2] percentiles.
        seed: Random seed.

    Returns:
        Tuple (point estimate on full data, lower, upper).
    """
    y, s = _check_inputs(y_true, y_score)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_idx(pos, neg, rng)
        vals[i] = metric(y[idx], s[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, s)), float(lo), float(hi)


def paired_bootstrap_diff(
    y_true: object,
    score_a: object,
    score_b: object,
    metric: Metric,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired stratified bootstrap of metric(a) - metric(b).

    Args:
        y_true: Binary labels.
        score_a: Scores of model A.
        score_b: Scores of model B.
        metric: Callable `(y_true, y_score) -> float`.
        n_boot: Number of resamples.
        alpha: Total tail mass.
        seed: Random seed.

    Returns:
        Dict with diff, lower, upper and share_a_better (share of resamples with diff > 0).
    """
    y, a = _check_inputs(y_true, score_a)
    _, b = _check_inputs(y_true, score_b)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_idx(pos, neg, rng)
        diffs[i] = metric(y[idx], a[idx]) - metric(y[idx], b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "diff": float(metric(y, a) - metric(y, b)),
        "lower": float(lo),
        "upper": float(hi),
        "share_a_better": float((diffs > 0).mean()),
    }


def calibration_table(y_true: object, y_score: object, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-bin calibration table.

    Args:
        y_true: Binary labels.
        y_score: Scores.
        n_bins: Requested number of quantile bins; tied edges merge bins.

    Returns:
        Frame with columns bin, n, mean_score, observed_rate.
    """
    y, s = _check_inputs(y_true, y_score)
    if s.min() == s.max():
        bins = pd.Series(0, index=range(len(s)))
    else:
        bins = pd.qcut(pd.Series(s), q=n_bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"bin": bins, "y": y, "s": s})
    out = (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_score=("s", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    out["bin"] = out["bin"].astype(int)
    return out


def expected_calibration_error(y_true: object, y_score: object, n_bins: int = 10) -> float:
    """Expected calibration error over quantile bins.

    Args:
        y_true: Binary labels.
        y_score: Scores.
        n_bins: Requested number of quantile bins.

    Returns:
        n-weighted mean of |mean_score - observed_rate|.
    """
    t = calibration_table(y_true, y_score, n_bins)
    gap = (t["mean_score"] - t["observed_rate"]).abs()
    return float((gap * t["n"]).sum() / t["n"].sum())


def psi(expected: Sequence[float], actual: Sequence[float], n_bins: int = 10) -> float:
    """Population stability index of `actual` against `expected`.

    Args:
        expected: Reference sample; its finite quantiles define the bin edges.
        actual: Comparison sample.
        n_bins: Requested number of quantile bins.

    Returns:
        PSI value; NaN forms its own bin when present in either sample.

    Raises:
        ValueError: If either sample is empty or `expected` has no finite values.
    """
    e = np.asarray(expected, dtype=float)
    a = np.asarray(actual, dtype=float)
    if e.size == 0 or a.size == 0:
        raise ValueError("expected and actual must be non-empty")
    finite = e[np.isfinite(e)]
    if finite.size == 0:
        raise ValueError("expected has no finite values")
    inner = np.unique(np.quantile(finite, np.linspace(0, 1, n_bins + 1)[1:-1]))
    edges = np.concatenate([[-np.inf], inner, [np.inf]])

    def shares(x: np.ndarray) -> np.ndarray:
        """Return bin shares of `x`, with a trailing NaN bin when either sample has NaN.

        Args:
            x: Sample to bin.

        Returns:
            Array of shares floored at 1e-6.
        """
        counts = np.histogram(x[~np.isnan(x)], bins=edges)[0].astype(float)
        if has_nan:
            counts = np.append(counts, np.isnan(x).sum())
        return np.maximum(counts / len(x), _PSI_FLOOR)

    has_nan = bool(np.isnan(e).any() or np.isnan(a).any())
    pe, pa = shares(e), shares(a)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def cutoff_table(y_true: object, y_score: object, approval_rates: Sequence[float]) -> pd.DataFrame:
    """Approve the lowest-score share of applicants and report outcomes per rate.

    Args:
        y_true: Binary labels (1 = bad).
        y_score: Scores (higher = riskier).
        approval_rates: Shares in (0, 1] of lowest scores approved.

    Returns:
        Frame with approval_rate, threshold, n_approved, bad_rate_approved, n_declined,
        bad_rate_declined, share_of_bads_declined. Bad rates are NaN for an empty group.

    Raises:
        ValueError: If a rate is outside (0, 1].
    """
    y, s = _check_inputs(y_true, y_score)
    total_bad = y.sum()
    rows = []
    for rate in approval_rates:
        if not 0 < rate <= 1:
            raise ValueError(f"approval rate must be in (0, 1]: {rate}")
        thr = float(np.quantile(s, rate))
        ok = s <= thr
        n_ok, n_dec = int(ok.sum()), int((~ok).sum())
        rows.append(
            {
                "approval_rate": float(rate),
                "threshold": thr,
                "n_approved": n_ok,
                "bad_rate_approved": float(y[ok].mean()) if n_ok else float("nan"),
                "n_declined": n_dec,
                "bad_rate_declined": float(y[~ok].mean()) if n_dec else float("nan"),
                "share_of_bads_declined": float(y[~ok].sum() / total_bad),
            }
        )
    return pd.DataFrame(rows)


def breakeven_pd(lgd: float, margin: float) -> float:
    """Probability of default where expected loss equals expected margin.

    Args:
        lgd: Loss given default, positive.
        margin: Margin earned on a good loan, positive.

    Returns:
        margin / (margin + lgd).

    Raises:
        ValueError: If either input is not finite and positive.
    """
    if not (np.isfinite(lgd) and np.isfinite(margin) and lgd > 0 and margin > 0):
        raise ValueError(f"lgd and margin must be finite and > 0: {lgd}, {margin}")
    return margin / (margin + lgd)
