"""Step logic of the training pipeline; the Metaflow flow only orchestrates these functions."""

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.config import (
    APPLICATION_COLUMNS,
    DEFAULT_PROTOCOL,
    FEATURE_SETS,
    NUMERIC_INPUTS,
    SELECTABLE_FEATURE_SETS,
    TARGET,
    TIME,
    WEIGHT,
    SplitProtocol,
)
from credit_risk.data import data_sha256, load_raw
from credit_risk.evaluation import (
    binary_metrics,
    bootstrap_ci,
    breakeven_pd,
    brier_skill_score,
    calibration_table,
    cutoff_table,
    expected_calibration_error,
    paired_bootstrap_diff,
    psi,
)
from credit_risk.features import clean_special_values
from credit_risk.models import (
    CALIBRATION_METHODS,
    FAMILIES,
    ScoreCalibrator,
    calibrate,
    cross_validate,
    fit_model,
    positive_scores,
)
from credit_risk.packaging import load_artifact, save_artifact
from credit_risk.schema import SchemaError
from credit_risk.splits import assign_split, expanding_quarter_folds

logger = logging.getLogger(__name__)

_TIER = {"base_rate": 0, "logreg": 1, "lgbm": 2}
_SWEEP_KEYS = {"name", "family", "feature_set", "params", "weighted"}


@dataclass(frozen=True)
class Candidate:
    """One model configuration in the sweep.

    Attributes:
        name: Unique label.
        family: One of `credit_risk.models.FAMILIES`.
        feature_set: Key of `SELECTABLE_FEATURE_SETS`.
        params: Family hyperparameters.
        weighted: Train with `sample_weight` from the data.
    """

    name: str
    family: str
    feature_set: str
    params: dict
    weighted: bool = False


@dataclass
class CandidateResult:
    """A trained candidate with its cross-validation and validation scores.

    Attributes:
        candidate: The configuration.
        cv_mean: Mean ROC AUC over the expanding-quarter folds.
        cv_se: Standard error of the fold AUCs.
        fold_scores: ROC AUC per fold.
        model: Pipeline fitted on the full train frame.
        val_scores: Raw model scores on the validation frame.
        val_auc: ROC AUC of `val_scores`.
    """

    candidate: Candidate
    cv_mean: float
    cv_se: float
    fold_scores: list[float]
    model: Any
    val_scores: np.ndarray
    val_auc: float


def load_sweep(path: Path) -> list[Candidate]:
    """Read the candidate sweep from a JSON list.

    Args:
        path: JSON file holding a list of objects with the keys of `Candidate`.

    Returns:
        Candidates in file order.

    Raises:
        ValueError: On an unknown family, a feature set that is not selectable, duplicate
            names, missing or unexpected keys, or a top level that is not a list.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError(f"sweep must be a JSON list: {path}")
    candidates = []
    for item in raw:
        keys = set(item)
        if not {"name", "family", "feature_set", "params"} <= keys or keys - _SWEEP_KEYS:
            raise ValueError(f"sweep entry has missing or unexpected keys: {sorted(keys)}")
        candidates.append(Candidate(**item))
    names = [c.name for c in candidates]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate candidate names: {duplicates}")
    for c in candidates:
        if c.family not in FAMILIES:
            raise ValueError(f"{c.name}: unknown family {c.family!r}; choose from {FAMILIES}")
        if c.feature_set not in SELECTABLE_FEATURE_SETS:
            raise ValueError(
                f"{c.name}: feature set {c.feature_set!r} not in {SELECTABLE_FEATURE_SETS}"
            )
    return candidates


def load_and_validate(path: Path) -> tuple[pd.DataFrame, str]:
    """Load the raw CSV with validation and hash it.

    Args:
        path: CSV file.

    Returns:
        Tuple (raw frame, SHA-256 of the file).
    """
    return load_raw(path), data_sha256(path)


def split_frames(
    df: pd.DataFrame, protocol: SplitProtocol = DEFAULT_PROTOCOL
) -> dict[str, pd.DataFrame]:
    """Split by vintage into train, val, gap and test frames.

    Args:
        df: Raw frame with `vintage` and the target.
        protocol: Split boundaries.

    Returns:
        Dict with keys train, val, gap, test; each has a fresh RangeIndex.

    Raises:
        ValueError: If any split is empty.
    """
    labels = assign_split(df[TIME], protocol)
    frames = {}
    for name in ("train", "val", "gap", "test"):
        frame = df[labels == name].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"split {name!r} is empty under protocol {protocol}")
        logger.info("split %s: %d rows, default rate %.4f", name, len(frame), frame[TARGET].mean())
        frames[name] = frame
    return frames


def train_candidate(
    candidate: Candidate, frames: Mapping[str, pd.DataFrame], seed: int = 0
) -> CandidateResult:
    """Cross-validate on train, fit on train and score validation.

    Args:
        candidate: Configuration to train.
        frames: Output of `split_frames`; only train and val are read.
        seed: Random seed.

    Returns:
        The fitted candidate with CV and validation scores.
    """
    train, val = frames["train"], frames["val"]
    X = train[list(APPLICATION_COLUMNS)]
    y = train[TARGET]
    weight = train[WEIGHT] if candidate.weighted else None
    folds = expanding_quarter_folds(train[TIME], y)
    cv = cross_validate(
        candidate.family,
        candidate.feature_set,
        candidate.params,
        X,
        y,
        folds,
        sample_weight=weight,
        seed=seed,
        allow_leaky=False,
    )
    model = fit_model(
        candidate.family,
        candidate.feature_set,
        candidate.params,
        X,
        y,
        sample_weight=weight,
        seed=seed,
        allow_leaky=False,
    )
    val_scores = positive_scores(model, val[list(APPLICATION_COLUMNS)])
    val_auc = float(roc_auc_score(val[TARGET], val_scores))
    logger.info("%s: cv %.4f +- %.4f, val AUC %.4f", candidate.name, cv.cv_mean, cv.cv_se, val_auc)
    return CandidateResult(
        candidate, cv.cv_mean, cv.cv_se, cv.fold_scores, model, val_scores, val_auc
    )


def select_champion(
    results: list[CandidateResult],
) -> tuple[CandidateResult, CandidateResult | None, str]:
    """Pick the simplest candidate within one standard error of the best CV mean.

    Args:
        results: All trained candidates.

    Returns:
        Tuple (champion, challenger, explanation). The challenger is the best-CV result other
        than the champion, or None if there is only one result.

    Raises:
        ValueError: If `results` is empty.
    """
    if not results:
        raise ValueError("no candidate results to select from")
    best = max(results, key=lambda r: r.cv_mean)
    floor = best.cv_mean - best.cv_se
    eligible = [r for r in results if r.cv_mean >= floor]
    champion = min(
        eligible,
        key=lambda r: (
            _TIER[r.candidate.family],
            len(FEATURE_SETS[r.candidate.feature_set]),
            -r.cv_mean,
        ),
    )
    others = [r for r in results if r is not champion]
    challenger = max(others, key=lambda r: r.cv_mean) if others else None
    explanation = (
        f"best cv {best.cv_mean:.4f} ({best.candidate.name}), se {best.cv_se:.4f}, "
        f"floor {floor:.4f}; {len(eligible)} of {len(results)} eligible; champion "
        f"{champion.candidate.name} (cv {champion.cv_mean:.4f}, "
        f"tier {_TIER[champion.candidate.family]}, "
        f"{len(FEATURE_SETS[champion.candidate.feature_set])} features)"
    )
    logger.info(explanation)
    return champion, challenger, explanation


def compare_champion_challenger(
    champion: CandidateResult,
    challenger: CandidateResult,
    y_val: object,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of the validation AUC difference, champion minus challenger.

    Args:
        champion: Selected candidate.
        challenger: Runner-up.
        y_val: Validation labels.
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        Dict with diff, lower, upper, share_a_better, champion_val_auc, challenger_val_auc.
    """
    out = paired_bootstrap_diff(
        y_val, champion.val_scores, challenger.val_scores, roc_auc_score, n_boot=n_boot, seed=seed
    )
    out["champion_val_auc"] = champion.val_auc
    out["challenger_val_auc"] = challenger.val_auc
    return out


def calibrate_and_cutoff(
    result: CandidateResult,
    frames: Mapping[str, pd.DataFrame],
    method: Literal["logit_intercept", "logit_platt", "sigmoid", "isotonic"] = "sigmoid",
    lgd: float = 0.6,
    margin: float = 0.10,
    approval_rates: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> dict:
    """Calibrate the champion on validation and tabulate cutoffs.

    The before/after Brier and ECE values are computed on the validation frame that the
    calibrator was fitted on, so `*_after` is in-sample for the calibrator and optimistic.

    Args:
        result: Champion with a model fitted on train.
        frames: Output of `split_frames`; only val is read.
        method: Calibration method.
        lgd: Loss given default.
        margin: Margin earned on a good loan.
        approval_rates: Approval shares for the cutoff table.

    Returns:
        Dict with calibrated_model, val_scores_calibrated, brier_before, brier_after,
        ece_before, ece_after, breakeven_pd, cutoff_table, approval_rate_at_breakeven.
    """
    val = frames["val"]
    X, y = val[list(APPLICATION_COLUMNS)], val[TARGET]
    calibrated = calibrate(result.model, X, y, method)
    after = positive_scores(calibrated, X)
    be = breakeven_pd(lgd, margin)
    out = {
        "calibrated_model": calibrated,
        "val_scores_calibrated": after,
        "brier_before": binary_metrics(y, result.val_scores)["brier"],
        "brier_after": binary_metrics(y, after)["brier"],
        "ece_before": expected_calibration_error(y, result.val_scores),
        "ece_after": expected_calibration_error(y, after),
        "breakeven_pd": be,
        "cutoff_table": cutoff_table(y, after, approval_rates),
        "approval_rate_at_breakeven": float((after <= be).mean()),
    }
    logger.info(
        "calibrated (%s): brier %.5f -> %.5f, ece %.4f -> %.4f, breakeven PD %.4f",
        method,
        out["brier_before"],
        out["brier_after"],
        out["ece_before"],
        out["ece_after"],
        be,
    )
    return out


def _one_se_choice(table: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """Apply the one-standard-error rule to a method table.

    Args:
        table: Frame with columns method (in priority order, simplest first), brier and
            se_vs_best (standard error of the per-row loss difference to the best method).

    Returns:
        Tuple (table with diff_vs_best and eligible added, chosen method, best method). The
        chosen method is the first one whose Brier is within one standard error of the best.
    """
    out = table.copy()
    best = str(out.loc[out["brier"].idxmin(), "method"])
    out["diff_vs_best"] = out["brier"] - out["brier"].min()
    out["eligible"] = out["diff_vs_best"] <= out["se_vs_best"]
    chosen = str(out.loc[out["eligible"].idxmax(), "method"])
    return out, chosen, best


def select_calibrator(
    model: object, val: pd.DataFrame, methods: tuple[str, ...] = CALIBRATION_METHODS
) -> dict:
    """Choose a calibration method by time-ordered two-fold validation inside `val`.

    Folds: A is the first three vintages present, B the rest. Each method is fitted on A and
    scored on B, then fitted on B and scored on A; the out-of-fold predictions are pooled.

    Args:
        model: Fitted model with `predict_proba` over raw application columns.
        val: Validation frame with application columns, the target and the vintage.
        methods: Candidate methods, simplest first.

    Returns:
        Dict with table (one row per method: method, brier, log_loss, ece, diff_vs_best,
        se_vs_best, eligible), chosen (first method within one paired standard error of the
        lowest pooled Brier), best (lowest pooled Brier) and fold_vintages ({"A": [...],
        "B": [...]}).

    Raises:
        ValueError: If `val` has fewer than 4 vintages.
    """
    vintages = sorted(int(v) for v in val[TIME].unique())
    if len(vintages) < 4:
        raise ValueError(f"need at least 4 validation vintages, got {len(vintages)}")
    fold_vintages = {"A": vintages[:3], "B": vintages[3:]}
    in_a = val[TIME].isin(fold_vintages["A"]).to_numpy()
    X, y = val[list(APPLICATION_COLUMNS)], val[TARGET].to_numpy()
    sq_err: dict[str, np.ndarray] = {}
    rows = []
    for method in methods:
        pooled = np.empty(len(val))
        for fit_mask in (in_a, ~in_a):
            cal = calibrate(model, X[fit_mask], pd.Series(y[fit_mask]), method)
            pooled[~fit_mask] = positive_scores(cal, X[~fit_mask])
        sq_err[method] = (pooled - y) ** 2
        rows.append(
            {
                "method": method,
                "brier": float(sq_err[method].mean()),
                "log_loss": binary_metrics(y, pooled)["log_loss"],
                "ece": expected_calibration_error(y, pooled),
            }
        )
    table = pd.DataFrame(rows)
    best_err = sq_err[str(table.loc[table["brier"].idxmin(), "method"])]
    table["se_vs_best"] = [
        float((sq_err[m] - best_err).std(ddof=1) / np.sqrt(len(y))) for m in table["method"]
    ]
    table, chosen, best = _one_se_choice(table)
    logger.info("calibrator selection:\n%s\nchosen %s (best %s)", table.to_string(), chosen, best)
    return {
        "table": table[
            ["method", "brier", "log_loss", "ece", "diff_vs_best", "se_vs_best", "eligible"]
        ],
        "chosen": chosen,
        "best": best,
        "fold_vintages": fold_vintages,
    }


def select_calibrator_blocks(
    p: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    methods: tuple[str, ...] = ("logit_intercept", "logit_platt", "isotonic"),
) -> dict:
    """Choose a calibration method by leave-one-block-out over raw scores.

    Each method is fitted on every block but one and scored on the held-out block; the
    out-of-fold predictions are pooled and the one-standard-error rule picks the method.

    Args:
        p: Raw positive-class probabilities.
        y: Binary target aligned to `p`.
        blocks: Block id per row (for example a quarter).
        methods: Candidate methods, simplest first.

    Returns:
        Dict with table (one row per method: method, brier, log_loss, ece, diff_vs_best,
        se_vs_best, eligible), chosen (first method within one paired standard error of the
        lowest pooled Brier), best (lowest pooled Brier) and blocks (sorted block ids).

    Raises:
        ValueError: If there are fewer than 3 blocks or a block lacks one of the classes.
    """
    p, y, blocks = np.asarray(p, dtype=float), np.asarray(y), np.asarray(blocks)
    ids = sorted(np.unique(blocks).tolist())
    if len(ids) < 3:
        raise ValueError(f"need at least 3 blocks, got {len(ids)}")
    for b in ids:
        if set(np.unique(y[blocks == b]).tolist()) != {0, 1}:
            raise ValueError(f"block {b} must contain both classes")
    sq_err: dict[str, np.ndarray] = {}
    rows = []
    for method in methods:
        pooled = np.empty(len(y))
        for b in ids:
            held = blocks == b
            cal = ScoreCalibrator(method).fit(p[~held], y[~held])
            pooled[held] = cal.transform(p[held])
        sq_err[method] = (pooled - y) ** 2
        rows.append(
            {
                "method": method,
                "brier": float(sq_err[method].mean()),
                "log_loss": binary_metrics(y, pooled)["log_loss"],
                "ece": expected_calibration_error(y, pooled),
            }
        )
    table = pd.DataFrame(rows)
    best_err = sq_err[str(table.loc[table["brier"].idxmin(), "method"])]
    table["se_vs_best"] = [
        float((sq_err[m] - best_err).std(ddof=1) / np.sqrt(len(y))) for m in table["method"]
    ]
    table, chosen, best = _one_se_choice(table)
    logger.info(
        "block calibrator selection:\n%s\nchosen %s (best %s)", table.to_string(), chosen, best
    )
    return {
        "table": table[
            ["method", "brier", "log_loss", "ece", "diff_vs_best", "se_vs_best", "eligible"]
        ],
        "chosen": chosen,
        "best": best,
        "blocks": ids,
    }


def _vintage_table(y: pd.Series, pd_hat: np.ndarray, vintage: pd.Series) -> pd.DataFrame:
    """Summarize outcomes and predictions per vintage.

    Args:
        y: Binary labels.
        pd_hat: Predicted PDs aligned to `y`.
        vintage: Vintage per row.

    Returns:
        Frame with vintage, n, defaults, observed_rate, mean_pd, auc (NaN if one class).
    """
    rows = []
    for v, idx in vintage.groupby(vintage).groups.items():
        yv = y.loc[idx]
        pv = pd_hat[y.index.get_indexer(idx)]
        auc = float(roc_auc_score(yv, pv)) if yv.nunique() >= 2 else float("nan")
        rows.append(
            {
                "vintage": int(v),
                "n": len(yv),
                "defaults": int(yv.sum()),
                "observed_rate": float(yv.mean()),
                "mean_pd": float(pv.mean()),
                "auc": auc,
            }
        )
    return pd.DataFrame(rows)


def evaluate_holdout(
    model: object,
    frame: pd.DataFrame,
    name: str,
    *,
    threshold: float,
    reference_rate: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Score a frozen calibrated model on one holdout frame.

    Args:
        model: Calibrated model; never refitted here.
        frame: Holdout frame (gap or test).
        name: Label for logging and the result.
        threshold: Approve applicants with PD at or below this value.
        reference_rate: Constant prediction for the Brier skill (the training default rate).
        n_boot: Bootstrap resamples for the AUC interval.
        seed: Random seed.

    Returns:
        Dict with name, metrics, brier_skill, auc_ci (lower, upper), ece, calibration_table,
        mean_pd, observed_rate, per_vintage, threshold, approval_rate, bad_rate_approved.
    """
    logger.info("scoring holdout %s", name)
    y = frame[TARGET]
    scores = positive_scores(model, frame[list(APPLICATION_COLUMNS)])
    _, lo, hi = bootstrap_ci(y, scores, roc_auc_score, n_boot=n_boot, seed=seed)
    approved = scores <= threshold
    return {
        "name": name,
        "metrics": binary_metrics(y, scores),
        "brier_skill": brier_skill_score(y, scores, reference_rate),
        "auc_ci": (lo, hi),
        "ece": expected_calibration_error(y, scores),
        "calibration_table": calibration_table(y, scores),
        "mean_pd": float(scores.mean()),
        "observed_rate": float(y.mean()),
        "per_vintage": _vintage_table(y, scores, frame[TIME]),
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y[approved].mean()) if approved.any() else float("nan"),
    }


def stability_report(model: object, frames: Mapping[str, pd.DataFrame]) -> dict:
    """Population stability of the calibrated score and of each numeric input.

    Reads only application columns, never the target.

    Args:
        model: Calibrated model.
        frames: Output of `split_frames`.

    Returns:
        Dict with score_psi (train vs val, gap, test) and feature_psi (DataFrame of feature,
        psi for train vs test, sorted descending).
    """
    cols = list(APPLICATION_COLUMNS)
    base = positive_scores(model, frames["train"][cols])
    score_psi = {
        k: psi(base, positive_scores(model, frames[k][cols])) for k in ("val", "gap", "test")
    }
    train_clean = clean_special_values(frames["train"])
    test_clean = clean_special_values(frames["test"])
    feature_psi = pd.DataFrame(
        {
            "feature": list(NUMERIC_INPUTS),
            "psi": [psi(train_clean[c], test_clean[c]) for c in NUMERIC_INPUTS],
        }
    )
    feature_psi = feature_psi.sort_values("psi", ascending=False).reset_index(drop=True)
    return {"score_psi": score_psi, "feature_psi": feature_psi}


def _jsonable(obj: object) -> object:
    """Convert numpy, pandas and non-finite values to strict-JSON types.

    Args:
        obj: Nested structure of dicts, lists, tuples, arrays, frames and scalars.

    Returns:
        Equivalent structure of dict, list, str, int, float, bool and None.
    """
    if isinstance(obj, pd.DataFrame):
        return _jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def package_model(
    calibrated_model: object,
    out_dir: Path,
    *,
    candidate: Candidate,
    metrics: Mapping[str, Any],
    data_hash: str,
    protocol: SplitProtocol,
    threshold: float,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Save the calibrated model with metadata.

    Args:
        calibrated_model: Calibrated, frozen model.
        out_dir: New artifact directory.
        candidate: Configuration of the champion.
        metrics: Evaluation results; converted to JSON types (NaN becomes null).
        data_hash: SHA-256 of the training data file.
        protocol: Split protocol used.
        threshold: Frozen approval cutoff on the calibrated PD.
        extra: Additional JSON-serializable metadata.

    Returns:
        `out_dir`.
    """
    metadata = {
        "model_name": candidate.name,
        "candidate": asdict(candidate),
        "input_columns": list(FEATURE_SETS[candidate.feature_set]),
        "split_protocol": asdict(protocol),
        "data_sha256": data_hash,
        "threshold": float(threshold),
        "metrics": _jsonable(metrics),
        "extra": _jsonable(dict(extra or {})),
    }
    return save_artifact(calibrated_model, metadata, Path(out_dir))


def score_applications(artifact_dir: Path, df: pd.DataFrame) -> pd.Series:
    """Score applications with a saved artifact.

    Args:
        artifact_dir: Directory written by `package_model`.
        df: Applications with at least the artifact's input columns.

    Returns:
        Calibrated PDs indexed like `df`.

    Raises:
        SchemaError: If an input column is missing.
    """
    model, meta = load_artifact(Path(artifact_dir))
    cols = meta["input_columns"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    return pd.Series(positive_scores(model, df[cols]), index=df.index, name="pd")
