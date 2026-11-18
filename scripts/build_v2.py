"""Build v2: monotone LightGBM tuned on cross-validated log loss, with a pooled calibrator.

The search minimizes CV log loss on the training window (202101-202206). The calibrator is chosen
and fit on out-of-fold training scores plus validation scores, after a scale check that keeps only
folds whose model sits on the same logit scale as the final model. The result is scored once on
the 2023 gap year next to v1 and v1.1. The 2024 test window is dropped on load and never used.

Usage:
    uv run python scripts/build_v2.py --register-only             # write the manifest and stop
    uv run python scripts/build_v2.py                             # search, build and score
    uv run python scripts/build_v2.py --smoke --out-root /tmp/x   # 3 trials, one seed, temp output
"""

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from recalibrate import (
    LGD,
    MARGIN,
    RegistrationError,
    _markdown,
    check_registration,
    summary_table,
)
from run_experiments import lgbm_space
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from credit_risk.config import APPLICATION_COLUMNS, DEFAULT_PROTOCOL, ID, TARGET, TIME
from credit_risk.data import load_raw
from credit_risk.evaluation import breakeven_pd, cutoff_table
from credit_risk.models import (
    CVResult,
    PrecalibratedModel,
    ScoreCalibrator,
    cross_validate,
    fit_model,
    positive_scores,
)
from credit_risk.packaging import file_sha256, load_artifact
from credit_risk.pipeline import (
    Candidate,
    _jsonable,
    evaluate_holdout,
    package_model,
    select_calibrator_blocks,
)
from credit_risk.splits import assign_split, expanding_quarter_folds
from credit_risk.tuning import tune

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_risk_data_enhanced.csv"
LOCK_PATH = ROOT / "uv.lock"
V1_DIR = ROOT / "models" / "pd-lgbm-no-apr-v1"
V11_DIR = ROOT / "models" / "pd-lgbm-no-apr-v1.1"
OUT_DIR = ROOT / "models" / "pd-lgbm-no-apr-v2"
RUN_DIR = ROOT / "runs" / "R13-lgbm-no-apr-logloss-v2"
PREDICTIONS = ROOT / "reports" / "gap_predictions_v2.csv"

RUN_ID = "R13-lgbm-no-apr-logloss-v2"
CANDIDATE_NAME = "R13-lgbm-no-apr-logloss"
FEATURE_SET = "no_apr"
SEEDS = (0, 1, 2)
MIN_DELTA = 0.0002
WINDOW = 60
CAP = 400
SMOKE_TRIALS = 3
MODEL_SEED = 0
N_BOOT = 2000
APPROVAL_RATES = (0.80, 0.85, 0.90, 0.95)
SLOPE_RANGE = (0.9, 1.1)
SHIFT_MAX = 0.1
AUC_MARGIN = 0.01
CALIBRATION_METHODS = ("logit_intercept", "logit_platt", "isotonic")

MANIFEST_BODY = """## Question
Does tuning on log loss, with a calibrator fit on about three times the defaults, give a PD
that is better calibrated than v1.1 without losing ranking?

## Status
Registered after the 2024 test window was scored for v1 and v1.1. Development uses data
through 2022-12 only. The 2023 gap year (scored before for v1 and v1.1, never used to choose
anything) is scored once for v2. The 2024 window is not loaded. Results are published
whatever they show.

## Protocol
- Model: monotone LightGBM, no_apr feature set, same search space as R03/R09.
- Search: TPE on CV mean log loss (expanding-quarter-4 folds on 202101-202206), seeds 0-2,
  min_delta 0.0002, window 60, hard cap 400. Best over seeds.
- Final model: fit on 202101-202206 with the chosen parameters.
- Scale check: each fold model and the final model score validation; a fold's out-of-fold
  scores are pooled only if the OLS slope of final logit on fold logit is in [0.9, 1.1] and
  the mean logit shift is at most 0.1.
- Calibrator: candidates log-odds shift, log-odds Platt, isotonic (simplest first). Leave one
  quarter out over the pooled out-of-fold quarters and the two validation quarters; simplest
  method within one paired standard error of the lowest pooled Brier. Refit on all pooled rows.
- Cutoff: break-even PD 0.10 / 0.70.
- Evaluation: 2023 gap year, once, against v1 and v1.1. Brier skill against the training
  default rate, with bootstrap intervals; paired bootstrap of v2 minus v1.1.

## Decision rule
v2 replaces v1.1 as the recommended model if the paired 95% CI of gap-year Brier (v2 minus
v1.1) is entirely below zero and the mean paired AUC difference is at least -0.01. Otherwise
v1.1 stays and v2 is reported.

## Expected
A larger or more flexible model than v1, less compression at the bottom of the scale, and a
higher gap-year Brier skill than v1.1 (+1.3%).
"""


def manifest_text(data_hash: str, lock_hash: str, today: str) -> str:
    """Render the manifest written before the run.

    Args:
        data_hash: SHA-256 of the data file.
        lock_hash: SHA-256 of uv.lock.
        today: ISO date of registration.

    Returns:
        Markdown text.
    """
    return (
        f"# Run manifest: {RUN_ID}\n\n"
        f"## Run\n\n- Run id: {RUN_ID}\n- Date: {today}\n"
        f"- Family and feature set: lgbm, {FEATURE_SET}\n\n"
        f"{MANIFEST_BODY}\n"
        f"## Reproducibility\n\n- Data hash: {data_hash}\n- Environment lock: {lock_hash}\n"
    )


def register(run_dir: Path, data_path: Path, lock_path: Path, today: str) -> Path:
    """Write the manifest, or confirm that an identical one exists.

    Args:
        run_dir: Run folder holding manifest.md.
        data_path: Data file to hash.
        lock_path: Environment lock file to hash.
        today: ISO date of registration.

    Returns:
        Path of the manifest.

    Raises:
        FileExistsError: If a manifest with different content exists.
    """
    path = run_dir / "manifest.md"
    text = manifest_text(file_sha256(data_path), file_sha256(lock_path), today)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different manifest: {path}")
        logger.info("manifest already registered: %s", path)
        return path
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("registered %s", path)
    return path


def load_pre_test_frames(path: Path) -> dict[str, pd.DataFrame]:
    """Load the data, drop every test-window row at once and split the rest.

    Args:
        path: Raw data CSV.

    Returns:
        Dict with keys train, val and gap, each with a fresh RangeIndex.

    Raises:
        ValueError: If a test-window row survives or a split is empty.
    """
    df = load_raw(path)
    df = df[df[TIME] < DEFAULT_PROTOCOL.test_start].reset_index(drop=True)
    if (df[TIME] >= DEFAULT_PROTOCOL.test_start).any():
        raise ValueError("test-window rows present after the drop")
    labels = assign_split(df[TIME])
    frames = {}
    for name in ("train", "val", "gap"):
        frame = df[labels == name].reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"split {name!r} is empty")
        logger.info("split %s: %d rows, default rate %.4f", name, len(frame), frame[TARGET].mean())
        frames[name] = frame
    return frames


def quarter_label(vintage: object) -> np.ndarray:
    """Label YYYYMM vintages with their calendar quarter, for example "2021Q4".

    Args:
        vintage: YYYYMM integers.

    Returns:
        Array of quarter labels.
    """
    v = np.asarray(vintage)
    return np.array([f"{x // 100}Q{(x % 100 - 1) // 3 + 1}" for x in v])


def search_seeds(
    cv: Callable[[dict[str, Any]], CVResult],
    run_dir: Path,
    *,
    seeds: tuple[int, ...],
    cap: int,
    window: int,
    min_delta: float,
) -> list[dict[str, Any]]:
    """Minimize CV log loss over `lgbm_space` once per seed.

    Args:
        cv: Function from parameters to a CVResult.
        run_dir: Folder for the per-seed trial logs.
        seeds: Sampler seeds.
        cap: Hard cap on trials per seed.
        window: Plateau window.
        min_delta: Minimum improvement that resets the plateau counter.

    Returns:
        One dict per seed with seed, best, trials, stop_reason and params.
    """
    out = []
    for seed in seeds:
        tuning = tune(
            lambda trial: cv(lgbm_space(trial)).ll_mean,
            "minimize",
            min_delta=min_delta,
            window=window,
            cap=cap,
            seed=seed,
            log_path=run_dir / f"trials_seed{seed}.csv",
        )
        out.append(
            {
                "seed": seed,
                "best": tuning.best_value,
                "trials": tuning.n_trials,
                "stop_reason": tuning.stop_reason,
                "params": {**tuning.best_params, "subsample_freq": 1},
            }
        )
        logger.info(
            "seed %d: best log loss %.5f after %d trials (%s)",
            seed,
            tuning.best_value,
            tuning.n_trials,
            tuning.stop_reason,
        )
    return out


def check_fold_scale(final_logit: np.ndarray, fold_logit: np.ndarray) -> dict[str, Any]:
    """Compare a fold model's validation logits with the final model's.

    Args:
        final_logit: Final model log-odds on the validation rows.
        fold_logit: Fold model log-odds on the same rows.

    Returns:
        Dict with slope (OLS of final on fold), shift (mean final minus fold) and pass.
    """
    x = np.asarray(fold_logit, dtype=float)
    z = np.asarray(final_logit, dtype=float)
    slope = float(np.cov(x, z, ddof=1)[0, 1] / np.var(x, ddof=1))
    shift = float(np.mean(z - x))
    ok = SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1] and abs(shift) <= SHIFT_MAX
    return {"slope": slope, "shift": shift, "pass": bool(ok)}


def _logit(p: np.ndarray) -> np.ndarray:
    """Log-odds with the same clipping as the calibrators.

    Args:
        p: Probabilities.

    Returns:
        Log-odds.
    """
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def paired_bootstrap(
    y: np.ndarray,
    scores: Mapping[str, np.ndarray],
    *,
    reference_rate: float,
    a: str,
    b: str,
    n_boot: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Resample rows with replacement; report skill intervals and paired differences.

    Every statistic uses the same resamples.

    Args:
        y: Binary labels.
        scores: Model label to predicted PD.
        reference_rate: Constant prediction for the Brier skill.
        a: Label of the model on the left of the differences.
        b: Label of the model on the right.
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        Dict with skill (label to point, lower, upper), brier_diff and auc_diff (a minus b: point,
        mean over resamples, lower, upper).
    """
    y = np.asarray(y)
    n = len(y)
    err = {m: (np.asarray(s) - y) ** 2 for m, s in scores.items()}
    rng = np.random.default_rng(seed)
    skill = {m: np.empty(n_boot) for m in scores}
    brier_diff = np.empty(n_boot)
    auc_diff = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ref = float(np.mean((reference_rate - y[idx]) ** 2))
        for m in scores:
            skill[m][i] = 1.0 - err[m][idx].mean() / ref
        brier_diff[i] = err[a][idx].mean() - err[b][idx].mean()
        auc_diff[i] = roc_auc_score(y[idx], scores[a][idx]) - roc_auc_score(y[idx], scores[b][idx])
    ref_full = float(np.mean((reference_rate - y) ** 2))

    def interval(vals: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    out: dict[str, Any] = {
        "skill": {
            m: {
                "point": float(1.0 - err[m].mean() / ref_full),
                "lower": interval(skill[m])[0],
                "upper": interval(skill[m])[1],
            }
            for m in scores
        }
    }
    point_auc = float(roc_auc_score(y, scores[a]) - roc_auc_score(y, scores[b]))
    out["brier_diff"] = {
        "point": float(err[a].mean() - err[b].mean()),
        "mean": float(brier_diff.mean()),
        "lower": interval(brier_diff)[0],
        "upper": interval(brier_diff)[1],
    }
    out["auc_diff"] = {
        "point": point_auc,
        "mean": float(auc_diff.mean()),
        "lower": interval(auc_diff)[0],
        "upper": interval(auc_diff)[1],
    }
    return out


def decide(paired: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the registered decision rule.

    Args:
        paired: Output of `paired_bootstrap` with a = v2 and b = v1.1.

    Returns:
        Dict with brier_diff_ci_hi, auc_diff_mean and v2_preferred.
    """
    hi = float(paired["brier_diff"]["upper"])
    auc = float(paired["auc_diff"]["mean"])
    return {
        "brier_diff_ci_hi": hi,
        "auc_diff_mean": auc,
        "v2_preferred": bool(hi < 0 and auc >= -AUC_MARGIN),
    }


def _fmt(table: pd.DataFrame) -> str:
    """Render a frame as a markdown table.

    Args:
        table: Frame to render.

    Returns:
        Markdown text.
    """
    return _markdown(table)


def run_full(
    *,
    data_path: Path,
    lock_path: Path,
    v1_dir: Path,
    v11_dir: Path,
    out_dir: Path,
    run_dir: Path,
    predictions_path: Path,
    seeds: tuple[int, ...] = SEEDS,
    cap: int = CAP,
    window: int = WINDOW,
    min_delta: float = MIN_DELTA,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    """Search, build v2, score the gap year once and write all outputs.

    Args:
        data_path: Raw data CSV.
        lock_path: Environment lock file whose hash the manifest registered.
        v1_dir: Frozen v1 artifact directory (read only).
        v11_dir: Frozen v1.1 artifact directory (read only).
        out_dir: New v2 artifact directory.
        run_dir: Run folder; its manifest must already exist.
        predictions_path: Output CSV of gap-year predictions.
        seeds: Sampler seeds of the search.
        cap: Hard cap on trials per seed.
        window: Plateau window.
        min_delta: Minimum improvement that resets the plateau counter.
        n_boot: Bootstrap resamples.

    Returns:
        Result dictionary, also written to `run_dir / "result.json"`.

    Raises:
        FileNotFoundError: If the manifest was not registered first.
        RegistrationError: If the registered hashes differ from the current files.
        FileExistsError: If a result, trial log, the v2 artifact or the predictions exist.
    """
    manifest = run_dir / "manifest.md"
    if not manifest.exists():
        raise FileNotFoundError(f"register the run first (--register-only): {manifest}")
    check_registration(manifest, data_path, lock_path)
    if (run_dir / "result.json").exists():
        raise FileExistsError(f"result already exists: {run_dir / 'result.json'}")
    logs = sorted(run_dir.glob("trials_seed*.csv"))
    if logs:
        raise FileExistsError(f"trial logs already exist: {logs[0]}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {out_dir}")
    if predictions_path.exists():
        raise FileExistsError(f"predictions already exist: {predictions_path}")

    frames = load_pre_test_frames(data_path)
    train, val, gap = frames["train"], frames["val"], frames["gap"]
    cols = list(APPLICATION_COLUMNS)
    x_train, y_train = train[cols], train[TARGET]
    folds = expanding_quarter_folds(train[TIME], y_train)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def cv(params: dict[str, Any]) -> CVResult:
        return cross_validate("lgbm", FEATURE_SET, params, x_train, y_train, folds, seed=MODEL_SEED)

    per_seed = search_seeds(cv, run_dir, seeds=seeds, cap=cap, window=window, min_delta=min_delta)
    best = min(per_seed, key=lambda r: r["best"])
    params = best["params"]
    final_cv = cv(params)
    stops = {r["stop_reason"] for r in per_seed}
    search = {
        "per_seed": per_seed,
        "trials": sum(r["trials"] for r in per_seed),
        "stop_reason": "cap" if "cap" in stops else "plateau",
        "spread": max(r["best"] for r in per_seed) - min(r["best"] for r in per_seed),
        "best_seed": best["seed"],
        "params": params,
        "ll_mean": final_cv.ll_mean,
        "ll_se": final_cv.ll_se,
        "fold_log_loss": final_cv.fold_log_loss,
        "cv_auc_mean": final_cv.cv_mean,
        "cv_auc_se": final_cv.cv_se,
        "fold_auc": final_cv.fold_scores,
    }

    final = fit_model("lgbm", FEATURE_SET, params, x_train, y_train, seed=MODEL_SEED)
    x_val, y_val = val[cols], val[TARGET].to_numpy()
    val_raw = positive_scores(final, x_val)
    final_logit = _logit(val_raw)

    scale_rows = []
    for i, (tr, va) in enumerate(folds):
        fold_model = fit_model(
            "lgbm", FEATURE_SET, params, x_train.iloc[tr], y_train.iloc[tr], seed=MODEL_SEED
        )
        check = check_fold_scale(final_logit, _logit(positive_scores(fold_model, x_val)))
        quarter = str(quarter_label(train[TIME].iloc[va])[0])
        scale_rows.append({"fold": i, "quarter": quarter, **check})
        logger.info("scale check fold %d (%s): %s", i, quarter, check)

    oof = final_cv.oof.to_numpy()
    p_parts, y_parts, b_parts = [], [], []
    for row, (_, va) in zip(scale_rows, folds, strict=True):
        if row["pass"]:
            p_parts.append(oof[va])
            y_parts.append(y_train.to_numpy()[va])
            b_parts.append(quarter_label(train[TIME].iloc[va]))
    half = val[TIME].to_numpy() <= 202209
    p_parts.append(val_raw)
    y_parts.append(y_val)
    b_parts.append(np.where(half, "2022Q3", "2022Q4"))
    p_pool, y_pool, b_pool = (np.concatenate(x) for x in (p_parts, y_parts, b_parts))
    selection = select_calibrator_blocks(p_pool, y_pool, b_pool, CALIBRATION_METHODS)
    chosen = selection["chosen"]
    calibrator = ScoreCalibrator(chosen).fit(p_pool, y_pool)
    v2 = PrecalibratedModel(final, calibrator)
    logger.info("calibrator %s fit on %d rows (%d defaults)", chosen, len(y_pool), y_pool.sum())

    cutoff = breakeven_pd(LGD, MARGIN)
    val_scores = positive_scores(v2, x_val)
    validation = {
        "note": "validation rows are in-sample for the calibrator",
        "cutoff_table": cutoff_table(y_val, val_scores, APPROVAL_RATES),
        "approval_rate_at_breakeven": float((val_scores <= cutoff).mean()),
    }

    v1, _ = load_artifact(v1_dir)
    v11, _ = load_artifact(v11_dir)
    models = {"v1": v1, "v1.1": v11, "v2": v2}
    reference_rate = float(y_train.mean())
    evals = {
        label: evaluate_holdout(
            m,
            gap,
            f"gap/{label}",
            threshold=cutoff,
            reference_rate=reference_rate,
            n_boot=n_boot,
        )
        for label, m in models.items()
    }
    gap_scores = {label: positive_scores(m, gap[cols]) for label, m in models.items()}
    paired = paired_bootstrap(
        gap[TARGET].to_numpy(),
        gap_scores,
        reference_rate=reference_rate,
        a="v2",
        b="v1.1",
        n_boot=n_boot,
    )
    decision = decide(paired)

    predictions = pd.DataFrame(
        {
            ID: gap[ID],
            TIME: gap[TIME],
            TARGET: gap[TARGET],
            "pd_v1": gap_scores["v1"],
            "pd_v1_1": gap_scores["v1.1"],
            "pd_v2": gap_scores["v2"],
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    summary = summary_table(evals)
    scale_table = pd.DataFrame(scale_rows)
    package_model(
        v2,
        out_dir,
        candidate=Candidate(CANDIDATE_NAME, "lgbm", FEATURE_SET, params),
        metrics={"validation": validation, "gap": evals["v2"]},
        data_hash=file_sha256(data_path),
        protocol=DEFAULT_PROTOCOL,
        threshold=cutoff,
        extra={
            "calibration": chosen,
            "selection_table": selection["table"],
            "calibration_blocks": selection["blocks"],
            "scale_check": scale_table,
            "decision": decision,
            "reference_rate": reference_rate,
            "lgd": LGD,
            "margin": MARGIN,
        },
    )

    result = {
        "run_id": RUN_ID,
        "family": "lgbm",
        "feature_set": FEATURE_SET,
        "params": params,
        "search": search,
        "val_auc_raw": float(roc_auc_score(y_val, val_raw)),
        "scale_check": scale_table,
        "calibration": {
            "chosen": chosen,
            "best": selection["best"],
            "blocks": selection["blocks"],
            "table": selection["table"],
            "n_rows": len(y_pool),
            "n_defaults": int(y_pool.sum()),
        },
        "reference_rate": reference_rate,
        "breakeven_pd": cutoff,
        "validation": validation,
        "gap_summary": summary,
        "evaluations": evals,
        "paired": paired,
        "decision": decision,
    }
    (run_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    write_result_section(manifest, result, summary, scale_table, evals)
    logger.info(
        "calibrator %s; v2_preferred %s; wrote %s", chosen, decision["v2_preferred"], run_dir
    )
    return result


def write_result_section(
    manifest: Path,
    result: Mapping[str, Any],
    summary: pd.DataFrame,
    scale_table: pd.DataFrame,
    evals: Mapping[str, Mapping[str, Any]],
) -> None:
    """Append the "## Result" section to the manifest.

    Args:
        manifest: Registered manifest.
        result: Result dictionary of `run_full`.
        summary: Gap-year summary table, one row per model.
        scale_table: Scale check per fold.
        evals: Model label to `evaluate_holdout` output.
    """
    search = result["search"]
    calib = result["calibration"]
    paired = result["paired"]
    dec = result["decision"]
    seeds = "".join(
        f"  - seed {r['seed']}: best log loss {r['best']:.5f} after {r['trials']} trials "
        f"({r['stop_reason']})\n"
        for r in search["per_seed"]
    )
    skill = pd.DataFrame(
        [
            {
                "model": m,
                "brier_skill": v["point"],
                "ci_lower": v["lower"],
                "ci_upper": v["upper"],
            }
            for m, v in paired["skill"].items()
        ]
    )
    diffs = pd.DataFrame(
        [
            {"statistic": f"{k} (v2 minus v1.1)", **{c: paired[k][c] for c in paired[k]}}
            for k in ("brier_diff", "auc_diff")
        ]
    )
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## Result\n\n### Search\n\n{seeds}"
            f"- Trials in total: {search['trials']} ({search['stop_reason']}); spread of per-seed "
            f"best across seeds {search['spread']:.5f}\n"
            f"- Final CV log loss {search['ll_mean']:.5f} (se {search['ll_se']:.5f}); CV AUC "
            f"{search['cv_auc_mean']:.4f} (se {search['cv_auc_se']:.4f})\n"
            f"- Validation AUC of the raw final model: {result['val_auc_raw']:.4f}\n"
            f"- Parameters: `{json.dumps(search['params'])}`\n"
            f"\n### Scale check\n\n{_fmt(scale_table)}"
            f"\n### Calibrator selection (leave one quarter out)\n\n{_fmt(calib['table'])}\n"
            f"- Blocks: {calib['blocks']}; {calib['n_rows']} rows, {calib['n_defaults']} defaults\n"
            f"- Lowest pooled Brier: {calib['best']}; chosen: {calib['chosen']}\n"
            f"- Break-even cutoff: {result['breakeven_pd']:.4f}; training-window default rate "
            f"{result['reference_rate']:.5f}\n"
            f"- Validation cutoff table is in-sample for the calibrator (see result.json)\n"
            f"\n### Gap year 2023\n\n{_fmt(summary)}"
            f"\n### Brier skill, 95% bootstrap interval\n\n{_fmt(skill)}"
            f"\n### Paired differences (v2 minus v1.1)\n\n{_fmt(diffs)}"
        )
        for label, ev in evals.items():
            fh.write(f"\n### Deciles, {label}\n\n{_fmt(ev['calibration_table'])}")
        fh.write(
            f"\n### Decision\n\n- Brier difference CI upper bound: {dec['brier_diff_ci_hi']:.6f}\n"
            f"- Mean paired AUC difference: {dec['auc_diff_mean']:.5f}\n"
            f"- v2_preferred: {dec['v2_preferred']}\n"
        )


def main() -> None:
    """Parse arguments and register the run or build v2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true", help="write the manifest and exit")
    parser.add_argument(
        "--smoke", action="store_true", help="3 trials, one seed, outputs under --out-root"
    )
    parser.add_argument("--out-root", type=Path, help="output root for --smoke")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()
    level = "DEBUG" if args.verbose else os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.register_only:
            if args.smoke:
                sys.exit("--register-only cannot be combined with --smoke")
            register(RUN_DIR, DATA_PATH, LOCK_PATH, date.today().isoformat())
        elif args.smoke:
            if args.out_root is None:
                sys.exit("--smoke needs --out-root")
            root = args.out_root
            run_dir = root / "runs" / RUN_ID
            register(run_dir, DATA_PATH, LOCK_PATH, date.today().isoformat())
            run_full(
                data_path=DATA_PATH,
                lock_path=LOCK_PATH,
                v1_dir=V1_DIR,
                v11_dir=V11_DIR,
                out_dir=root / "models" / "pd-lgbm-no-apr-v2",
                run_dir=run_dir,
                predictions_path=root / "reports" / "gap_predictions_v2.csv",
                seeds=(0,),
                cap=SMOKE_TRIALS,
            )
        else:
            if args.out_root is not None:
                sys.exit("--out-root is only for --smoke")
            run_full(
                data_path=DATA_PATH,
                lock_path=LOCK_PATH,
                v1_dir=V1_DIR,
                v11_dir=V11_DIR,
                out_dir=OUT_DIR,
                run_dir=RUN_DIR,
                predictions_path=PREDICTIONS,
            )
    except (FileExistsError, FileNotFoundError, RegistrationError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
