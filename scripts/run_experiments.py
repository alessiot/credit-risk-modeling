"""Run the registered experiments and write the leaderboard, manifests and trial logs.

Every run is cross-validated on the training window (expanding quarterly folds) and scored
once on the validation window. The test window is never loaded here.

Usage:
    uv run python scripts/run_experiments.py              # all runs
    uv run python scripts/run_experiments.py --runs R02   # a subset
    uv run python scripts/run_experiments.py --smoke      # 3 trials per search, no outputs kept
    uv run python scripts/run_experiments.py --references # post-test references R11, R12 only
    uv run python scripts/run_experiments.py --add-r13    # append R13 from its result.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from credit_risk.config import DEFAULT_PROTOCOL, TARGET, TIME, WEIGHT
from credit_risk.data import data_sha256, load_raw
from credit_risk.evaluation import binary_metrics
from credit_risk.models import CVResult, cross_validate, fit_model, positive_scores
from credit_risk.packaging import file_sha256
from credit_risk.splits import assign_split, expanding_quarter_folds
from credit_risk.tuning import tune

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_risk_data_enhanced.csv"
RUNS_DIR = ROOT / "runs"
LEADERBOARD = ROOT / "reports" / "leaderboard.csv"

DATA_VERSION = "enhanced-v1"
CV_SCHEME = "expanding-quarter-4"
METRIC = "roc_auc"
DIRECTION = "max"
SEED = 0
WINDOW = 30
LEADERBOARD_FIELDS = (
    "data_version",
    "split_id",
    "cv_scheme",
    "metric",
    "direction",
    "run_id",
    "family",
    "feature_set",
    "weighted",
    "selectable",
    "cv_mean",
    "cv_se",
    "val_score",
    "trials",
    "stop_reason",
    "selected",
    "test_score",
    "manifest",
)


def logreg_space(trial: optuna.Trial) -> dict[str, Any]:
    """Search space of the binned logistic regression.

    Args:
        trial: Optuna trial.

    Returns:
        Parameters for `build_model("logreg", ...)`.
    """
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "n_bins": trial.suggest_int("n_bins", 3, 20),
    }


def lgbm_space(trial: optuna.Trial) -> dict[str, Any]:
    """Search space of the monotone LightGBM.

    Args:
        trial: Optuna trial.

    Returns:
        LGBMClassifier keyword arguments.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 32),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
    }


SPACES: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "logreg": logreg_space,
    "lgbm": lgbm_space,
}
SPACE_TEXT = {
    "logreg": "C log-uniform [1e-3, 10]; n_bins int [3, 20]",
    "lgbm": (
        "n_estimators [50, 800]; learning_rate log [0.01, 0.2]; num_leaves [2, 32]; "
        "min_child_samples log [20, 500]; reg_lambda log [1e-3, 30]; "
        "colsample_bytree [0.4, 1]; subsample [0.5, 1] with subsample_freq 1"
    ),
}
DEFAULT_PARAMS = {"logreg": {"C": 1.0, "n_bins": 10}, "lgbm": {}}


@dataclass(frozen=True)
class RunSpec:
    """One registered run.

    Attributes:
        run_id: Identifier, also the folder under runs/.
        family: Model family.
        feature_set: Key of FEATURE_SETS.
        hypothesis: What the run tests.
        pass_criterion: Measurement that means yes.
        kill_criterion: Measurement that means no.
        cap: Hard cap on trials; 0 means a single fit with fixed params.
        params_from: Run whose best parameters are reused (ablations); None otherwise.
        weighted: Whether `sample_weight` is the training weight.
        selectable: Whether the run may be selected (False for the leakage ablation).
        fixed_params: Parameters for a single fit when there is no search and no params_from;
            for a grid run, the parameters shared by every grid point.
        grid: Parameter name to values for an exhaustive grid search; None otherwise.
    """

    run_id: str
    family: str
    feature_set: str
    hypothesis: str
    pass_criterion: str
    kill_criterion: str
    cap: int = 0
    params_from: str | None = None
    weighted: bool = False
    selectable: bool = True
    fixed_params: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, tuple[Any, ...]] | None = None


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "R00-base-rate",
        "base_rate",
        "full",
        "A constant prediction at the training default rate sets the floor (AUC 0.5).",
        "Runs and scores AUC 0.5.",
        "n/a (baseline).",
    ),
    RunSpec(
        "R01-fico-only",
        "logreg",
        "fico_only",
        "FICO alone, binned, is the single-feature benchmark any model must beat.",
        "CV AUC clearly above 0.5.",
        "n/a (baseline).",
        fixed_params={"C": 1.0, "n_bins": 10},
    ),
    RunSpec(
        "R02-logreg",
        "logreg",
        "full",
        "A binned L2 logistic regression on all application features beats FICO alone.",
        "CV AUC above R01 by more than one cv_se.",
        "CV AUC within one cv_se of R01 after a converged search.",
        cap=150,
    ),
    RunSpec(
        "R03-lgbm",
        "lgbm",
        "full",
        "Monotone LightGBM captures threshold effects and interactions beyond R02.",
        "CV AUC above R02 by more than one cv_se.",
        "CV AUC within one cv_se of R02 after a converged search.",
        cap=200,
    ),
    RunSpec(
        "R04-logreg-no-apr",
        "logreg",
        "no_apr",
        "APR adds no rank-ordering power (audit: no relation to FICO or default).",
        "CV AUC within one cv_se of R02.",
        "CV AUC below R02 by more than one cv_se.",
        params_from="R02-logreg",
    ),
    RunSpec(
        "R05-lgbm-no-apr",
        "lgbm",
        "no_apr",
        "APR adds no rank-ordering power to LightGBM.",
        "CV AUC within one cv_se of R03.",
        "CV AUC below R03 by more than one cv_se.",
        params_from="R03-lgbm",
    ),
    RunSpec(
        "R06-lgbm-weighted",
        "lgbm",
        "full",
        "Training with sample_weight (about 3x on defaults) does not improve ranking.",
        "CV AUC within one cv_se of R03.",
        "CV AUC differs from R03 by more than one cv_se.",
        params_from="R03-lgbm",
        weighted=True,
    ),
    RunSpec(
        "R07-lgbm-leaky",
        "lgbm",
        "leaky",
        "Post-origination columns inflate AUC toward 1; never selectable.",
        "CV AUC far above R03, showing the inflation.",
        "n/a (leakage ablation).",
        params_from="R03-lgbm",
        selectable=False,
    ),
)


SENSITIVITY_MIN_DELTA = 0.001
SENSITIVITY_WINDOW = 60
SENSITIVITY_CAP = 400
SENSITIVITY_SEEDS = (0, 1, 2)
SENSITIVITY_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "R08-lgbm-sensitivity",
        "lgbm",
        "full",
        "R03's search stopped on a minimum delta (0.013) coarser than the selection margin; a "
        "search with a 0.001 delta shows where LightGBM with APR converges.",
        "Best CV AUC stable across three seeds within 0.001.",
        "n/a (sensitivity check, run after the test pass, never selectable).",
        cap=SENSITIVITY_CAP,
        selectable=False,
    ),
    RunSpec(
        "R09-lgbm-no-apr-sensitivity",
        "lgbm",
        "no_apr",
        "R05 reused R03's parameters; tuning it directly shows whether the selection holds "
        "when both feature sets are searched to convergence.",
        "Best CV AUC stable across three seeds within 0.001.",
        "n/a (sensitivity check, run after the test pass, never selectable).",
        cap=SENSITIVITY_CAP,
        selectable=False,
    ),
)

REFERENCE_RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        "R11-lgbm-no-apr-untuned",
        "lgbm",
        "no_apr",
        "Reference run after the test pass: untuned LightGBM, monotone constraints on, shows "
        "how much of the tuned model's validation result tuning explains. LightGBM defaults "
        "apply: n_estimators 100, learning_rate 0.1, num_leaves 31, min_child_samples 20, "
        "reg_lambda 0, subsample 1, colsample_bytree 1.",
        "Reported as a reference; CV and validation only.",
        "n/a (reference run, never selectable, test window not read).",
        selectable=False,
    ),
    RunSpec(
        "R12-scorecard-no-apr",
        "scorecard",
        "no_apr",
        "Reference run after the test pass: a supervised-bin scorecard (tree-based bins, "
        "one-hot, L2 logistic regression) shows what a conventional scorecard reaches without "
        "APR.",
        "Reported as a reference; CV and validation only.",
        "n/a (reference run, never selectable, test window not read).",
        selectable=False,
        fixed_params={"min_bin_fraction": 0.05},
        grid={"max_bins": (4, 6, 8), "C": (0.01, 0.1, 1.0)},
    ),
)


def write_text(path: Path, text: str) -> None:
    """Write text, creating parent folders.

    Args:
        path: Destination.
        text: Content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def manifest_head(spec: RunSpec, data_hash: str, lock_hash: str, min_delta: str) -> str:
    """Render the manifest sections written before the run.

    Args:
        spec: Run specification.
        data_hash: SHA-256 of the data file.
        lock_hash: SHA-256 of uv.lock.
        min_delta: Minimum delta of the stopping rule, as text.

    Returns:
        Markdown text.
    """
    if spec.grid:
        points = len(list(product(*spec.grid.values())))
        grid_text = "; ".join(f"{k} in {list(v)}" for k, v in spec.grid.items())
        search = (
            f"- Search space: exhaustive grid of {points} configurations, {grid_text}; "
            f"fixed {json.dumps(spec.fixed_params)}\n"
            f"- Sampler: none\n- Run log: trials.csv\n\n"
            f"## Stopping rule\n\n- Window: n/a\n- Minimum delta: n/a\n"
            f"- Hard cap: {points} configurations (grid)\n"
        )
    elif spec.cap:
        search = (
            f"- Search space: {SPACE_TEXT[spec.family]}\n"
            f"- Sampler: TPE through Optuna (seed {SEED})\n"
            f"- Run log: trials.csv\n\n"
            f"## Stopping rule\n\n- Window: {WINDOW} trials\n- Minimum delta: {min_delta}\n"
            f"- Hard cap: {spec.cap} trials\n"
        )
    else:
        source = f"best parameters of {spec.params_from}" if spec.params_from else "fixed"
        params = "" if spec.params_from else f" {json.dumps(spec.fixed_params)}"
        search = (
            f"- Search space: none, single fit with {source} parameters{params}\n"
            f"- Sampler: none\n- Run log: none\n\n"
            f"## Stopping rule\n\n- Window: n/a\n- Minimum delta: n/a\n- Hard cap: 1 fit\n"
        )
    return (
        f"# Run manifest: {spec.run_id}\n\n"
        f"## Run\n\n- Run id: {spec.run_id}\n- Date: {date.today().isoformat()}\n"
        f"- Family and feature set: {spec.family}, {spec.feature_set}\n"
        f"- Training weight: {'sample_weight' if spec.weighted else 'none'}\n\n"
        f"## Hypothesis\n\n{spec.hypothesis}\n\n"
        f"## Pass and kill criteria\n\n- Pass: {spec.pass_criterion}\n"
        f"- Kill: {spec.kill_criterion}\n\n"
        f"## Protocol\n\n- `data_version`: {DATA_VERSION}\n"
        f"- `split_id`: {DEFAULT_PROTOCOL.split_id}\n- `cv_scheme`: {CV_SCHEME}\n"
        f"- `metric`: {METRIC}\n- `direction`: {DIRECTION}\n\n"
        f"## Reproducibility\n\n- Data hash: {data_hash}\n- Environment lock: {lock_hash}\n"
        f"- Seeds: {SEED}\n\n## Search\n\n{search}"
    )


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the data and return the training and validation windows.

    Returns:
        (train, val) frames. Gap and test rows are dropped here and never used.
    """
    df = load_raw(DATA_PATH)
    split = assign_split(df[TIME])
    train = df[split == "train"].reset_index(drop=True)
    val = df[split == "val"].reset_index(drop=True)
    logger.info(
        "train %d rows (%d defaults), val %d rows (%d defaults)",
        len(train),
        int(train[TARGET].sum()),
        len(val),
        int(val[TARGET].sum()),
    )
    return train, val


def run_one(
    spec: RunSpec,
    train: pd.DataFrame,
    val: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    results: dict[str, dict[str, Any]],
    hashes: tuple[str, str],
    smoke: bool,
) -> dict[str, Any]:
    """Run one registered experiment and write its manifest and result.

    Args:
        spec: Run specification.
        train: Training window.
        val: Validation window.
        folds: Expanding-window folds over `train`.
        results: Results of runs already done (for params_from).
        hashes: (data hash, lock hash).
        smoke: If True, cap searches at 3 trials and write nothing.

    Returns:
        Result dictionary (also written to runs/<run_id>/result.json unless smoke).
    """
    run_dir = RUNS_DIR / spec.run_id
    y = train[TARGET]
    weight = train[WEIGHT] if spec.weighted else None

    def cv(params: dict[str, Any]) -> CVResult:
        return cross_validate(
            spec.family,
            spec.feature_set,
            params,
            train,
            y,
            folds,
            sample_weight=weight,
            seed=SEED,
            allow_leaky=spec.feature_set == "leaky",
        )

    if spec.grid:
        if not smoke:
            write_text(run_dir / "manifest.md", manifest_head(spec, *hashes, "n/a"))
            log_path = run_dir / "trials.csv"
            log_path.unlink(missing_ok=True)
        configs = [
            dict(zip(spec.grid, point, strict=True)) for point in product(*spec.grid.values())
        ]
        if smoke:
            configs = configs[:3]
        scored = [cv({**spec.fixed_params, **config}) for config in configs]
        best_i = int(np.argmax([r.cv_mean for r in scored]))
        params = {**spec.fixed_params, **configs[best_i]}
        trials, stop_reason = len(configs), "none"
        if not smoke:
            best_so_far = np.maximum.accumulate([r.cv_mean for r in scored])
            with log_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["trial", "value", "best_so_far", "params"])
                for i, (config, r) in enumerate(zip(configs, scored, strict=True)):
                    writer.writerow(
                        [i, r.cv_mean, best_so_far[i], json.dumps(config, sort_keys=True)]
                    )
    elif spec.cap:
        default = cv(DEFAULT_PARAMS[spec.family])
        min_delta = default.cv_se
        delta_text = (
            f"{min_delta:.5f} (cv_se of the default configuration, CV AUC {default.cv_mean:.4f})"
        )
        if not smoke:
            write_text(run_dir / "manifest.md", manifest_head(spec, *hashes, delta_text))
            log_path = run_dir / "trials.csv"
            log_path.unlink(missing_ok=True)
        else:
            log_path = None
        space = SPACES[spec.family]
        tuning = tune(
            lambda trial: cv(space(trial)).cv_mean,
            "maximize",
            min_delta=min_delta,
            window=WINDOW,
            cap=3 if smoke else spec.cap,
            seed=SEED,
            log_path=log_path,
        )
        params = {**tuning.best_params}
        if spec.family == "lgbm":
            params["subsample_freq"] = 1
        trials, stop_reason = tuning.n_trials, tuning.stop_reason
    else:
        if not smoke:
            write_text(run_dir / "manifest.md", manifest_head(spec, *hashes, "n/a"))
        params = (
            dict(results[spec.params_from]["params"]) if spec.params_from else spec.fixed_params
        )
        trials, stop_reason = 1, "none"

    final_cv = cv(params)
    model = fit_model(
        spec.family,
        spec.feature_set,
        params,
        train,
        y,
        sample_weight=weight,
        seed=SEED,
        allow_leaky=spec.feature_set == "leaky",
    )
    val_metrics = binary_metrics(val[TARGET], positive_scores(model, val))
    result = {
        "run_id": spec.run_id,
        "family": spec.family,
        "feature_set": spec.feature_set,
        "weighted": spec.weighted,
        "selectable": spec.selectable,
        "params": params,
        "fold_scores": final_cv.fold_scores,
        "cv_mean": final_cv.cv_mean,
        "cv_se": final_cv.cv_se,
        "val_metrics": val_metrics,
        "trials": trials,
        "stop_reason": stop_reason,
    }
    logger.info(
        "%s: cv %.4f +/- %.4f, val %.4f, trials %d, stop %s",
        spec.run_id,
        final_cv.cv_mean,
        final_cv.cv_se,
        val_metrics["roc_auc"],
        trials,
        stop_reason,
    )
    if not smoke:
        write_text(run_dir / "result.json", json.dumps(result, indent=2))
        folds_text = ", ".join(f"{s:.4f}" for s in final_cv.fold_scores)
        with (run_dir / "manifest.md").open("a", encoding="utf-8") as fh:
            fh.write(
                f"\n## Result\n\n- Trials run: {trials}\n- Stop reason: {stop_reason}\n"
                f"- Best value and cv_se: {final_cv.cv_mean:.4f}, {final_cv.cv_se:.4f} "
                f"(folds: {folds_text})\n"
                f"- Validation AUC: {val_metrics['roc_auc']:.4f}\n"
                f"- Parameters: `{json.dumps(params)}`\n"
                f"- Leaderboard row: {spec.run_id}\n"
            )
    return result


def convergence_check(
    spec: RunSpec,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    result: dict[str, Any],
    seeds: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """Rerun a tuned search with other seeds, double window and double cap.

    The search counts as converged when no rerun beats the recorded best CV AUC by more than
    the minimum delta (the cv_se of the default configuration).

    Args:
        spec: Specification of a tuned run (cap > 0).
        train: Training window.
        folds: Expanding-window folds over `train`.
        result: Recorded result of the run.
        seeds: Sampler seeds of the reruns.

    Returns:
        Dictionary with the minimum delta, per-seed best values, trials, stop reasons and the
        status ("converged" or "unconverged"). Also appended to the run manifest.
    """
    run_dir = RUNS_DIR / spec.run_id
    y = train[TARGET]

    def cv(params: dict[str, Any]) -> CVResult:
        return cross_validate(spec.family, spec.feature_set, params, train, y, folds, seed=SEED)

    min_delta = cv(DEFAULT_PARAMS[spec.family]).cv_se
    space = SPACES[spec.family]
    reruns = []
    for seed in seeds:
        log_path = run_dir / f"trials_seed{seed}.csv"
        log_path.unlink(missing_ok=True)
        rerun = tune(
            lambda trial: cv(space(trial)).cv_mean,
            "maximize",
            min_delta=min_delta,
            window=2 * WINDOW,
            cap=2 * spec.cap,
            seed=seed,
            log_path=log_path,
        )
        reruns.append(
            {
                "seed": seed,
                "best": rerun.best_value,
                "trials": rerun.n_trials,
                "stop_reason": rerun.stop_reason,
            }
        )
        logger.info(
            "%s convergence seed %d: best %.4f after %d trials (%s)",
            spec.run_id,
            seed,
            rerun.best_value,
            rerun.n_trials,
            rerun.stop_reason,
        )
    beaten = [r for r in reruns if r["best"] > result["cv_mean"] + min_delta]
    status = "unconverged" if beaten else "converged"
    lines = "".join(
        f"  - seed {r['seed']}: best {r['best']:.4f} after {r['trials']} trials "
        f"({r['stop_reason']}), log trials_seed{r['seed']}.csv\n"
        for r in reruns
    )
    with (run_dir / "manifest.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## Convergence check\n\n- Best-so-far curve: trials.csv (column best_so_far)\n"
            f"- Rerun with a larger cap: window {2 * WINDOW}, cap {2 * spec.cap}, "
            f"minimum delta {min_delta:.5f}\n{lines}"
            f"- Status: {status} (recorded best {result['cv_mean']:.4f})\n"
        )
    return {"min_delta": min_delta, "reruns": reruns, "status": status}


def sensitivity_search(
    spec: RunSpec,
    train: pd.DataFrame,
    val: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    hashes: tuple[str, str],
) -> dict[str, Any]:
    """Search one family and feature set with a fine stopping rule over several seeds.

    The minimum delta (0.001) is finer than the selection margins it informs. The best
    configuration over all seeds is refit and scored on validation. The test window is not read.

    Args:
        spec: Sensitivity run specification.
        train: Training window.
        val: Validation window.
        folds: Expanding-window folds over `train`.
        hashes: (data hash, lock hash).

    Returns:
        Result dictionary, also written to runs/<run_id>/result.json.
    """
    run_dir = RUNS_DIR / spec.run_id
    y = train[TARGET]
    delta_text = (
        f"{SENSITIVITY_MIN_DELTA} (fixed, finer than the selection margin); window "
        f"{SENSITIVITY_WINDOW}; seeds {list(SENSITIVITY_SEEDS)}"
    )
    write_text(run_dir / "manifest.md", manifest_head(spec, *hashes, delta_text))

    def cv(params: dict[str, Any]) -> CVResult:
        return cross_validate(spec.family, spec.feature_set, params, train, y, folds, seed=SEED)

    space = SPACES[spec.family]
    per_seed = []
    for seed in SENSITIVITY_SEEDS:
        log_path = run_dir / f"trials_seed{seed}.csv"
        log_path.unlink(missing_ok=True)
        tuning = tune(
            lambda trial: cv(space(trial)).cv_mean,
            "maximize",
            min_delta=SENSITIVITY_MIN_DELTA,
            window=SENSITIVITY_WINDOW,
            cap=spec.cap,
            seed=seed,
            log_path=log_path,
        )
        params = {**tuning.best_params, "subsample_freq": 1}
        per_seed.append(
            {
                "seed": seed,
                "best": tuning.best_value,
                "trials": tuning.n_trials,
                "stop_reason": tuning.stop_reason,
                "params": params,
            }
        )
        logger.info(
            "%s seed %d: best %.4f after %d trials (%s)",
            spec.run_id,
            seed,
            tuning.best_value,
            tuning.n_trials,
            tuning.stop_reason,
        )
    best = max(per_seed, key=lambda r: r["best"])
    final_cv = cv(best["params"])
    model = fit_model(spec.family, spec.feature_set, best["params"], train, y, seed=SEED)
    val_metrics = binary_metrics(val[TARGET], positive_scores(model, val))
    stops = {r["stop_reason"] for r in per_seed}
    result = {
        "run_id": spec.run_id,
        "family": spec.family,
        "feature_set": spec.feature_set,
        "weighted": False,
        "selectable": False,
        "params": best["params"],
        "fold_scores": final_cv.fold_scores,
        "cv_mean": final_cv.cv_mean,
        "cv_se": final_cv.cv_se,
        "val_metrics": val_metrics,
        "trials": sum(r["trials"] for r in per_seed),
        "stop_reason": "cap" if "cap" in stops else "plateau",
        "per_seed": per_seed,
    }
    write_text(run_dir / "result.json", json.dumps(result, indent=2))
    lines = "".join(
        f"  - seed {r['seed']}: best {r['best']:.4f} after {r['trials']} trials "
        f"({r['stop_reason']})\n"
        for r in per_seed
    )
    spread = max(r["best"] for r in per_seed) - min(r["best"] for r in per_seed)
    with (run_dir / "manifest.md").open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## Result\n\n{lines}- Best over seeds: {final_cv.cv_mean:.4f} "
            f"(cv_se {final_cv.cv_se:.4f}); spread across seeds {spread:.4f}\n"
            f"- Validation AUC: {val_metrics['roc_auc']:.4f}\n"
            f"- Parameters: `{json.dumps(best['params'])}`\n"
        )
    return result


def write_leaderboard(results: list[dict[str, Any]]) -> None:
    """Write reports/leaderboard.csv with one row per run.

    The `selected` and `test_score` columns are written by the analysis notebook after the
    test pass. They are carried over from the existing leaderboard so that adding runs does not
    erase the recorded selection.

    Args:
        results: Run results in registration order.
    """
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, dict[str, str]] = {}
    if LEADERBOARD.exists():
        with LEADERBOARD.open(newline="", encoding="utf-8") as fh:
            recorded = {row["run_id"]: row for row in csv.DictReader(fh)}
    with LEADERBOARD.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "data_version": DATA_VERSION,
                    "split_id": DEFAULT_PROTOCOL.split_id,
                    "cv_scheme": CV_SCHEME,
                    "metric": METRIC,
                    "direction": DIRECTION,
                    "run_id": r["run_id"],
                    "family": r["family"],
                    "feature_set": r["feature_set"],
                    "weighted": "yes" if r["weighted"] else "no",
                    "selectable": "yes" if r["selectable"] else "no",
                    "cv_mean": f"{r['cv_mean']:.5f}",
                    "cv_se": f"{r['cv_se']:.5f}",
                    "val_score": f"{r['val_metrics']['roc_auc']:.5f}",
                    "trials": r["trials"],
                    "stop_reason": r["stop_reason"],
                    "selected": recorded.get(r["run_id"], {}).get("selected", "no"),
                    "test_score": recorded.get(r["run_id"], {}).get("test_score", ""),
                    "manifest": f"../runs/{r['run_id']}/manifest.md",
                }
            )
    logger.info("wrote %s", LEADERBOARD)


def add_r13(result_path: Path, leaderboard: Path) -> None:
    """Append the R13 row to an existing leaderboard, leaving every other row unchanged.

    Args:
        result_path: runs/R13.../result.json written by scripts/build_v2.py.
        leaderboard: Leaderboard CSV to extend.

    Raises:
        FileExistsError: If the leaderboard already has the R13 row.
    """
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with leaderboard.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if any(r["run_id"] == result["run_id"] for r in rows):
        raise FileExistsError(f"{result['run_id']} is already on {leaderboard}")
    search = result["search"]
    rows.append(
        {
            "data_version": DATA_VERSION,
            "split_id": DEFAULT_PROTOCOL.split_id,
            "cv_scheme": CV_SCHEME,
            "metric": METRIC,
            "direction": DIRECTION,
            "run_id": result["run_id"],
            "family": result["family"],
            "feature_set": result["feature_set"],
            "weighted": "no",
            "selectable": "no",
            "cv_mean": f"{search['cv_auc_mean']:.5f}",
            "cv_se": f"{search['cv_auc_se']:.5f}",
            "val_score": f"{result['val_auc_raw']:.5f}",
            "trials": search["trials"],
            "stop_reason": search["stop_reason"],
            "selected": "no",
            "test_score": "",
            "manifest": f"../runs/{result['run_id']}/manifest.md",
        }
    )
    with leaderboard.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("added %s to %s", result["run_id"], leaderboard)


def main() -> None:
    """Parse arguments and run the registered experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", help="run_id prefixes to run (default: all)")
    parser.add_argument("--smoke", action="store_true", help="3 trials per search, no outputs")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="post-test sensitivity searches (R08, R09) with a fine stopping rule",
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="rerun the tuned searches with two more seeds, double window and cap",
    )
    parser.add_argument(
        "--references",
        action="store_true",
        help="post-test reference runs (R11, R12) on CV and validation only; runs only these",
    )
    parser.add_argument(
        "--add-r13",
        action="store_true",
        help="append R13 to the leaderboard from its result.json; runs nothing else",
    )
    args = parser.parse_args()
    level = "DEBUG" if args.verbose else os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if args.add_r13:
        add_r13(RUNS_DIR / "R13-lgbm-no-apr-logloss-v2" / "result.json", LEADERBOARD)
        return

    train, val = load_frames()
    folds = expanding_quarter_folds(train[TIME], train[TARGET])
    latest = max(int(train[TIME].iloc[np.concatenate(f)].max()) for f in folds)
    if latest > DEFAULT_PROTOCOL.train_end:
        raise RuntimeError(f"fold vintage {latest} is after train_end {DEFAULT_PROTOCOL.train_end}")
    hashes = (data_sha256(DATA_PATH), file_sha256(ROOT / "uv.lock"))
    selected = [s for s in RUNS if not args.runs or any(s.run_id.startswith(p) for p in args.runs)]
    done: dict[str, dict[str, Any]] = {}
    for path in RUNS_DIR.glob("*/result.json"):
        prior = json.loads(path.read_text(encoding="utf-8"))
        done[prior["run_id"]] = prior
    if args.sensitivity:
        for spec in SENSITIVITY_RUNS:
            done[spec.run_id] = sensitivity_search(spec, train, val, folds, hashes)
        write_leaderboard(
            [
                done[s.run_id]
                for s in (*RUNS, *SENSITIVITY_RUNS, *REFERENCE_RUNS)
                if s.run_id in done
            ]
        )
        return
    if args.references:
        for spec in REFERENCE_RUNS:
            done[spec.run_id] = run_one(spec, train, val, folds, done, hashes, args.smoke)
        if not args.smoke:
            write_leaderboard(
                [
                    done[s.run_id]
                    for s in (*RUNS, *SENSITIVITY_RUNS, *REFERENCE_RUNS)
                    if s.run_id in done
                ]
            )
        return
    if args.convergence:
        for spec in selected:
            if spec.cap:
                check = convergence_check(spec, train, folds, done[spec.run_id])
                done[spec.run_id]["convergence"] = check
                write_text(
                    RUNS_DIR / spec.run_id / "result.json", json.dumps(done[spec.run_id], indent=2)
                )
        return
    for spec in selected:
        if spec.params_from and spec.params_from not in done:
            sys.exit(f"{spec.run_id} needs {spec.params_from} to have run first")
        done[spec.run_id] = run_one(spec, train, val, folds, done, hashes, args.smoke)
    if not args.smoke:
        write_leaderboard(
            [
                done[s.run_id]
                for s in (*RUNS, *SENSITIVITY_RUNS, *REFERENCE_RUNS)
                if s.run_id in done
            ]
        )


if __name__ == "__main__":
    main()
