"""Run the registered experiments and write the leaderboard, manifests and trial logs.

Every run is cross-validated on the training window (expanding quarterly folds) and scored
once on the validation window. The test window is never loaded here.

Usage:
    uv run python scripts/run_experiments.py              # all runs
    uv run python scripts/run_experiments.py --runs R02   # a subset
    uv run python scripts/run_experiments.py --smoke      # 3 trials per search, no outputs kept
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
        fixed_params: Parameters for a single fit when there is no search and no params_from.
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
    if spec.cap:
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
            allow_leaky=not spec.selectable,
        )

    if spec.cap:
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
        allow_leaky=not spec.selectable,
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


def write_leaderboard(results: list[dict[str, Any]]) -> None:
    """Write reports/leaderboard.csv with one row per run (none selected yet).

    Args:
        results: Run results in registration order.
    """
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
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
                    "selected": "no",
                    "test_score": "",
                    "manifest": f"../runs/{r['run_id']}/manifest.md",
                }
            )
    logger.info("wrote %s", LEADERBOARD)


def main() -> None:
    """Parse arguments and run the registered experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", help="run_id prefixes to run (default: all)")
    parser.add_argument("--smoke", action="store_true", help="3 trials per search, no outputs")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()
    level = "DEBUG" if args.verbose else os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    for spec in selected:
        if spec.params_from and spec.params_from not in done:
            sys.exit(f"{spec.run_id} needs {spec.params_from} to have run first")
        done[spec.run_id] = run_one(spec, train, val, folds, done, hashes, args.smoke)
    if not args.smoke:
        write_leaderboard([done[s.run_id] for s in RUNS if s.run_id in done])


if __name__ == "__main__":
    main()
