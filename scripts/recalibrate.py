"""Build the v1.1 artifact: the frozen v1 base model with a calibrator chosen on validation.

The base model is the pipeline inside the frozen v1 artifact, unchanged. The calibration method is
chosen with two time-ordered folds inside the validation window, refit on all of validation, and
the result is scored once on the gap year and the test window next to v1 and the raw scores.

Usage:
    uv run python scripts/recalibrate.py --register-only   # write the manifest and stop
    uv run python scripts/recalibrate.py                   # build, score and write the results
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.frozen import FrozenEstimator

from credit_risk.config import APPLICATION_COLUMNS, DEFAULT_PROTOCOL, ID, TARGET, TIME
from credit_risk.evaluation import breakeven_pd, cutoff_table
from credit_risk.models import calibrate, positive_scores
from credit_risk.packaging import file_sha256, load_artifact
from credit_risk.pipeline import (
    Candidate,
    _jsonable,
    evaluate_holdout,
    load_and_validate,
    package_model,
    select_calibrator,
    split_frames,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_risk_data_enhanced.csv"
LOCK_PATH = ROOT / "uv.lock"
V1_DIR = ROOT / "models" / "pd-lgbm-no-apr-v1"
OUT_DIR = ROOT / "models" / "pd-lgbm-no-apr-v1.1"
RUN_DIR = ROOT / "runs" / "R10-calibration-v1.1"
PREDICTIONS = ROOT / "reports" / "test_predictions_v1.1.csv"

RUN_ID = "R10-calibration-v1.1"
LGD = 0.6
MARGIN = 0.10
APPROVAL_RATES = (0.80, 0.85, 0.90, 0.95)

MANIFEST_BODY = """## Question
The v1 calibrator (Platt on the raw probability) loses to a constant on test Brier. Which
calibration shape, chosen only from data mature at a 2024-01 decision date, should v1.1 use,
and how does it score on the gap year and test?

## Status
Registered after the test window was scored for v1 (post-hoc). The base model is the frozen
v1 pipeline, unchanged. The results below are published whatever they show.

## Protocol
- Input: raw scores of the frozen v1 base model on validation 202207-202212.
- Candidates, simplest first: logit_intercept (log-odds shift), logit_platt (intercept and
  slope on log-odds), sigmoid (v1 method), isotonic.
- Selection: two time-ordered folds inside validation (202207-202209, 202210-202212). Fit on
  one, score the other, pool the out-of-fold predictions. Choose the simplest method whose
  pooled Brier is within one paired standard error of the lowest.
- Refit the chosen method on all of validation. Cutoff: break-even PD 0.10 / 0.70.
- Score the 2023 gap year and the 2024 test window once. Brier skill is reported against the
  training-window default rate, the constant available at the decision date.
- Not used: any 2023 or 2024 label for fitting or choosing.

## Expected
Shape fix lowers Brier at the bottom of the scale. Test Brier skill stays near zero because
the 2024 level shift was not knowable at the decision date.
"""


class RegistrationError(RuntimeError):
    """Raised when the registered manifest does not match the current inputs."""


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
        f"- Family and feature set: lgbm, no_apr (frozen v1 base model)\n\n"
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


def check_registration(manifest: Path, data_path: Path, lock_path: Path) -> None:
    """Verify that the registered hashes match the current data and lock files.

    Args:
        manifest: Registered manifest.
        data_path: Data file.
        lock_path: Environment lock file.

    Raises:
        RegistrationError: If a hash line is missing or differs from the current file.
    """
    text = manifest.read_text(encoding="utf-8")
    for label, path in (("Data hash", data_path), ("Environment lock", lock_path)):
        found = re.search(rf"^- {label}: (\S+)$", text, flags=re.MULTILINE)
        if found is None:
            raise RegistrationError(f"{manifest} has no '{label}' line")
        if found.group(1) != file_sha256(path):
            raise RegistrationError(f"{label} in {manifest} differs from the current {path}")


def frozen_base(model: object) -> object:
    """Return the fitted base pipeline inside a calibrated v1 artifact.

    Args:
        model: Loaded v1 model, a CalibratedClassifierCV around a FrozenEstimator.

    Returns:
        The fitted pipeline, unchanged.

    Raises:
        ValueError: If the model does not wrap a FrozenEstimator.
    """
    inner = model.calibrated_classifiers_[0].estimator
    if not isinstance(inner, FrozenEstimator):
        raise ValueError(f"expected a FrozenEstimator inside the artifact, got {type(inner)}")
    return inner.estimator


def _decile_row(ev: dict[str, Any]) -> dict[str, float]:
    """Pick the bottom and top calibration decile from a holdout evaluation.

    Args:
        ev: Output of `evaluate_holdout`.

    Returns:
        Dict with bottom and top predicted and observed rates.
    """
    t = ev["calibration_table"]
    lo, hi = t.iloc[0], t.iloc[-1]
    return {
        "bottom_pred": float(lo["mean_score"]),
        "bottom_obs": float(lo["observed_rate"]),
        "top_pred": float(hi["mean_score"]),
        "top_obs": float(hi["observed_rate"]),
    }


def summary_table(evals: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Summarize one split for several models, one row per model.

    Args:
        evals: Model label to `evaluate_holdout` output.

    Returns:
        Frame with model, auc, brier, brier_skill, ece, mean_pd, observed, decile rates, approval
        and bad rate at the cutoff.
    """
    rows = []
    for label, ev in evals.items():
        rows.append(
            {
                "model": label,
                "auc": ev["metrics"]["roc_auc"],
                "brier": ev["metrics"]["brier"],
                "brier_skill": ev["brier_skill"],
                "ece": ev["ece"],
                "mean_pd": ev["mean_pd"],
                "observed": ev["observed_rate"],
                **_decile_row(ev),
                "approval": ev["approval_rate"],
                "bad_rate_approved": ev["bad_rate_approved"],
            }
        )
    return pd.DataFrame(rows)


def _markdown(table: pd.DataFrame) -> str:
    """Render a frame as a pipe table with 5-significant-digit floats.

    Args:
        table: Frame to render.

    Returns:
        Markdown table text.
    """
    head = "| " + " | ".join(table.columns) + " |\n"
    rule = "|" + "|".join("---" for _ in table.columns) + "|\n"
    body = ""
    for row in table.itertuples(index=False):
        cells = [f"{v:.5g}" if isinstance(v, float) else str(v) for v in row]
        body += "| " + " | ".join(cells) + " |\n"
    return head + rule + body


def run_full(
    *,
    data_path: Path,
    lock_path: Path,
    v1_dir: Path,
    out_dir: Path,
    run_dir: Path,
    predictions_path: Path,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Build v1.1, score gap and test once, and write all outputs.

    Args:
        data_path: Raw data CSV.
        lock_path: Environment lock file whose hash the manifest registered.
        v1_dir: Frozen v1 artifact directory (read only).
        out_dir: New v1.1 artifact directory.
        run_dir: Run folder; its manifest must already exist.
        predictions_path: Output CSV of v1 and v1.1 test predictions.
        n_boot: Bootstrap resamples for the AUC intervals.

    Returns:
        Result dictionary, also written to `run_dir / "result.json"`.

    Raises:
        FileNotFoundError: If the manifest was not registered first.
        RegistrationError: If the registered hashes differ from the current files.
        FileExistsError: If a result, the v1.1 artifact or the predictions already exist.
    """
    manifest = run_dir / "manifest.md"
    if not manifest.exists():
        raise FileNotFoundError(f"register the run first (--register-only): {manifest}")
    check_registration(manifest, data_path, lock_path)
    if (run_dir / "result.json").exists():
        raise FileExistsError(f"result already exists: {run_dir / 'result.json'}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {out_dir}")
    if predictions_path.exists():
        raise FileExistsError(f"predictions already exist: {predictions_path}")
    df, data_hash = load_and_validate(data_path)
    frames = split_frames(df, DEFAULT_PROTOCOL)
    v1, meta = load_artifact(v1_dir)
    base = frozen_base(v1)
    val = frames["val"]
    cols = list(APPLICATION_COLUMNS)

    selection = select_calibrator(base, val)
    chosen = selection["chosen"]
    final = calibrate(base, val[cols], val[TARGET], chosen)
    cutoff = breakeven_pd(LGD, MARGIN)
    val_scores = positive_scores(final, val[cols])
    validation = {
        "cutoff_table": cutoff_table(val[TARGET], val_scores, APPROVAL_RATES),
        "approval_rate_at_breakeven": float((val_scores <= cutoff).mean()),
    }
    reference_rate = float(frames["train"][TARGET].mean())
    models = {"raw": base, "v1": v1, "v1.1": final}
    evals = {
        split: {
            label: evaluate_holdout(
                m,
                frames[split],
                f"{split}/{label}",
                threshold=cutoff,
                reference_rate=reference_rate,
                n_boot=n_boot,
            )
            for label, m in models.items()
        }
        for split in ("gap", "test")
    }

    test = frames["test"]
    predictions = pd.DataFrame(
        {
            ID: test[ID],
            TIME: test[TIME],
            TARGET: test[TARGET],
            "pd_v1": positive_scores(v1, test[cols]),
            "pd_v1_1": positive_scores(final, test[cols]),
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)

    package_model(
        final,
        out_dir,
        candidate=Candidate(**meta["candidate"]),
        metrics={
            "validation": validation,
            "gap": evals["gap"]["v1.1"],
            "test": evals["test"]["v1.1"],
        },
        data_hash=data_hash,
        protocol=DEFAULT_PROTOCOL,
        threshold=cutoff,
        extra={
            "calibration": chosen,
            "selection_table": selection["table"],
            "fold_vintages": selection["fold_vintages"],
            "v1_model_sha256": meta["model_sha256"],
            "reference_rate": reference_rate,
            "lgd": LGD,
            "margin": MARGIN,
        },
    )

    summaries = {split: summary_table(evals[split]) for split in evals}
    result = {
        "run_id": RUN_ID,
        "chosen": chosen,
        "best": selection["best"],
        "fold_vintages": selection["fold_vintages"],
        "selection_table": selection["table"],
        "reference_rate": reference_rate,
        "breakeven_pd": cutoff,
        "validation": validation,
        "summary": summaries,
        "evaluations": evals,
    }
    (run_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2), encoding="utf-8")
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## Result\n\n### Selection (out-of-fold on validation)\n\n"
            f"{_markdown(selection['table'])}\n"
            f"- Folds: {selection['fold_vintages']}\n"
            f"- Lowest pooled Brier: {selection['best']}\n- Chosen: {chosen}\n"
            f"- Break-even cutoff: {cutoff:.4f}; training-window default rate "
            f"{reference_rate:.5f}\n"
        )
        for split in ("gap", "test"):
            fh.write(f"\n### {split}\n\n{_markdown(summaries[split])}")
    logger.info("chosen %s; wrote %s", chosen, run_dir)
    return result


def main() -> None:
    """Parse arguments and register the run or build v1.1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true", help="write the manifest and exit")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging")
    args = parser.parse_args()
    level = "DEBUG" if args.verbose else os.environ.get("LOG_LEVEL", "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.register_only:
            register(RUN_DIR, DATA_PATH, LOCK_PATH, date.today().isoformat())
        else:
            run_full(
                data_path=DATA_PATH,
                lock_path=LOCK_PATH,
                v1_dir=V1_DIR,
                out_dir=OUT_DIR,
                run_dir=RUN_DIR,
                predictions_path=PREDICTIONS,
            )
    except (FileExistsError, FileNotFoundError, RegistrationError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
