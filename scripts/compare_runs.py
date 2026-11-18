"""Compare recent successful CreditRiskFlow runs via the Metaflow Client API.

Usage:
    uv run python scripts/compare_runs.py [--last 5] [--with-test]
    uv run python scripts/compare_runs.py --run-id 12
"""

import argparse
import sys
from itertools import islice
from pathlib import Path

import pandas as pd
from metaflow import Flow, Run, namespace

FLOW = "CreditRiskFlow"
ROOT = Path(__file__).resolve().parents[1]


def short_path(path: str) -> str:
    """Show a path relative to the repository root when it lies inside it.

    Args:
        path: Absolute or relative path.

    Returns:
        The path relative to the repo root, or unchanged when it is outside.
    """
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else path


def summarize(run: Run, with_test: bool = False) -> dict:
    """Extract the comparison fields of one run.

    Test-window metrics are left out unless asked for, so that variants are compared on
    validation and gap-year results rather than on the test window.

    Args:
        run: A successful CreditRiskFlow run.
        with_test: Also include test AUC with its interval and test ECE; NaN and blank for a
            run that did not score the test window.

    Returns:
        Dict with run id, created time, parameters, champion, CV, validation and gap-year
        metrics, optionally test metrics, and the artifact path.
    """
    d = run.data
    gap = d.gap_eval
    row = {
        "run_id": run.id,
        "created": run.created_at.strftime("%Y-%m-%d %H:%M"),
        "sweep": short_path(d.sweep),
        "train_end": d.train_end,
        "calibration": d.calibration,
        "champion": d.champion.candidate.name,
        "cv_auc": round(d.champion.cv_mean, 4),
        "val_auc": round(d.champion.val_auc, 4),
        "gap_auc": round(gap["metrics"]["roc_auc"], 4),
        "gap_ece": round(gap["ece"], 4),
    }
    if with_test:
        test = d.test_eval
        if test is None:
            row["test_auc"] = float("nan")
            row["test_auc_ci"] = ""
            row["test_ece"] = float("nan")
        else:
            row["test_auc"] = round(test["metrics"]["roc_auc"], 4)
            row["test_auc_ci"] = f"[{test['auc_ci'][0]:.4f}, {test['auc_ci'][1]:.4f}]"
            row["test_ece"] = round(test["ece"], 4)
    row["artifact"] = short_path(d.artifact_path)
    return row


def main() -> None:
    """Parse arguments and write the run table or one run's per-vintage test table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=int, default=5, help="number of successful runs")
    parser.add_argument("--run-id", help="show the per-vintage test table of this run")
    parser.add_argument(
        "--with-test", action="store_true", help="add test-window metrics to the run table"
    )
    args = parser.parse_args()
    namespace(None)
    flow = Flow(FLOW)
    if args.run_id:
        test = flow[args.run_id].data.test_eval
        if test is None:
            sys.exit(f"run {args.run_id} did not score the test window")
        table = test["per_vintage"]
    else:
        runs = islice((r for r in flow.runs() if r.successful), args.last)
        table = pd.DataFrame([summarize(r, args.with_test) for r in runs])
    sys.stdout.write(table.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
