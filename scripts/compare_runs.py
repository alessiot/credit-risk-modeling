"""Compare recent successful CreditRiskFlow runs via the Metaflow Client API.

Usage:
    uv run python scripts/compare_runs.py [--last 5]
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


def summarize(run: Run) -> dict:
    """Extract the comparison fields of one run.

    Args:
        run: A successful CreditRiskFlow run.

    Returns:
        Dict with run id, created time, parameters, champion, CV AUC, test AUC with CI,
        test ECE and artifact path.
    """
    d = run.data
    test = d.test_eval
    return {
        "run_id": run.id,
        "created": run.created_at.strftime("%Y-%m-%d %H:%M"),
        "sweep": short_path(d.sweep),
        "train_end": d.train_end,
        "calibration": d.calibration,
        "champion": d.champion.candidate.name,
        "cv_auc": round(d.champion.cv_mean, 4),
        "test_auc": round(test["metrics"]["roc_auc"], 4),
        "test_auc_ci": f"[{test['auc_ci'][0]:.4f}, {test['auc_ci'][1]:.4f}]",
        "test_ece": round(test["ece"], 4),
        "artifact": short_path(d.artifact_path),
    }


def main() -> None:
    """Parse arguments and write the run table or one run's per-vintage test table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last", type=int, default=5, help="number of successful runs")
    parser.add_argument("--run-id", help="show the per-vintage test table of this run")
    args = parser.parse_args()
    namespace(None)
    flow = Flow(FLOW)
    if args.run_id:
        table = flow[args.run_id].data.test_eval["per_vintage"]
    else:
        runs = islice((r for r in flow.runs() if r.successful), args.last)
        table = pd.DataFrame([summarize(r) for r in runs])
    sys.stdout.write(table.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
