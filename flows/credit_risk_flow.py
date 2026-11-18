"""Metaflow pipeline: validate, split, sweep, select, calibrate, evaluate, package.

Run with `uv run python flows/credit_risk_flow.py run`. Step logic lives in
`credit_risk.pipeline`; this file only orchestrates.
"""

import logging
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
from metaflow import FlowSpec, Parameter, card, current, retry, step
from metaflow.cards import Markdown, Table

from credit_risk import pipeline
from credit_risk.config import DEFAULT_PROTOCOL, FEATURE_SETS, TARGET, SplitProtocol
from credit_risk.models import CALIBRATION_METHODS

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]


def _configure_logging() -> None:
    """Configure root logging from LOG_LEVEL (default INFO).

    Called at the top of every step because each foreach task runs in its own process.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


class CreditRiskFlow(FlowSpec):
    """Out-of-time credit-risk pipeline with a champion/challenger selection."""

    data_path = Parameter(
        "data_path",
        default=str(_ROOT / "data" / "credit_risk_data_enhanced.csv"),
        help="Input CSV",
    )
    sweep = Parameter(
        "sweep", default=str(_ROOT / "flows" / "sweep.json"), help="JSON list of candidates"
    )
    train_end = Parameter("train_end", default=DEFAULT_PROTOCOL.train_end, type=int)
    val_start = Parameter("val_start", default=DEFAULT_PROTOCOL.val_start, type=int)
    val_end = Parameter("val_end", default=DEFAULT_PROTOCOL.val_end, type=int)
    test_start = Parameter("test_start", default=DEFAULT_PROTOCOL.test_start, type=int)
    test_end = Parameter("test_end", default=DEFAULT_PROTOCOL.test_end, type=int)
    calibration = Parameter(
        "calibration", default="logit_intercept", help=f"One of {', '.join(CALIBRATION_METHODS)}"
    )
    score_test = Parameter(
        "score_test",
        default=False,
        type=bool,
        help="Also score the test window (default: gap year only)",
    )
    lgd = Parameter("lgd", default=0.6, type=float, help="Loss given default")
    margin = Parameter("margin", default=0.10, type=float, help="Margin on a good loan")
    seed = Parameter("seed", default=0, type=int)
    output_dir = Parameter(
        "output_dir", default=str(_ROOT / "outputs" / "models"), help="Artifact root"
    )

    @step
    def start(self) -> None:
        """Load and validate the data, build the splits and read the sweep."""
        _configure_logging()
        if self.calibration not in CALIBRATION_METHODS:
            raise ValueError(
                f"calibration must be one of {CALIBRATION_METHODS}, got {self.calibration!r}"
            )
        self.protocol = SplitProtocol(
            train_end=self.train_end,
            val_start=self.val_start,
            val_end=self.val_end,
            test_start=self.test_start,
            test_end=self.test_end,
            split_id=DEFAULT_PROTOCOL.split_id,
        )
        if self.protocol != DEFAULT_PROTOCOL:
            self.protocol = replace(self.protocol, split_id="custom")
        df, self.data_hash = pipeline.load_and_validate(Path(self.data_path))
        self.frames = pipeline.split_frames(df, self.protocol)
        self.split_counts = {k: len(v) for k, v in self.frames.items()}
        self.candidates = pipeline.load_sweep(Path(self.sweep))
        logger.info("data sha256 %s; splits %s", self.data_hash, self.split_counts)
        self.next(self.train, foreach="candidates")

    @retry(times=1)
    @step
    def train(self) -> None:
        """Train one candidate (runs once per sweep entry)."""
        _configure_logging()
        self.result = pipeline.train_candidate(self.input, self.frames, seed=self.seed)
        self.next(self.select)

    @step
    def select(self, inputs: list) -> None:
        """Join the sweep, pick the champion and compare it with the challenger."""
        _configure_logging()
        self.merge_artifacts(inputs, exclude=["result"])
        results = [i.result for i in inputs]
        self.champion, self.challenger, self.explanation = pipeline.select_champion(results)
        self.comparison = (
            None
            if self.challenger is None
            else pipeline.compare_champion_challenger(
                self.champion, self.challenger, self.frames["val"][TARGET], seed=self.seed
            )
        )
        self.leaderboard = pd.DataFrame(
            {
                "name": r.candidate.name,
                "family": r.candidate.family,
                "feature_set": r.candidate.feature_set,
                "n_features": len(FEATURE_SETS[r.candidate.feature_set]),
                "weighted": r.candidate.weighted,
                "cv_mean": r.cv_mean,
                "cv_se": r.cv_se,
                "val_auc": r.val_auc,
            }
            for r in results
        ).sort_values("cv_mean", ascending=False, ignore_index=True)
        logger.info("leaderboard:\n%s", self.leaderboard.to_string())
        self.next(self.calibrate)

    @step
    def calibrate(self) -> None:
        """Calibrate the champion on validation and freeze the breakeven cutoff."""
        _configure_logging()
        self.calib = pipeline.calibrate_and_cutoff(
            self.champion,
            self.frames,
            method=self.calibration,
            lgd=self.lgd,
            margin=self.margin,
        )
        self.threshold = self.calib["breakeven_pd"]
        self.next(self.evaluate)

    @card
    @step
    def evaluate(self) -> None:
        """Score the gap year with the frozen model, and the test window only if asked."""
        _configure_logging()
        model = self.calib["calibrated_model"]
        reference_rate = float(self.frames["train"][TARGET].mean())
        self.gap_eval = pipeline.evaluate_holdout(
            model,
            self.frames["gap"],
            "gap",
            threshold=self.threshold,
            reference_rate=reference_rate,
            seed=self.seed,
        )
        self.test_eval = (
            pipeline.evaluate_holdout(
                model,
                self.frames["test"],
                "test",
                threshold=self.threshold,
                reference_rate=reference_rate,
                seed=self.seed,
            )
            if self.score_test
            else None
        )
        self.stability = pipeline.stability_report(model, self.frames)
        rows = []
        for ev in (e for e in (self.gap_eval, self.test_eval) if e is not None):
            rows.append(
                [
                    ev["name"],
                    f"{ev['metrics']['roc_auc']:.4f}",
                    f"[{ev['auc_ci'][0]:.4f}, {ev['auc_ci'][1]:.4f}]",
                    f"{ev['metrics']['brier']:.5f}",
                    f"{ev['brier_skill']:.4f}",
                    f"{ev['ece']:.4f}",
                    f"{ev['mean_pd']:.4f}",
                    f"{ev['observed_rate']:.4f}",
                    f"{ev['approval_rate']:.3f}",
                    f"{ev['bad_rate_approved']:.4f}",
                ]
            )
        current.card.append(Markdown(f"## Champion: {self.champion.candidate.name}"))
        current.card.append(Markdown(self.explanation))
        if not self.score_test:
            current.card.append(Markdown("The test window was not scored in this run."))
        current.card.append(
            Table(
                rows,
                headers=[
                    "split",
                    "AUC",
                    "AUC 95% CI",
                    "Brier",
                    "Brier skill",
                    "ECE",
                    "mean PD",
                    "observed",
                    "approval",
                    "bad rate approved",
                ],
            )
        )
        self.next(self.end)

    @step
    def end(self) -> None:
        """Save the versioned model artifact."""
        _configure_logging()
        path = Path(self.output_dir) / str(current.run_id)
        metrics = {
            "validation": {
                "champion_cv_mean": self.champion.cv_mean,
                "champion_cv_se": self.champion.cv_se,
                "champion_val_auc": self.champion.val_auc,
                "brier_before": self.calib["brier_before"],
                "brier_after": self.calib["brier_after"],
                "ece_before": self.calib["ece_before"],
                "ece_after": self.calib["ece_after"],
                "cutoff_table": self.calib["cutoff_table"],
                "approval_rate_at_breakeven": self.calib["approval_rate_at_breakeven"],
            },
            "gap": self.gap_eval,
            "test": self.test_eval,
            "stability": self.stability,
        }
        extra = {
            "run_id": str(current.run_id),
            "explanation": self.explanation,
            "leaderboard": self.leaderboard,
            "comparison": self.comparison,
            "calibration": self.calibration,
            "score_test": self.score_test,
            "lgd": self.lgd,
            "margin": self.margin,
            "seed": self.seed,
            "sweep": self.sweep,
        }
        self.artifact_path = str(
            pipeline.package_model(
                self.calib["calibrated_model"],
                path,
                candidate=self.champion.candidate,
                metrics=metrics,
                data_hash=self.data_hash,
                protocol=self.protocol,
                threshold=self.threshold,
                extra=extra,
            )
        )
        logger.info("artifact: %s", self.artifact_path)


if __name__ == "__main__":
    CreditRiskFlow()
