"""Optuna search with a plateau stopping rule and a hard cap."""

import csv
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import optuna
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Outcome of a tuning run.

    Attributes:
        best_value: Best objective value.
        best_params: Parameters of the best trial.
        n_trials: Number of completed trials.
        stop_reason: "plateau" or "cap".
        history: Frame with trial, value, best_so_far, params (JSON string).
    """

    best_value: float
    best_params: dict
    n_trials: int
    stop_reason: Literal["plateau", "cap"]
    history: pd.DataFrame


def tune(
    objective: Callable[[optuna.Trial], float],
    direction: Literal["minimize", "maximize"],
    *,
    min_delta: float,
    window: int = 30,
    cap: int,
    seed: int = 0,
    log_path: Path | None = None,
) -> TuningResult:
    """Run a TPE search that stops on a plateau or at a hard cap.

    An improvement is a value better than the best so far by more than `min_delta`.

    Args:
        objective: Function of an Optuna trial returning the value to optimize.
        direction: "minimize" or "maximize".
        min_delta: Minimum improvement that resets the plateau counter.
        window: Consecutive completed trials without improvement that trigger a stop.
        cap: Maximum number of trials.
        seed: Seed for the TPE sampler.
        log_path: If given, one CSV row per trial is appended as it completes.

    Returns:
        A TuningResult.

    Raises:
        ValueError: If a trial ends in a state other than COMPLETE (failed, NaN or pruned).
    """
    sign = 1.0 if direction == "maximize" else -1.0
    study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=seed))
    rows: list[dict] = []
    state = {"best": None, "since": 0, "plateau": False}

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Record a finished trial, log it and apply the plateau rule.

        Args:
            study: The running study.
            trial: The trial that just finished.
        """
        if trial.state != optuna.trial.TrialState.COMPLETE:
            raise ValueError(
                f"trial {trial.number} ended in state {trial.state.name}, not COMPLETE"
            )
        value = float(trial.value)
        best = state["best"]
        if best is None or sign * (value - best) > min_delta:
            state["since"] = 0
        else:
            state["since"] += 1
        if best is None or sign * (value - best) > 0:
            state["best"] = value
        row = {
            "trial": trial.number,
            "value": value,
            "best_so_far": state["best"],
            "params": json.dumps(trial.params, sort_keys=True),
        }
        rows.append(row)
        if log_path is not None:
            new = not log_path.exists() or log_path.stat().st_size == 0
            with log_path.open("a", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    writer.writeheader()
                writer.writerow(row)
        if state["since"] >= window:
            state["plateau"] = True
            study.stop()

    study.optimize(objective, n_trials=cap, callbacks=[callback])
    reason: Literal["plateau", "cap"] = "plateau" if state["plateau"] else "cap"
    logger.info("tuning stopped (%s) after %s trials", reason, len(rows))
    return TuningResult(
        best_value=float(study.best_value),
        best_params=dict(study.best_params),
        n_trials=len(rows),
        stop_reason=reason,
        history=pd.DataFrame(rows, columns=["trial", "value", "best_so_far", "params"]),
    )
