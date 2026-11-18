import csv

import optuna
import pytest

from credit_risk.tuning import tune


def quad(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    return (x - 1) ** 2


def test_plateau_stop():
    res = tune(quad, "minimize", min_delta=1e-3, window=10, cap=300)
    assert res.stop_reason == "plateau"
    assert res.n_trials < 300
    assert abs(res.best_params["x"] - 1) < 0.5
    assert list(res.history.columns) == ["trial", "value", "best_so_far", "params"]
    assert len(res.history) == res.n_trials
    assert res.history["best_so_far"].is_monotonic_decreasing
    assert res.best_value == res.history["value"].min()
    # last `window` trials brought no improvement over the best before them
    h = res.history
    assert h["best_so_far"].iloc[-1] - h["best_so_far"].iloc[-11] > -1e-3


def test_cap_stop_and_determinism():
    r1 = tune(quad, "minimize", min_delta=0.0, window=1000, cap=15, seed=4)
    r2 = tune(quad, "minimize", min_delta=0.0, window=1000, cap=15, seed=4)
    assert r1.stop_reason == "cap" and r1.n_trials == 15
    assert r1.history["value"].tolist() == r2.history["value"].tolist()


def test_maximize():
    res = tune(lambda t: -((t.suggest_float("x", -5, 5) - 1) ** 2), "maximize",
               min_delta=1e-3, window=10, cap=100)  # fmt: skip
    assert res.best_value <= 0
    assert res.history["best_so_far"].is_monotonic_increasing


def test_log_path(tmp_path):
    log = tmp_path / "trials.csv"
    res = tune(quad, "minimize", min_delta=0.0, window=1000, cap=5, log_path=log)
    with log.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["trial", "value", "best_so_far", "params"]
    assert len(rows) == 1 + res.n_trials


def test_plateau_window_exact():
    values = iter([10.0, 5.0] + [5.0] * 100)
    res = tune(lambda t: next(values), "minimize", min_delta=0.0, window=5, cap=50)
    assert res.n_trials == 7
    assert res.stop_reason == "plateau"


def test_nan_and_pruned_trials_raise():
    with pytest.raises(ValueError, match="trial 0"):
        tune(lambda t: float("nan"), "minimize", min_delta=0.0, cap=5)

    def pruned(trial: optuna.Trial) -> float:
        raise optuna.TrialPruned

    with pytest.raises(ValueError, match="trial 0"):
        tune(pruned, "minimize", min_delta=0.0, cap=5)


def test_log_header_once_across_runs(tmp_path):
    log = tmp_path / "t.csv"
    tune(quad, "minimize", min_delta=0.0, window=1000, cap=3, log_path=log)
    tune(quad, "minimize", min_delta=0.0, window=1000, cap=3, log_path=log)
    lines = log.read_text().splitlines()
    assert len(lines) == 7
    assert sum(line.startswith("trial,") for line in lines) == 1
