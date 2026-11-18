import numpy as np
import pandas as pd
import pytest

from credit_risk.config import SplitProtocol
from credit_risk.splits import assign_split, expanding_quarter_folds


def test_assign_split():
    v = pd.Series([202101, 202206, 202207, 202212, 202301, 202312, 202401, 202409])
    assert assign_split(v).tolist() == [
        "train", "train", "val", "val", "gap", "gap", "test", "test"
    ]  # fmt: skip


def test_assign_split_after_test_end():
    with pytest.raises(ValueError, match="after test_end"):
        assign_split(pd.Series([202410]))


def _frame():
    months = [m for y in (2021, 2022) for m in range(1, 13) if y * 100 + m <= 202206]
    v = pd.Series(
        np.repeat([2021 * 100 + m for m in range(1, 13)] + [202201 + i for i in range(6)], 10)
    )
    y = pd.Series(np.tile([0, 1], len(v) // 2))
    assert months
    return v, y


def test_expanding_folds_structure():
    v, y = _frame()
    q = (v // 100) * 4 + ((v % 100) - 1) // 3
    folds = expanding_quarter_folds(v, y, n_folds=4, min_positives=5)
    assert len(folds) == 4
    uq = sorted(q.unique())
    assert len(uq) == 6
    for k, (tr, va) in enumerate(folds):
        assert set(q.iloc[va]) == {uq[2 + k]}
        assert set(q.iloc[tr]) == set(uq[: 2 + k])
        assert not set(tr) & set(va)


def test_expanding_folds_errors():
    v, y = _frame()
    with pytest.raises(ValueError, match="positives"):
        expanding_quarter_folds(v, y, n_folds=4, min_positives=50)
    with pytest.raises(ValueError, match="quarters"):
        expanding_quarter_folds(v, y, n_folds=6, min_positives=1)


def test_protocol_december_rollover():
    p = SplitProtocol(train_end=202212, val_start=202301, val_end=202306)
    assert p.val_start == 202301


@pytest.mark.parametrize(
    "kwargs",
    [
        {"val_start": 202208},  # gap after train_end
        {"val_start": 202206},  # overlaps train
        {"train_end": 202213, "val_start": 202214},  # invalid month
        {"val_end": 202206},  # val_end before val_start
        {"test_start": 202212},  # test_start not after val_end
        {"test_end": 202312},  # test_end before test_start
    ],
)
def test_protocol_rejects(kwargs):
    with pytest.raises(ValueError):
        SplitProtocol(**kwargs)
