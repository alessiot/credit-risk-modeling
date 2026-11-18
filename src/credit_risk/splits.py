"""Out-of-time split assignment and expanding-window folds."""

import logging

import numpy as np
import pandas as pd

from credit_risk.config import DEFAULT_PROTOCOL, SplitProtocol

logger = logging.getLogger(__name__)


def assign_split(vintage: pd.Series, protocol: SplitProtocol = DEFAULT_PROTOCOL) -> pd.Series:
    """Label each row train, val, gap or test by vintage.

    Args:
        vintage: YYYYMM integers.
        protocol: Split boundaries.

    Returns:
        String Series aligned to `vintage`.

    Raises:
        ValueError: If any vintage is after `protocol.test_end`.
    """
    if (vintage > protocol.test_end).any():
        raise ValueError(f"vintages after test_end {protocol.test_end} present")
    labels = np.select(
        [
            vintage <= protocol.train_end,
            vintage <= protocol.val_end,
            vintage < protocol.test_start,
        ],
        ["train", "val", "gap"],
        default="test",
    )
    return pd.Series(labels, index=vintage.index, dtype="str")


def expanding_quarter_folds(
    vintage: pd.Series, y: pd.Series, *, n_folds: int = 4, min_positives: int = 50
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window folds over calendar quarters.

    The last `n_folds` quarters are the validation quarters, one per fold; each fold trains
    on all earlier quarters.

    Args:
        vintage: YYYYMM integers.
        y: Binary target aligned to `vintage`.
        n_folds: Number of folds.
        min_positives: Minimum positives required in every validation quarter.

    Returns:
        List of (train_idx, val_idx) positional index arrays.

    Raises:
        ValueError: If there are fewer than n_folds + 1 quarters or a validation quarter has
            too few positives.
    """
    v = np.asarray(vintage)
    target = np.asarray(y)
    quarter = (v // 100) * 4 + ((v % 100) - 1) // 3
    quarters = np.unique(quarter)
    if len(quarters) < n_folds + 1:
        raise ValueError(f"need at least {n_folds + 1} quarters, got {len(quarters)}")
    folds = []
    for vq in quarters[-n_folds:]:
        val_idx = np.flatnonzero(quarter == vq)
        n_pos = int(target[val_idx].sum())
        if n_pos < min_positives:
            raise ValueError(
                f"validation quarter {vq // 4}Q{vq % 4 + 1} has {n_pos} positives "
                f"(< {min_positives})"
            )
        folds.append((np.flatnonzero(quarter < vq), val_idx))
    return folds
