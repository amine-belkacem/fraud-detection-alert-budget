"""Temporal train / validation / test split.

A random split mixes transactions from the second day into training, so the
model gets to see the future it is evaluated on. Here the rows are cut in time
order: train is the oldest 60 %, validation the next 20 %, test the most recent
20 %. Each cut is moved forward until the timestamp changes, so that no second
of activity is shared between two sets (the dataset has many transactions
stamped with the same second, including exact duplicates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudbudget.data import TIME

DEFAULT_FRACTIONS = (0.6, 0.2, 0.2)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _cut_after_timestamp(times: np.ndarray, index: int) -> int:
    """First position >= index whose timestamp differs from the one before it."""
    if index <= 0 or index >= len(times):
        return index
    # searchsorted(side="right") jumps past every row sharing times[index - 1].
    return int(np.searchsorted(times, times[index - 1], side="right"))


def temporal_split(
    df: pd.DataFrame, fractions: tuple[float, float, float] = DEFAULT_FRACTIONS
) -> TemporalSplit:
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError(f"need three positive fractions, got {fractions}")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    times = df[TIME].to_numpy()
    if not np.all(np.diff(times) >= 0):
        raise ValueError("rows must be sorted by Time before splitting")

    n = len(df)
    cut_val = _cut_after_timestamp(times, int(n * fractions[0]))
    cut_test = _cut_after_timestamp(times, int(n * (fractions[0] + fractions[1])))
    if not 0 < cut_val < cut_test < n:
        raise ValueError(f"split produced an empty set (cuts at {cut_val}, {cut_test} of {n})")

    return TemporalSplit(
        train=df.iloc[:cut_val],
        val=df.iloc[cut_val:cut_test],
        test=df.iloc[cut_test:],
    )
