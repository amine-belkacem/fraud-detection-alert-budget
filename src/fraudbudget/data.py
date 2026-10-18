"""Loading and sanity checks for the ULB / Worldline credit card fraud dataset.

The CSV has 31 columns: Time (seconds since the first transaction), V1..V28
(PCA components, anonymised by the data provider), Amount (in euros) and Class
(1 = fraud). Every check below raises a ValueError that says what was expected,
so a corrupted or truncated download fails loudly instead of silently skewing
the results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TIME = "Time"
AMOUNT = "Amount"
TARGET = "Class"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
COLUMNS = [TIME, *PCA_FEATURES, AMOUNT, TARGET]

# Counts of the published dataset (Dal Pozzolo et al., 2015).
EXPECTED_ROWS = 284_807
EXPECTED_FRAUDS = 492


def validate_schema(df: pd.DataFrame) -> None:
    """Check columns, types and value ranges. Works on any number of rows."""
    if list(df.columns) != COLUMNS:
        missing = [c for c in COLUMNS if c not in df.columns]
        extra = [c for c in df.columns if c not in COLUMNS]
        raise ValueError(f"unexpected columns: missing={missing}, extra={extra}")

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        raise ValueError(f"{n_missing} missing values, expected none")

    non_numeric = [c for c in COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"non-numeric columns: {non_numeric}")

    labels = set(np.unique(df[TARGET]))
    if not labels <= {0, 1}:
        raise ValueError(f"Class must be 0 or 1, found {sorted(labels)}")

    if (df[AMOUNT] < 0).any():
        raise ValueError("negative transaction amount")

    # The temporal split relies on this: rows must already be in time order.
    if not df[TIME].is_monotonic_increasing:
        raise ValueError("Time is not sorted in increasing order")


def check_counts(df: pd.DataFrame) -> None:
    """Check that the full dataset was downloaded (not a partial file)."""
    n_rows, n_frauds = len(df), int(df[TARGET].sum())
    if n_rows != EXPECTED_ROWS or n_frauds != EXPECTED_FRAUDS:
        raise ValueError(
            f"got {n_rows} rows / {n_frauds} frauds, "
            f"expected {EXPECTED_ROWS} / {EXPECTED_FRAUDS}"
        )


def load_dataset(path: str | Path, check_full: bool = True) -> pd.DataFrame:
    """Read the CSV, validate it and return it with Class as int64.

    I first stored Class as int8 to save memory. It overflowed silently as soon
    as a script computed 100 * frauds per hour (600 does not fit in 8 bits), so
    the label now keeps a full-width integer type.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found, run `python scripts/download_data.py` first"
        )
    df = pd.read_csv(path)
    validate_schema(df)
    if check_full:
        check_counts(df)
    df[TARGET] = df[TARGET].astype("int64")
    return df
