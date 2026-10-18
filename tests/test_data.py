"""Tests for the dataset checks. They run on a small synthetic frame, so the CI
never needs the real 150 MB CSV; the last test uses it only if it is present."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudbudget.data import (
    COLUMNS,
    EXPECTED_FRAUDS,
    EXPECTED_ROWS,
    check_counts,
    load_dataset,
    validate_schema,
)

REAL_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard.csv"


def synthetic_frame(n_rows: int = 50, n_frauds: int = 2, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n_rows, len(COLUMNS))), columns=COLUMNS)
    df["Time"] = np.arange(n_rows, dtype=float)
    df["Amount"] = rng.uniform(0, 500, n_rows)
    df["Class"] = 0
    df.loc[: n_frauds - 1, "Class"] = 1
    return df


def expect_rejected(df: pd.DataFrame, fragment: str) -> None:
    try:
        validate_schema(df)
    except ValueError as err:
        assert fragment in str(err), (
            f"ECHEC : message obtenu '{err}', attendu un message contenant '{fragment}'"
        )
    else:
        pytest.fail(f"ECHEC : frame acceptée, attendu un rejet ('{fragment}')")


def test_valid_frame_is_accepted():
    validate_schema(synthetic_frame())


def test_missing_column_is_rejected():
    expect_rejected(synthetic_frame().drop(columns="V7"), "missing=['V7']")


def test_missing_value_is_rejected():
    df = synthetic_frame()
    df.loc[3, "V2"] = np.nan
    expect_rejected(df, "1 missing values")


def test_label_outside_0_1_is_rejected():
    df = synthetic_frame()
    df.loc[5, "Class"] = 2
    expect_rejected(df, "Class must be 0 or 1")


def test_negative_amount_is_rejected():
    df = synthetic_frame()
    df.loc[0, "Amount"] = -1.0
    expect_rejected(df, "negative transaction amount")


def test_unsorted_time_is_rejected():
    df = synthetic_frame()
    df.loc[[10, 11], "Time"] = [11.0, 10.0]
    expect_rejected(df, "Time is not sorted")


def test_check_counts_rejects_partial_file():
    # A 50-row synthetic frame is obviously not the full dataset.
    with pytest.raises(ValueError, match=f"expected {EXPECTED_ROWS} / {EXPECTED_FRAUDS}"):
        check_counts(synthetic_frame())


def test_load_dataset_on_small_csv(tmp_path):
    path = tmp_path / "small.csv"
    synthetic_frame(n_rows=20, n_frauds=3).to_csv(path, index=False)
    df = load_dataset(path, check_full=False)
    got = (len(df), int(df["Class"].sum()), str(df["Class"].dtype))
    expected = (20, 3, "int8")
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


@pytest.mark.skipif(not REAL_CSV.exists(), reason="real dataset not downloaded")
def test_real_dataset_counts():
    # 492 / 284 807 = 0.1727 %, the prevalence quoted everywhere for this dataset.
    df = load_dataset(REAL_CSV)
    prevalence = df["Class"].mean()
    expected = 492 / 284_807
    assert abs(prevalence - expected) < 1e-12, (
        f"ECHEC prévalence : obtenu {prevalence}, attendu {expected}"
    )
