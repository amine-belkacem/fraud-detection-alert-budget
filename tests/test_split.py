"""Tests for the temporal split."""

import numpy as np
import pandas as pd
import pytest

from fraudbudget.split import temporal_split


def frame_with_times(times):
    return pd.DataFrame({"Time": np.asarray(times, dtype=float), "Class": 0})


def test_cuts_move_past_shared_timestamps_hand_computed():
    # 10 rows, fractions 60/20/20.
    # First cut: int(10 * 0.6) = 6. Row 5 has Time 3 and so does row 6,
    #   so the cut moves to the first row after the last Time 3 -> position 7.
    # Second cut: int(10 * 0.8) = 8. Row 7 has Time 4, row 8 has Time 5,
    #   no shared second -> stays at 8.
    # Sizes: train 7, val 1, test 2.
    s = temporal_split(frame_with_times([0, 1, 1, 2, 3, 3, 3, 4, 5, 6]))
    got = (len(s.train), len(s.val), len(s.test))
    expected = (7, 1, 2)
    assert got == expected, f"ECHEC tailles : obtenu {got}, attendu {expected}"


def test_no_cut_needed_gives_exact_fractions():
    # Distinct timestamps: 100 rows -> 60 / 20 / 20 exactly.
    s = temporal_split(frame_with_times(range(100)))
    got = (len(s.train), len(s.val), len(s.test))
    assert got == (60, 20, 20), f"ECHEC tailles : obtenu {got}, attendu (60, 20, 20)"


def test_sets_are_ordered_disjoint_and_complete():
    rng = np.random.default_rng(1)
    times = np.sort(rng.integers(0, 500, size=1000))  # lots of shared seconds
    s = temporal_split(frame_with_times(times))

    assert s.train["Time"].max() < s.val["Time"].min(), (
        f"ECHEC : train finit à {s.train['Time'].max()}, val commence à {s.val['Time'].min()}"
    )
    assert s.val["Time"].max() < s.test["Time"].min(), (
        f"ECHEC : val finit à {s.val['Time'].max()}, test commence à {s.test['Time'].min()}"
    )
    total = len(s.train) + len(s.val) + len(s.test)
    assert total == 1000, f"ECHEC : {total} lignes au total, attendu 1000"
    shared = set(s.train.index) & (set(s.val.index) | set(s.test.index))
    assert not shared, f"ECHEC : {len(shared)} lignes dans deux ensembles"


def test_unsorted_input_is_rejected():
    with pytest.raises(ValueError, match="sorted by Time"):
        temporal_split(frame_with_times([0, 2, 1, 3, 4]))


@pytest.mark.parametrize("fractions", [(0.6, 0.2), (0.6, 0.3, 0.2), (0.8, 0.2, 0.0)])
def test_bad_fractions_are_rejected(fractions):
    with pytest.raises(ValueError):
        temporal_split(frame_with_times(range(10)), fractions)
