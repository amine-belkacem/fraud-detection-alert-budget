"""Tests for the prevalence stress test helpers. Expected values worked out by hand."""

import numpy as np
import pytest

from fraudbudget.stress import expected_precision, n_frauds_for_prevalence, subsample_frauds


@pytest.mark.parametrize("n_legit, prevalence, expected", [
    # 0.001 x 56,887 / 0.999 = 56.94 -> 57
    (56_887, 0.001, 57),
    # 0.0005 x 56,887 / 0.9995 = 28.46 -> 28
    (56_887, 0.0005, 28),
    # 0.0002 x 56,887 / 0.9998 = 11.38 -> 11
    (56_887, 0.0002, 11),
    # 0.5 x 10 / 0.5 = 10: half and half
    (10, 0.5, 10),
])
def test_n_frauds_for_prevalence_hand_computed(n_legit, prevalence, expected):
    got = n_frauds_for_prevalence(n_legit, prevalence)
    assert got == expected, f"ECHEC n fraudes @ {prevalence} : obtenu {got}, attendu {expected}"


def test_subsample_keeps_every_legit_row():
    y = np.r_[np.ones(20, dtype=int), np.zeros(80, dtype=int)]
    idx = subsample_frauds(y, 5, np.random.default_rng(0))
    got = (int((y[idx] == 0).sum()), int((y[idx] == 1).sum()), len(np.unique(idx)))
    # 80 legit kept, 5 frauds, no index drawn twice -> 85 distinct rows.
    expected = (80, 5, 85)
    assert got == expected, f"ECHEC (légit, fraudes, distincts) : obtenu {got}, attendu {expected}"


def test_subsample_rejects_more_frauds_than_available():
    with pytest.raises(ValueError, match="between 1 and 2"):
        subsample_frauds([1, 1, 0], 3, np.random.default_rng(0))


def test_expected_precision_hand_computed():
    # Recall 0.8 on 75 frauds -> 60 TP; with 46 FP: 60 / 106 = 0.5660...
    # Same threshold, 11 frauds -> 8.8 TP expected; 8.8 / (8.8 + 46) = 0.1606...
    for n_frauds, expected in [(75, 60 / 106), (11, 8.8 / 54.8)]:
        got = expected_precision(0.8, n_frauds, 46)
        assert abs(got - expected) < 1e-12, (
            f"ECHEC précision @ {n_frauds} fraudes : obtenu {got}, attendu {expected}"
        )
