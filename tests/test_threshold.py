"""Tests for threshold selection and calibration. Expected values worked out by hand."""

import numpy as np
import pytest

from fraudbudget.calibration import brier, reliability
from fraudbudget.threshold import cost_curve, threshold_for_budget, threshold_for_cost, threshold_for_fpr

# 4 transactions, sorted by score: fraud 100 EUR, legit, fraud 30 EUR, legit.
Y = np.array([1, 0, 1, 0])
S = np.array([.9, .8, .7, .1])
AMOUNT = np.array([100.0, 5.0, 30.0, 5.0])


def test_cost_curve_hand_computed():
    # Total fraud = 130 EUR, 10 EUR per alert.
    #   threshold  alerts  missed  review  total
    #   inf        0       130     0       130   (flag nothing)
    #   0.9        1       30      10      40
    #   0.8        2       30      20      50
    #   0.7        3       0       30      30
    #   0.1        4       0       40      40    (flag everything)
    c = cost_curve(Y, S, AMOUNT, cost_per_alert=10.0)
    expected = {
        "seuils": [np.inf, .9, .8, .7, .1],
        "alertes": [0, 1, 2, 3, 4],
        "coût total": [130, 40, 50, 30, 40],
    }
    got = {"seuils": c.thresholds.tolist(), "alertes": c.n_alerts.tolist(),
           "coût total": c.total_eur.tolist()}
    for key in expected:
        assert np.allclose(got[key], expected[key]), (
            f"ECHEC {key} : obtenu {got[key]}, attendu {expected[key]}"
        )


@pytest.mark.parametrize("cost_per_alert, expected", [
    (10.0, 0.7),     # table above: minimum 30 EUR at 0.7
    # 40 EUR/alert: inf 130 | 0.9: 30+40=70 | 0.8: 30+80=110 | 0.7: 120 | 0.1: 160
    (40.0, 0.9),
    # 200 EUR/alert: inf 130 | 0.9: 30+200=230 | ... -> flagging nothing is cheapest
    (200.0, np.inf),
])
def test_threshold_for_cost_hand_computed(cost_per_alert, expected):
    got = threshold_for_cost(Y, S, AMOUNT, cost_per_alert)
    assert got == expected, f"ECHEC seuil @ {cost_per_alert} EUR : obtenu {got}, attendu {expected}"


def test_cost_tie_prefers_fewer_alerts():
    # At 0 EUR per alert only missed frauds count: 130, 30, 30, 0, 0.
    # The minimum (0 EUR) is reached at 0.7 and again at 0.1; the rule must pick
    # 0.7, which bothers one customer fewer.
    got = threshold_for_cost(Y, S, AMOUNT, cost_per_alert=0.0)
    assert got == 0.7, f"ECHEC : obtenu {got}, attendu 0.7"


def test_cost_curve_keeps_ties_together():
    # Two rows with the same score: only "nothing" or "both" are possible.
    # inf: 100 EUR missed | 0.5: 2 alerts x 10 = 20 EUR
    c = cost_curve([1, 0], [.5, .5], [100.0, 1.0], cost_per_alert=10.0)
    got = (c.n_alerts.tolist(), c.total_eur.tolist())
    assert got == ([0, 2], [100.0, 20.0]), f"ECHEC : obtenu {got}, attendu ([0, 2], [100.0, 20.0])"


def test_fpr_and_budget_rules_hand_computed():
    # FPR rule, max 50 %: 2 legit rows -> at most 1 false positive.
    #   0.9 (TP) -> 0.8 (FP=1) -> 0.7 (TP, FP=1) -> 0.1 would be FP=2. Threshold 0.7.
    got = threshold_for_fpr(Y, S, max_fpr=0.5)
    assert got == 0.7, f"ECHEC seuil FPR : obtenu {got}, attendu 0.7"
    # Budget rule, 50,000 per 100k on 4 rows -> 2 alerts -> threshold 0.8.
    got = threshold_for_budget(Y, S, alerts_per_100k=50_000)
    assert got == 0.8, f"ECHEC seuil budget : obtenu {got}, attendu 0.8"


def test_brier_hand_computed():
    # ((1 - 0.8)^2 + (0 - 0.4)^2) / 2 = (0.04 + 0.16) / 2 = 0.10
    got = brier([1, 0], [0.8, 0.4])
    assert abs(got - 0.10) < 1e-12, f"ECHEC Brier : obtenu {got}, attendu 0.10"


def test_reliability_bins_hand_computed():
    # Edges (0, 0.5, 1.0):
    #   [0, 0.5)  : p = 0.1, 0.2, 0.3   labels 0, 0, 1 -> n 3, 1 fraud, mean p 0.2, rate 1/3
    #   [0.5, 1.0]: p = 0.6, 1.0        labels 1, 1    -> n 2, 2 frauds, mean p 0.8, rate 1
    bins = reliability([0, 0, 1, 1, 1], [.1, .2, .3, .6, 1.0], edges=(0.0, 0.5, 1.0))
    got = [(b.n, b.n_fraud, round(b.mean_predicted, 12), round(b.observed_rate, 12)) for b in bins]
    expected = [(3, 1, 0.2, round(1 / 3, 12)), (2, 2, 0.8, 1.0)]
    assert got == expected, f"ECHEC bins : obtenu {got}, attendu {expected}"


def test_reliability_skips_empty_bins():
    bins = reliability([0, 1], [.1, .2], edges=(0.0, 0.5, 1.0))
    assert len(bins) == 1, f"ECHEC : obtenu {len(bins)} bins, attendu 1"
