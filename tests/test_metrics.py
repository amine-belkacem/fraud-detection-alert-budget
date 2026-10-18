"""Tests for the metrics. Every expected value is worked out by hand in the comments."""

import numpy as np
import pytest

from fraudbudget.metrics import (
    alerts_for_budget,
    business_cost,
    confusion,
    pr_auc,
    recall_at_budget,
    recall_at_fpr,
)

# Shared toy case: 10 transactions, 3 frauds, sorted by decreasing score.
#   score : .95  .90  .85  .80  .30  .20  .10  .05  .04  .01
#   label :  F    F    -    F    -    -    -    -    -    -
Y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
S = np.array([.95, .90, .85, .80, .30, .20, .10, .05, .04, .01])


def close(got, expected, name, tol=1e-12):
    assert abs(got - expected) < tol, f"ECHEC {name} : obtenu {got}, attendu {expected}"


def test_confusion_and_accuracy_trap():
    # "Never flag anything" on 998 legit + 2 frauds:
    # TP=0, FP=0, FN=2, TN=998 -> accuracy = 998/1000 = 0.998, recall = 0.
    y = np.r_[np.ones(2), np.zeros(998)]
    c = confusion(y, np.zeros(1000))
    assert (c.tp, c.fp, c.fn, c.tn) == (0, 0, 2, 998), f"ECHEC confusion : obtenu {c}"
    close(c.accuracy, 0.998, "accuracy")
    close(c.recall, 0.0, "recall")
    close(c.precision, 0.0, "precision (rien de signalé)")


def test_pr_auc_hand_computed():
    # Ranking: F, -, F, -  (2 frauds).
    # AP = sum over each fraud of (recall step) x (precision at that rank)
    #    = 1/2 x 1/1  +  1/2 x 2/3  = 0.5 + 0.3333... = 0.8333...
    close(pr_auc([1, 0, 1, 0], [.9, .8, .7, .1]), 5 / 6, "PR-AUC")


def test_pr_auc_perfect_ranking_is_one():
    # Every fraud scored above every legit row -> precision 1 at each fraud -> AP = 1.
    close(pr_auc(Y, Y.astype(float)), 1.0, "PR-AUC classement parfait")


def test_recall_at_fpr_hand_computed():
    # 7 legit transactions. max_fpr = 0.15 -> at most 0.15 x 7 = 1.05, so 1 FP.
    # Going down the list: F, F, - (FP=1), F (TP=3, FP=1) -> recall 3/3,
    # FPR 1/7. The next row (0.30) would be the 2nd FP -> stop at threshold 0.80.
    p = recall_at_fpr(Y, S, max_fpr=0.15)
    close(p.recall, 1.0, "rappel @ FPR 15 %")
    close(p.fpr, 1 / 7, "FPR atteint")
    close(p.threshold, 0.80, "seuil")
    assert p.n_flagged == 4, f"ECHEC alertes : obtenu {p.n_flagged}, attendu 4"


def test_recall_at_fpr_no_false_positive_allowed():
    # max_fpr = 0.10 -> 0.10 x 7 = 0.7, so 0 FP allowed.
    # Only the first two rows (both frauds) fit -> recall 2/3, threshold 0.90.
    p = recall_at_fpr(Y, S, max_fpr=0.10)
    close(p.recall, 2 / 3, "rappel @ FPR 10 %")
    close(p.threshold, 0.90, "seuil")


def test_recall_at_fpr_keeps_ties_together():
    # The top score 0.5 is shared by a fraud and a legit row. With max_fpr = 0
    # flagging the fraud would also flag the legit one -> nothing can be flagged.
    p = recall_at_fpr([1, 0, 1], [.5, .5, .2], max_fpr=0.0)
    close(p.recall, 0.0, "rappel avec ex-aequo")
    assert p.threshold == np.inf, f"ECHEC seuil : obtenu {p.threshold}, attendu inf"


def test_alerts_for_budget_hand_computed():
    # 56,962 transactions at 100 alerts per 100k -> 56,962 x 100 / 100,000 = 56.96 -> 56.
    got = alerts_for_budget(56_962, 100)
    assert got == 56, f"ECHEC capacité : obtenu {got}, attendu 56"


def test_recall_at_budget_hand_computed():
    # 10 transactions at 20,000 alerts per 100k -> capacity 2.
    # Top 2 rows are both frauds -> recall 2/3.
    # With capacity 4 (40,000 per 100k): F, F, -, F -> recall 3/3, FPR 1/7.
    p2 = recall_at_budget(Y, S, alerts_per_100k=20_000)
    close(p2.recall, 2 / 3, "rappel @ 2 alertes")
    p4 = recall_at_budget(Y, S, alerts_per_100k=40_000)
    close(p4.recall, 1.0, "rappel @ 4 alertes")
    close(p4.fpr, 1 / 7, "FPR @ 4 alertes")


def test_recall_at_budget_never_exceeds_capacity_with_ties():
    # Capacity 2 but the top three scores are equal -> cannot take them all,
    # cannot split them -> 0 alerts.
    p = recall_at_budget([1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [.9, .9, .9, .1, .1, .1, .1, .1, .1, .1], alerts_per_100k=20_000)
    assert p.n_flagged == 0, f"ECHEC alertes : obtenu {p.n_flagged}, attendu 0"


def test_business_cost_hand_computed():
    # Frauds of 10, 20, 70 and 500 EUR; only the 500 EUR one is flagged.
    # Five alerts in total (1 fraud + 4 legit) at 5 EUR each.
    # Missed: 10 + 20 + 70 = 100 EUR. Review: 5 x 5 = 25 EUR. Total 125 EUR.
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    amount = [10, 20, 70, 500, 1, 2, 3, 4]
    flagged = [0, 0, 0, 1, 1, 1, 1, 1]
    c = business_cost(y, flagged, amount, cost_per_alert=5.0)
    close(c.missed_fraud_eur, 100.0, "fraude manquée")
    close(c.review_eur, 25.0, "coût de revue")
    close(c.total_eur, 125.0, "coût total")


@pytest.mark.parametrize("y, s", [([0, 0, 0], [.1, .2, .3]), ([1, 0], [.1]), ([1, 0], [np.nan, .2])])
def test_invalid_inputs_are_rejected(y, s):
    with pytest.raises(ValueError):
        pr_auc(y, s)
