"""Choosing the decision threshold on validation, then freezing it for test.

Three rules, matching three ways a fraud team can state its constraint:
  - "at most X % of honest customers bothered"      -> threshold_for_fpr
  - "the analysts can review N alerts per 100k"      -> threshold_for_budget
  - "lose as little money as possible"               -> threshold_for_cost

Each returns a score threshold. Applied to test, it does not guarantee the same
FPR or alert count there: the score distribution moves between days. Measuring
that gap is the point of choosing on validation instead of on test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fraudbudget.metrics import recall_at_budget, recall_at_fpr


def threshold_for_fpr(y_val, s_val, max_fpr: float) -> float:
    return recall_at_fpr(y_val, s_val, max_fpr).threshold


def threshold_for_budget(y_val, s_val, alerts_per_100k: float) -> float:
    return recall_at_budget(y_val, s_val, alerts_per_100k).threshold


@dataclass(frozen=True)
class CostCurve:
    """Cost of flagging every row with score >= thresholds[i].

    The first point is threshold = +inf (flag nothing). Thresholds then go down
    through every distinct score, the last one flagging every row.
    """
    thresholds: np.ndarray
    n_alerts: np.ndarray
    missed_eur: np.ndarray
    review_eur: np.ndarray

    @property
    def total_eur(self) -> np.ndarray:
        return self.missed_eur + self.review_eur


def cost_curve(y, scores, amount, cost_per_alert: float) -> CostCurve:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(amount, dtype=float)
    if not y.shape == s.shape == a.shape:
        raise ValueError(f"length mismatch: y={y.size}, scores={s.size}, amount={a.size}")
    if cost_per_alert < 0:
        raise ValueError(f"cost_per_alert must be >= 0, got {cost_per_alert}")

    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    caught_eur = np.cumsum((a * y)[order])
    # Last row of every group of equal scores: ties are flagged together.
    ends = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    total_fraud_eur = float((a * y).sum())

    n_alerts = np.r_[0, ends + 1]
    return CostCurve(
        thresholds=np.r_[np.inf, s_sorted[ends]],
        n_alerts=n_alerts,
        missed_eur=total_fraud_eur - np.r_[0.0, caught_eur[ends]],
        review_eur=cost_per_alert * n_alerts,
    )


def threshold_for_cost(y_val, s_val, amount_val, cost_per_alert: float) -> float:
    """Threshold with the lowest simulated cost on validation (inf = flag nothing).

    On a tie the higher threshold wins: same cost, fewer customers bothered.
    """
    curve = cost_curve(y_val, s_val, amount_val, cost_per_alert)
    return float(curve.thresholds[int(np.argmin(curve.total_eur))])
