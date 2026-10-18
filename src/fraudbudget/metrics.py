"""Metrics that make sense when 0.17 % of the rows are positive.

Convention everywhere: a higher score means "more likely fraud", and a
transaction is flagged when score >= threshold. Ties are never split: all rows
with the same score are flagged together, because a real system cannot flag
half of the transactions that received an identical score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

TRANSACTIONS_PER_BUDGET_UNIT = 100_000


def _as_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"y_true has {y.size} values, scores has {s.size}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if y.sum() == 0 or y.sum() == y.size:
        raise ValueError("need at least one positive and one negative")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinite values")
    return y, s


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn)

    @property
    def precision(self) -> float:
        # Nothing flagged: precision is undefined, reported as 0 rather than NaN.
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn)

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)


def confusion(y_true, flagged) -> Confusion:
    y = np.asarray(y_true).astype(bool).ravel()
    f = np.asarray(flagged).astype(bool).ravel()
    if y.shape != f.shape:
        raise ValueError(f"y_true has {y.size} values, flagged has {f.size}")
    return Confusion(
        tp=int(np.sum(y & f)), fp=int(np.sum(~y & f)),
        fn=int(np.sum(y & ~f)), tn=int(np.sum(~y & ~f)),
    )


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve, computed as average precision.

    A random ranking scores about the prevalence (0.0017 here), not 0.5.
    """
    y, s = _as_arrays(y_true, scores)
    return float(average_precision_score(y, s))


def _cumulative_counts(y: np.ndarray, s: np.ndarray):
    """TP and FP counts for every distinct threshold, highest score first."""
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # Index of the last row of each group of equal scores.
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tp = np.cumsum(y_sorted)[last_of_group]
    fp = np.cumsum(1 - y_sorted)[last_of_group]
    return s_sorted[last_of_group], tp, fp


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float  # np.inf means "flag nothing"
    recall: float
    fpr: float
    n_flagged: int


def recall_at_fpr(y_true, scores, max_fpr: float) -> OperatingPoint:
    """Best recall reachable while keeping the false positive rate <= max_fpr."""
    if not 0 <= max_fpr <= 1:
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")
    y, s = _as_arrays(y_true, scores)
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())

    thresholds, tp, fp = _cumulative_counts(y, s)
    # Compare counts, not rates, to avoid float rounding at the boundary.
    allowed = fp <= max_fpr * n_neg + 1e-9
    if not allowed.any():
        return OperatingPoint(threshold=np.inf, recall=0.0, fpr=0.0, n_flagged=0)
    # Recall only grows as the threshold goes down, so the last allowed point wins.
    i = np.flatnonzero(allowed)[-1]
    return OperatingPoint(
        threshold=float(thresholds[i]), recall=tp[i] / n_pos,
        fpr=fp[i] / n_neg, n_flagged=int(tp[i] + fp[i]),
    )


def alerts_for_budget(n_transactions: int, alerts_per_100k: float) -> int:
    """How many alerts the analysts can review for this volume of transactions."""
    if alerts_per_100k < 0:
        raise ValueError(f"budget must be >= 0, got {alerts_per_100k}")
    return int(n_transactions * alerts_per_100k // TRANSACTIONS_PER_BUDGET_UNIT)


def recall_at_budget(y_true, scores, alerts_per_100k: float) -> OperatingPoint:
    """Best recall when at most `alerts_per_100k` alerts per 100k transactions fit.

    Because ties are flagged together, the number of alerts can end up below the
    budget, never above it.
    """
    y, s = _as_arrays(y_true, scores)
    capacity = alerts_for_budget(y.size, alerts_per_100k)

    thresholds, tp, fp = _cumulative_counts(y, s)
    fits = (tp + fp) <= capacity
    if not fits.any():
        return OperatingPoint(threshold=np.inf, recall=0.0, fpr=0.0, n_flagged=0)
    i = np.flatnonzero(fits)[-1]
    n_neg = int(y.size - y.sum())
    return OperatingPoint(
        threshold=float(thresholds[i]), recall=tp[i] / int(y.sum()),
        fpr=fp[i] / n_neg, n_flagged=int(tp[i] + fp[i]),
    )


@dataclass(frozen=True)
class Cost:
    missed_fraud_eur: float  # amount of every fraud that was not flagged
    review_eur: float        # analyst time, paid for every alert (true or false)

    @property
    def total_eur(self) -> float:
        return self.missed_fraud_eur + self.review_eur


def business_cost(y_true, flagged, amount, cost_per_alert: float) -> Cost:
    """Simulated cost of a set of decisions.

    A missed fraud costs its amount. Every alert costs `cost_per_alert`, an
    assumption about analyst time, not a figure from the dataset.
    """
    if cost_per_alert < 0:
        raise ValueError(f"cost_per_alert must be >= 0, got {cost_per_alert}")
    y = np.asarray(y_true).astype(bool).ravel()
    f = np.asarray(flagged).astype(bool).ravel()
    a = np.asarray(amount, dtype=float).ravel()
    if not y.shape == f.shape == a.shape:
        raise ValueError(f"length mismatch: y={y.size}, flagged={f.size}, amount={a.size}")
    return Cost(
        missed_fraud_eur=float(a[y & ~f].sum()),
        review_eur=float(cost_per_alert * f.sum()),
    )
