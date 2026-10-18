"""Are the scores usable as probabilities?

Class weighting and resampling both tell the model that frauds are far more
common than they are, so its outputs can overshoot. That does not hurt ranking
metrics (PR-AUC, recall at a budget) but it matters as soon as someone reads
"0.7" as "70 % chance of fraud", or multiplies it by an amount to get an
expected loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import brier_score_loss

# Probability bins spaced by decades. Quantile bins do not work here: with
# 0.13 % positives, nine bins out of ten hold no fraud at all.
DEFAULT_EDGES = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0)


def brier(y_true, proba) -> float:
    """Mean squared gap between predicted probability and outcome (lower is better)."""
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)))


@dataclass(frozen=True)
class ReliabilityBin:
    low: float
    high: float
    n: int
    n_fraud: int
    mean_predicted: float
    observed_rate: float


def reliability(y_true, proba, edges=DEFAULT_EDGES) -> list[ReliabilityBin]:
    """Observed fraud rate against mean predicted probability, per bin.

    Bins are [low, high), except the last one which includes 1.0. Empty bins
    are skipped.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y_true has {y.size} values, proba has {p.size}")
    if p.min() < 0 or p.max() > 1:
        raise ValueError("probabilities must lie in [0, 1]")

    bins = []
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        mask = (p >= low) & ((p <= high) if last else (p < high))
        if mask.any():
            bins.append(ReliabilityBin(
                low=low, high=high, n=int(mask.sum()), n_fraud=int(y[mask].sum()),
                mean_predicted=float(p[mask].mean()), observed_rate=float(y[mask].mean()),
            ))
    return bins
