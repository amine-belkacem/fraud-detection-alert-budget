"""Stress test: what happens to the metrics when fraud gets rarer?

The test slice is kept whole for legitimate transactions and only a random
subset of its frauds is kept, so the model, its scores and its threshold do not
change; only the base rate does. ROC-AUC compares frauds with legit rows one
pair at a time, so it does not see the base rate. Precision does: the false
positives stay the same while the true positives shrink.
"""

from __future__ import annotations

import numpy as np


def n_frauds_for_prevalence(n_legit: int, prevalence: float) -> int:
    """Number of frauds that gives `prevalence` alongside `n_legit` legit rows.

    prevalence = k / (k + n_legit)  =>  k = prevalence * n_legit / (1 - prevalence)
    """
    if not 0 < prevalence < 1:
        raise ValueError(f"prevalence must be in (0, 1), got {prevalence}")
    return int(round(prevalence * n_legit / (1 - prevalence)))


def subsample_frauds(y, n_frauds: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of every legit row plus `n_frauds` frauds drawn without replacement."""
    y = np.asarray(y).astype(int)
    frauds = np.flatnonzero(y == 1)
    if not 1 <= n_frauds <= frauds.size:
        raise ValueError(f"n_frauds must be between 1 and {frauds.size}, got {n_frauds}")
    kept = rng.choice(frauds, size=n_frauds, replace=False)
    return np.sort(np.r_[np.flatnonzero(y == 0), kept])


def expected_precision(recall: float, n_frauds: float, false_positives: float) -> float:
    """Precision at a frozen threshold when only the number of frauds changes.

    True positives scale with the frauds (recall x n_frauds), false positives
    do not depend on them at all.
    """
    tp = recall * n_frauds
    return tp / (tp + false_positives) if tp + false_positives > 0 else 0.0
