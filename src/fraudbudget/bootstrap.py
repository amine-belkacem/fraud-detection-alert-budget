"""Bootstrap confidence intervals on the test slice.

With 75 frauds in the test slice, one fraud more or less caught moves recall by
1.3 points, so a single number per model says little. Each replicate redraws the
test rows with replacement, frauds and legit transactions separately
(stratified), so every replicate keeps exactly 75 frauds. All models are
evaluated on the same replicates: the difference between two models is then
measured pairwise, which is much tighter than comparing two separate intervals.

Limit: the bootstrap captures the uncertainty due to *which* transactions ended
up in the test slice. It says nothing about the drift to a third day, and
stratifying slightly understates the variance compared to plain resampling.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

Metric = Callable[[np.ndarray, np.ndarray], float]


def stratified_indices(y, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) array of row indices, each row keeping the class counts of y."""
    y = np.asarray(y).astype(int)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    return np.hstack([
        rng.choice(pos, size=(n_boot, pos.size), replace=True),
        rng.choice(neg, size=(n_boot, neg.size), replace=True),
    ])


def percentile_interval(values, level: float = 0.95) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    low, high = np.percentile(np.asarray(values, dtype=float), [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_metric(
    y, scores: dict[str, np.ndarray], metric: Metric, n_boot: int = 1000, seed: int = 0,
) -> dict[str, np.ndarray]:
    """Metric value on every replicate, for every model (same replicates for all)."""
    y = np.asarray(y).astype(int)
    idx = stratified_indices(y, n_boot, seed)
    return {
        name: np.array([metric(y[rows], s[rows]) for rows in idx])
        for name, s in scores.items()
    }


def summarise(
    y, scores: dict[str, np.ndarray], replicates: dict[str, np.ndarray], metric: Metric,
    level: float = 0.95,
) -> pd.DataFrame:
    rows = []
    for name, s in scores.items():
        low, high = percentile_interval(replicates[name], level)
        rows.append({"model": name, "value": metric(np.asarray(y), s), "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows).set_index("model")


def paired_differences(
    replicates: dict[str, np.ndarray], pairs: Iterable[tuple[str, str]], level: float = 0.95,
) -> pd.DataFrame:
    """For each (a, b): interval of metric(a) - metric(b) over the shared replicates.

    share_a_better is the fraction of replicates where a beats b. An interval
    that contains 0 means the test slice cannot separate the two models.
    """
    rows = []
    for a, b in pairs:
        diff = replicates[a] - replicates[b]
        low, high = percentile_interval(diff, level)
        rows.append({
            "a": a, "b": b, "mean_diff": float(diff.mean()), "ci_low": low, "ci_high": high,
            "share_a_better": float(np.mean(diff > 0)),
        })
    return pd.DataFrame(rows)
