"""Turn a vector of scores into the table of numbers reported in the README."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from fraudbudget import config
from fraudbudget.metrics import (
    alerts_for_budget,
    business_cost,
    confusion,
    pr_auc,
    recall_at_budget,
    recall_at_fpr,
)


def evaluate_scores(
    y_true,
    scores,
    amount,
    threshold: float = config.DEFAULT_THRESHOLD,
    max_fpr: float = config.MAX_FPR,
    alerts_per_100k: float = config.ALERTS_PER_100K,
    alerts_per_100k_tight: float = config.ALERTS_PER_100K_TIGHT,
    cost_per_alert: float = config.COST_PER_ALERT_EUR,
    probabilistic: bool = True,
) -> dict[str, float]:
    """Ranking metrics for every model; threshold metrics only when the scores
    are probabilities (an Isolation Forest score has no meaningful 0.5)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    at_threshold = confusion(y, s >= threshold)
    at_fpr = recall_at_fpr(y, s, max_fpr)
    at_budget = recall_at_budget(y, s, alerts_per_100k)
    at_tight = recall_at_budget(y, s, alerts_per_100k_tight)
    capacity = alerts_for_budget(y.size, alerts_per_100k)
    capacity_tight = alerts_for_budget(y.size, alerts_per_100k_tight)
    n_pos = int(y.sum())
    cost = business_cost(y, s >= threshold, amount, cost_per_alert)

    row = {
        "accuracy@0.5": at_threshold.accuracy,
        "precision@0.5": at_threshold.precision,
        "recall@0.5": at_threshold.recall,
        "alerts@0.5": at_threshold.tp + at_threshold.fp,
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": pr_auc(y, s),
        f"recall@fpr{max_fpr:g}": at_fpr.recall,
        f"recall@budget{alerts_per_100k:g}": at_budget.recall,
        # Even a perfect model cannot catch more frauds than there are alerts.
        "budget_recall_ceiling": min(capacity, n_pos) / n_pos,
        f"recall@budget{alerts_per_100k_tight:g}": at_tight.recall,
        "tight_budget_recall_ceiling": min(capacity_tight, n_pos) / n_pos,
        "cost_eur@0.5": cost.total_eur,
    }
    if not probabilistic:
        for key in ("accuracy@0.5", "precision@0.5", "recall@0.5", "alerts@0.5", "cost_eur@0.5"):
            row[key] = float("nan")
    return row
