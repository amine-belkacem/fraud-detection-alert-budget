"""Tests for the bootstrap helpers."""

import numpy as np

from fraudbudget.bootstrap import (
    bootstrap_metric,
    paired_differences,
    percentile_interval,
    stratified_indices,
    summarise,
)
from fraudbudget.metrics import pr_auc

RNG = np.random.default_rng(0)
Y = np.r_[np.ones(15, dtype=int), np.zeros(985, dtype=int)]
NOISY = Y * 0.5 + RNG.uniform(0, 1, Y.size)  # frauds tend to score higher


def test_stratified_indices_keep_class_counts():
    # 15 frauds and 985 legit in every replicate, by construction.
    idx = stratified_indices(Y, n_boot=50, seed=1)
    assert idx.shape == (50, 1000), f"ECHEC forme : obtenu {idx.shape}, attendu (50, 1000)"
    frauds = Y[idx].sum(axis=1)
    assert np.all(frauds == 15), f"ECHEC fraudes par tirage : obtenu {np.unique(frauds)}, attendu 15"


def test_percentile_interval_hand_computed():
    # Values 1..1000, 95 % interval with linear interpolation (numpy default):
    # 2.5th percentile at position 0.025 x 999 = 24.975 -> 25 + 0.975 = 25.975
    # 97.5th percentile at position 0.975 x 999 = 974.025 -> 975 + 0.025 = 975.025
    low, high = percentile_interval(np.arange(1, 1001))
    for name, got, expected in [("borne basse", low, 25.975), ("borne haute", high, 975.025)]:
        assert abs(got - expected) < 1e-9, f"ECHEC {name} : obtenu {got}, attendu {expected}"


def test_perfect_ranking_has_a_degenerate_interval():
    # Frauds always above legit rows -> PR-AUC = 1 on every replicate.
    scores = {"perfect": Y.astype(float)}
    reps = bootstrap_metric(Y, scores, pr_auc, n_boot=30, seed=0)
    table = summarise(Y, scores, reps, pr_auc)
    got = tuple(table.loc["perfect", ["value", "ci_low", "ci_high"]])
    assert got == (1.0, 1.0, 1.0), f"ECHEC : obtenu {got}, attendu (1.0, 1.0, 1.0)"


def test_identical_models_have_zero_paired_difference():
    # Same scores on the same replicates -> difference exactly 0 everywhere.
    reps = bootstrap_metric(Y, {"a": NOISY, "b": NOISY.copy()}, pr_auc, n_boot=30, seed=0)
    d = paired_differences(reps, [("a", "b")]).iloc[0]
    got = (d.ci_low, d.ci_high, d.share_a_better)
    assert got == (0.0, 0.0, 0.0), f"ECHEC : obtenu {got}, attendu (0.0, 0.0, 0.0)"


def test_better_model_wins_most_replicates():
    # "good" separates the classes by +3 instead of +0.5: it should win nearly always.
    good = Y * 3.0 + RNG.uniform(0, 1, Y.size)
    reps = bootstrap_metric(Y, {"good": good, "noisy": NOISY}, pr_auc, n_boot=100, seed=0)
    d = paired_differences(reps, [("good", "noisy")]).iloc[0]
    assert d.ci_low > 0, f"ECHEC : borne basse obtenue {d.ci_low}, attendu > 0"


def test_same_seed_same_replicates():
    a = bootstrap_metric(Y, {"m": NOISY}, pr_auc, n_boot=20, seed=7)["m"]
    b = bootstrap_metric(Y, {"m": NOISY}, pr_auc, n_boot=20, seed=7)["m"]
    assert np.array_equal(a, b), "ECHEC : deux tirages avec la même graine diffèrent"
