"""Tests for the model pipelines and the evaluation table."""

import numpy as np
import pandas as pd
import pytest

from fraudbudget.evaluate import evaluate_scores
from fraudbudget.models import FEATURES, MODEL_NAMES, make_model


def synthetic_xy(n=2000, n_fraud=20, seed=0):
    # Frauds are shifted on V1 so that a linear model can separate them.
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    X["Amount"] = rng.exponential(50, n)
    y = np.zeros(n, dtype=int)
    y[:n_fraud] = 1
    X.loc[y == 1, "V1"] += 4
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_scores(name):
    X, y = synthetic_xy()
    proba = make_model(name).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(y),), f"ECHEC forme : obtenu {proba.shape}, attendu {(len(y),)}"
    assert np.all((proba >= 0) & (proba <= 1)), "ECHEC : probabilités hors [0, 1]"


def test_dummy_scores_are_the_training_prevalence():
    # strategy="prior" returns the class frequency seen in training: 20 / 2000 = 0.01.
    X, y = synthetic_xy()
    proba = make_model("dummy").fit(X, y).predict_proba(X)[:, 1]
    assert np.allclose(proba, 0.01), f"ECHEC : obtenu {np.unique(proba)}, attendu 0.01"


def test_logreg_separates_the_shifted_class():
    # Sanity check, not a performance claim: on this easy synthetic case the
    # ranking must be far better than random (PR-AUC of random = 0.01).
    X, y = synthetic_xy()
    proba = make_model("logreg_balanced").fit(X, y).predict_proba(X)[:, 1]
    got = evaluate_scores(y, proba, X["Amount"])["pr_auc"]
    assert got > 0.5, f"ECHEC PR-AUC : obtenu {got}, attendu > 0.5"


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("random_forest_typo")


def test_evaluate_scores_on_flag_nothing():
    # 4 legit + 1 fraud of 80 EUR, all scores 0.1 (below 0.5): nothing flagged.
    # accuracy = 4/5 = 0.8, recall = 0, cost = 80 EUR missed + 0 review = 80 EUR.
    # Budget of 100 per 100k on 5 rows -> capacity 0 -> ceiling 0.
    r = evaluate_scores([0, 0, 0, 0, 1], [.1] * 5, [5, 5, 5, 5, 80])
    for key, expected in [("accuracy@0.5", 0.8), ("recall@0.5", 0.0),
                          ("cost_eur@0.5", 80.0), ("budget_recall_ceiling", 0.0)]:
        assert abs(r[key] - expected) < 1e-12, f"ECHEC {key} : obtenu {r[key]}, attendu {expected}"
