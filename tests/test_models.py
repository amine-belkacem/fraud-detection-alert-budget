"""Tests for the model pipelines and the evaluation table."""

import numpy as np
import pandas as pd
import pytest

from fraudbudget.evaluate import evaluate_scores
from fraudbudget.models import FEATURES, MODEL_NAMES, fraud_scores, make_model


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
    scores = fraud_scores(make_model(name).fit(X, y), X)
    assert scores.shape == (len(y),), f"ECHEC forme : obtenu {scores.shape}, attendu {(len(y),)}"
    assert np.isfinite(scores).all(), "ECHEC : scores non finis"
    if name != "isolation_forest":
        assert np.all((scores >= 0) & (scores <= 1)), "ECHEC : probabilités hors [0, 1]"


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
    # Budgets of 200 and 100 per 100k on 5 rows -> capacity 0 -> ceilings 0.
    r = evaluate_scores([0, 0, 0, 0, 1], [.1] * 5, [5, 5, 5, 5, 80])
    for key, expected in [("accuracy@0.5", 0.8), ("recall@0.5", 0.0),
                          ("cost_eur@0.5", 80.0), ("budget_recall_ceiling", 0.0),
                          ("tight_budget_recall_ceiling", 0.0)]:
        assert abs(r[key] - expected) < 1e-12, f"ECHEC {key} : obtenu {r[key]}, attendu {expected}"


@pytest.mark.parametrize("name, expected_frauds", [
    # 1980 legit, 20 frauds, ratio 0.1:
    # SMOTE adds frauds until frauds = 0.1 x 1980 = 198 (legit unchanged).
    # Undersampling removes legit until legit = 20 / 0.1 = 200 (frauds unchanged).
    ("xgboost_smote", (1980, 198)),
    ("xgboost_undersample", (200, 20)),
])
def test_resampling_happens_on_training_data(name, expected_frauds):
    X, y = synthetic_xy()
    pipe = make_model(name, resample_ratio=0.1)
    Xt = pipe[:-2].fit_transform(X)            # preprocessing only
    _, y_res = pipe[-2].fit_resample(Xt, y)    # the sampler step
    got = (int((y_res == 0).sum()), int((y_res == 1).sum()))
    assert got == expected_frauds, f"ECHEC (légit, fraudes) : obtenu {got}, attendu {expected_frauds}"


@pytest.mark.parametrize("name", ["xgboost_smote", "xgboost_undersample"])
def test_resampling_never_touches_scored_rows(name):
    # Samplers in an imblearn Pipeline run in fit() only: scoring 500 rows must
    # return 500 scores, not a resampled count.
    X, y = synthetic_xy()
    pipe = make_model(name).fit(X, y)
    scores = fraud_scores(pipe, X.iloc[:500])
    assert scores.shape == (500,), f"ECHEC : obtenu {scores.shape}, attendu (500,)"


def test_isolation_forest_flags_the_shifted_points():
    # Frauds are shifted by +4 on V1, so they are outliers even without labels.
    # Their mean anomaly score must exceed the legit one.
    X, y = synthetic_xy()
    s = fraud_scores(make_model("isolation_forest").fit(X), X)
    assert s[y == 1].mean() > s[y == 0].mean(), (
        f"ECHEC : score moyen fraudes {s[y == 1].mean():.3f} <= légit {s[y == 0].mean():.3f}"
    )


def test_threshold_metrics_are_blank_for_unsupervised_scores():
    r = evaluate_scores([0, 0, 0, 0, 1], [.1, .2, .3, .4, .9], [5] * 5, probabilistic=False)
    assert np.isnan(r["accuracy@0.5"]), f"ECHEC : obtenu {r['accuracy@0.5']}, attendu NaN"
    assert abs(r["pr_auc"] - 1.0) < 1e-12, f"ECHEC PR-AUC : obtenu {r['pr_auc']}, attendu 1.0"
