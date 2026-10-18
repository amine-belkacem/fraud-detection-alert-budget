"""Model pipelines. Every preprocessing step lives inside the pipeline, so it is
fitted on the training set only and simply applied to validation and test."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraudbudget.data import AMOUNT, PCA_FEATURES

# Time is left out on purpose: "seconds since the first transaction" over two
# days cannot generalise to a third day, and the test set is the third slice.
FEATURES = [*PCA_FEATURES, AMOUNT]


def _preprocessing() -> ColumnTransformer:
    # Amounts range from 0 to about 25,000 EUR with a long tail; log1p brings
    # them to a scale comparable with the PCA components before standardising.
    amount = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("pca", StandardScaler(), PCA_FEATURES),
        ("amount", amount, [AMOUNT]),
    ])


def make_model(name: str, seed: int = 0) -> Pipeline:
    if name == "dummy":
        # Always predicts the majority class: the "99.8 % accurate" model.
        clf = DummyClassifier(strategy="prior")
    elif name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
    elif name == "logreg_balanced":
        # Each fraud weighs n_legit / n_fraud times more in the loss (about 470x here).
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"unknown model '{name}'")
    return Pipeline([("prep", _preprocessing()), ("clf", clf)])


MODEL_NAMES = ("dummy", "logreg", "logreg_balanced")
