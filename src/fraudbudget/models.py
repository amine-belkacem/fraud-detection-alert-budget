"""Model pipelines. Every preprocessing step lives inside the pipeline, so it is
fitted on the training set only and simply applied to validation and test.

Resampling (SMOTE, undersampling) uses imbalanced-learn's Pipeline: samplers in
it only run during fit(), so validation and test rows are never resampled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from fraudbudget.data import AMOUNT, PCA_FEATURES

# Time is left out on purpose: "seconds since the first transaction" over two
# days cannot generalise to a third day, and the test set is the third slice.
FEATURES = [*PCA_FEATURES, AMOUNT]

MODEL_NAMES = (
    "dummy", "logreg", "logreg_balanced",
    "isolation_forest", "xgboost", "xgboost_smote", "xgboost_undersample",
)

# Models whose output is not a probability: only ranking metrics apply to them.
UNSUPERVISED = ("isolation_forest",)

# Starting point for XGBoost before tuning on the validation set.
XGB_DEFAULTS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
# Fraud / legit ratio after resampling (imblearn's sampling_strategy).
RESAMPLE_RATIO_DEFAULT = 0.1
IFOREST_DEFAULTS: dict[str, Any] = {"n_estimators": 300, "max_samples": 256}


def _preprocessing() -> ColumnTransformer:
    # Amounts range from 0 to about 25,000 EUR with a long tail; log1p brings
    # them to a scale comparable with the PCA components before standardising.
    # Trees do not need it, but SMOTE does: it interpolates between neighbours,
    # so distances must not be dominated by the raw amount.
    amount = Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("pca", StandardScaler(), PCA_FEATURES),
        ("amount", amount, [AMOUNT]),
    ])


def _xgb(seed: int, scale_pos_weight: float, params: dict[str, Any] | None) -> XGBClassifier:
    return XGBClassifier(
        **{**XGB_DEFAULTS, **(params or {})},
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=seed,
        n_jobs=-1,
    )


def make_model(
    name: str,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    resample_ratio: float = RESAMPLE_RATIO_DEFAULT,
    pos_weight: float = 1.0,
):
    """Build an unfitted pipeline.

    pos_weight is XGBoost's scale_pos_weight for the plain "xgboost" model; the
    caller passes n_legit / n_fraud of its training set. The resampled variants
    keep it at 1 because the resampling already rebalances the classes.
    """
    if name == "dummy":
        # Always predicts the majority class: the "99.8 % accurate" model.
        return Pipeline([("prep", _preprocessing()), ("clf", DummyClassifier(strategy="prior"))])
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        return Pipeline([("prep", _preprocessing()), ("clf", clf)])
    if name == "logreg_balanced":
        # Each fraud weighs n_legit / n_fraud times more in the loss (about 470x here).
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        return Pipeline([("prep", _preprocessing()), ("clf", clf)])
    if name == "isolation_forest":
        # Trained without labels: it only learns what "usual" looks like.
        clf = IsolationForest(**{**IFOREST_DEFAULTS, **(params or {})}, random_state=seed, n_jobs=-1)
        return Pipeline([("prep", _preprocessing()), ("clf", clf)])
    if name == "xgboost":
        return Pipeline([("prep", _preprocessing()), ("clf", _xgb(seed, pos_weight, params))])
    if name == "xgboost_smote":
        sampler = SMOTE(sampling_strategy=resample_ratio, k_neighbors=5, random_state=seed)
        return ImbPipeline([("prep", _preprocessing()), ("resample", sampler),
                            ("clf", _xgb(seed, 1.0, params))])
    if name == "xgboost_undersample":
        sampler = RandomUnderSampler(sampling_strategy=resample_ratio, random_state=seed)
        return ImbPipeline([("prep", _preprocessing()), ("resample", sampler),
                            ("clf", _xgb(seed, 1.0, params))])
    raise ValueError(f"unknown model '{name}'")


def fraud_scores(model, X) -> np.ndarray:
    """Higher = more suspicious, whatever the model type."""
    if isinstance(model[-1], IsolationForest):
        # score_samples is high for normal points, so flip the sign.
        return -model.score_samples(X)
    return model.predict_proba(X)[:, 1]
