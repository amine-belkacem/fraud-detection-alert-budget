"""J2: small hyperparameter search, scored on the validation slice only.

Usage:
    python scripts/tune.py

The validation slice holds 57 frauds, so a large search would end up fitting
those 57 rows rather than the problem. The grid is deliberately small, and the
test slice is never touched here. Writes results/tuning.csv (every trial) and
results/best_params.json (read by scripts/run_models.py).
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudbudget import config  # noqa: E402
from fraudbudget.data import TARGET, load_dataset  # noqa: E402
from fraudbudget.metrics import pr_auc  # noqa: E402
from fraudbudget.models import FEATURES, fraud_scores, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402

DATA = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS = ROOT / "results"

XGB_GRID = {"max_depth": [3, 4, 6], "n_estimators": [200, 500], "learning_rate": [0.05, 0.1]}
RESAMPLE_RATIOS = [0.01, 0.1, 0.5]
IFOREST_GRID = {"max_samples": [256, 4096], "max_features": [0.5, 1.0]}


def grid(spec: dict) -> list[dict]:
    return [dict(zip(spec, values)) for values in itertools.product(*spec.values())]


def main() -> int:
    split = temporal_split(load_dataset(DATA))
    X_tr, y_tr = split.train[FEATURES], split.train[TARGET].to_numpy()
    X_va, y_va = split.val[FEATURES], split.val[TARGET].to_numpy()
    imbalance = (y_tr == 0).sum() / (y_tr == 1).sum()
    print(f"train legit/fraud ratio = {imbalance:.1f}")

    trials = []

    def run(model_name: str, label: dict, **kwargs) -> float:
        t0 = time.perf_counter()
        model = make_model(model_name, seed=config.SEED, **kwargs).fit(X_tr, y_tr)
        score = pr_auc(y_va, fraud_scores(model, X_va))
        trials.append({"model": model_name, **label, "val_pr_auc": score,
                       "fit_s": time.perf_counter() - t0})
        print(f"{model_name:20s} {label}  val PR-AUC {score:.4f}")
        return score

    # 1. XGBoost tree parameters, with and without class weighting.
    best_xgb = (-1.0, None, None)
    for params in grid(XGB_GRID):
        for weight_name, weight in [("1", 1.0), ("ratio", imbalance)]:
            s = run("xgboost", {**params, "pos_weight": weight_name}, params=params, pos_weight=weight)
            if s > best_xgb[0]:
                best_xgb = (s, params, weight_name)
    _, xgb_params, xgb_weight = best_xgb

    # 2. Resampled variants reuse the best tree parameters; only the ratio is tuned,
    #    so any difference comes from the resampling, not from different trees.
    best_ratio = {}
    for name in ("xgboost_smote", "xgboost_undersample"):
        scores = {r: run(name, {**xgb_params, "resample_ratio": r}, params=xgb_params, resample_ratio=r)
                  for r in RESAMPLE_RATIOS}
        best_ratio[name] = max(scores, key=scores.get)

    # 3. Isolation Forest (labels are not used for fitting, only for scoring here).
    if_scores = {}
    for params in grid(IFOREST_GRID):
        if_scores[json.dumps(params)] = run("isolation_forest", params, params=params)
    if_params = json.loads(max(if_scores, key=if_scores.get))

    best = {
        "train_imbalance_ratio": imbalance,
        "xgboost": {"params": xgb_params, "pos_weight": xgb_weight},
        "xgboost_smote": {"params": xgb_params, "resample_ratio": best_ratio["xgboost_smote"]},
        "xgboost_undersample": {"params": xgb_params,
                                "resample_ratio": best_ratio["xgboost_undersample"]},
        "isolation_forest": {"params": if_params},
    }
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(trials).to_csv(RESULTS / "tuning.csv", index=False, float_format="%.5f")
    (RESULTS / "best_params.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
