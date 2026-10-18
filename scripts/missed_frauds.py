"""J2: do the supervised models catch the same frauds, or different ones?

Usage:
    python scripts/missed_frauds.py

If the models missed different frauds, combining them would help. If they all
miss the same ones, the limit is in the features, not in the choice of model.
Uses seed 0 and the 200-alerts-per-100k budget. Writes results/missed_frauds.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fraudbudget import config  # noqa: E402
from fraudbudget.data import AMOUNT, TARGET, TIME, load_dataset  # noqa: E402
from fraudbudget.metrics import recall_at_budget  # noqa: E402
from fraudbudget.models import FEATURES, fraud_scores, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402
from run_models import model_kwargs  # noqa: E402

MODELS = ("logreg_balanced", "xgboost", "xgboost_smote", "xgboost_undersample")


def main() -> int:
    best = json.loads((ROOT / "results" / "best_params.json").read_text())
    split = temporal_split(load_dataset(ROOT / "data" / "raw" / "creditcard.csv"))
    test = split.test
    y = test[TARGET].to_numpy()

    caught = pd.DataFrame(index=test.index[y == 1])
    for name in MODELS:
        model = make_model(name, seed=config.SEED, **model_kwargs(name, best))
        model.fit(split.train[FEATURES], split.train[TARGET])
        s = fraud_scores(model, test[FEATURES])
        point = recall_at_budget(y, s, config.ALERTS_PER_100K)
        caught[name] = (s >= point.threshold)[y == 1]

    caught["n_models"] = caught[list(MODELS)].sum(axis=1)
    caught[AMOUNT] = test.loc[caught.index, AMOUNT]
    caught["hour"] = test.loc[caught.index, TIME] / 3600
    caught.to_csv(ROOT / "results" / "missed_frauds.csv", float_format="%.2f")

    all_four = caught["n_models"] == len(MODELS)
    none = caught["n_models"] == 0
    print(f"{len(caught)} frauds in test: caught by all {len(MODELS)} models = {all_four.sum()}, "
          f"by none = {none.sum()}, by some but not all = {(~all_four & ~none).sum()}")
    print(f"amount missed by every model: {caught.loc[none, AMOUNT].sum():.2f} EUR "
          f"(median {caught.loc[none, AMOUNT].median():.2f}, "
          f"vs {caught.loc[all_four, AMOUNT].median():.2f} for frauds all models catch)")
    print("amounts missed by all:", np.sort(caught.loc[none, AMOUNT].to_numpy()).round(2).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
