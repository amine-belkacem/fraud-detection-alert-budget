"""J4: does the time order matter on this dataset?

Usage:
    python scripts/run_drift.py      # needs results/best_params.json (scripts/tune.py)

Three checks:
  1. Random vs temporal split: same model, same set sizes (60 % train, 20 %
     test), only the way rows are assigned changes. 5 random splits.
  2. Activity per hour: transaction volume and fraud rate over the 48 hours.
  3. Adversarial validation: can a classifier tell training rows from test
     rows using the features alone? ROC-AUC 0.5 means no visible drift. Done
     twice: test slice vs the whole train slice, and test slice vs the same
     clock hours one day earlier, to separate day-to-day drift from the mix of
     hours of the day.

Writes results/split_comparison.csv, results/hourly_activity.csv,
results/adversarial_validation.csv, results/feature_drift.csv and
figures/hourly_activity.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fraudbudget import config  # noqa: E402
from fraudbudget.plotting import GRID, INK, INK_MUTED, TEAL, VIOLET  # noqa: E402
from fraudbudget.data import COLUMNS, TARGET, TIME, load_dataset  # noqa: E402
from fraudbudget.metrics import pr_auc, recall_at_fpr  # noqa: E402
from fraudbudget.models import FEATURES, fraud_scores, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402
from run_models import model_kwargs  # noqa: E402

DATA = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ("logreg_balanced", "xgboost")
RANDOM_SEEDS = (0, 1, 2, 3, 4)
SECONDS_PER_HOUR = 3600


def fit_and_score(name: str, best: dict, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    kwargs = model_kwargs(name, best)
    if name == "xgboost" and best["xgboost"]["pos_weight"] == "ratio":
        # Recompute the weight on this particular training set.
        kwargs["pos_weight"] = (train[TARGET] == 0).sum() / train[TARGET].sum()
    model = make_model(name, seed=config.SEED, **kwargs).fit(train[FEATURES], train[TARGET])
    s = fraud_scores(model, test[FEATURES])
    y = test[TARGET].to_numpy()
    return {"pr_auc": pr_auc(y, s), "recall@fpr0.001": recall_at_fpr(y, s, config.MAX_FPR).recall,
            "test_frauds": int(y.sum())}


def exact_copies_across(train: pd.DataFrame, test: pd.DataFrame) -> tuple[int, int]:
    """(rows, frauds) of test that have an identical twin (every column, Time included) in train."""
    key = lambda d: pd.util.hash_pandas_object(d[COLUMNS], index=False)  # noqa: E731
    copied = key(test).isin(set(key(train))).to_numpy()
    return int(copied.sum()), int(test[TARGET].to_numpy()[copied].sum())


def feature_drift(train: pd.DataFrame, test: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Per feature: ROC-AUC of the raw value at telling test rows from train rows
    (0.5 = same distribution), and its correlation with Time over the two days."""
    y = np.r_[np.zeros(len(train)), np.ones(len(test))]
    rows = []
    for f in FEATURES:
        auc = roc_auc_score(y, np.r_[train[f].to_numpy(), test[f].to_numpy()])
        rows.append({"feature": f, "train_vs_test_auc": auc, "separation": abs(auc - 0.5),
                     "corr_with_time": float(np.corrcoef(full[TIME], full[f])[0, 1])})
    return pd.DataFrame(rows).sort_values("separation", ascending=False)


def adversarial_auc(a: pd.DataFrame, b: pd.DataFrame) -> tuple[float, float]:
    """3-fold ROC-AUC of a classifier separating rows of `a` (0) from rows of `b` (1)."""
    X = pd.concat([a[FEATURES], b[FEATURES]], ignore_index=True)
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist",
                        random_state=config.SEED, n_jobs=-1)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED)
    scores = cross_val_score(clf, X, y, cv=folds, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main() -> int:
    best = json.loads((RESULTS / "best_params.json").read_text())
    df = load_dataset(DATA)
    split = temporal_split(df)
    n_train, n_test = len(split.train), len(split.test)

    # 1. Random vs temporal split.
    rows = []
    for name in MODELS:
        copies, fraud_copies = exact_copies_across(split.train, split.test)
        rows.append({"model": name, "split": "temporal", "seed": None,
                     "exact_copies_in_train": copies, "fraud_copies_in_train": fraud_copies,
                     **fit_and_score(name, best, split.train, split.test)})
        for seed in RANDOM_SEEDS:
            train, rest = train_test_split(df, train_size=n_train, stratify=df[TARGET], random_state=seed)
            test, _ = train_test_split(rest, train_size=n_test, stratify=rest[TARGET], random_state=seed)
            copies, fraud_copies = exact_copies_across(train, test)
            rows.append({"model": name, "split": "random", "seed": seed,
                         "exact_copies_in_train": copies, "fraud_copies_in_train": fraud_copies,
                         **fit_and_score(name, best, train, test)})
        print(f"split comparison done: {name}")
    comp = pd.DataFrame(rows)
    comp.to_csv(RESULTS / "split_comparison.csv", index=False, float_format="%.6f")
    print("\n" + comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + comp.groupby(["model", "split"])[["pr_auc", "recall@fpr0.001"]]
          .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4f}"))

    # 2. Activity per hour.
    hour = (df[TIME] // SECONDS_PER_HOUR).astype(int)
    hourly = df.groupby(hour).agg(transactions=(TARGET, "size"), frauds=(TARGET, "sum"))
    hourly["fraud_rate_pct"] = 100 * hourly["frauds"] / hourly["transactions"]
    hourly.index.name = "hour"
    hourly.to_csv(RESULTS / "hourly_activity.csv", float_format="%.5f")

    # 3. Adversarial validation.
    test_start_h = split.test[TIME].min() / SECONDS_PER_HOUR
    same_hours_day1 = df[(df[TIME] >= (test_start_h - 24) * SECONDS_PER_HOUR)
                         & (df[TIME] < 24 * SECONDS_PER_HOUR)]
    adv_rows = []
    for label, reference in [("train slice (hours 0 to 33.4)", split.train),
                             (f"same clock hours on day 1 ({test_start_h - 24:.1f} to 24)", same_hours_day1)]:
        auc, std = adversarial_auc(reference, split.test)
        adv_rows.append({"reference": label, "reference_rows": len(reference), "test_rows": len(split.test),
                         "roc_auc_mean": auc, "roc_auc_std": std})
        print(f"adversarial validation vs {label}: ROC-AUC {auc:.4f} +/- {std:.4f}")
    pd.DataFrame(adv_rows).to_csv(RESULTS / "adversarial_validation.csv", index=False, float_format="%.5f")
    drift = feature_drift(split.train, split.test, df)
    drift.to_csv(RESULTS / "feature_drift.csv", index=False, float_format="%.4f")
    print("\n" + drift.head(6).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    plot_hourly(hourly, split)
    print(f"\nwrote {FIGURES / 'hourly_activity.png'}")
    return 0


def plot_hourly(hourly: pd.DataFrame, split) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True)
    x = hourly.index.to_numpy() + 0.5
    ax1.bar(x, hourly["transactions"] / 1000, width=0.85, color=TEAL)
    ax1.set_ylabel("Transactions\n(thousands / hour)", color=INK_MUTED, fontsize=9)
    ax2.bar(x, hourly["fraud_rate_pct"], width=0.85, color=VIOLET)
    ax2.set_ylabel("Fraud rate (%)", color=INK_MUTED, fontsize=9)
    ax2.set_xlabel("Hours since the first transaction", color=INK_MUTED, fontsize=9)
    bounds = [split.val[TIME].min() / SECONDS_PER_HOUR, split.test[TIME].min() / SECONDS_PER_HOUR]
    for ax in (ax1, ax2):
        for b in bounds:
            ax.axvline(b, color=INK, linewidth=1, linestyle="--")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    top = ax1.get_ylim()[1]
    for xpos, label in [(bounds[0] / 2, "train"), ((bounds[0] + bounds[1]) / 2, "val"),
                        ((bounds[1] + 48) / 2, "test")]:
        ax1.text(xpos, top * 0.98, label, ha="center", va="top", fontsize=9, color=INK)
    ax2.set_xlim(0, 48)
    ax2.set_xticks(range(0, 49, 6))
    ax1.set_title("Two days of card activity: volume follows the day, fraud rate peaks at night",
                  fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "hourly_activity.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
