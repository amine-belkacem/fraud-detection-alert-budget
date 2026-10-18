"""J2: every model on the test slice, with seed spread and bootstrap intervals.

Usage:
    python scripts/tune.py        # once, writes results/best_params.json
    python scripts/run_models.py

Each model is trained on the train slice with the parameters picked on
validation, for 5 seeds. The seed spread shows how much the training itself
varies; the bootstrap (1000 stratified replicates of the test slice, seed 0
scores) shows how much the *test set* lets us conclude. Writes
results/models_seeds.csv, results/models_bootstrap.csv,
results/models_paired.csv and figures/models_comparison.png.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudbudget import config  # noqa: E402
from fraudbudget.bootstrap import bootstrap_metric, paired_differences, summarise  # noqa: E402
from fraudbudget.plotting import GREY, INK, INK_MUTED, TEAL, style_axis  # noqa: E402
from fraudbudget.data import AMOUNT, TARGET, load_dataset  # noqa: E402
from fraudbudget.evaluate import evaluate_scores  # noqa: E402
from fraudbudget.metrics import alerts_for_budget, pr_auc, recall_at_budget, recall_at_fpr  # noqa: E402
from fraudbudget.models import FEATURES, MODEL_NAMES, UNSUPERVISED, fraud_scores, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402

DATA = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 1000

LABELS = {
    "dummy": "Dummy", "logreg": "LogReg", "logreg_balanced": "LogReg weighted",
    "isolation_forest": "Isolation Forest", "xgboost": "XGBoost weighted",
    "xgboost_smote": "XGBoost + SMOTE", "xgboost_undersample": "XGBoost + undersampling",
}
PAIRS = [
    ("xgboost", "logreg_balanced"),
    ("xgboost_smote", "xgboost"),
    ("xgboost_undersample", "xgboost"),
    ("xgboost", "isolation_forest"),
]
ACCENT, NEUTRAL = TEAL, GREY


def model_kwargs(name: str, best: dict) -> dict:
    """Constructor arguments for a model, from the tuning results."""
    if name in ("xgboost_smote", "xgboost_undersample"):
        return {"params": best[name]["params"], "resample_ratio": best[name]["resample_ratio"]}
    if name == "xgboost":
        weight = best["train_imbalance_ratio"] if best["xgboost"]["pos_weight"] == "ratio" else 1.0
        return {"params": best["xgboost"]["params"], "pos_weight": weight}
    if name == "isolation_forest":
        return {"params": best["isolation_forest"]["params"]}
    return {}


def interval_panel(ax, table: pd.DataFrame, order: list[str], title: str, xlabel: str,
                   reference: float | None = None, reference_label: str = "") -> None:
    y = np.arange(len(order))
    t = table.loc[order]
    colors = [NEUTRAL if m in ("dummy", "isolation_forest") else ACCENT for m in order]
    ax.hlines(y, t["ci_low"], t["ci_high"], colors=colors, linewidth=2)
    ax.scatter(t["value"], y, s=42, color=colors, zorder=3, edgecolors="white", linewidths=1.5)
    for yi, (v, hi) in enumerate(zip(t["value"], t["ci_high"])):
        ax.text(hi + 0.015, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
    if reference is not None:
        ax.axvline(reference, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.text(reference, len(order) - 0.4, reference_label, fontsize=8, color=INK_MUTED,
                ha="right", va="bottom")
    ax.set_yticks(y, [LABELS[m] for m in order])
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, len(order) - 0.1)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    style_axis(ax, grid_axis="x")


def main() -> int:
    best = json.loads((RESULTS / "best_params.json").read_text())
    split = temporal_split(load_dataset(DATA))
    X_tr, y_tr = split.train[FEATURES], split.train[TARGET].to_numpy()
    X_te, y_te = split.test[FEATURES], split.test[TARGET].to_numpy()
    amount_te = split.test[AMOUNT].to_numpy()

    seed_rows, scores_seed0 = [], {}
    for name in MODEL_NAMES:
        for seed in SEEDS:
            model = make_model(name, seed=seed, **model_kwargs(name, best)).fit(X_tr, y_tr)
            s = fraud_scores(model, X_te)
            row = evaluate_scores(y_te, s, amount_te, probabilistic=name not in UNSUPERVISED)
            seed_rows.append({"model": name, "seed": seed, **row})
            if seed == SEEDS[0]:
                scores_seed0[name] = s
        print(f"trained {name}")

    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(RESULTS / "models_seeds.csv", index=False, float_format="%.6f")
    budget_key = f"recall@budget{config.ALERTS_PER_100K:g}"
    fpr_key = f"recall@fpr{config.MAX_FPR:g}"
    spread = seeds.groupby("model", sort=False)[["pr_auc", fpr_key, budget_key]].agg(["mean", "std"])
    print("\nseed spread (5 seeds)\n" + spread.to_string(float_format=lambda v: f"{v:.4f}"))

    metrics = {
        "pr_auc": pr_auc,
        fpr_key: lambda y, s: recall_at_fpr(y, s, config.MAX_FPR).recall,
        budget_key: lambda y, s: recall_at_budget(y, s, config.ALERTS_PER_100K).recall,
    }
    boot_tables, paired_tables = [], []
    for metric_name, fn in metrics.items():
        reps = bootstrap_metric(y_te, scores_seed0, fn, n_boot=N_BOOT, seed=config.SEED)
        boot_tables.append(summarise(y_te, scores_seed0, reps, fn).assign(metric=metric_name))
        paired_tables.append(paired_differences(reps, PAIRS).assign(metric=metric_name))
        print(f"bootstrap done: {metric_name}")
    boot = pd.concat(boot_tables).reset_index()
    paired = pd.concat(paired_tables)
    boot.to_csv(RESULTS / "models_bootstrap.csv", index=False, float_format="%.6f")
    paired.to_csv(RESULTS / "models_paired.csv", index=False, float_format="%.6f")
    print("\n" + boot.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n" + paired.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Figure: PR-AUC and recall under budget, each with its 95 % interval.
    pr = boot[boot.metric == "pr_auc"].set_index("model")
    rb = boot[boot.metric == budget_key].set_index("model")
    order = list(pr.sort_values("value").index)
    capacity = alerts_for_budget(len(y_te), config.ALERTS_PER_100K)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    interval_panel(ax1, pr, order, "PR-AUC on the test slice", "PR-AUC (95 % bootstrap interval)",
                   reference=y_te.mean(), reference_label="random  ")
    ceiling = min(capacity, int(y_te.sum())) / int(y_te.sum())
    interval_panel(ax2, rb, order,
                   f"Frauds caught with {capacity} alerts ({config.ALERTS_PER_100K:g} per 100k)",
                   "Recall under alert budget (95 % bootstrap interval)",
                   reference=ceiling if ceiling < 1 else None, reference_label="ceiling  ")
    fig.tight_layout()
    fig.savefig(FIGURES / "models_comparison.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {FIGURES / 'models_comparison.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
