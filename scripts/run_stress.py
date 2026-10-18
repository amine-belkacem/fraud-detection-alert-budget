"""J4: prevalence stress test. Same model, same threshold, rarer fraud.

Usage:
    python scripts/run_stress.py     # needs results/best_params.json (scripts/tune.py)

The threshold is the FPR-0.1 % one chosen on validation (J3), frozen. On the
test slice every legit transaction is kept and only part of the frauds, drawn
at random 200 times per prevalence level. Writes results/stress_test.csv,
results/stress_test_summary.csv and figures/stress_test.png.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fraudbudget import config  # noqa: E402
from fraudbudget.plotting import GRID, INK, INK_MUTED, MAGENTA, TEAL, VIOLET  # noqa: E402
from fraudbudget.data import TARGET, load_dataset  # noqa: E402
from fraudbudget.metrics import confusion, pr_auc  # noqa: E402
from fraudbudget.models import FEATURES, fraud_scores, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402
from fraudbudget.stress import expected_precision, n_frauds_for_prevalence, subsample_frauds  # noqa: E402
from fraudbudget.threshold import threshold_for_fpr  # noqa: E402
from run_models import model_kwargs  # noqa: E402

DATA = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ("logreg_balanced", "xgboost")
LABELS = {"logreg_balanced": "LogReg weighted", "xgboost": "XGBoost weighted"}
PREVALENCES = (0.001, 0.0005, 0.0002)  # plus the test slice as it is
N_REPEATS = 200
METRIC_STYLE = {
    "roc_auc": ("ROC-AUC", MAGENTA),
    "pr_auc": ("PR-AUC", TEAL),
    "precision": ("Precision at the frozen threshold", VIOLET),
}


def main() -> int:
    best = json.loads((RESULTS / "best_params.json").read_text())
    split = temporal_split(load_dataset(DATA))
    X_tr, y_tr = split.train[FEATURES], split.train[TARGET].to_numpy()
    y_va, y_te = split.val[TARGET].to_numpy(), split.test[TARGET].to_numpy()
    n_legit, n_fraud_full = int((y_te == 0).sum()), int(y_te.sum())
    levels = [(y_te.mean(), n_fraud_full)] + [(p, n_frauds_for_prevalence(n_legit, p)) for p in PREVALENCES]

    rng = np.random.default_rng(config.SEED)
    rows, expected_rows = [], []
    for name in MODELS:
        model = make_model(name, seed=config.SEED, **model_kwargs(name, best)).fit(X_tr, y_tr)
        s_va = fraud_scores(model, split.val[FEATURES])
        s_te = fraud_scores(model, split.test[FEATURES])
        threshold = threshold_for_fpr(y_va, s_va, config.MAX_FPR)
        full = confusion(y_te, s_te >= threshold)

        for prevalence, k in levels:
            # The full test slice is a single configuration: no need to redraw it.
            repeats = 1 if k == n_fraud_full else N_REPEATS
            for rep in range(repeats):
                idx = np.arange(y_te.size) if k == n_fraud_full else subsample_frauds(y_te, k, rng)
                y, s = y_te[idx], s_te[idx]
                c = confusion(y, s >= threshold)
                rows.append({"model": name, "target_prevalence": prevalence, "n_frauds": k, "repeat": rep,
                             "roc_auc": roc_auc_score(y, s), "pr_auc": pr_auc(y, s),
                             "precision": c.precision, "recall": c.recall, "false_positives": c.fp})
            expected_rows.append({"model": name, "n_frauds": k,
                                  "expected_precision": expected_precision(full.recall, k, full.fp)})
        print(f"{name}: threshold {threshold:.4g}, full test recall {full.recall:.3f}, FP {full.fp}")

    raw = pd.DataFrame(rows)
    raw.to_csv(RESULTS / "stress_test.csv", index=False, float_format="%.6f")
    summary = (raw.groupby(["model", "n_frauds"])
               [["target_prevalence", "roc_auc", "pr_auc", "precision", "recall"]]
               .agg(["median", lambda v: v.quantile(0.05), lambda v: v.quantile(0.95)]))
    summary.columns = [f"{m}_{s}" for m, s in
                       zip(summary.columns.get_level_values(0), ["median", "p05", "p95"] * 5)]
    summary = summary.reset_index().merge(pd.DataFrame(expected_rows), on=["model", "n_frauds"])
    summary = summary.sort_values(["model", "n_frauds"], ascending=[True, False])
    summary.to_csv(RESULTS / "stress_test_summary.csv", index=False, float_format="%.6f")
    cols = ["model", "n_frauds", "target_prevalence_median", "roc_auc_median", "pr_auc_median",
            "precision_median", "expected_precision", "recall_median"]
    print("\n" + summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    plot(summary)
    print(f"\nwrote {FIGURES / 'stress_test.png'}")
    return 0


def plot(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.3), sharey=True)
    for ax, name in zip(axes, MODELS):
        d = summary[summary.model == name].sort_values("target_prevalence_median")
        x = d["target_prevalence_median"].to_numpy() * 100
        for metric, (label, color) in METRIC_STYLE.items():
            ax.fill_between(x, d[f"{metric}_p05"], d[f"{metric}_p95"], color=color, alpha=0.15, linewidth=0)
            ax.plot(x, d[f"{metric}_median"], color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.plot(x, d["expected_precision"], color=INK, linewidth=1, linestyle=":",
                label="Precision predicted from the FP count (dotted)")
        for xi, k in zip(x, d["n_frauds"]):
            ax.text(xi, -0.07, f"{k} frauds", ha="center", fontsize=7.5, color=INK_MUTED)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:.2f} %" for v in x])
        ax.minorticks_off()
        ax.set_ylim(-0.1, 1.03)
        ax.set_xlabel("Fraud prevalence in the test slice (log scale)", color=INK_MUTED, fontsize=9)
        ax.set_title(LABELS[name], fontsize=10.5, color=INK, loc="left")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Median over 200 draws (band: 5th to 95th percentile)", color=INK_MUTED, fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="lower center", ncol=4)
    fig.suptitle("Same model, same threshold, rarer fraud: ROC-AUC stays flat, precision falls",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIGURES / "stress_test.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
