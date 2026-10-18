"""J1: baseline models on the temporal split, and the accuracy trap in numbers.

Usage:
    python scripts/run_baseline.py

Writes results/split_summary.csv, results/baseline_test.csv and
figures/baseline_accuracy_trap.png. Models are trained on the train slice only
and evaluated on the test slice (the most recent 20 % of transactions).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fraudbudget import config  # noqa: E402
from fraudbudget.data import AMOUNT, TARGET, TIME, load_dataset  # noqa: E402
from fraudbudget.evaluate import evaluate_scores  # noqa: E402
from fraudbudget.models import FEATURES, MODEL_NAMES, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402

DATA = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

LABELS = {"dummy": "Dummy (always legit)", "logreg": "Logistic regression",
          "logreg_balanced": "Logistic regression, class-weighted"}
COLORS = {"dummy": "#8a8986", "logreg": "#2a78d6", "logreg_balanced": "#eb6834"}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def split_summary(split) -> pd.DataFrame:
    rows = []
    for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        rows.append({
            "set": name, "rows": len(part), "frauds": int(part[TARGET].sum()),
            "prevalence_pct": 100 * part[TARGET].mean(),
            "start_h": part[TIME].min() / 3600, "end_h": part[TIME].max() / 3600,
            "fraud_amount_eur": part.loc[part[TARGET] == 1, AMOUNT].sum(),
        })
    return pd.DataFrame(rows)


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot(table: pd.DataFrame, curves: dict, prevalence: float, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.3]})

    # Left: accuracy says all three models are excellent, recall says otherwise.
    names = list(table.index)
    x = np.arange(len(names))
    width = 0.36
    acc = table["accuracy@0.5"].to_numpy() * 100
    rec = table["recall@0.5"].to_numpy() * 100
    b1 = ax1.bar(x - width / 2 - 0.01, acc, width, color="#2a78d6", label="Accuracy")
    b2 = ax1.bar(x + width / 2 + 0.01, rec, width, color="#eb6834", label="Recall (frauds caught)")
    for bars, fmt in [(b1, "{:.2f}"), (b2, "{:.0f}")]:
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     fmt.format(bar.get_height()), ha="center", fontsize=8.5, color=INK)
    ax1.set_xticks(x, ["Dummy", "LogReg", "LogReg\nweighted"])
    ax1.set_ylim(0, 112)
    ax1.set_ylabel("% at threshold 0.5", color=INK_MUTED, fontsize=9)
    ax1.set_title("Accuracy hides that the dummy catches nothing", fontsize=10.5, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    style_axis(ax1)
    ax1.grid(axis="x", visible=False)

    # Right: precision-recall curves against the prevalence line (= random ranking).
    for name, (precision, recall) in curves.items():
        ax2.plot(recall, precision, color=COLORS[name], linewidth=2,
                 label=f"{LABELS[name]}  (PR-AUC {table.loc[name, 'pr_auc']:.3f})")
    ax2.axhline(prevalence, color=COLORS["dummy"], linewidth=1.5, linestyle="--",
                label=f"Dummy / random ranking  (PR-AUC = prevalence {prevalence:.4f})")
    ax2.set_xlim(0, 1.0)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel("Recall", color=INK_MUTED, fontsize=9)
    ax2.set_ylabel("Precision", color=INK_MUTED, fontsize=9)
    ax2.set_title("Precision-recall on the test slice (last 20 % in time)",
                  fontsize=10.5, color=INK, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower left")
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    df = load_dataset(DATA)
    split = temporal_split(df)
    summary = split_summary(split)
    summary.to_csv(RESULTS / "split_summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    X_train, y_train = split.train[FEATURES], split.train[TARGET].to_numpy()
    X_test, y_test = split.test[FEATURES], split.test[TARGET].to_numpy()

    rows, curves = {}, {}
    for name in MODEL_NAMES:
        model = make_model(name, seed=config.SEED).fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
        rows[name] = evaluate_scores(y_test, scores, split.test[AMOUNT])
        if name != "dummy":
            precision, recall, _ = precision_recall_curve(y_test, scores)
            curves[name] = (precision, recall)

    table = pd.DataFrame(rows).T
    table.index.name = "model"
    table.to_csv(RESULTS / "baseline_test.csv", float_format="%.6f")
    print()
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    plot(table, curves, prevalence=y_test.mean(), out=FIGURES / "baseline_accuracy_trap.png")
    print(f"\nwrote {FIGURES / 'baseline_accuracy_trap.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
