"""J3: pick the threshold on validation, freeze it, measure on test.

Usage:
    python scripts/tune.py           # once, writes results/best_params.json
    python scripts/run_thresholds.py

For each model (seed 0, tuned parameters) the threshold is chosen on the
validation slice with three rules (FPR cap, alert budget, minimum cost), then
applied unchanged to the test slice. Also: bootstrap interval on the test cost,
sensitivity to the assumed cost per alert, and probability calibration.

Writes results/thresholds_test.csv, results/thresholds_cost_ci.csv,
results/cost_sensitivity.csv, results/calibration.csv and three figures.
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
sys.path.insert(0, str(ROOT / "scripts"))
from fraudbudget import config  # noqa: E402
from fraudbudget.bootstrap import percentile_interval, stratified_indices  # noqa: E402
from fraudbudget.calibration import brier, reliability  # noqa: E402
from fraudbudget.data import AMOUNT, TARGET, load_dataset  # noqa: E402
from fraudbudget.metrics import business_cost, confusion  # noqa: E402
from fraudbudget.models import FEATURES, fraud_scores, make_model  # noqa: E402
from fraudbudget.split import temporal_split  # noqa: E402
from fraudbudget.threshold import (  # noqa: E402
    cost_curve,
    threshold_for_budget,
    threshold_for_cost,
    threshold_for_fpr,
)
from run_models import model_kwargs  # noqa: E402

DATA = ROOT / "data" / "raw" / "creditcard.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
N_BOOT = 1000
COSTS_PER_ALERT = (1.0, 2.0, 5.0, 10.0, 20.0)

# xgboost_plain is the same tuned XGBoost without class weighting: it is the
# reference for calibration, since nothing in its training distorts the base rate.
MODELS = ("logreg_balanced", "xgboost", "xgboost_plain", "xgboost_smote", "xgboost_undersample")
LABELS = {
    "logreg_balanced": "LogReg weighted", "xgboost": "XGBoost weighted",
    "xgboost_plain": "XGBoost, no weighting", "xgboost_smote": "XGBoost + SMOTE",
    "xgboost_undersample": "XGBoost + undersampling",
}
COLORS = {
    "logreg_balanced": "#8a8986", "xgboost": "#2a78d6", "xgboost_plain": "#1baf7a",
    "xgboost_smote": "#eb6834", "xgboost_undersample": "#6250d6",
}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def build(name: str, best: dict):
    if name == "xgboost_plain":
        return make_model("xgboost", seed=config.SEED, params=best["xgboost"]["params"], pos_weight=1.0)
    return make_model(name, seed=config.SEED, **model_kwargs(name, best))


def apply_threshold(y, s, amount, threshold: float, cost_per_alert: float) -> dict:
    flagged = s >= threshold
    c = confusion(y, flagged)
    cost = business_cost(y, flagged, amount, cost_per_alert)
    return {
        "threshold": threshold, "alerts": c.tp + c.fp, "frauds_caught": c.tp,
        "recall": c.recall, "precision": c.precision, "fpr": c.fpr,
        "missed_eur": cost.missed_fraud_eur, "review_eur": cost.review_eur,
        "cost_eur": cost.total_eur,
    }


def style_axis(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> int:
    best = json.loads((RESULTS / "best_params.json").read_text())
    split = temporal_split(load_dataset(DATA))
    X_tr, y_tr = split.train[FEATURES], split.train[TARGET].to_numpy()
    X_va, y_va, a_va = split.val[FEATURES], split.val[TARGET].to_numpy(), split.val[AMOUNT].to_numpy()
    X_te, y_te, a_te = split.test[FEATURES], split.test[TARGET].to_numpy(), split.test[AMOUNT].to_numpy()
    C = config.COST_PER_ALERT_EUR

    val_scores, test_scores = {}, {}
    for name in MODELS:
        model = build(name, best).fit(X_tr, y_tr)
        val_scores[name] = fraud_scores(model, X_va)
        test_scores[name] = fraud_scores(model, X_te)
        print(f"trained {name}")

    # 1. Thresholds chosen on validation, applied to test.
    rows = [
        {"model": "flag_nothing", "rule": "-", **apply_threshold(y_te, y_te * 0.0, a_te, np.inf, C)},
        {"model": "flag_everything", "rule": "-", **apply_threshold(y_te, y_te * 0.0, a_te, 0.0, C)},
    ]
    frozen_cost_threshold = {}
    for name in MODELS:
        sv, st = val_scores[name], test_scores[name]
        rules = {
            "default_0.5": config.DEFAULT_THRESHOLD,
            f"fpr_{config.MAX_FPR:g}": threshold_for_fpr(y_va, sv, config.MAX_FPR),
            f"budget_{config.ALERTS_PER_100K:g}": threshold_for_budget(y_va, sv, config.ALERTS_PER_100K),
            "min_cost": threshold_for_cost(y_va, sv, a_va, C),
        }
        frozen_cost_threshold[name] = rules["min_cost"]
        for rule, t in rules.items():
            on_val = apply_threshold(y_va, sv, a_va, t, C)
            rows.append({"model": name, "rule": rule, **apply_threshold(y_te, st, a_te, t, C),
                         "val_alerts": on_val["alerts"], "val_fpr": on_val["fpr"]})
        # Hindsight: the best threshold had we been allowed to look at test.
        oracle = threshold_for_cost(y_te, st, a_te, C)
        rows.append({"model": name, "rule": "oracle_on_test", **apply_threshold(y_te, st, a_te, oracle, C)})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "thresholds_test.csv", index=False, float_format="%.6f")
    cols = ["model", "rule", "alerts", "frauds_caught", "recall", "precision", "fpr", "cost_eur", "val_alerts"]
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 2. Bootstrap on test with the frozen min-cost threshold. Savings are paired
    #    with "flag nothing" on the same replicate.
    idx = stratified_indices(y_te, N_BOOT, config.SEED)
    nothing = np.array([a_te[r][y_te[r] == 1].sum() for r in idx])
    ci_rows = []
    for name in MODELS:
        flagged = test_scores[name] >= frozen_cost_threshold[name]
        costs = np.array([business_cost(y_te[r], flagged[r], a_te[r], C).total_eur for r in idx])
        point = business_cost(y_te, flagged, a_te, C).total_eur
        saving = 1 - costs / nothing
        lo, hi = percentile_interval(costs)
        slo, shi = percentile_interval(saving)
        ci_rows.append({"model": name, "cost_eur": point, "cost_ci_low": lo, "cost_ci_high": hi,
                        "saving_vs_nothing": 1 - point / a_te[y_te == 1].sum(),
                        "saving_ci_low": slo, "saving_ci_high": shi})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(RESULTS / "thresholds_cost_ci.csv", index=False, float_format="%.6f")
    print("\n" + ci.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # 3. Sensitivity to the cost per alert: re-pick the threshold on validation for each value.
    sens_rows = []
    for c_alert in COSTS_PER_ALERT:
        base = {"cost_per_alert": c_alert,
                "flag_nothing_eur": float(a_te[y_te == 1].sum()),
                "flag_everything_eur": c_alert * len(y_te)}
        for name in MODELS:
            t = threshold_for_cost(y_va, val_scores[name], a_va, c_alert)
            r = apply_threshold(y_te, test_scores[name], a_te, t, c_alert)
            sens_rows.append({**base, "model": name, "threshold": t, "alerts": r["alerts"],
                              "recall": r["recall"], "cost_eur": r["cost_eur"]})
    sens = pd.DataFrame(sens_rows)
    sens["ratio_to_nothing"] = sens["cost_eur"] / sens["flag_nothing_eur"]
    sens.to_csv(RESULTS / "cost_sensitivity.csv", index=False, float_format="%.6f")
    print("\n" + sens.pivot(index="cost_per_alert", columns="model", values="cost_eur")
          .to_string(float_format=lambda v: f"{v:.0f}"))

    # 4. Calibration on test.
    cal_rows, bin_rows, curves = [], [], {}
    for name in MODELS:
        p = test_scores[name]
        curves[name] = reliability(y_te, p)
        bin_rows += [{"model": name, **vars(b)} for b in curves[name]]
        cal_rows.append({"model": name, "brier": brier(y_te, p), "mean_proba": float(p.mean()),
                         "observed_rate": float(y_te.mean())})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(RESULTS / "calibration.csv", index=False, float_format="%.8f")
    bins = pd.DataFrame(bin_rows)
    bins.to_csv(RESULTS / "calibration_bins.csv", index=False, float_format="%.8f")
    print("\n" + cal.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print("\n" + bins.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    plot_cost_curve(y_te, test_scores["xgboost"], a_te, table, C)
    plot_sensitivity(sens)
    plot_calibration(curves)
    print("\nfigures written")
    return 0


def plot_cost_curve(y, s, amount, table: pd.DataFrame, c_alert: float) -> None:
    curve = cost_curve(y, s, amount, c_alert)
    keep = curve.n_alerts > 0
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.plot(curve.n_alerts[keep], curve.total_eur[keep], color="#2a78d6", linewidth=2,
            label="XGBoost weighted, every possible threshold")
    nothing = float(amount[y == 1].sum())
    ax.axhline(nothing, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(1.1, nothing * 1.04, f"flag nothing: {nothing:,.0f} EUR", fontsize=8.5, color=INK_MUTED)
    markers = {"default_0.5": ("0.5 (default)", "s"), "min_cost": ("min cost on val", "o"),
               f"budget_{config.ALERTS_PER_100K:g}": ("alert budget on val", "D"),
               "oracle_on_test": ("best in hindsight", "*")}
    sub = table[table.model == "xgboost"].set_index("rule")
    for rule, (label, marker) in markers.items():
        r = sub.loc[rule]
        ax.scatter(max(r.alerts, 1), r.cost_eur, marker=marker, s=70 if marker != "*" else 140,
                   color="#eb6834" if rule != "oracle_on_test" else INK, zorder=3,
                   edgecolors="white", linewidths=1.2,
                   label=f"{label}: {int(r.alerts)} alerts, {r.cost_eur:,.0f} EUR")
    ax.set_xscale("log")
    ax.set_ylim(0, nothing * 1.6)
    ax.set_xlabel("Alerts raised on the test slice (log scale)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel(f"Simulated cost (EUR), {c_alert:g} EUR per alert", color=INK_MUTED, fontsize=9)
    ax.set_title("Cost against number of alerts, test slice", fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_curve.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for name in MODELS:
        d = sens[sens.model == name]
        ax.plot(d.cost_per_alert, d.ratio_to_nothing, marker="o", markersize=5, linewidth=2,
                color=COLORS[name], label=LABELS[name])
    ax.axhline(1.0, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(COSTS_PER_ALERT[0], 1.02, "flag nothing", fontsize=8.5, color=INK_MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_xticks(COSTS_PER_ALERT, [f"{c:g}" for c in COSTS_PER_ALERT])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Assumed cost per alert (EUR, log scale)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("Test cost / cost of flagging nothing", color=INK_MUTED, fontsize=9)
    ax.set_title("Threshold re-picked on validation for each cost assumption",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "cost_sensitivity.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_calibration(curves: dict) -> None:
    floor = 1e-5  # bins with no fraud are drawn hollow on this line (log axes cannot show 0)
    xlims, ylims = (1e-7, 1.4), (floor * 0.6, 1.4)
    fig, axes = plt.subplots(1, len(curves), figsize=(13, 3.6), sharex=True, sharey=True)
    for ax, (name, bins) in zip(axes, curves.items()):
        ax.plot(xlims, xlims, color=INK_MUTED, linestyle="--", linewidth=1)
        pred = np.array([b.mean_predicted for b in bins])
        obs = np.array([b.observed_rate for b in bins])
        pos = obs > 0
        ax.plot(pred[pos], obs[pos], color=COLORS[name], linewidth=2, marker="o", markersize=5)
        ax.scatter(pred[~pos], np.full((~pos).sum(), floor), facecolors="none",
                   edgecolors=COLORS[name], s=30, linewidths=1.5)
        for b in bins:
            ax.annotate(f"{b.n_fraud}/{b.n}", (b.mean_predicted, max(b.observed_rate, floor)),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color=INK_MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        ax.set_title(LABELS[name], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("Mean predicted probability", color=INK_MUTED, fontsize=8.5)
        style_axis(ax)
    axes[0].set_ylabel("Observed fraud rate", color=INK_MUTED, fontsize=8.5)
    fig.suptitle("Reliability on the test slice (bins by decade, labels = frauds / transactions; "
                 "hollow = no fraud in bin)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
