#!/usr/bin/env python3
"""Generate focused preview images from pipeline output CSVs.

Usage:
    python scripts/generate_preview_assets.py

Requires (produced by scripts/run_pipeline.py):
    outputs/routing_metrics.csv
    outputs/routed_tickets.csv
    outputs/threshold_sweep.csv

Optional (produced only when data/eval/eval_tickets.csv exists):
    outputs/eval_comparison.csv
    outputs/eval_per_class_metrics.csv

Generates:
    assets/dashboard_overview.png  — routing KPIs, stage breakdown, queue distribution
    assets/policy_tradeoff.png     — threshold cost-coverage operating curve
    assets/model_evaluation.png    — ML vs keyword baseline (only when eval files present)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
ASSETS = PROJECT_ROOT / "assets"

_STAGE_LABELS = {
    "rule_based": "Rule-based",
    "ml_high_confidence": "ML High-Conf",
    "llm_reasoning": "LLM Fallback",
    "human_fallback": "Human Triage",
}


def _require(path: Path, image_name: str) -> bool:
    if not path.exists():
        print(f"[SKIP] {image_name}: required file missing: {path.relative_to(PROJECT_ROOT)}")
        return False
    return True


def _save(fig, path: Path, name: str) -> None:
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[OK]   {name} -> {path.relative_to(PROJECT_ROOT)}")


def generate_dashboard_overview() -> bool:
    """KPI tiles + routing stage pie + queue distribution bar."""
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import pandas as pd

    m_path = OUTPUTS / "routing_metrics.csv"
    r_path = OUTPUTS / "routed_tickets.csv"
    if not (_require(m_path, "dashboard_overview") and _require(r_path, "dashboard_overview")):
        return False

    metrics = pd.read_csv(m_path).iloc[0]
    routed = pd.read_csv(r_path)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "Support Ticket Routing — Operations Overview",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(
        2, 5, figure=fig,
        top=0.88, bottom=0.07, left=0.06, right=0.97,
        hspace=0.55, wspace=0.45,
    )

    # ── KPI tiles ──────────────────────────────────────────────────────────────
    kpis = [
        ("Tickets\nProcessed",         f"{int(metrics.get('tickets', 0)):,}",                      "#4C72B0"),
        ("Human Triage\nRate (proxy)",  f"{metrics.get('human_triage_rate', 0):.1%}",              "#DD8452"),
        ("LLM Invocation\nRate",        f"{metrics.get('llm_invocation_rate', 0):.1%}",            "#55A868"),
        ("Avg Routing\nConfidence",     f"{metrics.get('avg_routing_confidence', 0):.3f}",         "#C44E52"),
        ("Est. Cost /\nTicket (USD)",   f"${metrics.get('cost_per_ticket_usd_estimated', 0):.5f}", "#8172B2"),
    ]
    for col, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.05), 0.90, 0.90,
            boxstyle="round,pad=0.03",
            facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(0.5, 0.62, value, ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
        ax.text(0.5, 0.25, label, ha="center", va="center",
                fontsize=8.5, color="#333333", multialignment="center")

    # ── Routing stage breakdown pie ────────────────────────────────────────────
    stage_counts = routed["stage"].value_counts()
    stage_data = {_STAGE_LABELS.get(k, k): v for k, v in stage_counts.items()}
    colors_pie = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2"]

    ax_pie = fig.add_subplot(gs[1, :2])
    _, texts, autotexts = ax_pie.pie(
        list(stage_data.values()),
        labels=list(stage_data.keys()),
        autopct="%1.1f%%",
        colors=colors_pie[: len(stage_data)],
        startangle=90,
        pctdistance=0.78,
    )
    for t in texts:
        t.set_fontsize(9)
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("Routing Stage Breakdown", fontsize=11, pad=6)

    # ── Queue distribution horizontal bar ──────────────────────────────────────
    queue_counts = routed["route"].value_counts().sort_values()
    max_val = int(queue_counts.max())

    ax_bar = fig.add_subplot(gs[1, 2:])
    bars = ax_bar.barh(
        queue_counts.index.tolist(),
        queue_counts.values,
        color="#4C72B0", alpha=0.80,
    )
    ax_bar.set_xlabel("Ticket Count", fontsize=9)
    ax_bar.set_title("Queue Distribution", fontsize=11)
    ax_bar.tick_params(axis="y", labelsize=8)
    ax_bar.tick_params(axis="x", labelsize=8)
    for bar in bars:
        w = bar.get_width()
        ax_bar.text(
            w + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{int(w):,}", va="center", fontsize=7.5,
        )
    ax_bar.set_xlim(right=max_val * 1.14)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    _save(fig, ASSETS / "dashboard_overview.png", "dashboard_overview.png")
    plt.close(fig)
    return True


def generate_policy_tradeoff() -> bool:
    """Threshold sweep — stage rates and estimated cost vs confidence threshold."""
    import matplotlib.pyplot as plt
    import pandas as pd

    sweep_path = OUTPUTS / "threshold_sweep.csv"
    if not _require(sweep_path, "policy_tradeoff"):
        return False

    sweep = pd.read_csv(sweep_path)

    if "is_recommended_threshold" in sweep.columns:
        rec = sweep[sweep["is_recommended_threshold"].astype(bool)]
    else:
        rec = sweep.iloc[[]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Confidence Threshold — Cost–Coverage Policy Tradeoff\n"
        "(Estimated analytic metrics — no LLM calls required)",
        fontsize=12, fontweight="bold",
    )

    t = sweep["threshold_high"]

    ax1.plot(t, sweep["auto_routed_rate_estimated"],   "o-", color="#55A868", linewidth=2, label="Auto-routed")
    ax1.plot(t, sweep["llm_fallback_rate_estimated"],  "s-", color="#DD8452", linewidth=2, label="LLM Fallback")
    ax1.plot(t, sweep["human_fallback_rate_estimated"], "^-", color="#C44E52", linewidth=2, label="Human Triage")
    ax1.set_ylabel("Fraction of Tickets", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9, loc="center right")
    ax1.set_title("Routing Stage Rates vs High-Confidence Threshold", fontsize=10)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", alpha=0.3)

    cost_milli = sweep["cost_per_ticket_usd_estimated"] * 1_000
    ax2.plot(t, cost_milli, "D-", color="#8172B2", linewidth=2)
    ax2.set_xlabel("High-Confidence Threshold", fontsize=10)
    ax2.set_ylabel("Est. Cost / Ticket\n(×10⁻³ USD)", fontsize=10)
    ax2.set_title("Estimated Cost per Ticket vs Threshold", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", alpha=0.3)

    if not rec.empty:
        rec_t = float(rec.iloc[0]["threshold_high"])
        for ax in (ax1, ax2):
            ax.axvline(rec_t, color="#555555", linestyle="--", alpha=0.7, linewidth=1.5)
        ax1.text(
            rec_t, 1.02,
            f"Analytic guide ({rec_t:.2f})",
            ha="center", va="bottom", fontsize=8, color="#555555",
            transform=ax1.get_xaxis_transform(),
        )

    fig.tight_layout()
    _save(fig, ASSETS / "policy_tradeoff.png", "policy_tradeoff.png")
    plt.close(fig)
    return True


def generate_model_evaluation() -> bool:
    """ML vs keyword baseline — summary bars + per-class F1."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    eval_path = OUTPUTS / "eval_comparison.csv"
    pc_path   = OUTPUTS / "eval_per_class_metrics.csv"

    if not _require(eval_path, "model_evaluation"):
        return False

    row = pd.read_csv(eval_path).iloc[0]
    has_pc = pc_path.exists()

    ncols = 2 if has_pc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(13 if has_pc else 7, 6))
    if ncols == 1:
        axes = [axes]

    fig.suptitle(
        "Model Evaluation — Measured on Labeled Eval Set",
        fontsize=13, fontweight="bold",
    )

    # ── Summary: ML vs keyword baseline ───────────────────────────────────────
    ax = axes[0]
    width = 0.34
    metric_names = ["Accuracy", "Macro-F1", "Weighted-F1"]
    ml_vals = [
        float(row.get("ml_accuracy", 0)),
        float(row.get("ml_macro_f1", 0)),
        float(row.get("ml_weighted_f1", 0)),
    ]
    kw_vals = [
        float(row.get("keyword_baseline_accuracy", 0)),
        float(row.get("keyword_macro_f1", 0)),
        float(row.get("keyword_weighted_f1", 0)),
    ]
    x = np.arange(len(metric_names))
    b_ml = ax.bar(x - width / 2, ml_vals, width, label="ML (TF-IDF + LR)", color="#4C72B0", alpha=0.85)
    b_kw = ax.bar(x + width / 2, kw_vals, width, label="Keyword Baseline",  color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("ML vs Keyword Baseline", fontsize=11)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    for bar in list(b_ml) + list(b_kw):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.012, f"{h:.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    n = int(row.get("n_eval_samples", 0))
    ax.text(0.5, -0.10, f"n = {n} labeled tickets",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")

    # ── Per-class F1 ───────────────────────────────────────────────────────────
    if has_pc:
        pc = pd.read_csv(pc_path)
        exclude = {"macro avg", "weighted avg", "accuracy"}
        real = pc[~pc["label"].isin(exclude)]
        ml_f1 = real[real["model"] == "ml_tfidf_lr"].set_index("label")["f1_score"]
        kw_f1 = real[real["model"] == "keyword_baseline"].set_index("label")["f1_score"]
        all_labels = sorted(set(ml_f1.index) | set(kw_f1.index))

        ax2 = axes[1]
        x2 = np.arange(len(all_labels))
        ax2.bar(x2 - width / 2, [float(ml_f1.get(l, 0)) for l in all_labels],
                width, label="ML",      color="#4C72B0", alpha=0.85)
        ax2.bar(x2 + width / 2, [float(kw_f1.get(l, 0)) for l in all_labels],
                width, label="Keyword", color="#DD8452", alpha=0.85)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=8.5)
        ax2.set_ylabel("F1 Score", fontsize=10)
        ax2.set_ylim(0, 1.08)
        ax2.set_title("Per-Class F1", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, ASSETS / "model_evaluation.png", "model_evaluation.png")
    plt.close(fig)
    return True


def main() -> None:
    print("Generating preview assets from pipeline outputs...")
    print(f"  Source: {OUTPUTS}")
    print(f"  Target: {ASSETS}\n")

    results = [
        generate_dashboard_overview(),
        generate_policy_tradeoff(),
        generate_model_evaluation(),
    ]

    generated = sum(results)
    if generated == 0:
        print(
            "\nNo images generated. Run the pipeline first:\n"
            "  python scripts/run_pipeline.py\n"
            "Then re-run:\n"
            "  python scripts/generate_preview_assets.py"
        )
        sys.exit(1)

    print(f"\n{generated}/3 image(s) generated in assets/")
    if generated < 3:
        print("(Some images skipped — see [SKIP] messages above.)")


if __name__ == "__main__":
    main()
