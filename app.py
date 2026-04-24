import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from llm_support_routing.config import MODELS_DIR, OUTPUTS_DIR, RoutingThresholds

st.set_page_config(page_title="LLM Support Ticket Routing", layout="wide")
st.title("LLM-powered Support Ticket Routing Dashboard")
st.caption(
    "Support operations routing system (not a chatbot): rules → calibrated ML → "
    "low-confidence LLM classification → human triage."
)

metrics_path = OUTPUTS_DIR / "routing_metrics.csv"
routed_path = OUTPUTS_DIR / "routed_tickets.csv"
report_path = OUTPUTS_DIR / "training_report.txt"
sweep_path = OUTPUTS_DIR / "threshold_sweep.csv"
eval_summary_path = OUTPUTS_DIR / "eval_comparison.csv"
eval_per_class_path = OUTPUTS_DIR / "eval_per_class_metrics.csv"
eval_confusion_path = OUTPUTS_DIR / "eval_confusion_matrix.csv"
model_path = MODELS_DIR / "issue_type_tfidf_lr.joblib"
urgency_model_path = MODELS_DIR / "urgency_tfidf_lr.joblib"

if not metrics_path.exists() or not routed_path.exists():
    st.warning("Run: `python scripts/run_pipeline.py --download` first to generate outputs.")
    st.stop()

metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
routed = pd.read_csv(routed_path)

st.caption(
    "**Note:** Measured eval artifacts (accuracy, F1, confusion matrix) appear only when "
    "`data/eval/eval_tickets.csv` exists. Threshold recommendations are analytic estimates "
    "based on ML confidence distribution and are not automatically applied."
)

# ── KPI tiles ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
human_triage_rate = metrics.get("human_triage_rate", metrics.get("escalation_rate", 0.0))
col1.metric("Tickets", int(metrics["tickets"]))
col2.metric(
    "Human Triage Rate",
    f"{human_triage_rate:.2%}",
    help="Fraction routed to human_triage_queue — manual review / fallback rate. "
         "This is a routing proxy, not a true production escalation rate.",
)
col3.metric("LLM Invocation", f"{metrics['llm_invocation_rate']:.2%}",
            help="Fraction of tickets sent to the LLM classification stage (low-confidence band).")
col4.metric("Avg Confidence", f"{metrics.get('avg_routing_confidence', 0):.2%}")
est_cost = metrics.get("cost_per_ticket_usd_estimated", metrics.get("cost_per_ticket_usd", 0.0))
col5.metric(
    "Est. Cost / Ticket (USD)",
    f"${est_cost:.5f}",
    help="Estimated cost based on LLM invocation rate × per-call cost + fixed infra cost. "
         "Not measured billing.",
)

# ── Stage breakdown ───────────────────────────────────────────────────────────
st.subheader("Routing Stage Breakdown")
stage_counts = routed["stage"].value_counts().reset_index()
stage_counts.columns = ["stage", "count"]
st.plotly_chart(
    px.pie(stage_counts, names="stage", values="count", hole=0.4),
    use_container_width=True,
)

with ops_tab:
    st.subheader("Routing Operations Overview")
    st.caption(
        "These KPIs reflect production-style routing behavior. "
        "`human_fallback_rate` is shown as human-triage rate (not a true escalation metric)."
    )

    # ── KPI tiles ────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Tickets", int(metrics["tickets"]))
    col2.metric("Human Triage Rate", f"{metrics['human_fallback_rate']:.2%}")
    col3.metric("LLM Invocation Rate", f"{metrics['llm_invocation_rate']:.2%}")
    col4.metric("Avg Routing Confidence", f"{metrics.get('avg_routing_confidence', 0):.2%}")
    est_cost = metrics.get("cost_per_ticket_usd_estimated", metrics.get("cost_per_ticket_usd", 0.0))
    col5.metric("Est. Cost / Ticket (USD)", f"${est_cost:.5f}")

    st.info(
        "Metric semantics: measured = labeled-ground-truth evaluation; estimated = policy/cost approximations; "
        "proxy = heuristic-derived labels (e.g., complexity)."
    )

    # ── Stage breakdown ───────────────────────────────────────────────────────
    st.subheader("Routing Stage Breakdown")
    stage_counts = routed["stage"].value_counts().reset_index()
    stage_counts.columns = ["stage", "count"]
    st.plotly_chart(
        px.pie(stage_counts, names="stage", values="count", hole=0.4),
        use_container_width=True,
    )

    # ── Route distribution ────────────────────────────────────────────────────
    st.subheader("Queue Distribution")
    route_counts = routed["route"].value_counts().reset_index()
    route_counts.columns = ["route", "count"]
    st.plotly_chart(px.bar(route_counts, x="route", y="count", color="route"), use_container_width=True)

    # ── Urgency breakdown (if column present) ────────────────────────────────
    if "llm_urgency" in routed.columns and routed["llm_urgency"].ne("").any():
        st.subheader("Urgency Distribution")
        urg = routed[routed["llm_urgency"].ne("")]["llm_urgency"].value_counts().reset_index()
        urg.columns = ["urgency", "count"]
        order = ["critical", "high", "medium", "low"]
        urg["urgency"] = pd.Categorical(urg["urgency"], categories=order, ordered=True)
        urg = urg.sort_values("urgency")
        st.plotly_chart(
            px.bar(urg, x="urgency", y="count", color="urgency",
                   color_discrete_map={"critical": "#d62728", "high": "#ff7f0e",
                                       "medium": "#2ca02c", "low": "#1f77b4"}),
            use_container_width=True,
        )

    # ── Confidence histogram ──────────────────────────────────────────────────
    st.subheader("Confidence Distribution by Stage")
    st.plotly_chart(
        px.histogram(routed, x="confidence", color="stage", nbins=40, barmode="overlay", opacity=0.7),
        use_container_width=True,
    )

# ── Threshold sweep (estimated — no live LLM calls) ──────────────────────────
if sweep_path.exists():
    st.subheader("Confidence Threshold Sweep — Cost–Coverage Tradeoff (Estimated)")
    sweep = pd.read_csv(sweep_path)
    _rate_cols = [c for c in ["auto_routed_rate_estimated", "llm_fallback_rate_estimated",
                               "human_fallback_rate_estimated"] if c in sweep.columns]
    if not _rate_cols:
        _rate_cols = [c for c in ["auto_routed_rate", "llm_fallback_rate", "human_fallback_rate"]
                      if c in sweep.columns]
    fig_sweep = px.line(
        sweep.melt(id_vars="threshold_high", value_vars=_rate_cols),
        x="threshold_high", y="value", color="variable",
        labels={"threshold_high": "High-confidence threshold", "value": "Rate (estimated)", "variable": ""},
        title="Estimated routing stage rates vs. confidence threshold",
    )

    _cost_col = "cost_per_ticket_usd_estimated" if "cost_per_ticket_usd_estimated" in sweep.columns else "est_cost_per_ticket_usd"
    if _cost_col in sweep.columns:
        fig_cost = px.line(
            sweep, x="threshold_high", y=_cost_col,
            labels={"threshold_high": "High-confidence threshold", _cost_col: "Est. cost / ticket (USD)"},
            title="Estimated cost per ticket vs. confidence threshold",
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    if "is_recommended_threshold" in sweep.columns and sweep["is_recommended_threshold"].any():
        rec = sweep[sweep["is_recommended_threshold"]].iloc[0]
        st.info(
            f"**Analytic recommendation** (cost–coverage guide, not automatically applied): "
            f"high={rec['threshold_high']:.2f}, low={rec['threshold_low']:.2f} — "
            f"auto-route≈{rec['auto_routed_rate_estimated']:.1%}, "
            f"human-fallback≈{rec['human_fallback_rate_estimated']:.1%}. "
            f"Apply via `--high-threshold` / `--low-threshold`."
        )

# ── Evaluation artifacts (measured on labeled eval set) ──────────────────────
if eval_summary_path.exists():
    st.subheader("Labeled Eval Set: ML vs Keyword Baseline (Measured)")
    eval_summary = pd.read_csv(eval_summary_path)
    st.dataframe(eval_summary, use_container_width=True)

if eval_per_class_path.exists():
    st.subheader("Per-Class Metrics (Measured)")
    per_class = pd.read_csv(eval_per_class_path)
    display_cols = [c for c in ["model", "label", "precision", "recall", "f1_score", "support"]
                    if c in per_class.columns]
    st.dataframe(per_class[display_cols], use_container_width=True)

if eval_confusion_path.exists():
    st.subheader("Confusion Matrix Heatmap (Measured)")
    cm = pd.read_csv(eval_confusion_path)
    model_choice = st.selectbox("Confusion matrix model", sorted(cm["model"].unique().tolist()))
    cm_model = cm[cm["model"] == model_choice]
    cm_pivot = cm_model.pivot(index="actual_label", columns="predicted_label", values="count").fillna(0)
    st.plotly_chart(
        px.imshow(cm_pivot, text_auto=True, color_continuous_scale="Blues",
                  labels={"x": "Predicted", "y": "Actual", "color": "Count"}),
        use_container_width=True,
    )

if not eval_summary_path.exists() and not eval_per_class_path.exists() and not eval_confusion_path.exists():
    st.caption(
        "Measured eval artifacts not found. Add `data/eval/eval_tickets.csv` "
        "and rerun the pipeline to generate them."
    )

# ── Human-fallback enrichment (shown when --enrich-human-with-llm was used) ──
enrich_cols = ["suggested_path", "should_escalate", "reason"]
if all(c in routed.columns for c in enrich_cols) and routed[enrich_cols[0]].ne("").any():
    st.subheader("Human-Fallback Enrichment — LLM Resolution Guidance")
    enriched = routed[routed["stage"] == "human_fallback"][
        ["text", "confidence", "llm_urgency"] + enrich_cols + ["llm_summary"]
    ].copy()
    enriched = enriched[enriched["suggested_path"].ne("")]
    if not enriched.empty:
        st.dataframe(enriched.reset_index(drop=True), use_container_width=True)
    else:
        st.caption("No enriched human-fallback tickets in this run.")

# ── Training report ───────────────────────────────────────────────────────────
if report_path.exists():
    with st.expander("Model Training Report (all classifiers)"):
        st.code(report_path.read_text(encoding="utf-8"))

# ── Interactive routing ───────────────────────────────────────────────────────
st.subheader("Route a Ticket (Live)")
if model_path.exists():
    from llm_support_routing.models import load_model
    from llm_support_routing.routing import route_ticket

    @st.cache_resource
    def _load_issue_model():
        return load_model(str(model_path))

    @st.cache_resource
    def _load_urgency_model():
        return load_model(str(urgency_model_path)) if urgency_model_path.exists() else None

    ticket_text = st.text_area("Paste a support ticket:", height=120)
    enrich_live = st.checkbox(
        "Enrich human fallback with LLM guidance",
        value=False,
        help="If the ticket lands in human_fallback, calls llm_resolution_and_escalation() "
             "and llm_summarize_ticket(). Requires a configured OPENAI_API_KEY; incurs one "
             "extra LLM call.",
    )
    if st.button("Route") and ticket_text.strip():
        with st.spinner("Routing..."):
            decision = route_ticket(
                ticket_text,
                _load_issue_model(),
                urgency_model=_load_urgency_model(),
                enrich_human=enrich_live,
            )
        st.success(f"**Queue:** {decision.route}")
        cols = st.columns(3)
        cols[0].metric("Stage", decision.stage)
        cols[1].metric("Confidence", f"{decision.confidence:.2%}")
        cols[2].metric("Issue Type", decision.metadata.get("issue_type", "—"))
        if decision.metadata.get("urgency"):
            st.info(
                f"Urgency: **{decision.metadata['urgency']}**"
                + (f" | Complexity: **{decision.metadata['complexity']}**"
                   if decision.metadata.get("complexity") else "")
            )
        if decision.stage == "human_fallback" and decision.metadata.get("suggested_path"):
            st.subheader("LLM Resolution Guidance")
            g1, g2 = st.columns(2)
            g1.write(f"**Suggested path:** {decision.metadata['suggested_path']}")
            g2.write(f"**Should escalate:** {decision.metadata['should_escalate']}")
            if decision.metadata.get("reason"):
                st.write(f"**Reason:** {decision.metadata['reason']}")
            if decision.metadata.get("llm_summary"):
                st.write(f"**Summary:** {decision.metadata['llm_summary']}")
else:
    st.info("Train the model first (`python scripts/run_pipeline.py`) to enable live routing.")
