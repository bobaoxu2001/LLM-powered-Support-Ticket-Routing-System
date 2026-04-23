import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from llm_support_routing.config import MODELS_DIR, OUTPUTS_DIR

st.set_page_config(page_title="LLM Support Ticket Routing", layout="wide")
st.title("LLM-powered Support Ticket Routing Dashboard")

metrics_path = OUTPUTS_DIR / "routing_metrics.csv"
routed_path = OUTPUTS_DIR / "routed_tickets.csv"
report_path = OUTPUTS_DIR / "training_report.txt"
sweep_path = OUTPUTS_DIR / "threshold_sweep.csv"
model_path = MODELS_DIR / "issue_type_tfidf_lr.joblib"
urgency_model_path = MODELS_DIR / "urgency_tfidf_lr.joblib"

if not metrics_path.exists() or not routed_path.exists():
    st.warning("Run: `python scripts/run_pipeline.py --download` first to generate outputs.")
    st.stop()

metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
routed = pd.read_csv(routed_path)

# ── KPI tiles ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Tickets", int(metrics["tickets"]))
col2.metric("Escalation Rate", f"{metrics['escalation_rate']:.2%}")
col3.metric("LLM Invocation", f"{metrics['llm_invocation_rate']:.2%}")
col4.metric("Avg Confidence", f"{metrics.get('avg_routing_confidence', 0):.2%}")
col5.metric("Cost / Ticket (USD)", f"${metrics['cost_per_ticket_usd']:.5f}")

# ── Stage breakdown ───────────────────────────────────────────────────────────
st.subheader("Routing Stage Breakdown")
stage_counts = routed["stage"].value_counts().reset_index()
stage_counts.columns = ["stage", "count"]
st.plotly_chart(
    px.pie(stage_counts, names="stage", values="count", hole=0.4),
    use_container_width=True,
)

# ── Route distribution ────────────────────────────────────────────────────────
st.subheader("Queue Distribution")
route_counts = routed["route"].value_counts().reset_index()
route_counts.columns = ["route", "count"]
st.plotly_chart(px.bar(route_counts, x="route", y="count", color="route"), use_container_width=True)

# ── Urgency breakdown (if column present) ────────────────────────────────────
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

# ── Confidence histogram ──────────────────────────────────────────────────────
st.subheader("Confidence Distribution by Stage")
st.plotly_chart(
    px.histogram(routed, x="confidence", color="stage", nbins=40, barmode="overlay", opacity=0.7),
    use_container_width=True,
)

# ── Routing flow ──────────────────────────────────────────────────────────────
st.subheader("Routing Flow (Stage → Queue)")
flow = routed.groupby(["stage", "route"], as_index=False).size()
st.plotly_chart(
    px.parallel_categories(flow, dimensions=["stage", "route"], color="size",
                           color_continuous_scale=px.colors.sequential.Viridis),
    use_container_width=True,
)

# ── Threshold sweep ───────────────────────────────────────────────────────────
if sweep_path.exists():
    st.subheader("Confidence Threshold Sweep (Cost–Coverage Tradeoff)")
    sweep = pd.read_csv(sweep_path)
    fig_sweep = px.line(
        sweep.melt(id_vars="threshold_high",
                   value_vars=["auto_routed_rate", "llm_fallback_rate", "human_fallback_rate"]),
        x="threshold_high", y="value", color="variable",
        labels={"threshold_high": "High-confidence threshold", "value": "Rate", "variable": ""},
        title="Routing stage rates vs. confidence threshold",
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    fig_cost = px.line(
        sweep, x="threshold_high", y="est_cost_per_ticket_usd",
        labels={"threshold_high": "High-confidence threshold", "est_cost_per_ticket_usd": "Est. cost / ticket (USD)"},
        title="Estimated cost per ticket vs. confidence threshold",
    )
    st.plotly_chart(fig_cost, use_container_width=True)

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
    if st.button("Route") and ticket_text.strip():
        with st.spinner("Routing..."):
            decision = route_ticket(
                ticket_text,
                _load_issue_model(),
                urgency_model=_load_urgency_model(),
            )
        st.success(f"**Queue:** {decision.route}")
        cols = st.columns(3)
        cols[0].metric("Stage", decision.stage)
        cols[1].metric("Confidence", f"{decision.confidence:.2%}")
        cols[2].metric("Issue Type", decision.metadata.get("issue_type", "—"))
        if decision.metadata.get("urgency"):
            st.info(f"Urgency: **{decision.metadata['urgency']}** | Complexity: **{decision.metadata.get('complexity', '—')}**")
else:
    st.info("Train the model first (`python scripts/run_pipeline.py`) to enable live routing.")
