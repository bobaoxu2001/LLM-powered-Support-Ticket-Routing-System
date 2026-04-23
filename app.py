import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))


import pandas as pd
import plotly.express as px
import streamlit as st

from llm_support_routing.config import OUTPUTS_DIR

st.set_page_config(page_title="LLM Support Ticket Routing", layout="wide")
st.title("LLM-powered Support Ticket Routing Dashboard")

metrics_path = OUTPUTS_DIR / "routing_metrics.csv"
routed_path = OUTPUTS_DIR / "routed_tickets.csv"

if not metrics_path.exists() or not routed_path.exists():
    st.warning("Run: `python scripts/run_pipeline.py --download` first to generate outputs.")
    st.stop()

metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
routed = pd.read_csv(routed_path)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Tickets", int(metrics["tickets"]))
col2.metric("Escalation Rate", f"{metrics['escalation_rate']:.2%}")
col3.metric("LLM Invocation", f"{metrics['llm_invocation_rate']:.2%}")
col4.metric("Cost / Ticket (USD)", f"${metrics['cost_per_ticket_usd']:.4f}")

st.subheader("Routing Flow (Sankey-like via grouped transitions)")
flow = routed.groupby(["stage", "route"], as_index=False).size()
fig_flow = px.parallel_categories(flow, dimensions=["stage", "route"], color="size")
st.plotly_chart(fig_flow, use_container_width=True)

st.subheader("Route Distribution")
route_counts = routed["route"].value_counts().reset_index()
route_counts.columns = ["route", "count"]
st.plotly_chart(px.bar(route_counts, x="route", y="count"), use_container_width=True)

st.subheader("Confidence Distribution")
st.plotly_chart(px.histogram(routed, x="confidence", nbins=30), use_container_width=True)
