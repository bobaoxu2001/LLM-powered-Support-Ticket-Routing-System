from __future__ import annotations

import pandas as pd


def routing_metrics(routed_df: pd.DataFrame) -> dict[str, float]:
    total = len(routed_df)
    escalation_rate = float((routed_df["route"] == "human_triage_queue").mean()) if total else 0.0
    llm_rate = float((routed_df["stage"] == "llm_reasoning").mean()) if total else 0.0

    # rough cost model in USD/ticket
    cost_per_ticket = 0.001 + (0.02 * llm_rate)

    return {
        "tickets": float(total),
        "escalation_rate": escalation_rate,
        "llm_invocation_rate": llm_rate,
        "cost_per_ticket_usd": cost_per_ticket,
    }
