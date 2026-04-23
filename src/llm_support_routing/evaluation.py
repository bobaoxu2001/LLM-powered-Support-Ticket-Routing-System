from __future__ import annotations

import pandas as pd


# GPT-4.1-mini pricing (as of 2025): $0.40/1M input tokens, $1.60/1M output tokens.
# Assuming ~200 input tokens + ~50 output tokens per LLM classify call.
_LLM_COST_PER_CALL_USD = (200 * 0.40 + 50 * 1.60) / 1_000_000
_INFRA_COST_PER_TICKET_USD = 0.0001


def routing_metrics(routed_df: pd.DataFrame) -> dict[str, float]:
    total = len(routed_df)
    if total == 0:
        return {"tickets": 0.0}

    escalation_rate = float((routed_df["route"] == "human_triage_queue").mean())
    llm_rate = float((routed_df["stage"] == "llm_reasoning").mean())
    rule_rate = float((routed_df["stage"] == "rule_based").mean())
    ml_rate = float((routed_df["stage"] == "ml_high_confidence").mean())
    human_rate = float((routed_df["stage"] == "human_fallback").mean())
    avg_confidence = float(routed_df["confidence"].mean())

    # Cost model: infra baseline + LLM token cost only for tickets that hit the LLM
    cost_per_ticket = _INFRA_COST_PER_TICKET_USD + llm_rate * _LLM_COST_PER_CALL_USD

    metrics: dict[str, float] = {
        "tickets": float(total),
        "escalation_rate": escalation_rate,
        "llm_invocation_rate": llm_rate,
        "rule_based_rate": rule_rate,
        "ml_high_confidence_rate": ml_rate,
        "human_fallback_rate": human_rate,
        "avg_routing_confidence": avg_confidence,
        "cost_per_ticket_usd": cost_per_ticket,
    }

    # Per-queue distribution
    for queue, frac in routed_df["route"].value_counts(normalize=True).items():
        metrics[f"queue_pct_{queue}"] = float(frac)

    return metrics
