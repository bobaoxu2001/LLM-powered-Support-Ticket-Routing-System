from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import RoutingThresholds
from .llm import llm_classify_ticket, llm_summarize_ticket
from .models import predict_with_confidence


@dataclass(slots=True)
class RoutingDecision:
    route: str
    stage: str
    confidence: float
    metadata: dict[str, Any]


RULE_ROUTING = {
    "billing": "billing_queue",
    "ads": "ads_ops_queue",
    "login": "identity_support_queue",
    "technical": "technical_support_queue",
    "account": "account_management_queue",
    "other": "general_support_queue",
}


def route_ticket(text: str, model, thresholds: RoutingThresholds = RoutingThresholds()) -> RoutingDecision:
    text_lower = text.lower()

    # 1) Rule-based fast path
    if "refund" in text_lower or "double charged" in text_lower:
        return RoutingDecision(
            route="billing_queue",
            stage="rule_based",
            confidence=0.99,
            metadata={"rule": "refund_or_double_charge"},
        )

    # 2) ML default routing
    labels, probs = predict_with_confidence(model, [text])
    label, prob = labels[0], float(probs[0])

    if prob >= thresholds.high_confidence:
        return RoutingDecision(
            route=RULE_ROUTING.get(label, "general_support_queue"),
            stage="ml_high_confidence",
            confidence=prob,
            metadata={"issue_type": label},
        )

    # 3) LLM for low-confidence reasoning band
    if prob <= thresholds.low_confidence:
        llm_result = llm_classify_ticket(text)
        # LLM call failed — escalate to human rather than silently misrouting
        if "_llm_error" in llm_result:
            return RoutingDecision(
                route="human_triage_queue",
                stage="human_fallback",
                confidence=prob,
                metadata={"issue_type": label, "reason": "llm_unavailable"},
            )
        issue_type = llm_result.get("issue_type", "other")
        return RoutingDecision(
            route=RULE_ROUTING.get(issue_type, "general_support_queue"),
            stage="llm_reasoning",
            confidence=prob,
            metadata=llm_result,
        )

    # 4) Human fallback for uncertain middle-confidence band
    return RoutingDecision(
        route="human_triage_queue",
        stage="human_fallback",
        confidence=prob,
        metadata={"issue_type": label},
    )


def route_dataframe(df: pd.DataFrame, model, summarize_human: bool = False) -> pd.DataFrame:
    """Route a DataFrame of tickets.

    Runs ML inference in a single batched call, then invokes the LLM only for
    the low-confidence subset. Set summarize_human=True to generate LLM summaries
    for tickets sent to human agents (adds latency proportional to human-fallback rate).
    """
    texts = df["text"].tolist()
    thresholds = RoutingThresholds()

    # Single batched ML inference call
    all_labels, all_probs = predict_with_confidence(model, texts)

    records = []
    for text, label, prob in zip(texts, all_labels, all_probs):
        prob = float(prob)
        text_lower = text.lower()

        # 1) Rule-based
        if "refund" in text_lower or "double charged" in text_lower:
            records.append({
                "text": text,
                "route": "billing_queue",
                "stage": "rule_based",
                "confidence": 0.99,
                "llm_issue_type": "",
                "llm_urgency": "",
                "llm_summary": "",
            })
            continue

        # 2) High-confidence ML
        if prob >= thresholds.high_confidence:
            records.append({
                "text": text,
                "route": RULE_ROUTING.get(label, "general_support_queue"),
                "stage": "ml_high_confidence",
                "confidence": prob,
                "llm_issue_type": "",
                "llm_urgency": "",
                "llm_summary": "",
            })
            continue

        # 3) LLM for low-confidence
        if prob <= thresholds.low_confidence:
            llm_result = llm_classify_ticket(text)
            if "_llm_error" in llm_result:
                records.append({
                    "text": text,
                    "route": "human_triage_queue",
                    "stage": "human_fallback",
                    "confidence": prob,
                    "llm_issue_type": "",
                    "llm_urgency": "",
                    "llm_summary": "",
                })
            else:
                issue_type = llm_result.get("issue_type", "other")
                records.append({
                    "text": text,
                    "route": RULE_ROUTING.get(issue_type, "general_support_queue"),
                    "stage": "llm_reasoning",
                    "confidence": prob,
                    "llm_issue_type": issue_type,
                    "llm_urgency": llm_result.get("urgency", ""),
                    "llm_summary": "",
                })
            continue

        # 4) Human fallback — optionally generate a summary for the agent
        summary = llm_summarize_ticket(text) if summarize_human else ""
        records.append({
            "text": text,
            "route": "human_triage_queue",
            "stage": "human_fallback",
            "confidence": prob,
            "llm_issue_type": label,
            "llm_urgency": "",
            "llm_summary": summary,
        })

    return pd.DataFrame(records)
