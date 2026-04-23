from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import RULE_PATTERNS, RoutingThresholds
from .llm import llm_classify_ticket, llm_resolution_and_escalation, llm_summarize_ticket
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

# Urgency levels that warrant a priority-queue suffix
_PRIORITY_URGENCY = {"high", "critical"}


def _apply_urgency_suffix(route: str, urgency: str) -> str:
    """Append _priority to a queue name for high/critical urgency tickets."""
    if urgency in _PRIORITY_URGENCY:
        return f"{route}_priority"
    return route


def _rule_match(text_lower: str) -> tuple[str, str] | None:
    """Return (queue, issue_type) for the first matching rule pattern, or None."""
    for pattern, queue, issue_type in RULE_PATTERNS:
        if pattern in text_lower:
            return queue, issue_type
    return None


def route_ticket(
    text: str,
    model,
    thresholds: RoutingThresholds = RoutingThresholds(),
    urgency_model=None,
) -> RoutingDecision:
    text_lower = text.lower()

    # 1) Rule-based fast path — driven by RULE_PATTERNS in config
    match = _rule_match(text_lower)
    if match:
        queue, issue_type = match
        urgency = ""
        if urgency_model is not None:
            urgency_labels, _ = predict_with_confidence(urgency_model, [text])
            urgency = str(urgency_labels[0])
            queue = _apply_urgency_suffix(queue, urgency)
        return RoutingDecision(
            route=queue,
            stage="rule_based",
            confidence=0.99,
            metadata={"issue_type": issue_type, "urgency": urgency, "rule": "pattern_match"},
        )

    # 2) ML default routing
    labels, probs = predict_with_confidence(model, [text])
    label, prob = labels[0], float(probs[0])

    urgency = ""
    if urgency_model is not None:
        urgency_labels, _ = predict_with_confidence(urgency_model, [text])
        urgency = str(urgency_labels[0])

    if prob >= thresholds.high_confidence:
        queue = RULE_ROUTING.get(label, "general_support_queue")
        if urgency:
            queue = _apply_urgency_suffix(queue, urgency)
        return RoutingDecision(
            route=queue,
            stage="ml_high_confidence",
            confidence=prob,
            metadata={"issue_type": label, "urgency": urgency},
        )

    # 3) LLM for low-confidence reasoning band
    if prob <= thresholds.low_confidence:
        llm_result = llm_classify_ticket(text)
        if "_llm_error" in llm_result:
            return RoutingDecision(
                route="human_triage_queue",
                stage="human_fallback",
                confidence=prob,
                metadata={"issue_type": label, "urgency": urgency, "reason": "llm_unavailable"},
            )
        issue_type = llm_result.get("issue_type", "other")
        llm_urgency = llm_result.get("urgency", urgency)
        queue = RULE_ROUTING.get(issue_type, "general_support_queue")
        queue = _apply_urgency_suffix(queue, llm_urgency)
        return RoutingDecision(
            route=queue,
            stage="llm_reasoning",
            confidence=prob,
            metadata=llm_result,
        )

    # 4) Human fallback for uncertain middle-confidence band
    return RoutingDecision(
        route="human_triage_queue",
        stage="human_fallback",
        confidence=prob,
        metadata={"issue_type": label, "urgency": urgency},
    )


def route_dataframe(
    df: pd.DataFrame,
    model,
    urgency_model=None,
    thresholds: RoutingThresholds = RoutingThresholds(),
    enrich_human: bool = False,
    summarize_human: bool = False,
) -> pd.DataFrame:
    """Route a DataFrame of tickets.

    Runs ML inference in a single batched call, then invokes the LLM only for
    the low-confidence subset.

    Args:
        urgency_model: when provided, urgency predictions append _priority
            suffix to queues for high/critical tickets.
        thresholds: confidence gates for ML and LLM stages.  Pass an explicit
            RoutingThresholds to apply thresholds from --high-threshold /
            --low-threshold CLI flags; the default matches config defaults.
        enrich_human: when True, calls llm_resolution_and_escalation() and
            llm_summarize_ticket() for every human-fallback ticket.  Adds
            llm_summary, suggested_path, should_escalate, and reason columns.
            Incurs one extra LLM call per human-fallback ticket; keep False
            (default) in high-volume or cost-sensitive runs.
        summarize_human: legacy flag — superseded by enrich_human.  When True
            and enrich_human is False, generates only llm_summary.
    """
    texts = df["text"].tolist()

    # Single batched ML inference call
    all_labels, all_probs = predict_with_confidence(model, texts)

    # Batch urgency inference if model provided
    all_urgency: list[str] = [""] * len(texts)
    if urgency_model is not None:
        urgency_labels, _ = predict_with_confidence(urgency_model, texts)
        all_urgency = [str(u) for u in urgency_labels]

    records = []
    for text, label, prob, urgency in zip(texts, all_labels, all_probs, all_urgency):
        prob = float(prob)
        text_lower = text.lower()

        # 1) Rule-based
        match = _rule_match(text_lower)
        if match:
            queue, issue_type = match
            if urgency:
                queue = _apply_urgency_suffix(queue, urgency)
            records.append({
                "text": text,
                "route": queue,
                "stage": "rule_based",
                "confidence": 0.99,
                "llm_issue_type": issue_type,
                "llm_urgency": urgency,
                "llm_summary": "",
                "suggested_path": "",
                "should_escalate": "",
                "reason": "",
            })
            continue

        # 2) High-confidence ML
        if prob >= thresholds.high_confidence:
            queue = RULE_ROUTING.get(label, "general_support_queue")
            if urgency:
                queue = _apply_urgency_suffix(queue, urgency)
            records.append({
                "text": text,
                "route": queue,
                "stage": "ml_high_confidence",
                "confidence": prob,
                "llm_issue_type": "",
                "llm_urgency": urgency,
                "llm_summary": "",
                "suggested_path": "",
                "should_escalate": "",
                "reason": "",
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
                    "llm_urgency": urgency,
                    "llm_summary": "",
                    "suggested_path": "",
                    "should_escalate": "",
                    "reason": "",
                })
            else:
                issue_type = llm_result.get("issue_type", "other")
                llm_urgency = llm_result.get("urgency", urgency)
                queue = RULE_ROUTING.get(issue_type, "general_support_queue")
                queue = _apply_urgency_suffix(queue, llm_urgency)
                records.append({
                    "text": text,
                    "route": queue,
                    "stage": "llm_reasoning",
                    "confidence": prob,
                    "llm_issue_type": issue_type,
                    "llm_urgency": llm_urgency,
                    "llm_summary": "",
                    "suggested_path": "",
                    "should_escalate": "",
                    "reason": "",
                })
            continue

        # 4) Human fallback — optionally enrich with LLM resolution guidance
        summary = ""
        suggested_path = ""
        should_escalate = ""
        reason = ""
        if enrich_human:
            resolution = llm_resolution_and_escalation(text)
            summary = llm_summarize_ticket(text)
            suggested_path = str(resolution.get("suggested_path", ""))
            should_escalate = str(resolution.get("should_escalate", ""))
            reason = str(resolution.get("reason", ""))
        elif summarize_human:
            summary = llm_summarize_ticket(text)

        records.append({
            "text": text,
            "route": "human_triage_queue",
            "stage": "human_fallback",
            "confidence": prob,
            "llm_issue_type": label,
            "llm_urgency": urgency,
            "llm_summary": summary,
            "suggested_path": suggested_path,
            "should_escalate": should_escalate,
            "reason": reason,
        })

    return pd.DataFrame(records)
