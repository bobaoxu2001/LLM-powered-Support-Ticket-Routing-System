from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .config import RoutingThresholds
from .llm import llm_classify_ticket
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
        issue_type = llm_result.get("issue_type", "other")
        return RoutingDecision(
            route=RULE_ROUTING.get(issue_type, "general_support_queue"),
            stage="llm_reasoning",
            confidence=prob,
            metadata=llm_result,
        )

    # 4) Human fallback for uncertain middle-confidence
    return RoutingDecision(
        route="human_triage_queue",
        stage="human_fallback",
        confidence=prob,
        metadata={"issue_type": label},
    )


def route_dataframe(df: pd.DataFrame, model) -> pd.DataFrame:
    decisions = [route_ticket(text, model) for text in df["text"].tolist()]
    return pd.DataFrame(
        {
            "text": df["text"],
            "route": [d.route for d in decisions],
            "stage": [d.stage for d in decisions],
            "confidence": [d.confidence for d in decisions],
        }
    )
