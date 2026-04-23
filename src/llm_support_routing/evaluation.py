from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .features import _keyword_issue_type, normalize_text
from .models import predict_with_confidence

# GPT-4.1-mini pricing (2025): $0.40/1M input tokens, $1.60/1M output tokens.
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

    for queue, frac in routed_df["route"].value_counts(normalize=True).items():
        metrics[f"queue_pct_{queue}"] = float(frac)

    return metrics


def evaluate_on_labeled_set(model, eval_path: Path) -> dict[str, object]:
    """Compare ML classifier against keyword-rule baseline on a labeled eval set.

    This is the core measurement that answers: does the ML model add signal
    beyond the keyword heuristics it was trained to approximate?

    Returns a dict with:
      - ml_accuracy, keyword_baseline_accuracy, ml_lift
      - per-class classification reports for both
      - label composition (how many real vs weak labels in the eval set)
    """
    eval_df = pd.read_csv(eval_path)
    if "text" not in eval_df.columns:
        eval_df["text"] = (
            eval_df.get("subject", pd.Series("", index=eval_df.index)).fillna("") + " " +
            eval_df.get("description", pd.Series("", index=eval_df.index)).fillna("")
        )
    eval_df["text"] = eval_df["text"].astype(str).map(normalize_text)

    true_labels = eval_df["issue_type"].tolist()

    # ML model predictions
    ml_preds, _ = predict_with_confidence(model, eval_df["text"].tolist())

    # Keyword-rule baseline predictions (same heuristic as add_weak_labels)
    keyword_preds = [_keyword_issue_type(t) for t in eval_df["text"].tolist()]

    ml_acc = float(accuracy_score(true_labels, ml_preds))
    kw_acc = float(accuracy_score(true_labels, keyword_preds))

    return {
        "n_eval_samples": len(eval_df),
        "ml_accuracy": ml_acc,
        "keyword_baseline_accuracy": kw_acc,
        "ml_lift_over_baseline": ml_acc - kw_acc,
        "ml_report": classification_report(true_labels, ml_preds, zero_division=0),
        "keyword_report": classification_report(true_labels, keyword_preds, zero_division=0),
    }


def threshold_sweep(texts: list[str], model) -> pd.DataFrame:
    """Sweep the high-confidence threshold from 0.50 to 0.95 using ML predictions only.

    For each threshold value reports the auto-route rate (ML stage), the estimated
    LLM fallback rate (low-confidence band using low = threshold * 0.65), the
    average confidence of auto-routed tickets, and the resulting estimated cost
    per ticket.  Use this to pick an operating point on the cost-coverage curve.
    """
    _, probs = predict_with_confidence(model, texts)
    rows = []
    for t_high in [round(v * 0.05 + 0.50, 2) for v in range(10)]:  # 0.50 … 0.95
        t_low = round(t_high * 0.65, 2)
        auto_mask = probs >= t_high
        llm_mask = probs < t_low
        human_mask = (probs >= t_low) & (probs < t_high)

        auto_rate = float(auto_mask.mean())
        llm_rate = float(llm_mask.mean())
        avg_conf = float(probs[auto_mask].mean()) if auto_mask.any() else 0.0
        cost = _INFRA_COST_PER_TICKET_USD + llm_rate * _LLM_COST_PER_CALL_USD

        rows.append({
            "threshold_high": t_high,
            "threshold_low": t_low,
            "auto_routed_rate": auto_rate,
            "llm_fallback_rate": llm_rate,
            "human_fallback_rate": float(human_mask.mean()),
            "avg_confidence_auto": avg_conf,
            "est_cost_per_ticket_usd": cost,
        })
    return pd.DataFrame(rows)
