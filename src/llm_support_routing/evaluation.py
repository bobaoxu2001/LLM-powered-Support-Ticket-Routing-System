from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

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

    # human_triage_rate: fraction of tickets sent to human_triage_queue.
    # This is a routing-system metric, not a true escalation rate.  True
    # escalation would require tracking which human-triaged tickets were
    # subsequently escalated to a senior agent or engineering team — data
    # not available at routing time.  escalation_rate is retained as a
    # backward-compatible alias and labeled as a proxy.
    human_triage_rate = float((routed_df["route"] == "human_triage_queue").mean())
    llm_rate = float((routed_df["stage"] == "llm_reasoning").mean())
    rule_rate = float((routed_df["stage"] == "rule_based").mean())
    ml_rate = float((routed_df["stage"] == "ml_high_confidence").mean())
    human_fallback_rate = float((routed_df["stage"] == "human_fallback").mean())
    avg_confidence = float(routed_df["confidence"].mean())

    # Estimated cost terms (not measured runtime billing).
    cost_per_ticket = _INFRA_COST_PER_TICKET_USD + llm_rate * _LLM_COST_PER_CALL_USD

    metrics: dict[str, float] = {
        "tickets": float(total),
        "human_triage_rate": human_triage_rate,
        "escalation_rate": human_triage_rate,  # proxy alias — see docstring above
        "llm_invocation_rate": llm_rate,
        "rule_based_rate": rule_rate,
        "ml_high_confidence_rate": ml_rate,
        "human_fallback_rate": human_fallback_rate,
        "avg_routing_confidence": avg_confidence,
        "cost_per_ticket_usd_estimated": cost_per_ticket,
        "infra_cost_per_ticket_usd_estimated": _INFRA_COST_PER_TICKET_USD,
        "llm_cost_per_call_usd_estimated": _LLM_COST_PER_CALL_USD,
    }

    for queue, frac in routed_df["route"].value_counts(normalize=True).items():
        metrics[f"queue_pct_{queue}"] = float(frac)

    return metrics


def _classification_report_df(true_labels: list[str], preds: list[str], model_name: str) -> pd.DataFrame:
    report = classification_report(true_labels, preds, zero_division=0, output_dict=True)
    rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            rows.append({
                "model": model_name,
                "label": label,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1-score", 0.0)),
                "support": float(metrics.get("support", 0.0)),
            })
    return pd.DataFrame(rows)


def _confusion_matrix_df(true_labels: list[str], preds: list[str], model_name: str) -> pd.DataFrame:
    labels = sorted(set(true_labels) | set(preds))
    matrix = confusion_matrix(true_labels, preds, labels=labels)
    cm = pd.DataFrame(matrix, index=labels, columns=labels).reset_index(names="actual_label")
    cm = cm.melt(id_vars="actual_label", var_name="predicted_label", value_name="count")
    cm.insert(0, "model", model_name)
    return cm


def evaluate_on_labeled_set(model, eval_path: Path) -> dict[str, object]:
    """Compare ML classifier against keyword-rule baseline on a labeled eval set.

    Returns both summary metrics and rich artifacts for downstream reporting:
      - accuracy + macro/weighted F1 for ML and baseline
      - per-class precision/recall/F1 tables
      - confusion-matrix tables
    """
    eval_df = pd.read_csv(eval_path)
    if "text" not in eval_df.columns:
        eval_df["text"] = (
            eval_df.get("subject", pd.Series("", index=eval_df.index)).fillna("") + " " +
            eval_df.get("description", pd.Series("", index=eval_df.index)).fillna("")
        )
    eval_df["text"] = eval_df["text"].astype(str).map(normalize_text)

    true_labels = eval_df["issue_type"].tolist()

    ml_preds, _ = predict_with_confidence(model, eval_df["text"].tolist())
    keyword_preds = [_keyword_issue_type(t) for t in eval_df["text"].tolist()]

    ml_acc = float(accuracy_score(true_labels, ml_preds))
    kw_acc = float(accuracy_score(true_labels, keyword_preds))

    ml_macro_f1 = float(f1_score(true_labels, ml_preds, average="macro", zero_division=0))
    kw_macro_f1 = float(f1_score(true_labels, keyword_preds, average="macro", zero_division=0))
    ml_weighted_f1 = float(f1_score(true_labels, ml_preds, average="weighted", zero_division=0))
    kw_weighted_f1 = float(f1_score(true_labels, keyword_preds, average="weighted", zero_division=0))

    per_class_df = pd.concat([
        _classification_report_df(true_labels, ml_preds.tolist(), "ml_tfidf_lr"),
        _classification_report_df(true_labels, keyword_preds, "keyword_baseline"),
    ], ignore_index=True)

    cm_df = pd.concat([
        _confusion_matrix_df(true_labels, ml_preds.tolist(), "ml_tfidf_lr"),
        _confusion_matrix_df(true_labels, keyword_preds, "keyword_baseline"),
    ], ignore_index=True)

    return {
        "n_eval_samples": len(eval_df),
        "ml_accuracy": ml_acc,
        "keyword_baseline_accuracy": kw_acc,
        "ml_lift_over_baseline": ml_acc - kw_acc,
        "ml_macro_f1": ml_macro_f1,
        "keyword_macro_f1": kw_macro_f1,
        "ml_macro_f1_lift_over_baseline": ml_macro_f1 - kw_macro_f1,
        "ml_weighted_f1": ml_weighted_f1,
        "keyword_weighted_f1": kw_weighted_f1,
        "ml_weighted_f1_lift_over_baseline": ml_weighted_f1 - kw_weighted_f1,
        "ml_report": classification_report(true_labels, ml_preds, zero_division=0),
        "keyword_report": classification_report(true_labels, keyword_preds, zero_division=0),
        "per_class_metrics_df": per_class_df,
        "confusion_matrix_df": cm_df,
    }


def threshold_sweep(
    texts: list[str],
    model,
    *,
    auto_weight: float = 1.0,
    human_penalty: float = 1.25,
    llm_penalty: float = 0.5,
) -> pd.DataFrame:
    """Sweep confidence thresholds and expose cost/coverage tradeoffs.

    Stage shares are estimated from ML confidence only (no live LLM calls).
    The recommendation score is a weighted policy objective:

        score = auto_weight * auto_rate
              - human_penalty * human_fallback_rate
              - llm_penalty   * llm_fallback_rate

    Default weights reflect a policy that values automation (1.0) and
    penalises human triage (1.25×) more than LLM invocations (0.5×).
    Adjust to match your SLA and cost priorities:
      - Increase human_penalty if human queue capacity is the bottleneck.
      - Increase llm_penalty if LLM API cost is the primary constraint.

    This score is an analytic estimate only.  It does not account for
    routing accuracy, queue SLA, or measured post-routing outcomes.
    Apply recommended thresholds explicitly via --high-threshold /
    --low-threshold in run_pipeline.py; they are not applied automatically.
    """
    _SWEEP_COLUMNS = [
        "threshold_high", "threshold_low",
        "auto_routed_rate_estimated", "llm_fallback_rate_estimated", "human_fallback_rate_estimated",
        "avg_confidence_auto", "cost_per_ticket_usd_estimated",
        "score_automation_gain", "score_human_penalty", "score_llm_penalty",
        "threshold_recommendation_score", "weight_auto", "weight_human_penalty", "weight_llm_penalty",
        "auto_routed_rate", "llm_fallback_rate", "human_fallback_rate",
        "est_cost_per_ticket_usd", "is_recommended_threshold",
    ]
    if not texts:
        return pd.DataFrame(columns=_SWEEP_COLUMNS)

    _, probs = predict_with_confidence(model, texts)
    rows = []
    for t_high in [round(v * 0.05 + 0.50, 2) for v in range(10)]:  # 0.50 … 0.95
        t_low = round(t_high * 0.65, 2)
        auto_mask = probs >= t_high
        llm_mask = probs < t_low
        human_mask = (probs >= t_low) & (probs < t_high)

        auto_rate = float(auto_mask.mean())
        llm_rate = float(llm_mask.mean())
        human_rate = float(human_mask.mean())
        avg_conf = float(probs[auto_mask].mean()) if auto_mask.any() else 0.0
        cost = _INFRA_COST_PER_TICKET_USD + llm_rate * _LLM_COST_PER_CALL_USD

        automation_gain   = auto_weight * auto_rate
        human_cost        = human_penalty * human_rate
        llm_cost          = llm_penalty * llm_rate
        score             = automation_gain - human_cost - llm_cost

        rows.append({
            "threshold_high": t_high,
            "threshold_low": t_low,
            "auto_routed_rate_estimated": auto_rate,
            "llm_fallback_rate_estimated": llm_rate,
            "human_fallback_rate_estimated": human_rate,
            "avg_confidence_auto": avg_conf,
            "cost_per_ticket_usd_estimated": cost,
            # Score breakdown — lets operators see what is driving the recommendation
            "score_automation_gain": automation_gain,
            "score_human_penalty": human_cost,
            "score_llm_penalty": llm_cost,
            "threshold_recommendation_score": score,
            # Sweep weights used (for reproducibility)
            "weight_auto": auto_weight,
            "weight_human_penalty": human_penalty,
            "weight_llm_penalty": llm_penalty,
        })

    sweep = pd.DataFrame(rows)
    # Backward-compatible aliases for earlier artifact consumers/tests.
    sweep["auto_routed_rate"] = sweep["auto_routed_rate_estimated"]
    sweep["llm_fallback_rate"] = sweep["llm_fallback_rate_estimated"]
    sweep["human_fallback_rate"] = sweep["human_fallback_rate_estimated"]
    sweep["est_cost_per_ticket_usd"] = sweep["cost_per_ticket_usd_estimated"]

    best_idx = int(sweep["threshold_recommendation_score"].idxmax())
    sweep["is_recommended_threshold"] = False
    sweep.loc[best_idx, "is_recommended_threshold"] = True
    return sweep
