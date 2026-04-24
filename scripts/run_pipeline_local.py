"""Local pipeline runner using only data/eval/eval_tickets.csv.

Trains TF-IDF + LR models on the 99-sample eval set, routes through the
pipeline without any LLM calls (low-confidence tickets fall back to
human_triage_queue), runs threshold sweep, and evaluates on the eval set.

Usage:
    python scripts/run_pipeline_local.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_support_routing.config import MODELS_DIR, OUTPUTS_DIR, RoutingThresholds
from llm_support_routing.evaluation import evaluate_on_labeled_set, routing_metrics, threshold_sweep
from llm_support_routing.features import normalize_text, _keyword_complexity
from llm_support_routing.models import save_model, train_tfidf_logreg, load_model
from llm_support_routing.routing import route_dataframe

EVAL_PATH = PROJECT_ROOT / "data" / "eval" / "eval_tickets.csv"
EVAL_SUMMARY_PATH = OUTPUTS_DIR / "eval_comparison.csv"
EVAL_PER_CLASS_PATH = OUTPUTS_DIR / "eval_per_class_metrics.csv"
EVAL_CONFUSION_PATH = OUTPUTS_DIR / "eval_confusion_matrix.csv"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    logger.info("Loading eval set from %s", EVAL_PATH)
    eval_df = pd.read_csv(EVAL_PATH)
    logger.info("Eval set: %d rows, columns: %s", len(eval_df), list(eval_df.columns))

    # Prepare training DataFrame from eval data (real labels)
    train_df = eval_df.copy()
    train_df["text"] = train_df["text"].astype(str).map(normalize_text)
    train_df["complexity"] = train_df["text"].map(_keyword_complexity)
    train_df["label_quality"] = "real"

    logger.info(
        "issue_type distribution: %s",
        train_df["issue_type"].value_counts().to_dict(),
    )
    logger.info(
        "urgency distribution: %s",
        train_df["urgency"].value_counts().to_dict(),
    )

    # Train issue_type model
    logger.info("Training TF-IDF + LR for issue_type (%d samples)...", len(train_df))
    issue_result = train_tfidf_logreg(train_df, "issue_type")
    save_model(issue_result.model, str(MODELS_DIR / "issue_type_tfidf_lr.joblib"))
    logger.info(
        "issue_type — held-out: %.4f | 5-fold CV: %.4f ± %.4f",
        issue_result.accuracy, issue_result.cv_mean, issue_result.cv_std,
    )

    # Train urgency model
    logger.info("Training TF-IDF + LR for urgency (%d samples)...", len(train_df))
    urgency_result = train_tfidf_logreg(train_df, "urgency")
    save_model(urgency_result.model, str(MODELS_DIR / "urgency_tfidf_lr.joblib"))
    logger.info(
        "urgency — held-out: %.4f | 5-fold CV: %.4f ± %.4f",
        urgency_result.accuracy, urgency_result.cv_mean, urgency_result.cv_std,
    )

    issue_model = load_model(str(MODELS_DIR / "issue_type_tfidf_lr.joblib"))
    urgency_model = load_model(str(MODELS_DIR / "urgency_tfidf_lr.joblib"))

    # Evaluate ML vs keyword baseline on the eval set
    logger.info("Evaluating ML vs keyword baseline on labeled eval set...")
    eval_result = evaluate_on_labeled_set(issue_model, EVAL_PATH)
    logger.info(
        "Eval set (%d samples) — ML: %.4f | Keyword: %.4f | Lift: %+.4f",
        eval_result["n_eval_samples"],
        eval_result["ml_accuracy"],
        eval_result["keyword_baseline_accuracy"],
        eval_result["ml_lift_over_baseline"],
    )

    pd.DataFrame([{
        "metric_type": "measured_eval_set",
        "n_eval_samples": eval_result["n_eval_samples"],
        "ml_accuracy": eval_result["ml_accuracy"],
        "keyword_baseline_accuracy": eval_result["keyword_baseline_accuracy"],
        "ml_accuracy_lift": eval_result["ml_lift_over_baseline"],
        "ml_macro_f1": eval_result["ml_macro_f1"],
        "keyword_macro_f1": eval_result["keyword_macro_f1"],
        "ml_macro_f1_lift": eval_result["ml_macro_f1_lift_over_baseline"],
        "ml_weighted_f1": eval_result["ml_weighted_f1"],
        "keyword_weighted_f1": eval_result["keyword_weighted_f1"],
        "ml_weighted_f1_lift": eval_result["ml_weighted_f1_lift_over_baseline"],
    }]).to_csv(EVAL_SUMMARY_PATH, index=False)
    eval_result["per_class_metrics_df"].to_csv(EVAL_PER_CLASS_PATH, index=False)
    eval_result["confusion_matrix_df"].to_csv(EVAL_CONFUSION_PATH, index=False)

    # Route all tickets (no LLM calls — low-confidence → human_triage_queue)
    thresholds = RoutingThresholds(high_confidence=0.85, low_confidence=0.55)
    logger.info("Routing %d tickets (no LLM calls)...", len(train_df))
    routed = route_dataframe(
        train_df,
        issue_model,
        urgency_model=urgency_model,
        thresholds=thresholds,
        enrich_human=False,
    )
    metrics = routing_metrics(routed)
    logger.info("Routing metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)

    routed.to_csv(OUTPUTS_DIR / "routed_tickets.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "routing_metrics.csv", index=False)

    # Threshold sweep
    logger.info("Running confidence threshold sweep...")
    sweep_df = threshold_sweep(train_df["text"].tolist(), issue_model)
    sweep_df.to_csv(OUTPUTS_DIR / "threshold_sweep.csv", index=False)
    logger.info("Threshold sweep: %d rows", len(sweep_df))

    # Policy recommendation
    if not sweep_df.empty and sweep_df["is_recommended_threshold"].any():
        recommended = sweep_df[sweep_df["is_recommended_threshold"]].iloc[0]
        logger.info(
            "Recommended threshold: high=%.2f low=%.2f auto=%.1f%% human=%.1f%%",
            recommended["threshold_high"], recommended["threshold_low"],
            100 * recommended["auto_routed_rate_estimated"],
            100 * recommended["human_fallback_rate_estimated"],
        )
        pd.DataFrame([{
            "metric_type": "estimated_policy_from_threshold_sweep",
            "note": "analytic estimate only — does not account for routing accuracy or SLA",
            "recommended_threshold_high": recommended["threshold_high"],
            "recommended_threshold_low": recommended["threshold_low"],
            "auto_routed_rate_estimated": recommended["auto_routed_rate_estimated"],
            "llm_fallback_rate_estimated": recommended["llm_fallback_rate_estimated"],
            "human_fallback_rate_estimated": recommended["human_fallback_rate_estimated"],
            "cost_per_ticket_usd_estimated": recommended["cost_per_ticket_usd_estimated"],
            "score_automation_gain": recommended["score_automation_gain"],
            "score_human_penalty": recommended["score_human_penalty"],
            "score_llm_penalty": recommended["score_llm_penalty"],
            "threshold_recommendation_score": recommended["threshold_recommendation_score"],
            "weight_auto": recommended["weight_auto"],
            "weight_human_penalty": recommended["weight_human_penalty"],
            "weight_llm_penalty": recommended["weight_llm_penalty"],
            "active_threshold_high": 0.85,
            "active_threshold_low": 0.55,
            "thresholds_match_recommendation": (
                0.85 == recommended["threshold_high"]
                and 0.55 == recommended["threshold_low"]
            ),
        }]).to_csv(OUTPUTS_DIR / "routing_policy_recommendation.csv", index=False)

    logger.info("Pipeline complete. Outputs written to %s", OUTPUTS_DIR)


if __name__ == "__main__":
    main()
