from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from llm_support_routing.config import DATA_PROCESSED, DATA_RAW, MODELS_DIR, OUTPUTS_DIR
from llm_support_routing.data import build_unified_ticket_table, download_kaggle_dataset, ensure_dirs, load_csvs
from llm_support_routing.evaluation import evaluate_on_labeled_set, routing_metrics, threshold_sweep
from llm_support_routing.features import add_weak_labels
from llm_support_routing.models import load_model, save_model, train_tfidf_logreg
from llm_support_routing.routing import route_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_tickets.csv"
EVAL_SUMMARY_PATH = OUTPUTS_DIR / "eval_comparison.csv"
EVAL_PER_CLASS_PATH = OUTPUTS_DIR / "eval_per_class_metrics.csv"
EVAL_CONFUSION_PATH = OUTPUTS_DIR / "eval_confusion_matrix.csv"


def _first_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return next(iter(frames.values()))


def main(download: bool) -> None:
    ensure_dirs()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    twitter_path = DATA_RAW / "twitter_support"
    tickets_path = DATA_RAW / "support_tickets"

    if download:
        logger.info("Downloading Kaggle datasets...")
        download_kaggle_dataset("thoughtvector/customer-support-on-twitter", twitter_path)
        download_kaggle_dataset("suraj520/customer-support-ticket-dataset", tickets_path)

    logger.info("Loading raw CSVs...")
    twitter_frames = load_csvs(twitter_path)
    ticket_frames = load_csvs(tickets_path)

    twitter_df = _first_frame(twitter_frames)
    tickets_df = _first_frame(ticket_frames)

    if twitter_df.empty or tickets_df.empty:
        raise RuntimeError(
            "Dataset files were not found. Use --download with configured Kaggle API credentials."
        )

    logger.info("Building unified ticket table...")
    unified = build_unified_ticket_table(twitter_df, tickets_df)
    logger.info("Unified table: %d rows (after inbound filter + dedup)", len(unified))

    logger.info("Applying labels (real where available, keyword fallback elsewhere)...")
    labeled = add_weak_labels(unified)

    real_count = int((labeled.get("label_quality", pd.Series()) == "real").sum())
    weak_count = int((labeled.get("label_quality", pd.Series()) == "weak").sum())
    logger.info(
        "Label composition: %d real labels (%.1f%%) | %d keyword labels (%.1f%%)",
        real_count, 100 * real_count / max(len(labeled), 1),
        weak_count, 100 * weak_count / max(len(labeled), 1),
    )

    labeled.to_csv(DATA_PROCESSED / "unified_labeled_tickets.csv", index=False)

    # ── Train one classifier per label dimension ──────────────────────────────
    report_lines: list[str] = []

    for label_col, model_stem in [
        ("issue_type", "issue_type_tfidf_lr"),
        ("urgency",    "urgency_tfidf_lr"),
        ("complexity", "complexity_tfidf_lr"),
    ]:
        logger.info("Training TF-IDF + LR for %s...", label_col)
        result = train_tfidf_logreg(labeled, label_col)
        save_model(result.model, str(MODELS_DIR / f"{model_stem}.joblib"))
        logger.info(
            "%s — held-out: %.4f | 5-fold CV: %.4f ± %.4f",
            label_col, result.accuracy, result.cv_mean, result.cv_std,
        )
        report_lines.append(f"=== {label_col} ===")
        report_lines.append(
            f"Held-out accuracy : {result.accuracy:.4f}\n"
            f"5-fold CV         : {result.cv_mean:.4f} ± {result.cv_std:.4f}\n"
            f"Label composition : {real_count} real / {weak_count} keyword"
        )
        report_lines.append(result.report)

    issue_model = load_model(str(MODELS_DIR / "issue_type_tfidf_lr.joblib"))
    urgency_model = load_model(str(MODELS_DIR / "urgency_tfidf_lr.joblib"))

    # ── Eval set: ML vs keyword baseline comparison ───────────────────────────
    if EVAL_SET_PATH.exists():
        logger.info("Evaluating ML model vs keyword baseline on labeled eval set...")
        eval_result = evaluate_on_labeled_set(issue_model, EVAL_SET_PATH)
        logger.info(
            "Eval set (%d samples) — ML: %.4f | Keyword baseline: %.4f | Lift: %+.4f",
            eval_result["n_eval_samples"],
            eval_result["ml_accuracy"],
            eval_result["keyword_baseline_accuracy"],
            eval_result["ml_lift_over_baseline"],
        )
        report_lines.append("=== Eval Set: ML vs Keyword Baseline ===")
        report_lines.append(
            f"Samples                 : {eval_result['n_eval_samples']}\n"
            f"ML accuracy             : {eval_result['ml_accuracy']:.4f}\n"
            f"Keyword baseline acc    : {eval_result['keyword_baseline_accuracy']:.4f}\n"
            f"ML accuracy lift        : {eval_result['ml_lift_over_baseline']:+.4f}\n"
            f"ML macro-F1             : {eval_result['ml_macro_f1']:.4f}\n"
            f"Keyword baseline macro-F1: {eval_result['keyword_macro_f1']:.4f}\n"
            f"ML macro-F1 lift        : {eval_result['ml_macro_f1_lift_over_baseline']:+.4f}\n"
        )
        report_lines.append("--- ML Model Report ---")
        report_lines.append(eval_result["ml_report"])
        report_lines.append("--- Keyword Baseline Report ---")
        report_lines.append(eval_result["keyword_report"])

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
    else:
        logger.info("No eval set found at %s — skipping baseline comparison.", EVAL_SET_PATH)
        # Avoid stale eval artifacts from previous runs appearing as fresh results.
        for stale in (EVAL_SUMMARY_PATH, EVAL_PER_CLASS_PATH, EVAL_CONFUSION_PATH):
            if stale.exists():
                stale.unlink()
                logger.info("Removed stale eval artifact: %s", stale.name)

    # ── Route a sample using issue + urgency models ───────────────────────────
    logger.info("Routing sample of 5000 tickets...")
    sample = labeled.sample(min(5000, len(labeled)), random_state=42)
    routed = route_dataframe(sample, issue_model, urgency_model=urgency_model)
    metrics = routing_metrics(routed)

    # ── Threshold sweep (ML-only, no LLM calls) ───────────────────────────────
    logger.info("Running confidence threshold sweep...")
    sweep_df = threshold_sweep(sample["text"].tolist(), issue_model)
    sweep_df.to_csv(OUTPUTS_DIR / "threshold_sweep.csv", index=False)

    # ── Suggested operating point from threshold sweep (estimated metrics) ───────
    if not sweep_df.empty and sweep_df["is_recommended_threshold"].any():
        recommended = sweep_df[sweep_df["is_recommended_threshold"]].iloc[0]
        pd.DataFrame([{
            "metric_type": "estimated_policy_from_threshold_sweep",
            "recommended_threshold_high": recommended["threshold_high"],
            "recommended_threshold_low": recommended["threshold_low"],
            "auto_routed_rate_estimated": recommended["auto_routed_rate_estimated"],
            "llm_fallback_rate_estimated": recommended["llm_fallback_rate_estimated"],
            "human_fallback_rate_estimated": recommended["human_fallback_rate_estimated"],
            "cost_per_ticket_usd_estimated": recommended["cost_per_ticket_usd_estimated"],
            "threshold_recommendation_score": recommended["threshold_recommendation_score"],
        }]).to_csv(OUTPUTS_DIR / "routing_policy_recommendation.csv", index=False)
    else:
        logger.info("Threshold sweep did not produce a recommendation (empty sample).")

    # ── Persist outputs ───────────────────────────────────────────────────────
    routed.to_csv(OUTPUTS_DIR / "routed_tickets.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "routing_metrics.csv", index=False)

    with open(OUTPUTS_DIR / "training_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("Pipeline complete.")
    for k, v in metrics.items():
        if not k.startswith("queue_pct_"):
            logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download Kaggle datasets before processing")
    args = parser.parse_args()
    main(download=args.download)
