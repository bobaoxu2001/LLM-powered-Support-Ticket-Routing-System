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

from llm_support_routing.config import DATA_PROCESSED, DATA_RAW, MODELS_DIR, OUTPUTS_DIR, RoutingThresholds
from llm_support_routing.data import build_unified_ticket_table, download_kaggle_dataset, ensure_dirs, load_csvs
from llm_support_routing.evaluation import evaluate_on_labeled_set, routing_metrics, threshold_sweep
from llm_support_routing.features import add_weak_labels
from llm_support_routing.models import load_model, save_model, train_tfidf_logreg
from llm_support_routing.routing import route_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_SET_PATH = PROJECT_ROOT / "data" / "eval" / "eval_tickets.csv"


def _first_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return next(iter(frames.values()))


def main(
    download: bool,
    high_threshold: float,
    low_threshold: float,
    enrich_human: bool,
) -> None:
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
        label_note = (
            "NOTE: complexity is a text-length proxy (word count bucketed into "
            "low/medium/high). It is not independently labeled or validated as a "
            "measure of semantic or resolution complexity."
            if label_col == "complexity" else ""
        )
        report_lines.append(
            f"Held-out accuracy : {result.accuracy:.4f}\n"
            f"5-fold CV         : {result.cv_mean:.4f} ± {result.cv_std:.4f}\n"
            f"Label composition : {real_count} real / {weak_count} keyword"
            + (f"\n{label_note}" if label_note else "")
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
            f"Samples              : {eval_result['n_eval_samples']}\n"
            f"ML accuracy          : {eval_result['ml_accuracy']:.4f}\n"
            f"Keyword baseline acc : {eval_result['keyword_baseline_accuracy']:.4f}\n"
            f"ML lift              : {eval_result['ml_lift_over_baseline']:+.4f}\n"
        )
        report_lines.append("--- ML Model Report ---")
        report_lines.append(eval_result["ml_report"])
        report_lines.append("--- Keyword Baseline Report ---")
        report_lines.append(eval_result["keyword_report"])

        pd.DataFrame([{
            "ml_accuracy": eval_result["ml_accuracy"],
            "keyword_baseline_accuracy": eval_result["keyword_baseline_accuracy"],
            "ml_lift": eval_result["ml_lift_over_baseline"],
            "n_eval_samples": eval_result["n_eval_samples"],
        }]).to_csv(OUTPUTS_DIR / "eval_comparison.csv", index=False)
    else:
        logger.info("No eval set found at %s — skipping baseline comparison.", EVAL_SET_PATH)

    # ── Route a sample using issue + urgency models ───────────────────────────
    thresholds = RoutingThresholds(
        high_confidence=high_threshold,
        low_confidence=low_threshold,
    )
    logger.info(
        "Routing sample of 5000 tickets (high_threshold=%.2f, low_threshold=%.2f)...",
        high_threshold, low_threshold,
    )
    if enrich_human:
        logger.info(
            "--enrich-human-with-llm enabled: human-fallback tickets will receive "
            "llm_resolution_and_escalation() + llm_summarize_ticket() calls."
        )
    sample = labeled.sample(min(5000, len(labeled)), random_state=42)
    routed = route_dataframe(
        sample,
        issue_model,
        urgency_model=urgency_model,
        thresholds=thresholds,
        enrich_human=enrich_human,
    )
    metrics = routing_metrics(routed)
    logger.info(
        "human_triage_rate=%.3f (proxy for escalation; true escalation requires "
        "downstream tracking)", metrics["human_triage_rate"],
    )

    # ── Threshold sweep (ML-only, no LLM calls) ───────────────────────────────
    logger.info("Running confidence threshold sweep (analytic estimate, no LLM calls)...")
    sweep_df = threshold_sweep(sample["text"].tolist(), issue_model)
    sweep_df.to_csv(OUTPUTS_DIR / "threshold_sweep.csv", index=False)

    # ── Suggested operating point from threshold sweep (analytic, estimated) ────
    # IMPORTANT: this recommendation is based on ML confidence distribution only.
    # It does not account for routing accuracy, queue SLA, or measured outcomes.
    # To apply a threshold, re-run with --high-threshold and --low-threshold.
    if not sweep_df.empty and sweep_df["is_recommended_threshold"].any():
        recommended = sweep_df[sweep_df["is_recommended_threshold"]].iloc[0]
        logger.info(
            "Threshold sweep recommendation (analytic estimate): "
            "high=%.2f low=%.2f auto-route=%.1f%% human-fallback=%.1f%% "
            "score=%.4f — apply via --high-threshold / --low-threshold",
            recommended["threshold_high"], recommended["threshold_low"],
            100 * recommended["auto_routed_rate_estimated"],
            100 * recommended["human_fallback_rate_estimated"],
            recommended["threshold_recommendation_score"],
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
            "active_threshold_high": high_threshold,
            "active_threshold_low": low_threshold,
            "thresholds_match_recommendation": (
                high_threshold == recommended["threshold_high"]
                and low_threshold == recommended["threshold_low"]
            ),
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
    parser = argparse.ArgumentParser(
        description="Run the support ticket routing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download Kaggle datasets before processing.",
    )
    parser.add_argument(
        "--high-threshold", type=float, default=0.85,
        help=(
            "Confidence threshold above which ML routes directly (no LLM). "
            "Tune using the threshold_sweep.csv recommendation; changes are "
            "not applied automatically from the sweep."
        ),
    )
    parser.add_argument(
        "--low-threshold", type=float, default=0.55,
        help=(
            "Confidence threshold below which the LLM is invoked for classification. "
            "Tickets in the (low, high) band go to human_triage_queue."
        ),
    )
    parser.add_argument(
        "--enrich-human-with-llm", action="store_true",
        help=(
            "Call llm_resolution_and_escalation() + llm_summarize_ticket() for every "
            "human-fallback ticket.  Adds suggested_path, should_escalate, reason, and "
            "llm_summary columns to routed_tickets.csv.  Incurs extra LLM calls "
            "proportional to the human-fallback rate; omit in cost-sensitive runs."
        ),
    )
    args = parser.parse_args()
    main(
        download=args.download,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        enrich_human=args.enrich_human_with_llm,
    )
