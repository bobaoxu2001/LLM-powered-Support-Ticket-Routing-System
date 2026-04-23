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
from llm_support_routing.evaluation import routing_metrics
from llm_support_routing.features import add_weak_labels
from llm_support_routing.models import save_model, train_tfidf_logreg
from llm_support_routing.routing import route_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
    logger.info("Unified table: %d rows", len(unified))

    logger.info("Generating weak labels...")
    labeled = add_weak_labels(unified)

    logger.info("Training TF-IDF + LR classifier...")
    result = train_tfidf_logreg(labeled, "issue_type")
    save_model(result.model, str(MODELS_DIR / "issue_type_tfidf_lr.joblib"))
    logger.info(
        "Held-out accuracy: %.4f | 5-fold CV: %.4f ± %.4f",
        result.accuracy, result.cv_mean, result.cv_std,
    )

    logger.info("Routing sample of 5000 tickets...")
    sample = labeled.sample(min(5000, len(labeled)), random_state=42)
    routed = route_dataframe(sample, result.model)
    metrics = routing_metrics(routed)

    labeled.to_csv(DATA_PROCESSED / "unified_labeled_tickets.csv", index=False)
    routed.to_csv(OUTPUTS_DIR / "routed_tickets.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "routing_metrics.csv", index=False)

    with open(OUTPUTS_DIR / "training_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Issue Type Held-Out Accuracy : {result.accuracy:.4f}\n")
        f.write(f"5-Fold CV Accuracy           : {result.cv_mean:.4f} ± {result.cv_std:.4f}\n\n")
        f.write(result.report)

    logger.info("Pipeline complete.")
    for k, v in metrics.items():
        logger.info("  %s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download Kaggle datasets before processing")
    args = parser.parse_args()
    main(download=args.download)
