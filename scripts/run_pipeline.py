from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from llm_support_routing.config import DATA_PROCESSED, DATA_RAW, MODELS_DIR, OUTPUTS_DIR
from llm_support_routing.data import build_unified_ticket_table, download_kaggle_dataset, ensure_dirs, load_csvs
from llm_support_routing.evaluation import routing_metrics
from llm_support_routing.features import add_weak_labels
from llm_support_routing.models import save_model, train_tfidf_logreg
from llm_support_routing.routing import route_dataframe


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
        download_kaggle_dataset("thoughtvector/customer-support-on-twitter", twitter_path)
        download_kaggle_dataset("suraj520/customer-support-ticket-dataset", tickets_path)

    twitter_frames = load_csvs(twitter_path)
    ticket_frames = load_csvs(tickets_path)

    twitter_df = _first_frame(twitter_frames)
    tickets_df = _first_frame(ticket_frames)

    if twitter_df.empty or tickets_df.empty:
        raise RuntimeError(
            "Dataset files were not found. Use --download with configured Kaggle API credentials."
        )

    unified = build_unified_ticket_table(twitter_df, tickets_df)
    labeled = add_weak_labels(unified)

    result = train_tfidf_logreg(labeled, "issue_type")
    save_model(result.model, str(MODELS_DIR / "issue_type_tfidf_lr.joblib"))

    routed = route_dataframe(labeled.sample(min(5000, len(labeled)), random_state=42), result.model)
    metrics = routing_metrics(routed)

    labeled.to_csv(DATA_PROCESSED / "unified_labeled_tickets.csv", index=False)
    routed.to_csv(OUTPUTS_DIR / "routed_tickets.csv", index=False)
    pd.DataFrame([metrics]).to_csv(OUTPUTS_DIR / "routing_metrics.csv", index=False)

    with open(OUTPUTS_DIR / "training_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Issue Type Accuracy: {result.accuracy:.4f}\n\n")
        f.write(result.report)

    print("Pipeline complete.")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download Kaggle datasets before processing")
    args = parser.parse_args()
    main(download=args.download)
