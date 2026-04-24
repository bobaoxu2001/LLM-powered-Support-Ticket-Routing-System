from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import DATA_RAW


DATASETS = {
    "twitter_support": "thoughtvector/customer-support-on-twitter",
    "support_tickets": "suraj520/customer-support-ticket-dataset",
}

# Maps raw Ticket Type values (lowercased) from suraj520/customer-support-ticket-dataset
# to our 6-class label schema. Extend as needed when new ticket types appear.
TICKET_TYPE_MAP: dict[str, str] = {
    "billing inquiry": "billing",
    "billing issue": "billing",
    "refund request": "billing",
    "payment issue": "billing",
    "cancellation request": "billing",
    "subscription issue": "billing",
    "technical issue": "technical",
    "technical support": "technical",
    "software bug": "technical",
    "hardware issue": "technical",
    "network problem": "technical",
    "service outage": "technical",
    "account issue": "account",
    "account access": "account",
    "account management": "account",
    "profile update": "account",
    "login issue": "login",
    "password reset": "login",
    "authentication issue": "login",
    "product inquiry": "other",
    "general inquiry": "other",
    "feature request": "other",
    "shipping problem": "other",
    "other": "other",
}

# Maps raw Ticket Priority values (lowercased) to our urgency schema.
TICKET_PRIORITY_MAP: dict[str, str] = {
    "critical": "critical",
    "high": "high",
    "medium": "medium",
    "low": "low",
}


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_slug: str, destination: Path) -> None:
    """Download a Kaggle dataset using Kaggle CLI.

    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment.
    """
    destination.mkdir(parents=True, exist_ok=True)
    cmd = f'kaggle datasets download -d {dataset_slug} -p "{destination}" --unzip'
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(
            f"Failed to download {dataset_slug}. Ensure Kaggle API credentials are configured."
        )


def load_csvs(folder: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for csv_path in folder.glob("*.csv"):
        frames[csv_path.stem] = pd.read_csv(csv_path)
    return frames


def _map_structured_tickets(tickets_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and extract real labels from the structured tickets dataset.

    Uses Ticket Type → category and Ticket Priority → priority when present,
    giving downstream feature engineering genuine ground-truth labels instead
    of keyword heuristics.  Rows without a recognized Ticket Type get
    label_source='weak' so the training loop can treat them separately.
    """
    df = tickets_df.copy()

    # Column mapping: Kaggle column name → our unified schema
    col_aliases = {
        "Ticket Subject": "subject",
        "Ticket Description": "description",
        "Ticket Type": "_raw_ticket_type",
        "Ticket Priority": "_raw_priority",
    }
    for src, dst in col_aliases.items():
        if src in df.columns:
            df[dst] = df[src]

    # Normalize subject / description
    if "subject" not in df.columns:
        df["subject"] = df.get("description", "")
    if "description" not in df.columns:
        df["description"] = df.get("subject", "")

    # Map Ticket Type → category (real label)
    if "_raw_ticket_type" in df.columns:
        mapped = df["_raw_ticket_type"].str.lower().str.strip().map(TICKET_TYPE_MAP)
        df["category"] = mapped.fillna("other")
        df["label_source"] = mapped.notna().map({True: "real", False: "weak"})
    else:
        df["category"] = ""
        df["label_source"] = "weak"

    # Map Ticket Priority → priority (real urgency label)
    if "_raw_priority" in df.columns:
        df["priority"] = df["_raw_priority"].str.lower().str.strip().map(TICKET_PRIORITY_MAP).fillna("")
    else:
        df["priority"] = ""

    df["source"] = "structured_tickets"
    return df


def build_unified_ticket_table(
    twitter_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create unified schema across real Twitter conversations + structured ticket data.

    Only inbound (customer-authored) Twitter messages are included to prevent
    agent-generated text from contaminating training labels.

    Structured tickets carry real Ticket Type / Ticket Priority labels where
    available (label_source='real'); Twitter messages fall back to keyword
    heuristics (label_source='weak').
    """
    twitter_df = twitter_df.copy()

    # Keep only customer-side messages
    if "inbound" in twitter_df.columns:
        twitter_df = twitter_df[twitter_df["inbound"] == True].copy()

    twitter_df["subject"] = twitter_df.get("text", "")
    twitter_df["description"] = twitter_df.get("text", "")
    twitter_df["category"] = "customer_message"
    twitter_df["priority"] = ""
    twitter_df["label_source"] = "weak"
    twitter_df["source"] = "twitter_support"

    tickets_df = _map_structured_tickets(tickets_df)

    shared_cols = ["subject", "description", "category", "priority", "label_source", "source"]

    def _select(df: pd.DataFrame) -> pd.DataFrame:
        out = df[[c for c in shared_cols if c in df.columns]].copy()
        for c in shared_cols:
            if c not in out.columns:
                out[c] = ""
        return out[shared_cols]

    unified = pd.concat([_select(twitter_df), _select(tickets_df)], ignore_index=True)
    unified = unified.dropna(subset=["description"]).reset_index(drop=True)
    # Sort so real-labeled rows come first before deduplication on description.
    # Without this, a weak Twitter row appearing before the same text in the
    # structured dataset would win the keep="first" and the real label would be
    # silently dropped, reducing ground-truth coverage.
    unified = unified.sort_values(
        "label_source", key=lambda s: s.map({"real": 0, "weak": 1}), kind="stable"
    )
    unified = unified.drop_duplicates(subset=["description"], keep="first").reset_index(drop=True)
    return unified
