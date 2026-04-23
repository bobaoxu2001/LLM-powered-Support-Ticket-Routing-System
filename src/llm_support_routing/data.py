from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import DATA_RAW


DATASETS = {
    "twitter_support": "thoughtvector/customer-support-on-twitter",
    "support_tickets": "suraj520/customer-support-ticket-dataset",
}


def ensure_dirs() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_slug: str, destination: Path) -> None:
    """Download a Kaggle dataset using Kaggle CLI.

    Requires KAGGLE_USERNAME and KAGGLE_KEY in environment.
    """
    destination.mkdir(parents=True, exist_ok=True)
    cmd = f"kaggle datasets download -d {dataset_slug} -p {destination} --unzip"
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


def build_unified_ticket_table(
    twitter_df: pd.DataFrame,
    tickets_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create unified schema across real Twitter conversations + structured ticket data."""
    twitter_df = twitter_df.copy()
    tickets_df = tickets_df.copy()

    twitter_df["subject"] = twitter_df.get("text", "")
    twitter_df["description"] = twitter_df.get("text", "")
    twitter_df["category"] = twitter_df.get("inbound", False).map({True: "customer_message", False: "agent_message"})
    twitter_df["source"] = "twitter_support"

    tickets_df["source"] = "structured_tickets"

    shared_cols = ["subject", "description", "category", "source"]

    twitter_min = twitter_df[[c for c in shared_cols if c in twitter_df.columns]].copy()
    for c in shared_cols:
        if c not in twitter_min.columns:
            twitter_min[c] = ""

    tickets_min = tickets_df[[c for c in shared_cols if c in tickets_df.columns]].copy()
    for c in shared_cols:
        if c not in tickets_min.columns:
            tickets_min[c] = ""

    unified = pd.concat([twitter_min[shared_cols], tickets_min[shared_cols]], ignore_index=True)
    unified = unified.dropna(subset=["description"]).reset_index(drop=True)
    return unified
