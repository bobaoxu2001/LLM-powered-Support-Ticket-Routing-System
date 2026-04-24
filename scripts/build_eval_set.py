"""Build an expanded eval set from structured Kaggle ticket metadata.

Label source: Ticket Type metadata from suraj520/customer-support-ticket-dataset.
Labels are NOT derived from keyword matching or model predictions.
Labels are NOT manually reviewed; they come from the Ticket Type column.

Dataset note:
  The suraj520 dataset is synthetic. All Ticket Descriptions contain a
  '{product_purchased}' placeholder (never substituted) and random additional
  text. Ticket Subject and Ticket Type assignments are random and do not
  reflect the actual description content. This means text-label alignment
  in the Kaggle-derived rows is limited.
  eval_label_source = 'metadata_ticket_type' identifies these rows; results
  should be interpreted as evaluation on metadata-labeled data, not a
  manually reviewed gold standard.

Output:
    data/eval/eval_tickets.csv          — merged eval set (300+ rows)
    data/eval/eval_set_summary.csv      — label distribution and source summary

Usage:
    python scripts/build_eval_set.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_support_routing.data import TICKET_PRIORITY_MAP, TICKET_TYPE_MAP
from llm_support_routing.features import normalize_text

RAW_TICKETS = PROJECT_ROOT / "data" / "raw" / "support_tickets" / "customer_support_tickets.csv"
ORIG_EVAL   = PROJECT_ROOT / "data" / "eval" / "eval_tickets.csv"
OUT_EVAL    = PROJECT_ROOT / "data" / "eval" / "eval_tickets.csv"
OUT_SUMMARY = PROJECT_ROOT / "data" / "eval" / "eval_set_summary.csv"

# Rows per issue_type to sample from Kaggle (keeps class imbalance manageable)
ROWS_PER_CLASS = 100
# Minimum cleaned-text length to accept a Kaggle row
MIN_TEXT_LEN = 100
# Random seed for reproducibility
SEED = 42


def _clean_description(raw: str) -> str:
    """Replace synthetic placeholders and normalise whitespace."""
    t = re.sub(r"\{product_purchased\}", "the product", str(raw))
    t = re.sub(r"\{error_message\}", "the error", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_kaggle_rows(path: Path) -> pd.DataFrame:
    """Extract and label rows from the structured Kaggle ticket dataset.

    Returns a DataFrame with columns:
        text, issue_type, urgency, eval_label_source, raw_ticket_type, raw_priority
    """
    print(f"Loading Kaggle structured tickets from {path.relative_to(PROJECT_ROOT)} …")
    raw = pd.read_csv(path)
    print(f"  Raw rows: {len(raw)}")

    # Build combined text: Subject + cleaned Description
    raw["_clean_desc"] = raw["Ticket Description"].map(_clean_description)
    raw["_text"] = (
        raw["Ticket Subject"].fillna("").str.strip()
        + ". "
        + raw["_clean_desc"]
    ).str.strip()

    # Filter: minimum length
    raw = raw[raw["_text"].str.len() >= MIN_TEXT_LEN].copy()
    print(f"  After length filter (>= {MIN_TEXT_LEN} chars): {len(raw)}")

    # Map Ticket Type → issue_type
    raw["_tt_lower"] = raw["Ticket Type"].str.lower().str.strip()
    raw["issue_type"] = raw["_tt_lower"].map(TICKET_TYPE_MAP)
    unmapped = raw["issue_type"].isna().sum()
    if unmapped:
        warnings.warn(f"{unmapped} rows have unmapped Ticket Type — dropping them")
    raw = raw.dropna(subset=["issue_type"]).copy()

    # Map Ticket Priority → urgency
    raw["urgency"] = (
        raw["Ticket Priority"]
        .str.lower()
        .str.strip()
        .map(TICKET_PRIORITY_MAP)
        .fillna("medium")
    )

    # Deduplicate on normalised text
    raw["_norm"] = raw["_text"].map(normalize_text)
    raw = raw.drop_duplicates(subset=["_norm"]).copy()
    print(f"  After dedup on normalised text: {len(raw)}")

    # Sample up to ROWS_PER_CLASS per issue_type
    samples = []
    for it in raw["issue_type"].unique():
        chunk = raw[raw["issue_type"] == it]
        n = min(ROWS_PER_CLASS, len(chunk))
        samples.append(chunk.sample(n, random_state=SEED))
        print(f"  Sampled {n:>3} rows for issue_type='{it}'")

    df = pd.concat(samples, ignore_index=True)

    return pd.DataFrame({
        "text":              df["_text"].values,
        "issue_type":        df["issue_type"].values,
        "urgency":           df["urgency"].values,
        "eval_label_source": "metadata_ticket_type",
        "raw_ticket_type":   df["Ticket Type"].values,
        "raw_priority":      df["Ticket Priority"].values,
    })


def load_original_eval(path: Path) -> pd.DataFrame:
    """Load the existing manually-written eval set and tag it."""
    df = pd.read_csv(path)
    if "eval_label_source" not in df.columns:
        df["eval_label_source"] = "manually_written"
    if "raw_ticket_type" not in df.columns:
        df["raw_ticket_type"] = ""
    if "raw_priority" not in df.columns:
        df["raw_priority"] = df.get("urgency", "")
    print(f"Loaded original eval set: {len(df)} rows")
    print(f"  issue_type distribution: {df['issue_type'].value_counts().to_dict()}")
    return df


def merge_and_dedup(original: pd.DataFrame, kaggle: pd.DataFrame) -> pd.DataFrame:
    """Combine original and Kaggle rows; deduplicate on normalised text."""
    orig_norm = set(original["text"].map(normalize_text))
    # Drop Kaggle rows whose normalised text already exists in original
    kaggle["_norm"] = kaggle["text"].map(normalize_text)
    kaggle_new = kaggle[~kaggle["_norm"].isin(orig_norm)].drop(columns=["_norm"])
    print(f"\nKaggle rows after overlap dedup: {len(kaggle_new)}")

    # Align columns
    all_cols = ["text", "issue_type", "urgency", "eval_label_source",
                "raw_ticket_type", "raw_priority"]
    for col in all_cols:
        if col not in original.columns:
            original[col] = ""
        if col not in kaggle_new.columns:
            kaggle_new[col] = ""

    merged = pd.concat(
        [original[all_cols], kaggle_new[all_cols]],
        ignore_index=True,
    )
    return merged


def write_summary(df: pd.DataFrame, path: Path) -> None:
    rows = []
    for src in df["eval_label_source"].unique():
        sub = df[df["eval_label_source"] == src]
        for it in df["issue_type"].unique():
            n = int((sub["issue_type"] == it).sum())
            rows.append({
                "eval_label_source": src,
                "issue_type": it,
                "count": n,
            })
    summary = pd.DataFrame(rows).sort_values(["eval_label_source", "issue_type"])
    summary.to_csv(path, index=False)

    print(f"\nEval set summary written to {path.relative_to(PROJECT_ROOT)}")
    print(summary.to_string(index=False))


def main() -> None:
    if not RAW_TICKETS.exists():
        print(
            f"ERROR: {RAW_TICKETS.relative_to(PROJECT_ROOT)} not found.\n"
            "Run: python scripts/run_pipeline.py --download"
        )
        sys.exit(1)

    original = load_original_eval(ORIG_EVAL)
    kaggle   = build_kaggle_rows(RAW_TICKETS)

    merged = merge_and_dedup(original, kaggle)

    print(f"\nFinal eval set: {len(merged)} rows")
    print("issue_type distribution:")
    print(merged["issue_type"].value_counts().to_string())
    print("eval_label_source distribution:")
    print(merged["eval_label_source"].value_counts().to_string())

    if len(merged) < 300:
        print(
            f"\nWARNING: Only {len(merged)} rows — target is 300+.\n"
            "The suraj520 Kaggle dataset is synthetic (template descriptions,\n"
            "random label assignments). See data/eval/eval_set_summary.csv."
        )

    OUT_EVAL.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_EVAL, index=False)
    print(f"\nWrote {len(merged)} rows to {OUT_EVAL.relative_to(PROJECT_ROOT)}")

    write_summary(merged, OUT_SUMMARY)

    print(
        "\nNOTE: Kaggle-derived rows (eval_label_source='metadata_ticket_type') use "
        "Ticket Type as the label. The suraj520 dataset is synthetic — descriptions "
        "are template-generated and Ticket Type assignments are random relative to "
        "text content. Treat results as metadata-labeled evaluation, not gold labels."
    )


if __name__ == "__main__":
    main()
