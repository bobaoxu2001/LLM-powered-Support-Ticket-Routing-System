#!/usr/bin/env python3
"""Clean supervised benchmark on metadata-derived labels from Kaggle ticket dataset.

Label source: 'Ticket Type' column from suraj520/customer-support-ticket-dataset.
Labels are NOT from keyword matching or model predictions.
The Kaggle dataset is synthetic — descriptions contain a '{product_purchased}'
placeholder and random text that does not reflect ticket type. Text-label alignment
is therefore limited; this is an honest evaluation on metadata-labeled data.

Trains:
  - keyword_baseline       rule-based pattern matching
  - ml_baseline            TF-IDF (word unigrams) + LogisticRegression
  - ml_improved            TF-IDF (word bigrams) + char-wb n-grams + balanced LR

Outputs:
  outputs/supervised_benchmark_comparison.csv   — per-model summary metrics
  outputs/supervised_benchmark_per_class.csv    — per-class F1/precision/recall
  outputs/supervised_benchmark_confusion_matrix.csv
  outputs/eval_comparison.csv                   — compat layer for generate_preview_assets.py
  outputs/eval_per_class_metrics.csv            — compat layer for generate_preview_assets.py

Usage:
    python scripts/run_supervised_benchmark.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "support_tickets" / "customer_support_tickets.csv"
OUTPUTS = PROJECT_ROOT / "outputs"

TICKET_TYPE_MAP: dict[str, str] = {
    "billing inquiry":    "billing",
    "refund request":     "billing",
    "payment issue":      "billing",
    "cancellation request": "billing",
    "subscription issue": "billing",
    "technical issue":    "technical",
    "technical support":  "technical",
    "software bug":       "technical",
    "hardware issue":     "technical",
    "network problem":    "technical",
    "service outage":     "technical",
    "account issue":      "account",
    "account management": "account",
    "profile update":     "account",
    "login issue":        "login",
    "password reset":     "login",
    "authentication issue": "login",
    "product inquiry":    "other",
}

# Keyword rules ordered by specificity — first match wins
KEYWORD_RULES: list[tuple[list[str], str]] = [
    (["double charge", "double charged", "overcharged", "duplicate charge",
      "charged twice", "refund", "invoice", "billing", "payment failed",
      "subscription fee", "unauthorized charge", "credit card charge"],
     "billing"),
    (["forgot password", "reset password", "account locked", "can't log in",
      "cannot log in", "login failed", "sign in", "password reset",
      "authentication failed", "locked out"],
     "login"),
    (["ad campaign", "impression drop", "ad spend", "advertising",
      "campaign paused", "cpc", "cpm", "ad click"],
     "ads"),
    (["app crash", "not working", "service outage", "technical issue",
      "error message", "software bug", "hardware", "network issue",
      "connection problem", "setup issue", "not loading", "broken",
      "product setup", "product issue"],
     "technical"),
    (["account settings", "account update", "profile update", "close account",
      "delete account", "account management", "cancel subscription",
      "cancellation"],
     "account"),
]


def keyword_predict(texts: list[str]) -> list[str]:
    """Rule-based classifier — no labels used, no model training."""
    preds = []
    for text in texts:
        t = text.lower()
        matched = "other"
        for keywords, label in KEYWORD_RULES:
            if any(kw in t for kw in keywords):
                matched = label
                break
        preds.append(matched)
    return preds


def _clean_text(raw: str) -> str:
    t = re.sub(r"\{product_purchased\}", "the product", str(raw))
    t = re.sub(r"\{error_message\}", "the error", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_dataset() -> pd.DataFrame:
    if not RAW_CSV.exists():
        print(f"[ERROR] Raw data not found: {RAW_CSV}", file=sys.stderr)
        print("Run: python scripts/run_pipeline.py --download", file=sys.stderr)
        sys.exit(1)

    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(raw):,} raw rows from {RAW_CSV.name}")

    # Build text
    raw["_subject"] = raw["Ticket Subject"].fillna("").astype(str).str.strip()
    raw["_desc"] = raw["Ticket Description"].fillna("").astype(str).apply(_clean_text)
    raw["text"] = raw["_subject"] + " " + raw["_desc"]

    # Map Ticket Type → issue_type
    raw["issue_type"] = (
        raw["Ticket Type"]
        .fillna("")
        .str.lower()
        .str.strip()
        .map(TICKET_TYPE_MAP)
    )

    df = raw[["text", "issue_type"]].copy()

    # Drop nulls and short texts
    df = df.dropna(subset=["issue_type"])
    df = df[df["text"].str.len() >= 20]

    # Deduplicate on text
    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    print(f"After dedup: {len(df):,} rows (dropped {before - len(df):,})")

    # Strip whitespace
    df["text"] = df["text"].str.strip()

    print("\nLabel distribution (full):")
    print(df["issue_type"].value_counts().to_string())

    return df.reset_index(drop=True)


def build_improved_pipeline() -> Pipeline:
    """TF-IDF word bigrams + char-wb ngrams → balanced LogisticRegression."""
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=50_000,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=30_000,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    return Pipeline([("features", features), ("clf", clf)])


def build_baseline_pipeline() -> Pipeline:
    vec = TfidfVectorizer(analyzer="word", min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=500, random_state=42)
    return Pipeline([("tfidf", vec), ("clf", clf)])


def evaluate(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    n_train: int,
) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro-F1:    {macro_f1:.4f}")
    print(f"  Weighted-F1: {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    return {
        "model_name": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train": n_train,
        "n_test": len(y_true),
        "metric_type": "measured_holdout_metadata_labels",
        "label_source": "metadata_ticket_type",
        "dataset_source": "Kaggle support tickets (suraj520)",
    }


def per_class_rows(name: str, y_true, y_pred, labels: list[str]) -> list[dict]:
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    rows = []
    for label in labels:
        d = report.get(label, {})
        rows.append({
            "model": name,
            "label": label,
            "precision": d.get("precision", 0.0),
            "recall": d.get("recall", 0.0),
            "f1_score": d.get("f1-score", 0.0),
            "support": d.get("support", 0),
        })
    return rows


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)

    # ── Load & split ───────────────────────────────────────────────────────────
    df = load_dataset()
    labels = sorted(df["issue_type"].unique().tolist())
    print(f"\nClasses: {labels}")

    X = df["text"].tolist()
    y = df["issue_type"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
    print("Test label distribution:")
    test_counts = pd.Series(y_test).value_counts()
    print(test_counts.to_string())

    # ── Train & evaluate ───────────────────────────────────────────────────────
    summary_rows: list[dict] = []
    pc_rows: list[dict] = []
    cm_rows: list[dict] = []

    # Keyword baseline (no training, no label leakage)
    kw_pred = keyword_predict(X_test)
    summary_rows.append(evaluate("keyword_baseline", y_test, kw_pred, labels, n_train=0))
    pc_rows.extend(per_class_rows("keyword_baseline", y_test, kw_pred, labels))
    cm = confusion_matrix(y_test, kw_pred, labels=labels)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm_rows.append({"model": "keyword_baseline", "true": true_label, "predicted": pred_label, "count": int(cm[i, j])})

    # Baseline ML
    baseline = build_baseline_pipeline()
    baseline.fit(X_train, y_train)
    bl_pred = baseline.predict(X_test)
    summary_rows.append(evaluate("ml_baseline", y_test, bl_pred, labels, n_train=len(X_train)))
    pc_rows.extend(per_class_rows("ml_baseline", y_test, bl_pred, labels))
    cm = confusion_matrix(y_test, bl_pred, labels=labels)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm_rows.append({"model": "ml_baseline", "true": true_label, "predicted": pred_label, "count": int(cm[i, j])})

    # Improved ML
    improved = build_improved_pipeline()
    improved.fit(X_train, y_train)
    imp_pred = improved.predict(X_test)
    summary_rows.append(evaluate("ml_improved", y_test, imp_pred, labels, n_train=len(X_train)))
    pc_rows.extend(per_class_rows("ml_improved", y_test, imp_pred, labels))
    cm = confusion_matrix(y_test, imp_pred, labels=labels)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm_rows.append({"model": "ml_improved", "true": true_label, "predicted": pred_label, "count": int(cm[i, j])})

    # ── Save benchmark CSVs ────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUTS / "supervised_benchmark_comparison.csv", index=False)
    print(f"\n[OK] outputs/supervised_benchmark_comparison.csv")

    pc_df = pd.DataFrame(pc_rows)
    pc_df.to_csv(OUTPUTS / "supervised_benchmark_per_class.csv", index=False)
    print(f"[OK] outputs/supervised_benchmark_per_class.csv")

    pd.DataFrame(cm_rows).to_csv(OUTPUTS / "supervised_benchmark_confusion_matrix.csv", index=False)
    print(f"[OK] outputs/supervised_benchmark_confusion_matrix.csv")

    # ── Compat layer for generate_preview_assets.py ────────────────────────────
    kw_row = next(r for r in summary_rows if r["model_name"] == "keyword_baseline")
    imp_row = next(r for r in summary_rows if r["model_name"] == "ml_improved")

    compat_row = {
        "ml_accuracy":               imp_row["accuracy"],
        "ml_macro_f1":               imp_row["macro_f1"],
        "ml_weighted_f1":            imp_row["weighted_f1"],
        "keyword_baseline_accuracy": kw_row["accuracy"],
        "keyword_macro_f1":          kw_row["macro_f1"],
        "keyword_weighted_f1":       kw_row["weighted_f1"],
        "n_eval_samples":            imp_row["n_test"],
        "label_source":              "metadata_ticket_type",
        "dataset_source":            "Kaggle support tickets (suraj520)",
        "metric_type":               "measured_holdout_metadata_labels",
    }
    pd.DataFrame([compat_row]).to_csv(OUTPUTS / "eval_comparison.csv", index=False)
    print(f"[OK] outputs/eval_comparison.csv  (compat layer)")

    # Per-class compat (include both ml_improved and keyword_baseline)
    compat_pc = [
        {**r, "model": "ml_tfidf_lr" if r["model"] == "ml_improved" else r["model"]}
        for r in pc_rows
        if r["model"] in ("ml_improved", "keyword_baseline")
    ]
    pd.DataFrame(compat_pc).to_csv(OUTPUTS / "eval_per_class_metrics.csv", index=False)
    print(f"[OK] outputs/eval_per_class_metrics.csv (compat layer)")

    print("\nBenchmark complete.")
    print(f"  Test set: {imp_row['n_test']} samples")
    print(f"  Keyword baseline — accuracy {kw_row['accuracy']:.3f}, macro-F1 {kw_row['macro_f1']:.3f}")
    print(f"  ML improved      — accuracy {imp_row['accuracy']:.3f}, macro-F1 {imp_row['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
