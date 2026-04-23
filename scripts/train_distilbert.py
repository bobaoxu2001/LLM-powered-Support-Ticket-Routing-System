from __future__ import annotations

"""Optional advanced model training script (DistilBERT).

This script is intentionally lightweight and can be extended for full fine-tuning.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_support_routing.config import DATA_PROCESSED


def main(model_name: str) -> None:
    path = DATA_PROCESSED / "unified_labeled_tickets.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first to create processed data.")

    df = pd.read_csv(path)
    labels = sorted(df["issue_type"].unique().tolist())
    label2id = {x: i for i, x in enumerate(labels)}
    df = df[["text", "issue_type"]].dropna().copy()
    df["label"] = df["issue_type"].map(label2id)

    dataset = Dataset.from_pandas(df[["text", "label"]].sample(min(5000, len(df)), random_state=42))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tok, batched=True)
    print("Prepared tokenized dataset for DistilBERT fine-tuning.")
    print(f"Rows: {len(dataset)}, Labels: {labels}")
    print("Next step: add Trainer + TrainingArguments for full training loop.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    args = parser.parse_args()
    main(args.model)
