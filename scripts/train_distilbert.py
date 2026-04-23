from __future__ import annotations

"""DistilBERT fine-tuning for issue type classification.

Produces a trained model in models/distilbert-issue-type/ and prints a
side-by-side accuracy comparison with the TF-IDF + LR baseline.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from llm_support_routing.config import DATA_PROCESSED, MODELS_DIR


def compute_metrics(pred: EvalPrediction) -> dict[str, float]:
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": float(accuracy_score(pred.label_ids, preds))}


def main(model_name: str, max_samples: int) -> None:
    path = DATA_PROCESSED / "unified_labeled_tickets.csv"
    if not path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first to create processed data.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = MODELS_DIR / "distilbert-issue-type"

    df = pd.read_csv(path)
    labels = sorted(df["issue_type"].dropna().unique().tolist())
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for i, x in enumerate(labels)}

    df = df[["text", "issue_type"]].dropna().copy()
    df["label"] = df["issue_type"].map(label2id)
    df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)

    split_idx = int(0.8 * len(df))
    train_ds = Dataset.from_pandas(df.iloc[:split_idx][["text", "label"]].reset_index(drop=True))
    eval_ds = Dataset.from_pandas(df.iloc[split_idx:][["text", "label"]].reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        fp16=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    print(f"Fine-tuning {model_name} on {len(train_ds)} train / {len(eval_ds)} eval tickets...")
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"\nEval accuracy: {eval_results['eval_accuracy']:.4f}")

    preds_output = trainer.predict(eval_ds)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = preds_output.label_ids
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max tickets to use for fine-tuning (default: 10000)")
    args = parser.parse_args()
    main(args.model, args.max_samples)
