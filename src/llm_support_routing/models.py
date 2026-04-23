from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class TrainingResult:
    model: Pipeline
    accuracy: float
    report: str



def train_tfidf_logreg(
    df: pd.DataFrame,
    label_col: str,
    random_state: int = 42,
) -> TrainingResult:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[label_col], test_size=0.2, random_state=random_state, stratify=df[label_col]
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds)
    return TrainingResult(model=pipeline, accuracy=acc, report=report)


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_with_confidence(model: Pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    labels = model.predict(texts)
    probs = model.predict_proba(texts).max(axis=1)
    return labels, probs
