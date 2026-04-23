from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


@dataclass(slots=True)
class TrainingResult:
    model: Pipeline
    accuracy: float
    report: str
    cv_mean: float
    cv_std: float


def train_tfidf_logreg(
    df: pd.DataFrame,
    label_col: str,
    random_state: int = 42,
    calibrate: bool = True,
) -> TrainingResult:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[label_col], test_size=0.2, random_state=random_state, stratify=df[label_col]
    )

    base_lr = LogisticRegression(max_iter=300, class_weight="balanced")
    clf = CalibratedClassifierCV(base_lr, cv=5, method="isotonic") if calibrate else base_lr

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
            ("clf", clf),
        ]
    )
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds)

    # 5-fold CV on full dataset for a stability estimate
    cv_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    cv_scores = cross_val_score(cv_pipeline, df["text"], df[label_col], cv=5, scoring="accuracy")

    return TrainingResult(
        model=pipeline,
        accuracy=acc,
        report=report,
        cv_mean=float(cv_scores.mean()),
        cv_std=float(cv_scores.std()),
    )


def save_model(model: Pipeline, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_with_confidence(model: Pipeline, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    labels = model.predict(texts)
    probs = model.predict_proba(texts).max(axis=1)
    return labels, probs
