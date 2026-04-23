from __future__ import annotations

import re

import pandas as pd


KEYWORD_MAP = {
    "billing": ["bill", "charge", "refund", "payment", "invoice"],
    "ads": ["ad", "campaign", "impression", "click"],
    "login": ["login", "password", "2fa", "signin", "locked"],
    "technical": ["error", "bug", "crash", "broken", "issue"],
    "account": ["account", "profile", "username", "verification"],
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate issue_type, urgency, complexity via weak supervision heuristics."""
    out = df.copy()
    out["text"] = (out["subject"].fillna("") + " " + out["description"].fillna(""))
    out["text"] = out["text"].astype(str).map(normalize_text)

    def infer_issue_type(text: str) -> str:
        for label, keywords in KEYWORD_MAP.items():
            if any(k in text for k in keywords):
                return label
        return "other"

    def infer_urgency(text: str) -> str:
        if any(k in text for k in ["urgent", "immediately", "asap", "cannot", "can't"]):
            return "high"
        if any(k in text for k in ["soon", "today", "help"]):
            return "medium"
        return "low"

    def infer_complexity(text: str) -> str:
        length = len(text.split())
        if length > 80:
            return "high"
        if length > 30:
            return "medium"
        return "low"

    out["issue_type"] = out["text"].map(infer_issue_type)
    out["urgency"] = out["text"].map(infer_urgency)
    out["complexity"] = out["text"].map(infer_complexity)
    return out
