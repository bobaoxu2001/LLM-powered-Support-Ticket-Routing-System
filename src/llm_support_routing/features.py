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

_VALID_ISSUE_TYPES = frozenset(KEYWORD_MAP) | {"other"}
_VALID_URGENCY = frozenset({"low", "medium", "high", "critical"})


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_issue_type(text: str) -> str:
    for label, keywords in KEYWORD_MAP.items():
        if any(k in text for k in keywords):
            return label
    return "other"


def _keyword_urgency(text: str) -> str:
    if any(k in text for k in ["urgent", "immediately", "asap", "cannot", "can't"]):
        return "high"
    if any(k in text for k in ["soon", "today", "help"]):
        return "medium"
    return "low"


def _keyword_complexity(text: str) -> str:
    n = len(text.split())
    if n > 80:
        return "high"
    if n > 30:
        return "medium"
    return "low"


def add_weak_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Assign issue_type, urgency, and complexity to every row.

    Priority order for issue_type:
      1. Real label from structured dataset (category column contains a known
         issue type such as "billing", "technical", etc.)
      2. Keyword heuristic fallback for Twitter / unlabeled rows.

    Priority order for urgency:
      1. Real label from structured dataset (priority column).
      2. Keyword heuristic fallback.

    complexity is always derived from text length (no real label exists for it).

    A label_quality column records 'real' or 'weak' so training code can
    weight or filter rows by label origin.
    """
    out = df.copy()
    out["text"] = (out["subject"].fillna("") + " " + out["description"].fillna(""))
    out["text"] = out["text"].astype(str).map(normalize_text)

    real_category = out.get("category", pd.Series("", index=out.index)).fillna("")
    real_priority = out.get("priority", pd.Series("", index=out.index)).fillna("")
    label_source = out.get("label_source", pd.Series("weak", index=out.index)).fillna("weak")

    # issue_type: use real label only when label_source='real' AND category is a known
    # type.  Without the label_source guard, unmapped ticket types that _map_structured_tickets
    # falls back to 'other' (with label_source='weak') would be treated as confirmed 'other'
    # labels and bypass keyword fallback, injecting systematic noise into training.
    is_real_issue = (label_source == "real") & real_category.isin(_VALID_ISSUE_TYPES)
    issue_type = real_category.where(is_real_issue, other=None)
    keyword_fallback = out["text"].map(_keyword_issue_type)
    out["issue_type"] = issue_type.combine_first(keyword_fallback)

    # urgency: same guard — only trust priority when label_source='real'
    is_real_urgency = (label_source == "real") & real_priority.isin(_VALID_URGENCY)
    urgency = real_priority.where(is_real_urgency, other=None)
    urgency_fallback = out["text"].map(_keyword_urgency)
    out["urgency"] = urgency.combine_first(urgency_fallback)

    # complexity: always keyword-derived (no real labels available)
    out["complexity"] = out["text"].map(_keyword_complexity)

    # Track label quality for training transparency
    out["label_quality"] = label_source

    return out
