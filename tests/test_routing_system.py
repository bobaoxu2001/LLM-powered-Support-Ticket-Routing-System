from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_support_routing.evaluation import routing_metrics
from llm_support_routing.features import add_weak_labels, normalize_text
from llm_support_routing.routing import RULE_ROUTING, route_ticket


# ── normalize_text ────────────────────────────────────────────────────────────

def test_normalize_lowercases():
    assert normalize_text("HELLO World") == "hello world"


def test_normalize_strips_urls():
    assert "http" not in normalize_text("visit https://example.com now")


def test_normalize_removes_punctuation():
    assert "!" not in normalize_text("help me!")


# ── add_weak_labels ───────────────────────────────────────────────────────────

def _df(text: str) -> pd.DataFrame:
    return pd.DataFrame([{"subject": text, "description": ""}])


def test_weak_label_billing():
    assert add_weak_labels(_df("I need a refund for my bill")).iloc[0]["issue_type"] == "billing"


def test_weak_label_login():
    assert add_weak_labels(_df("I forgot my password")).iloc[0]["issue_type"] == "login"


def test_weak_label_technical():
    assert add_weak_labels(_df("The app keeps crashing with an error")).iloc[0]["issue_type"] == "technical"


def test_weak_label_urgency_high():
    assert add_weak_labels(_df("I need this fixed asap")).iloc[0]["urgency"] == "high"


def test_weak_label_complexity_high():
    long_text = " ".join(["word"] * 90)
    assert add_weak_labels(_df(long_text)).iloc[0]["complexity"] == "high"


def test_weak_label_complexity_low():
    assert add_weak_labels(_df("short text")).iloc[0]["complexity"] == "low"


# ── routing_metrics ───────────────────────────────────────────────────────────

def _routed(routes, stages, confs):
    return pd.DataFrame({"route": routes, "stage": stages, "confidence": confs})


def test_routing_metrics_escalation_rate():
    df = _routed(
        ["human_triage_queue", "billing_queue", "billing_queue"],
        ["human_fallback", "ml_high_confidence", "rule_based"],
        [0.6, 0.9, 0.99],
    )
    m = routing_metrics(df)
    assert abs(m["escalation_rate"] - 1 / 3) < 1e-6


def test_routing_metrics_llm_rate():
    df = _routed(
        ["billing_queue", "billing_queue"],
        ["llm_reasoning", "ml_high_confidence"],
        [0.4, 0.9],
    )
    assert routing_metrics(df)["llm_invocation_rate"] == 0.5


def test_routing_metrics_empty_df():
    m = routing_metrics(pd.DataFrame({"route": [], "stage": [], "confidence": []}))
    assert m["tickets"] == 0.0


def test_routing_metrics_per_queue_keys():
    df = _routed(["billing_queue"], ["rule_based"], [0.99])
    m = routing_metrics(df)
    assert "queue_pct_billing_queue" in m


# ── route_ticket ──────────────────────────────────────────────────────────────

def _mock_model(label: str = "billing", prob: float = 0.95):
    model = MagicMock()
    model.predict.return_value = np.array([label])
    model.predict_proba.return_value = np.array([[prob]])
    return model


def test_route_ticket_rule_based_refund():
    decision = route_ticket("I need a refund please", _mock_model())
    assert decision.stage == "rule_based"
    assert decision.route == "billing_queue"


def test_route_ticket_rule_based_double_charge():
    decision = route_ticket("I was double charged twice", _mock_model())
    assert decision.stage == "rule_based"


def test_route_ticket_ml_high_confidence():
    decision = route_ticket("The app keeps crashing", _mock_model(label="technical", prob=0.92))
    assert decision.stage == "ml_high_confidence"
    assert decision.route == "technical_support_queue"


def test_route_ticket_human_fallback():
    decision = route_ticket("I have a general question", _mock_model(label="other", prob=0.70))
    assert decision.stage == "human_fallback"
    assert decision.route == "human_triage_queue"


@patch("llm_support_routing.routing.llm_classify_ticket")
def test_route_ticket_llm_reasoning(mock_llm):
    mock_llm.return_value = {"issue_type": "ads", "urgency": "high", "complexity": "low"}
    decision = route_ticket("My ad campaign dropped 80%", _mock_model(label="other", prob=0.40))
    assert decision.stage == "llm_reasoning"
    assert decision.route == "ads_ops_queue"
    mock_llm.assert_called_once()


@patch("llm_support_routing.routing.llm_classify_ticket")
def test_route_ticket_llm_error_escalates_to_human(mock_llm):
    mock_llm.return_value = {"issue_type": "other", "urgency": "low", "complexity": "low", "_llm_error": "timeout"}
    decision = route_ticket("Strange request", _mock_model(label="other", prob=0.40))
    assert decision.stage == "human_fallback"
    assert decision.route == "human_triage_queue"
