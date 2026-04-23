from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_support_routing.config import RULE_PATTERNS
from llm_support_routing.evaluation import routing_metrics, threshold_sweep
from llm_support_routing.features import add_weak_labels, normalize_text
from llm_support_routing.routing import RULE_ROUTING, _apply_urgency_suffix, _rule_match, route_ticket


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
    assert add_weak_labels(_df(" ".join(["word"] * 90))).iloc[0]["complexity"] == "high"


def test_weak_label_complexity_low():
    assert add_weak_labels(_df("short text")).iloc[0]["complexity"] == "low"


# ── rule engine ───────────────────────────────────────────────────────────────

def test_rule_match_refund():
    result = _rule_match("i need a refund please")
    assert result is not None
    assert result[0] == "billing_queue"


def test_rule_match_account_locked():
    result = _rule_match("my account locked and i cannot get in")
    assert result is not None
    assert result[0] == "identity_support_queue"


def test_rule_match_no_match():
    assert _rule_match("general question about the product") is None


def test_rule_patterns_non_empty():
    assert len(RULE_PATTERNS) >= 10


def test_apply_urgency_suffix_critical():
    assert _apply_urgency_suffix("billing_queue", "critical") == "billing_queue_priority"


def test_apply_urgency_suffix_low():
    assert _apply_urgency_suffix("billing_queue", "low") == "billing_queue"


# ── routing_metrics ───────────────────────────────────────────────────────────

def _routed(routes, stages, confs):
    return pd.DataFrame({"route": routes, "stage": stages, "confidence": confs})


def test_routing_metrics_escalation_rate():
    df = _routed(
        ["human_triage_queue", "billing_queue", "billing_queue"],
        ["human_fallback", "ml_high_confidence", "rule_based"],
        [0.6, 0.9, 0.99],
    )
    assert abs(routing_metrics(df)["escalation_rate"] - 1 / 3) < 1e-6


def test_routing_metrics_llm_rate():
    df = _routed(["billing_queue", "billing_queue"], ["llm_reasoning", "ml_high_confidence"], [0.4, 0.9])
    assert routing_metrics(df)["llm_invocation_rate"] == 0.5


def test_routing_metrics_empty_df():
    assert routing_metrics(pd.DataFrame({"route": [], "stage": [], "confidence": []}))["tickets"] == 0.0


def test_routing_metrics_per_queue_keys():
    df = _routed(["billing_queue"], ["rule_based"], [0.99])
    assert "queue_pct_billing_queue" in routing_metrics(df)


def test_routing_metrics_stage_rates():
    df = _routed(
        ["billing_queue", "billing_queue", "human_triage_queue"],
        ["rule_based", "ml_high_confidence", "human_fallback"],
        [0.99, 0.90, 0.65],
    )
    m = routing_metrics(df)
    assert abs(m["rule_based_rate"] - 1 / 3) < 1e-6
    assert abs(m["ml_high_confidence_rate"] - 1 / 3) < 1e-6


# ── threshold_sweep ───────────────────────────────────────────────────────────

def _mock_model_for_sweep(probs: list[float]):
    labels = np.array(["billing"] * len(probs))
    prob_matrix = np.array([[p] for p in probs])
    model = MagicMock()
    model.predict.return_value = labels
    model.predict_proba.return_value = prob_matrix
    return model


def test_threshold_sweep_returns_10_rows():
    model = _mock_model_for_sweep([0.3, 0.6, 0.9] * 10)
    sweep = threshold_sweep(["text"] * 30, model)
    assert len(sweep) == 10


def test_threshold_sweep_columns():
    model = _mock_model_for_sweep([0.5] * 5)
    cols = threshold_sweep(["text"] * 5, model).columns.tolist()
    assert "auto_routed_rate" in cols
    assert "llm_fallback_rate" in cols
    assert "est_cost_per_ticket_usd" in cols


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


def test_route_ticket_rule_based_account_locked():
    decision = route_ticket("My account locked and I can't get in", _mock_model())
    assert decision.stage == "rule_based"
    assert decision.route == "identity_support_queue"


def test_route_ticket_ml_high_confidence():
    decision = route_ticket("The app keeps crashing", _mock_model(label="technical", prob=0.92))
    assert decision.stage == "ml_high_confidence"
    assert decision.route == "technical_support_queue"


def test_route_ticket_human_fallback():
    decision = route_ticket("I have a general question", _mock_model(label="other", prob=0.70))
    assert decision.stage == "human_fallback"
    assert decision.route == "human_triage_queue"


def test_route_ticket_urgency_suffix():
    urgency_model = _mock_model(label="critical", prob=0.9)
    decision = route_ticket(
        "The app keeps crashing",
        _mock_model(label="technical", prob=0.92),
        urgency_model=urgency_model,
    )
    assert decision.route == "technical_support_queue_priority"


@patch("llm_support_routing.routing.llm_classify_ticket")
def test_route_ticket_llm_reasoning(mock_llm):
    mock_llm.return_value = {"issue_type": "ads", "urgency": "high", "complexity": "low"}
    # Text avoids all RULE_PATTERNS so routing falls through to the LLM band
    decision = route_ticket("My marketing spend keeps increasing unexpectedly", _mock_model(label="other", prob=0.40))
    assert decision.stage == "llm_reasoning"
    assert "ads_ops_queue" in decision.route
    mock_llm.assert_called_once()


@patch("llm_support_routing.routing.llm_classify_ticket")
def test_route_ticket_llm_error_escalates_to_human(mock_llm):
    mock_llm.return_value = {"issue_type": "other", "urgency": "low", "complexity": "low", "_llm_error": "timeout"}
    decision = route_ticket("Strange request", _mock_model(label="other", prob=0.40))
    assert decision.stage == "human_fallback"
    assert decision.route == "human_triage_queue"
