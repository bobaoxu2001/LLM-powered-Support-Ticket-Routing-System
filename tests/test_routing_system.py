from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_support_routing.config import RULE_PATTERNS
from llm_support_routing.data import TICKET_PRIORITY_MAP, TICKET_TYPE_MAP, build_unified_ticket_table
from llm_support_routing.evaluation import routing_metrics, threshold_sweep
from llm_support_routing.features import _keyword_issue_type, add_weak_labels, normalize_text
from llm_support_routing.routing import RULE_ROUTING, _apply_urgency_suffix, _rule_match, route_ticket


# ── normalize_text ────────────────────────────────────────────────────────────

def test_normalize_lowercases():
    assert normalize_text("HELLO World") == "hello world"


def test_normalize_strips_urls():
    assert "http" not in normalize_text("visit https://example.com now")


def test_normalize_removes_punctuation():
    assert "!" not in normalize_text("help me!")


# ── TICKET_TYPE_MAP / TICKET_PRIORITY_MAP coverage ───────────────────────────

def test_ticket_type_map_billing_variants():
    assert TICKET_TYPE_MAP["billing inquiry"] == "billing"
    assert TICKET_TYPE_MAP["refund request"] == "billing"
    assert TICKET_TYPE_MAP["cancellation request"] == "billing"


def test_ticket_type_map_technical_variants():
    assert TICKET_TYPE_MAP["technical issue"] == "technical"
    assert TICKET_TYPE_MAP["software bug"] == "technical"
    assert TICKET_TYPE_MAP["network problem"] == "technical"


def test_ticket_priority_map_all_levels():
    for level in ("critical", "high", "medium", "low"):
        assert TICKET_PRIORITY_MAP[level] == level


# ── build_unified_ticket_table ────────────────────────────────────────────────

def _make_twitter() -> pd.DataFrame:
    return pd.DataFrame([
        {"text": "My payment was doubled", "inbound": True},
        {"text": "Thank you for contacting us", "inbound": False},  # agent — should be filtered
    ])


def _make_tickets() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Ticket Subject": "Billing problem",
            "Ticket Description": "I was charged twice",
            "Ticket Type": "Billing inquiry",
            "Ticket Priority": "High",
        },
        {
            "Ticket Subject": "Login issue",
            "Ticket Description": "Cannot log in to my account",
            "Ticket Type": "Login issue",
            "Ticket Priority": "Medium",
        },
    ])


def test_build_filters_agent_messages():
    unified = build_unified_ticket_table(_make_twitter(), _make_tickets())
    assert not any("Thank you for contacting us" in str(d) for d in unified["description"])


def test_build_extracts_real_ticket_type():
    unified = build_unified_ticket_table(_make_twitter(), _make_tickets())
    structured = unified[unified["source"] == "structured_tickets"]
    # Billing inquiry → billing
    assert "billing" in structured["category"].values


def test_build_extracts_real_priority():
    unified = build_unified_ticket_table(_make_twitter(), _make_tickets())
    structured = unified[unified["source"] == "structured_tickets"]
    assert "high" in structured["priority"].values


def test_build_label_source_column():
    unified = build_unified_ticket_table(_make_twitter(), _make_tickets())
    assert "label_source" in unified.columns
    assert set(unified["label_source"].unique()).issubset({"real", "weak"})


def test_build_twitter_rows_are_weak():
    unified = build_unified_ticket_table(_make_twitter(), _make_tickets())
    twitter_rows = unified[unified["source"] == "twitter_support"]
    assert (twitter_rows["label_source"] == "weak").all()


def test_build_real_label_wins_dedup_over_weak():
    """When the same description exists in both sources, the real-labeled row wins."""
    shared_text = "I was charged twice for my subscription"
    twitter = pd.DataFrame([{"text": shared_text, "inbound": True}])
    tickets = pd.DataFrame([{
        "Ticket Subject": shared_text,
        "Ticket Description": shared_text,
        "Ticket Type": "Billing inquiry",
        "Ticket Priority": "High",
    }])
    unified = build_unified_ticket_table(twitter, tickets)
    matching = unified[unified["description"] == shared_text]
    assert len(matching) == 1
    assert matching.iloc[0]["label_source"] == "real"


# ── add_weak_labels ───────────────────────────────────────────────────────────

def _df(text: str, category: str = "", priority: str = "", label_source: str = "weak") -> pd.DataFrame:
    return pd.DataFrame([{
        "subject": text, "description": "", "category": category,
        "priority": priority, "label_source": label_source,
    }])


def test_weak_label_uses_real_category():
    row = add_weak_labels(_df("some unrelated text", category="billing", label_source="real"))
    assert row.iloc[0]["issue_type"] == "billing"


def test_weak_label_ignores_other_when_label_source_is_weak():
    """Unmapped ticket types fall back to keyword, not hard 'other' (Codex P2 fix)."""
    # category='other' + label_source='weak' means _map_structured_tickets couldn't map
    # the ticket type — keyword fallback should run, not lock in 'other'.
    row = add_weak_labels(_df("I need a refund for my bill", category="other", label_source="weak"))
    assert row.iloc[0]["issue_type"] == "billing"  # keyword wins over spurious 'other'


def test_weak_label_ignores_priority_when_label_source_is_weak():
    """A priority value with label_source='weak' must not override keyword urgency."""
    row = add_weak_labels(_df("I need this fixed asap", priority="low", label_source="weak"))
    assert row.iloc[0]["urgency"] == "high"  # keyword wins over spurious 'low'


def test_weak_label_uses_real_priority():
    row = add_weak_labels(_df("some text", priority="critical", label_source="real"))
    assert row.iloc[0]["urgency"] == "critical"


def test_weak_label_falls_back_to_keyword_for_twitter():
    row = add_weak_labels(_df("I need a refund for my bill", label_source="weak"))
    assert row.iloc[0]["issue_type"] == "billing"


def test_weak_label_login():
    assert add_weak_labels(_df("I forgot my password")).iloc[0]["issue_type"] == "login"


def test_weak_label_technical():
    assert add_weak_labels(_df("The app keeps crashing")).iloc[0]["issue_type"] == "technical"


def test_weak_label_urgency_high():
    assert add_weak_labels(_df("I need this fixed asap")).iloc[0]["urgency"] == "high"


def test_weak_label_complexity_high():
    assert add_weak_labels(_df(" ".join(["word"] * 90))).iloc[0]["complexity"] == "high"


def test_weak_label_complexity_low():
    assert add_weak_labels(_df("short text")).iloc[0]["complexity"] == "low"


def test_weak_label_label_quality_column():
    row = add_weak_labels(_df("some text", category="billing", label_source="real"))
    assert row.iloc[0]["label_quality"] == "real"


# ── keyword baseline ──────────────────────────────────────────────────────────

def test_keyword_baseline_billing():
    assert _keyword_issue_type("i need a refund for my bill") == "billing"


def test_keyword_baseline_other_for_ambiguous():
    assert _keyword_issue_type("i have a question") == "other"


# ── rule engine ───────────────────────────────────────────────────────────────

def test_rule_match_refund():
    result = _rule_match("i need a refund please")
    assert result is not None and result[0] == "billing_queue"


def test_rule_match_account_locked():
    result = _rule_match("my account locked and i cannot get in")
    assert result is not None and result[0] == "identity_support_queue"


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


def test_routing_metrics_human_triage_rate():
    df = _routed(
        ["human_triage_queue", "billing_queue", "billing_queue"],
        ["human_fallback", "ml_high_confidence", "rule_based"],
        [0.6, 0.9, 0.99],
    )
    m = routing_metrics(df)
    assert abs(m["human_triage_rate"] - 1 / 3) < 1e-6
    # escalation_rate kept as backward-compatible proxy alias
    assert m["escalation_rate"] == m["human_triage_rate"]


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
    model = MagicMock()
    model.predict.return_value = np.array(["billing"] * len(probs))
    model.predict_proba.return_value = np.array([[p] for p in probs])
    return model


def test_threshold_sweep_returns_10_rows():
    sweep = threshold_sweep(["text"] * 30, _mock_model_for_sweep([0.3, 0.6, 0.9] * 10))
    assert len(sweep) == 10


def test_threshold_sweep_columns():
    cols = threshold_sweep(["text"] * 5, _mock_model_for_sweep([0.5] * 5)).columns.tolist()
    assert "auto_routed_rate" in cols
    assert "llm_fallback_rate" in cols
    assert "est_cost_per_ticket_usd" in cols
    # Score breakdown columns
    assert "score_automation_gain" in cols
    assert "score_human_penalty" in cols
    assert "score_llm_penalty" in cols
    assert "weight_auto" in cols
    assert "is_recommended_threshold" in cols


def test_threshold_sweep_custom_weights():
    sweep = threshold_sweep(
        ["text"] * 20,
        _mock_model_for_sweep([0.9] * 20),  # all high confidence
        human_penalty=2.0,
        llm_penalty=1.0,
    )
    assert (sweep["weight_human_penalty"] == 2.0).all()
    assert (sweep["weight_llm_penalty"] == 1.0).all()
    assert sweep["is_recommended_threshold"].sum() == 1


def test_threshold_sweep_score_breakdown_sums():
    sweep = threshold_sweep(["text"] * 10, _mock_model_for_sweep([0.7] * 10))
    row = sweep.iloc[0]
    expected = row["score_automation_gain"] - row["score_human_penalty"] - row["score_llm_penalty"]
    assert abs(expected - row["threshold_recommendation_score"]) < 1e-9


# ── evaluate_on_labeled_set ───────────────────────────────────────────────────

def test_evaluate_on_labeled_set_keys():
    from llm_support_routing.evaluation import evaluate_on_labeled_set
    import tempfile, os

    # Write a tiny labeled CSV
    csv = "text,issue_type\nbill charge refund,billing\npassword login,login\nerror crash bug,technical\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv)
        tmp = Path(f.name)

    model = MagicMock()
    model.predict.return_value = np.array(["billing", "login", "technical"])
    model.predict_proba.return_value = np.array([[0.9], [0.8], [0.85]])

    result = evaluate_on_labeled_set(model, tmp)
    tmp.unlink()

    assert "ml_accuracy" in result
    assert "keyword_baseline_accuracy" in result
    assert "ml_lift_over_baseline" in result
    assert result["n_eval_samples"] == 3


# ── route_ticket ──────────────────────────────────────────────────────────────

def _mock_model(label: str = "billing", prob: float = 0.95):
    model = MagicMock()
    model.predict.return_value = np.array([label])
    model.predict_proba.return_value = np.array([[prob]])
    return model


def test_route_ticket_rule_based_refund():
    decision = route_ticket("I need a refund please", _mock_model())
    assert decision.stage == "rule_based" and decision.route == "billing_queue"


def test_route_ticket_rule_based_account_locked():
    decision = route_ticket("My account locked and I can't get in", _mock_model())
    assert decision.stage == "rule_based" and decision.route == "identity_support_queue"


def test_route_ticket_ml_high_confidence():
    decision = route_ticket("The app keeps crashing", _mock_model(label="technical", prob=0.92))
    assert decision.stage == "ml_high_confidence" and decision.route == "technical_support_queue"


def test_route_ticket_human_fallback():
    decision = route_ticket("I have a general question", _mock_model(label="other", prob=0.70))
    assert decision.stage == "human_fallback" and decision.route == "human_triage_queue"


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
    decision = route_ticket(
        "My marketing spend keeps increasing unexpectedly",
        _mock_model(label="other", prob=0.40),
    )
    assert decision.stage == "llm_reasoning" and "ads_ops_queue" in decision.route
    mock_llm.assert_called_once()


@patch("llm_support_routing.routing.llm_classify_ticket")
def test_route_ticket_llm_error_escalates_to_human(mock_llm):
    mock_llm.return_value = {"issue_type": "other", "urgency": "low", "complexity": "low", "_llm_error": "timeout"}
    decision = route_ticket("Strange request", _mock_model(label="other", prob=0.40))
    assert decision.stage == "human_fallback" and decision.route == "human_triage_queue"


def test_route_dataframe_custom_thresholds():
    from llm_support_routing.config import RoutingThresholds
    from llm_support_routing.routing import route_dataframe

    model = _mock_model(label="technical", prob=0.60)
    df = pd.DataFrame({"text": ["The application keeps throwing unexpected errors"]})
    # With default thresholds (high=0.85), prob=0.60 hits human_fallback
    result_default = route_dataframe(df, model)
    assert result_default.iloc[0]["stage"] == "human_fallback"
    # With low high-threshold (high=0.55), prob=0.60 hits ml_high_confidence
    result_low = route_dataframe(df, model, thresholds=RoutingThresholds(high_confidence=0.55, low_confidence=0.30))
    assert result_low.iloc[0]["stage"] == "ml_high_confidence"


def test_route_dataframe_enrich_human_columns():
    from llm_support_routing.routing import route_dataframe

    model = _mock_model(label="other", prob=0.70)  # human_fallback band
    df = pd.DataFrame({"text": ["I have a general question about the service"]})

    with patch("llm_support_routing.routing.llm_resolution_and_escalation") as mock_res, \
         patch("llm_support_routing.routing.llm_summarize_ticket") as mock_sum:
        mock_res.return_value = {"suggested_path": "billing_review", "should_escalate": False, "reason": "low priority"}
        mock_sum.return_value = "Customer has a general inquiry."
        result = route_dataframe(df, model, enrich_human=True)

    assert result.iloc[0]["suggested_path"] == "billing_review"
    assert result.iloc[0]["should_escalate"] == "False"
    assert "billing_review" in result.iloc[0]["suggested_path"]
    mock_res.assert_called_once()
    mock_sum.assert_called_once()
