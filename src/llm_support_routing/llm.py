from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_openai_client: OpenAI | None = None


def _client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


_FEW_SHOT_EXAMPLES = """\
Examples:
Ticket: "I was double billed this month and need a refund immediately"
{"issue_type": "billing", "urgency": "high", "complexity": "low"}

Ticket: "My ad campaign impressions dropped 80% overnight with no changes on my end"
{"issue_type": "ads", "urgency": "high", "complexity": "high"}

Ticket: "I forgot my password and the reset email isn't arriving"
{"issue_type": "login", "urgency": "medium", "complexity": "low"}
"""


def _parse_json_safe(text: str) -> dict[str, Any] | None:
    """Extract and parse the first JSON object from text, tolerating markdown fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def llm_classify_ticket(text: str) -> dict[str, Any]:
    prompt = (
        "You are a support operations assistant. Classify the ticket below.\n\n"
        f"{_FEW_SHOT_EXAMPLES}\n"
        "Return ONLY a JSON object with keys: issue_type (billing/ads/login/technical/account/other), "
        "urgency (low/medium/high/critical), complexity (low/medium/high).\n\n"
        f"Ticket: {text}"
    )
    try:
        response = _client().responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
            temperature=0.0,
        )
        result = _parse_json_safe(response.output_text)
        if result is None:
            raise ValueError(f"Unparseable LLM output: {response.output_text!r}")
        return result
    except Exception as exc:
        logger.warning("llm_classify_ticket failed (%s); returning error sentinel.", exc)
        return {"issue_type": "other", "urgency": "low", "complexity": "low", "_llm_error": str(exc)}


def llm_summarize_ticket(text: str) -> str:
    try:
        response = _client().responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=f"Summarize this support ticket in 1-2 sentences for a human agent:\n\n{text}",
            temperature=0.2,
        )
        return response.output_text.strip()
    except Exception as exc:
        logger.warning("llm_summarize_ticket failed (%s).", exc)
        return ""


def llm_resolution_and_escalation(text: str) -> dict[str, Any]:
    prompt = (
        "Given this customer ticket, propose a resolution path and whether it should be escalated.\n"
        "Return ONLY a JSON object with keys: suggested_path (string), should_escalate (true/false), reason (string).\n\n"
        f"Ticket: {text}"
    )
    try:
        response = _client().responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=prompt,
            temperature=0.2,
        )
        result = _parse_json_safe(response.output_text)
        if result is None:
            raise ValueError(f"Unparseable LLM output: {response.output_text!r}")
        return result
    except Exception as exc:
        logger.warning("llm_resolution_and_escalation failed (%s).", exc)
        return {"suggested_path": "human_review", "should_escalate": True, "reason": str(exc)}
