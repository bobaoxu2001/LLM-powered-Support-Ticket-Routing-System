from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_classify_ticket(text: str) -> dict[str, Any]:
    prompt = f"""
You are a support operations assistant.
Classify the ticket into:
- issue_type: one of billing, ads, login, technical, account, other
- urgency: one of low, medium, high, critical
- complexity: one of low, medium, high
Return strict JSON.
Ticket: {text}
"""
    response = _client().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
        temperature=0.0,
    )
    return json.loads(response.output_text)


def llm_summarize_ticket(text: str) -> str:
    response = _client().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=f"Summarize this support ticket in 1-2 sentences: {text}",
        temperature=0.2,
    )
    return response.output_text.strip()


def llm_resolution_and_escalation(text: str) -> dict[str, Any]:
    prompt = f"""
Given this customer ticket, propose a resolution path and whether it should be escalated.
Return strict JSON with keys: suggested_path, should_escalate, reason.
Ticket: {text}
"""
    response = _client().responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=prompt,
        temperature=0.2,
    )
    return json.loads(response.output_text)
