from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(slots=True)
class RoutingThresholds:
    high_confidence: float = 0.85
    low_confidence: float = 0.55


@dataclass(slots=True)
class Labels:
    issue_types: tuple[str, ...] = ("billing", "ads", "login", "technical", "account", "other")
    urgency_levels: tuple[str, ...] = ("low", "medium", "high", "critical")
    complexity_levels: tuple[str, ...] = ("low", "medium", "high")


# Configurable rule registry: (substring, destination_queue, issue_type_label).
# Evaluated in order; first match wins. Add entries here to extend the rule engine
# without touching routing logic.
RULE_PATTERNS: list[tuple[str, str, str]] = [
    ("double charged",      "billing_queue",           "billing"),
    ("overcharged",         "billing_queue",           "billing"),
    ("refund",              "billing_queue",           "billing"),
    ("invoice",             "billing_queue",           "billing"),
    ("account locked",      "identity_support_queue",  "login"),
    ("forgot password",     "identity_support_queue",  "login"),
    ("can't log in",        "identity_support_queue",  "login"),
    ("cannot log in",       "identity_support_queue",  "login"),
    ("reset password",      "identity_support_queue",  "login"),
    ("ad campaign",         "ads_ops_queue",           "ads"),
    ("impression drop",     "ads_ops_queue",           "ads"),
    ("app crash",           "technical_support_queue", "technical"),
    ("not working",         "technical_support_queue", "technical"),
    ("service outage",      "technical_support_queue", "technical"),
]
