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
