from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Action(StringEnum):
    NO_UPDATE = "no_update"
    RETRAIN_WITH_REPLAY = "retrain_with_replay"
    ROLLBACK = "rollback"
    ERROR = "error"


class DriftKind(StringEnum):
    NONE = "none"
    COVARIATE = "covariate"
    LABEL = "label"
    CONCEPT_SUSPECTED = "concept_suspected"
    MIXED = "mixed"


@dataclass(slots=True)
class DataWindow:
    index: int
    X: pd.DataFrame
    y: np.ndarray
    phase: str = "unknown"
    sample_ids: np.ndarray | None = None


@dataclass(slots=True)
class FeatureDriftResult:
    overall_score: float
    is_drift: bool
    per_feature: dict[str, dict[str, float]]
    top_features: list[str]
    data_quality_valid: bool = True
    data_quality_issues: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PerformanceDriftResult:
    current_score: float
    rolling_score: float
    baseline_score: float
    score_drop: float
    page_hinkley_alarm: bool
    is_drift: bool


@dataclass(slots=True)
class DriftReport:
    kind: DriftKind
    feature: FeatureDriftResult
    performance: PerformanceDriftResult
    label_shift: float
    is_actionable: bool
    explanation: str


@dataclass(slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class Decision:
    action: Action
    reason: str
    source: str = "rule"
    model: str | None = None
    confidence: float | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    latency_seconds: float = 0.0
    fallback_reason: str | None = None
    llm_attempted: bool = False
    llm_called: bool = False


@dataclass(slots=True)
class CandidateEvaluation:
    accepted: bool
    champion_recent_score: float
    challenger_recent_score: float
    recent_improvement: float
    champion_historical_score: float | None
    challenger_historical_score: float | None
    historical_drop: float
    reason: str


@dataclass(slots=True)
class BudgetSnapshot:
    updates_used: int
    max_updates: int
    training_seconds_used: float
    max_training_seconds: float
    labels_used: int
    max_labels: int
    tokens_used: int
    max_tokens: int
    budget_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyContext:
    window_index: int
    report: DriftReport
    budget: BudgetSnapshot
    requested_labels: int
    can_train: bool
    experiment_history: tuple[dict[str, Any], ...] = ()


@dataclass(slots=True)
class RunSummary:
    run_id: str
    windows_processed: int = 0
    successful_windows: int = 0
    updates_attempted: int = 0
    updates_deployed: int = 0
    updates_rejected: int = 0
    rollbacks: int = 0
    failures: int = 0
    skipped_windows: int = 0
    llm_attempts: int = 0
    llm_calls: int = 0
    llm_fallbacks: int = 0
    initial_score: float = 0.0
    final_score: float = 0.0
    mean_score: float = 0.0
    run_status: str = "running"
    action_counts: dict[str, int] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
