from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

from ..domain import Action, LLMUsage


@dataclass(frozen=True, slots=True)
class LLMDecisionRequest:
    window_index: int
    drift_kind: str
    feature_drift_score: float
    top_features: tuple[str, ...]
    current_score: float
    rolling_score: float
    baseline_score: float
    performance_drop: float
    label_shift: float
    remaining_updates: int
    remaining_training_seconds: float
    remaining_labels: int
    remaining_tokens: int
    requested_labels: int
    rule_reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LLMDecisionResponse:
    action: Action
    confidence: float
    reason: str
    model: str
    usage: LLMUsage
    latency_seconds: float


class PolicyLLMProvider(Protocol):
    def decide(self, request: LLMDecisionRequest) -> LLMDecisionResponse: ...
