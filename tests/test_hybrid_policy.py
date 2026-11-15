from __future__ import annotations

from dataclasses import dataclass

from industrial_evolution.config import LLMConfig, MonitoringConfig, PolicyConfig
from industrial_evolution.domain import (
    Action,
    BudgetSnapshot,
    DriftKind,
    DriftReport,
    FeatureDriftResult,
    LLMUsage,
    PerformanceDriftResult,
    PolicyContext,
)
from industrial_evolution.llm.provider import LLMDecisionRequest, LLMDecisionResponse
from industrial_evolution.policy.hybrid_policy import HybridPolicy
from industrial_evolution.policy.rule_policy import RuleBasedPolicy


@dataclass
class StubProvider:
    response: LLMDecisionResponse | None = None
    error: Exception | None = None
    calls: int = 0

    def decide(self, request: LLMDecisionRequest) -> LLMDecisionResponse:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def make_report(actionable: bool = True) -> DriftReport:
    return DriftReport(
        kind=DriftKind.MIXED if actionable else DriftKind.NONE,
        feature=FeatureDriftResult(1.5, actionable, {}, ["sensor_00"]),
        performance=PerformanceDriftResult(0.6, 0.62, 0.75, 0.13, True, actionable),
        label_shift=0.15 if actionable else 0.0,
        is_actionable=actionable,
        explanation="测试漂移",
    )


def make_context(
    window_index: int,
    *,
    actionable: bool = True,
    max_tokens: int = 5000,
    tokens_used: int = 0,
) -> PolicyContext:
    return PolicyContext(
        window_index=window_index,
        report=make_report(actionable),
        budget=BudgetSnapshot(
            updates_used=0,
            max_updates=6,
            training_seconds_used=0.5,
            max_training_seconds=120.0,
            labels_used=500,
            max_labels=5000,
            tokens_used=tokens_used,
            max_tokens=max_tokens,
        ),
        requested_labels=140,
        can_train=True,
    )


def make_policy(provider: StubProvider, **overrides) -> HybridPolicy:
    config = LLMConfig(**overrides)
    rule = RuleBasedPolicy(
        PolicyConfig(cooldown_windows=1, rollback_patience=2, rollback_drop=0.1),
        MonitoringConfig(
            psi_threshold=0.18,
            ks_threshold=0.18,
            performance_drop_threshold=0.06,
            label_shift_threshold=0.12,
            consecutive_windows=1,
            reference_update_rate=0.15,
        ),
    )
    return HybridPolicy(rule, provider, config)


def response(action: Action, confidence: float = 0.9) -> LLMDecisionResponse:
    return LLMDecisionResponse(
        action=action,
        confidence=confidence,
        reason="模型给出的受控理由",
        model="local-model",
        usage=LLMUsage(120, 30, 5, 150),
        latency_seconds=0.25,
    )


def test_rule_gate_avoids_llm_when_drift_is_not_actionable():
    provider = StubProvider(response(Action.RETRAIN_WITH_REPLAY))
    decision = make_policy(provider).decide(make_context(1, actionable=False))

    assert decision.action is Action.NO_UPDATE
    assert decision.source == "rule"
    assert provider.calls == 0


def test_high_confidence_llm_can_approve_rule_retraining():
    provider = StubProvider(response(Action.RETRAIN_WITH_REPLAY))
    decision = make_policy(provider).decide(make_context(1))

    assert decision.action is Action.RETRAIN_WITH_REPLAY
    assert decision.source == "llm"
    assert decision.llm_attempted
    assert decision.llm_called
    assert decision.usage.total_tokens == 150
    assert decision.model == "local-model"
    assert provider.calls == 1


def test_llm_veto_is_cached_without_extra_token_use():
    provider = StubProvider(response(Action.NO_UPDATE))
    policy = make_policy(provider, call_cooldown_windows=2)

    first = policy.decide(make_context(1))
    second = policy.decide(make_context(2))

    assert first.action is Action.NO_UPDATE
    assert first.source == "llm"
    assert first.usage.total_tokens == 150
    assert second.action is Action.NO_UPDATE
    assert second.source == "llm_cache"
    assert second.usage.total_tokens == 0
    assert provider.calls == 1


def test_low_confidence_and_provider_errors_fall_back_to_rules():
    low_provider = StubProvider(response(Action.NO_UPDATE, confidence=0.4))
    low_decision = make_policy(low_provider, min_confidence=0.65).decide(make_context(1))
    assert low_decision.action is Action.RETRAIN_WITH_REPLAY
    assert low_decision.source == "rule_fallback"
    assert low_decision.llm_attempted
    assert low_decision.llm_called
    assert low_decision.usage.total_tokens == 150

    error_provider = StubProvider(error=RuntimeError("temporary failure"))
    error_decision = make_policy(error_provider).decide(make_context(1))
    assert error_decision.action is Action.RETRAIN_WITH_REPLAY
    assert error_decision.source == "rule_fallback"
    assert "RuntimeError" in error_decision.fallback_reason
    assert error_decision.llm_attempted
    assert not error_decision.llm_called


def test_token_or_call_budget_prevents_api_access():
    token_provider = StubProvider(response(Action.RETRAIN_WITH_REPLAY))
    token_decision = make_policy(token_provider, max_tokens_per_call=800).decide(
        make_context(1, max_tokens=700)
    )
    assert token_decision.action is Action.RETRAIN_WITH_REPLAY
    assert token_decision.source == "rule_fallback"
    assert not token_decision.llm_attempted
    assert not token_decision.llm_called
    assert token_provider.calls == 0

    call_provider = StubProvider(response(Action.RETRAIN_WITH_REPLAY))
    call_decision = make_policy(call_provider, max_calls=0).decide(make_context(1))
    assert call_decision.action is Action.RETRAIN_WITH_REPLAY
    assert call_provider.calls == 0
