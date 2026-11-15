from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Any

from ..config import LLMConfig
from ..domain import Action, Decision, LLMUsage, PolicyContext
from ..llm.provider import LLMDecisionRequest, LLMDecisionResponse, PolicyLLMProvider
from .rule_policy import RuleBasedPolicy


class HybridPolicy:
    """规则先行、LLM受控参与、失败自动回退的更新策略。"""

    def __init__(
        self,
        rule_policy: RuleBasedPolicy,
        provider: PolicyLLMProvider,
        config: LLMConfig,
    ) -> None:
        self.rule_policy = rule_policy
        self.provider = provider
        self.config = config
        self.calls_used = 0
        self.fallbacks_used = 0
        self._cached_veto: LLMDecisionResponse | None = None
        self._cached_veto_until: int | None = None

    def decide(self, context: PolicyContext) -> Decision:
        rule_decision = self.rule_policy.decide(context)
        if rule_decision.action is not Action.RETRAIN_WITH_REPLAY:
            return rule_decision

        cached = self._cached_decision(context.window_index)
        if cached is not None:
            return cached

        if self.calls_used >= self.config.max_calls:
            return self._fallback(rule_decision, "LLM调用次数预算已耗尽")

        remaining_tokens = context.budget.max_tokens - context.budget.tokens_used
        if remaining_tokens < self.config.max_tokens_per_call:
            return self._fallback(rule_decision, "剩余Token不足以发起受控调用")

        request = self._build_request(context, rule_decision)
        self.calls_used += 1
        started = perf_counter()
        try:
            response = self.provider.decide(request)
        except Exception as exc:
            elapsed = perf_counter() - started
            usage = getattr(exc, "usage", None)
            if not isinstance(usage, LLMUsage):
                usage = LLMUsage()
            latency = float(getattr(exc, "latency_seconds", 0.0) or elapsed)
            model = getattr(exc, "model", None)
            return self._fallback(
                rule_decision,
                f"LLM不可用：{type(exc).__name__}: {exc}",
                usage=usage,
                model=model,
                llm_attempted=True,
                latency_seconds=latency,
            )

        if response.confidence < self.config.min_confidence:
            return self._fallback(
                rule_decision,
                f"LLM置信度 {response.confidence:.3f} 低于门槛",
                response=response,
                llm_attempted=True,
                llm_called=True,
            )

        decision = Decision(
            action=response.action,
            reason=response.reason,
            source="llm",
            model=response.model,
            confidence=response.confidence,
            usage=response.usage,
            latency_seconds=response.latency_seconds,
            llm_attempted=True,
            llm_called=True,
        )
        if response.action is Action.NO_UPDATE:
            self._cached_veto = response
            self._cached_veto_until = context.window_index + self.config.call_cooldown_windows
        return decision

    def mark_update(self, window_index: int) -> None:
        self.rule_policy.mark_update(window_index)
        self._cached_veto = None
        self._cached_veto_until = None

    def snapshot_state(self) -> dict[str, Any]:
        return {"rule_policy": self.rule_policy.snapshot_state()}

    def restore_state(self, state: dict[str, Any]) -> None:
        rule_state = state.get("rule_policy")
        if not isinstance(rule_state, dict):
            raise ValueError("混合策略快照缺少规则策略状态")
        self.rule_policy.restore_state(rule_state)
        self._cached_veto = None
        self._cached_veto_until = None

    def _cached_decision(self, window_index: int) -> Decision | None:
        if (
            self._cached_veto is None
            or self._cached_veto_until is None
            or window_index > self._cached_veto_until
        ):
            self._cached_veto = None
            self._cached_veto_until = None
            return None
        return Decision(
            action=Action.NO_UPDATE,
            reason=f"复用近期LLM否决：{self._cached_veto.reason}",
            source="llm_cache",
            model=self._cached_veto.model,
            confidence=self._cached_veto.confidence,
        )

    def _fallback(
        self,
        rule_decision: Decision,
        reason: str,
        *,
        response: LLMDecisionResponse | None = None,
        usage: LLMUsage | None = None,
        model: str | None = None,
        llm_attempted: bool = False,
        llm_called: bool = False,
        latency_seconds: float = 0.0,
    ) -> Decision:
        self.fallbacks_used += 1
        if response is None:
            return replace(
                rule_decision,
                source="rule_fallback",
                model=model or (self.config.model if llm_called else None),
                usage=usage or LLMUsage(),
                latency_seconds=latency_seconds,
                fallback_reason=reason,
                llm_attempted=llm_attempted,
                llm_called=llm_called,
            )
        return replace(
            rule_decision,
            source="rule_fallback",
            model=response.model,
            confidence=response.confidence,
            usage=response.usage,
            latency_seconds=response.latency_seconds,
            fallback_reason=reason,
            llm_attempted=llm_attempted,
            llm_called=llm_called,
        )

    @staticmethod
    def _build_request(
        context: PolicyContext,
        rule_decision: Decision,
    ) -> LLMDecisionRequest:
        budget = context.budget
        report = context.report
        performance = report.performance
        return LLMDecisionRequest(
            window_index=context.window_index,
            drift_kind=report.kind.value,
            feature_drift_score=report.feature.overall_score,
            top_features=tuple(report.feature.top_features),
            current_score=performance.current_score,
            rolling_score=performance.rolling_score,
            baseline_score=performance.baseline_score,
            performance_drop=performance.score_drop,
            label_shift=report.label_shift,
            remaining_updates=max(0, budget.max_updates - budget.updates_used),
            remaining_training_seconds=max(
                0.0,
                budget.max_training_seconds - budget.training_seconds_used,
            ),
            remaining_labels=max(0, budget.max_labels - budget.labels_used),
            remaining_tokens=max(0, budget.max_tokens - budget.tokens_used),
            requested_labels=context.requested_labels,
            rule_reason=rule_decision.reason,
        )
