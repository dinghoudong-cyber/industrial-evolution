from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from industrial_evolution.config import load_config
from industrial_evolution.data.synthetic import SyntheticDriftStream
from industrial_evolution.domain import Action, LLMUsage
from industrial_evolution.llm.provider import LLMDecisionRequest, LLMDecisionResponse
from industrial_evolution.orchestrator import EvolutionOrchestrator
from industrial_evolution.policy.hybrid_policy import HybridPolicy
from industrial_evolution.policy.rule_policy import RuleBasedPolicy
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker


@dataclass
class ApprovingProvider:
    calls: int = 0

    def decide(self, request: LLMDecisionRequest) -> LLMDecisionResponse:
        self.calls += 1
        return LLMDecisionResponse(
            action=Action.RETRAIN_WITH_REPLAY,
            confidence=0.9,
            reason="模拟策略模型批准规则训练建议",
            model="local-model",
            usage=LLMUsage(100, 20, 0, 120),
            latency_seconds=0.01,
        )


def test_hybrid_policy_records_tokens_without_exposing_credentials(tmp_path: Path):
    project = Path(__file__).parents[1]
    config = load_config(project / "configs" / "llm-local.yaml")
    config.tracking.run_root = tmp_path
    provider = ApprovingProvider()
    policy = HybridPolicy(
        RuleBasedPolicy(config.policy, config.monitoring),
        provider,
        config.llm,
    )
    local = LocalTracker(tmp_path)

    result = EvolutionOrchestrator(
        config,
        SyntheticDriftStream(config.stream, config.seed),
        CompositeTracker(local),
        policy=policy,
    ).run()

    assert provider.calls == 6
    assert result["llm_attempts"] == 6
    assert result["llm_calls"] == 6
    assert result["llm_fallbacks"] == 0
    assert result["budget"]["tokens_used"] == 720
    assert result["updates_attempted"] == 6
    assert result["updates_deployed"] == 4
    assert result["mean_score"] == 0.6762
    assert result["budget"]["labels_used"] == 4200

    events = local.events_path.read_text(encoding="utf-8")
    assert '"event_type": "llm_decision"' in events
    assert "LOCAL_LLM_API_KEY" not in events

    with local.metrics_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    llm_rows = [row for row in rows if int(row["llm_total_tokens"]) > 0]
    assert len(llm_rows) == 6
    assert sum(int(row["llm_total_tokens"]) for row in llm_rows) == 720


def test_actual_token_overrun_is_preserved_in_hypothesis_chain(tmp_path: Path):
    project = Path(__file__).parents[1]
    config = load_config(project / "configs" / "llm-local.yaml")
    config.tracking.run_root = tmp_path
    config.budget.max_tokens = 100
    config.llm.max_tokens_per_call = 64
    config.llm.max_calls = 1
    provider = ApprovingProvider()
    policy = HybridPolicy(
        RuleBasedPolicy(config.policy, config.monitoring),
        provider,
        config.llm,
    )
    local = LocalTracker(tmp_path)

    result = EvolutionOrchestrator(
        config,
        SyntheticDriftStream(config.stream, config.seed),
        CompositeTracker(local),
        policy=policy,
    ).run()

    assert provider.calls == 1
    assert result["llm_calls"] == 1
    assert result["failures"] >= 1
    assert result["budget"]["budget_violations"] >= 1

    events = [
        json.loads(line) for line in local.events_path.read_text(encoding="utf-8").splitlines()
    ]
    hypotheses = [event for event in events if event["event_type"] == "hypothesis_recorded"]
    overrun = next(
        event["payload"]
        for event in hypotheses
        if event["payload"]["outcome"]["status"] == "budget_exhausted"
        and event["payload"]["decision"]["llm_called"]
    )
    decision = overrun["decision"]
    assert decision["action"] == "error"
    assert decision["proposed_action"] == "retrain_with_replay"
    assert decision["usage"]["total_tokens"] == 120
    assert "Token" in decision["reason"]
    assert decision["fallback_reason"] is None

    window = overrun["window"]
    call_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "llm_decision" and event["payload"].get("window") == window
    )
    failure_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "window_failure" and event["payload"].get("window") == window
    )
    hypothesis_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "hypothesis_recorded" and event["payload"].get("window") == window
    )
    assert call_index < failure_index < hypothesis_index
