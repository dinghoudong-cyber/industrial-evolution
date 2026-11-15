import pytest

from industrial_evolution.budget import BudgetExceededError, BudgetLedger
from industrial_evolution.config import BudgetConfig, MonitoringConfig, PolicyConfig
from industrial_evolution.domain import (
    Action,
    DriftKind,
    DriftReport,
    FeatureDriftResult,
    PerformanceDriftResult,
    PolicyContext,
)
from industrial_evolution.policy.rule_policy import RuleBasedPolicy


def actionable_report() -> DriftReport:
    return DriftReport(
        kind=DriftKind.CONCEPT_SUSPECTED,
        feature=FeatureDriftResult(1.2, True, {}, ["sensor_00"]),
        performance=PerformanceDriftResult(0.6, 0.62, 0.8, 0.18, True, True),
        label_shift=0.0,
        is_actionable=True,
        explanation="测试漂移",
    )


def policy_context(
    window_index: int,
    can_train: bool = True,
    experiment_history: tuple[dict, ...] = (),
) -> PolicyContext:
    ledger = BudgetLedger(
        BudgetConfig(
            max_updates=10,
            max_training_seconds=100.0,
            max_labels=1000,
            max_tokens=5000,
        )
    )
    return PolicyContext(
        window_index=window_index,
        report=actionable_report(),
        budget=ledger.snapshot(),
        requested_labels=20,
        can_train=can_train,
        experiment_history=experiment_history,
    )


def test_policy_requires_confirmation_and_cooldown():
    monitoring = MonitoringConfig(
        psi_threshold=0.18,
        ks_threshold=0.18,
        performance_drop_threshold=0.06,
        label_shift_threshold=0.12,
        consecutive_windows=2,
        reference_update_rate=0.15,
    )
    policy = RuleBasedPolicy(
        PolicyConfig(cooldown_windows=2, rollback_patience=2, rollback_drop=0.1),
        monitoring,
    )

    assert policy.decide(policy_context(1)).action is Action.NO_UPDATE
    assert policy.decide(policy_context(2)).action is Action.RETRAIN_WITH_REPLAY
    policy.mark_update(2)
    assert policy.decide(policy_context(3)).action is Action.NO_UPDATE
    assert policy.decide(policy_context(4)).action is Action.NO_UPDATE
    assert policy.decide(policy_context(5)).action is Action.RETRAIN_WITH_REPLAY


def test_policy_uses_rejected_experiment_history_to_explain_retest():
    policy = RuleBasedPolicy(
        PolicyConfig(cooldown_windows=0, rollback_patience=2, rollback_drop=0.1),
        MonitoringConfig(
            psi_threshold=0.18,
            ks_threshold=0.18,
            performance_drop_threshold=0.06,
            label_shift_threshold=0.12,
            consecutive_windows=1,
            reference_update_rate=0.15,
        ),
    )
    history = (
        {
            "reference": "run-public:window:2",
            "drift_kind": DriftKind.CONCEPT_SUSPECTED.value,
            "top_features": ["sensor_00"],
            "action": Action.RETRAIN_WITH_REPLAY.value,
            "status": "rejected",
            "candidate_improvement": -0.03,
        },
    )

    decision = policy.decide(policy_context(3, experiment_history=history))

    assert decision.action is Action.RETRAIN_WITH_REPLAY
    assert "run-public:window:2" in decision.reason
    assert "历史守护集门槛" in decision.reason


def test_policy_retries_when_drift_signature_differs_from_failed_history():
    policy = RuleBasedPolicy(
        PolicyConfig(cooldown_windows=0, rollback_patience=2, rollback_drop=0.1),
        MonitoringConfig(
            psi_threshold=0.18,
            ks_threshold=0.18,
            performance_drop_threshold=0.06,
            label_shift_threshold=0.12,
            consecutive_windows=1,
            reference_update_rate=0.15,
        ),
    )
    history = (
        {
            "reference": "run-public:window:2",
            "drift_kind": DriftKind.CONCEPT_SUSPECTED.value,
            "top_features": ["another_sensor"],
            "action": Action.RETRAIN_WITH_REPLAY.value,
            "status": "rejected",
        },
    )

    decision = policy.decide(policy_context(3, experiment_history=history))

    assert decision.action is Action.RETRAIN_WITH_REPLAY


def test_latest_successful_experiment_supersedes_an_older_rejection():
    policy = RuleBasedPolicy(
        PolicyConfig(cooldown_windows=0, rollback_patience=2, rollback_drop=0.1),
        MonitoringConfig(
            psi_threshold=0.18,
            ks_threshold=0.18,
            performance_drop_threshold=0.06,
            label_shift_threshold=0.12,
            consecutive_windows=1,
            reference_update_rate=0.15,
        ),
    )
    signature = {
        "drift_kind": DriftKind.CONCEPT_SUSPECTED.value,
        "top_features": ["sensor_00"],
        "action": Action.RETRAIN_WITH_REPLAY.value,
    }
    history = (
        {
            **signature,
            "reference": "run-public:window:1",
            "status": "rejected",
        },
        {
            **signature,
            "reference": "run-public:window:2",
            "status": "deployed",
        },
    )

    decision = policy.decide(policy_context(3, experiment_history=history))

    assert decision.action is Action.RETRAIN_WITH_REPLAY


def test_budget_is_hard_limit_and_labels_are_deduplicated():
    ledger = BudgetLedger(
        BudgetConfig(
            max_updates=1,
            max_training_seconds=10.0,
            max_labels=3,
            max_tokens=0,
            label_budget_mode="enforce",
        )
    )
    assert ledger.acquire_labels(["a", "b"]) == 2
    assert ledger.acquire_labels(["a", "b"]) == 0
    assert ledger.acquire_labels(["b", "c"]) == 1
    can_acquire, reason, new_count = ledger.can_acquire_labels(["a", "b", "c", "c"])
    assert can_acquire
    assert new_count == 0
    can_acquire, reason, new_count = ledger.can_acquire_labels(["c", "d", "d"])
    assert not can_acquire
    assert "标注预算" in reason
    assert new_count == 1
    snapshot = ledger.snapshot()
    assert snapshot.labels_used == 3
    assert snapshot.budget_violations == 0
    with pytest.raises(BudgetExceededError, match="标注预算"):
        ledger.acquire_labels(["d"])

    ledger.record_update(seconds=1.0)
    allowed, reason = ledger.can_train()
    assert not allowed
    assert "更新次数" in reason
    with pytest.raises(BudgetExceededError, match="Token"):
        ledger.record_tokens(25)
    snapshot = ledger.snapshot()
    assert snapshot.labels_used == 3
    assert snapshot.tokens_used == 0
    assert snapshot.budget_violations == 1
