from __future__ import annotations

from typing import Any

from ..config import MonitoringConfig, PolicyConfig
from ..domain import Action, Decision, PolicyContext


class RuleBasedPolicy:
    def __init__(self, policy: PolicyConfig, monitoring: MonitoringConfig) -> None:
        self.policy = policy
        self.required_signals = monitoring.consecutive_windows
        self.signal_streak = 0
        self.last_update_window: int | None = None

    def decide(self, context: PolicyContext) -> Decision:
        report = context.report
        if report.is_actionable:
            self.signal_streak += 1
        else:
            self.signal_streak = 0
            return Decision(Action.NO_UPDATE, "当前窗口无可执行漂移")

        if self.signal_streak < self.required_signals:
            return Decision(
                Action.NO_UPDATE,
                f"等待漂移连续确认（{self.signal_streak}/{self.required_signals}）",
            )

        if self.last_update_window is not None:
            elapsed = context.window_index - self.last_update_window
            if elapsed <= self.policy.cooldown_windows:
                remaining = self.policy.cooldown_windows - elapsed + 1
                return Decision(Action.NO_UPDATE, f"更新冷却期内，还需等待 {remaining} 个窗口")

        if not context.can_train:
            return Decision(Action.NO_UPDATE, "漂移已确认，但训练预算不允许")

        prior_failure = self._prior_failed_experiment(context)
        if prior_failure is not None:
            return Decision(
                Action.RETRAIN_WITH_REPLAY,
                "历史同类实验未通过候选门槛；本窗口仍有新观测，继续验证候选，"
                "但必须通过近期提升和历史守护集门槛；"
                f"历史记录={prior_failure['reference']}，"
                f"上次候选增益={prior_failure.get('candidate_improvement')}",
            )

        return Decision(Action.RETRAIN_WITH_REPLAY, report.explanation)

    @staticmethod
    def _prior_failed_experiment(context: PolicyContext) -> dict[str, Any] | None:
        signature = (context.report.kind.value, tuple(context.report.feature.top_features))
        for entry in reversed(context.experiment_history):
            entry_signature = (
                entry.get("drift_kind"),
                tuple(entry.get("top_features", ())),
            )
            if (
                entry_signature == signature
                and entry.get("action") == Action.RETRAIN_WITH_REPLAY.value
            ):
                return entry if entry.get("status") == "rejected" else None
        return None

    def mark_update(self, window_index: int) -> None:
        self.last_update_window = window_index
        self.signal_streak = 0

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "signal_streak": self.signal_streak,
            "last_update_window": self.last_update_window,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.signal_streak = int(state.get("signal_streak", 0))
        last = state.get("last_update_window")
        self.last_update_window = None if last is None else int(last)
