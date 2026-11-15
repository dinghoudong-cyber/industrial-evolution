from __future__ import annotations

from collections.abc import Iterable

from .config import BudgetConfig
from .domain import BudgetSnapshot


class BudgetExceededError(RuntimeError):
    """资源消耗将突破配置上限。"""


class BudgetLedger:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.updates_used = 0
        self.training_seconds_used = 0.0
        self.labels_used = 0
        self.tokens_used = 0
        self.budget_violations = 0
        self._labeled_sample_ids: set[str] = set()

    @property
    def _enforce_labels(self) -> bool:
        """标签是否作为硬配额。按批次提供标签的运行模式下，配额记账与窗口推进解耦。"""
        return self.config.label_budget_mode == "enforce"

    def can_acquire_labels(self, sample_ids: Iterable[object]) -> tuple[bool, str, int]:
        normalized = {str(sample_id) for sample_id in sample_ids}
        new_count = len(normalized - self._labeled_sample_ids)
        if not self._enforce_labels:
            return True, "评测标签不占配额，仅记账", new_count
        if self.labels_used + new_count > self.config.max_labels:
            return False, "标注预算不足", new_count
        return True, "标注预算允许", new_count

    def acquire_labels(self, sample_ids: Iterable[object]) -> int:
        normalized = {str(sample_id) for sample_id in sample_ids}
        new_ids = normalized - self._labeled_sample_ids
        if self._enforce_labels and self.labels_used + len(new_ids) > self.config.max_labels:
            raise BudgetExceededError("标注预算不足，拒绝揭示本批标签")
        self._labeled_sample_ids.update(new_ids)
        self.labels_used += len(new_ids)
        return len(new_ids)

    def can_train(self, new_labels: int = 0, tokens: int = 0) -> tuple[bool, str]:
        if self.updates_used >= self.config.max_updates:
            return False, "更新次数预算已耗尽"
        if self.training_seconds_used >= self.config.max_training_seconds:
            return False, "训练时间预算已耗尽"
        if self._enforce_labels and self.labels_used + new_labels > self.config.max_labels:
            return False, "标注预算不足"
        if self.tokens_used + tokens > self.config.max_tokens:
            return False, "Token 预算不足"
        return True, "预算允许"

    def can_start_initial_training(self) -> tuple[bool, str]:
        if self.training_seconds_used >= self.config.max_training_seconds:
            return False, "训练时间预算已耗尽"
        return True, "预算允许"

    def record_initial_training(self, seconds: float, labels: int = 0) -> None:
        if labels:
            if self._enforce_labels and self.labels_used + labels > self.config.max_labels:
                raise BudgetExceededError("初始训练标签将突破标注预算")
            self.labels_used += max(0, labels)
        self._record_training_seconds(seconds)

    def record_update(self, seconds: float, new_labels: int = 0, tokens: int = 0) -> None:
        if self.updates_used >= self.config.max_updates:
            raise BudgetExceededError("更新次数预算已耗尽")
        if self._enforce_labels and self.labels_used + max(0, new_labels) > self.config.max_labels:
            raise BudgetExceededError("本次更新将突破标注预算")
        if self.tokens_used + max(0, tokens) > self.config.max_tokens:
            raise BudgetExceededError("本次更新将突破Token预算")
        self.updates_used += 1
        self.labels_used += max(0, new_labels)
        self.tokens_used += max(0, tokens)
        self._record_training_seconds(seconds)

    def record_tokens(self, tokens: int) -> None:
        tokens = max(0, tokens)
        if self.tokens_used + tokens > self.config.max_tokens:
            self.budget_violations += 1
            raise BudgetExceededError("实际Token用量超过剩余预算")
        self.tokens_used += tokens

    def _record_training_seconds(self, seconds: float) -> None:
        self.training_seconds_used += max(0.0, seconds)
        if self.training_seconds_used > self.config.max_training_seconds:
            self.budget_violations += 1
            raise BudgetExceededError("实际训练时间超过配置上限")

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            updates_used=self.updates_used,
            max_updates=self.config.max_updates,
            training_seconds_used=round(self.training_seconds_used, 6),
            max_training_seconds=self.config.max_training_seconds,
            labels_used=self.labels_used,
            max_labels=self.config.max_labels,
            tokens_used=self.tokens_used,
            max_tokens=self.config.max_tokens,
            budget_violations=self.budget_violations,
        )
