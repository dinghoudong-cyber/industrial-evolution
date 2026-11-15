from __future__ import annotations

from typing import Any

from ..config import MonitoringConfig
from ..domain import DriftKind, DriftReport, FeatureDriftResult, PerformanceDriftResult


class DriftDiagnoser:
    def __init__(
        self,
        config: MonitoringConfig,
        reference_label_distribution: dict[str, float],
    ) -> None:
        self.config = config
        self.reference_label_distribution = dict(reference_label_distribution)

    def diagnose(
        self,
        feature: FeatureDriftResult,
        performance: PerformanceDriftResult,
        current_label_distribution: dict[str, float],
    ) -> DriftReport:
        label_shift = self._distribution_shift(current_label_distribution)
        label_alarm = label_shift >= self.config.label_shift_threshold
        feature_alarm = feature.is_drift
        performance_alarm = performance.is_drift

        if performance_alarm and (feature_alarm or label_alarm):
            kind = DriftKind.MIXED
        elif performance_alarm:
            kind = DriftKind.CONCEPT_SUSPECTED
        elif feature_alarm and label_alarm:
            kind = DriftKind.MIXED
        elif feature_alarm:
            kind = DriftKind.COVARIATE
        elif label_alarm:
            kind = DriftKind.LABEL
        else:
            kind = DriftKind.NONE

        reasons: list[str] = []
        if feature_alarm:
            reasons.append(f"输入漂移，重点特征：{', '.join(feature.top_features)}")
        if performance_alarm:
            reasons.append(f"滚动性能下降 {performance.score_drop:.4f}")
        if label_alarm:
            reasons.append(f"标签分布变化 {label_shift:.4f}")
        if not reasons:
            reasons.append("未检测到达到阈值的漂移")

        return DriftReport(
            kind=kind,
            feature=feature,
            performance=performance,
            label_shift=round(label_shift, 6),
            is_actionable=kind is not DriftKind.NONE,
            explanation="；".join(reasons),
        )

    def _distribution_shift(self, current: dict[str, float]) -> float:
        labels = set(self.reference_label_distribution) | set(current)
        return 0.5 * sum(
            abs(current.get(label, 0.0) - self.reference_label_distribution.get(label, 0.0))
            for label in labels
        )

    def update_reference_label_distribution(self, current: dict[str, float]) -> None:
        rate = self.config.reference_update_rate
        labels = set(self.reference_label_distribution) | set(current)
        self.reference_label_distribution = {
            label: (1.0 - rate) * self.reference_label_distribution.get(label, 0.0)
            + rate * current.get(label, 0.0)
            for label in labels
        }

    def snapshot_state(self) -> dict[str, Any]:
        return {"reference_label_distribution": dict(self.reference_label_distribution)}

    def restore_state(self, state: dict[str, Any]) -> None:
        distribution = state.get("reference_label_distribution")
        if not isinstance(distribution, dict):
            raise ValueError("标签漂移快照缺少参考分布")
        self.reference_label_distribution = {
            str(label): float(rate) for label, rate in distribution.items()
        }
