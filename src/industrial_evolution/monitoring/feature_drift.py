from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ..config import MonitoringConfig
from ..domain import FeatureDriftResult

PSI_BINS = 10
TOP_FEATURE_COUNT = 3
MIN_DRIFTED_FEATURE_RATIO = 0.20
WASSERSTEIN_SEVERITY_SCALE = 0.35
OVERALL_DRIFT_THRESHOLD = 1.25


@dataclass(frozen=True, slots=True)
class _ReferenceStats:
    sorted_finite_values: np.ndarray
    scale: float
    psi_edges: np.ndarray
    psi_ratio: np.ndarray
    non_finite_rate: float


def _psi_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.array([-np.inf, float(np.median(reference)), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _histogram_ratio(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    return np.maximum(counts / max(1, counts.sum()), 1e-6)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = PSI_BINS,
) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = _psi_edges(reference, bins)
    ref_ratio = _histogram_ratio(reference, edges)
    cur_ratio = _histogram_ratio(current, edges)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def _population_stability_from_stats(
    reference: _ReferenceStats,
    current: np.ndarray,
) -> float:
    current = np.asarray(current, dtype=float)
    current = current[np.isfinite(current)]
    if len(reference.sorted_finite_values) == 0 or len(current) == 0:
        return 0.0
    current_ratio = _histogram_ratio(current, reference.psi_edges)
    return float(
        np.sum((current_ratio - reference.psi_ratio) * np.log(current_ratio / reference.psi_ratio))
    )


def _ks_statistic(sorted_reference: np.ndarray, current: np.ndarray) -> float:
    current = np.asarray(current, dtype=float)
    sorted_current = np.sort(current[np.isfinite(current)])
    if len(sorted_reference) == 0 or len(sorted_current) == 0:
        return 0.0

    sample_points = np.concatenate([sorted_reference, sorted_current])
    reference_cdf = np.searchsorted(sorted_reference, sample_points, side="right")
    current_cdf = np.searchsorted(sorted_current, sample_points, side="right")
    differences = np.abs(reference_cdf / len(sorted_reference) - current_cdf / len(sorted_current))
    return float(np.max(differences))


class FeatureDriftDetector:
    def __init__(self, reference: pd.DataFrame, config: MonitoringConfig) -> None:
        self.config = config
        self.reference = reference.copy().reset_index(drop=True)
        self._reference_stats: dict[str, _ReferenceStats] = {}
        self._refresh_reference_stats()

    def _refresh_reference_stats(self) -> None:
        self._reference_stats = {}
        for column in self.reference.columns:
            values = self.reference[column].to_numpy(dtype=float)
            finite_values = values[np.isfinite(values)]
            if len(finite_values):
                psi_edges = _psi_edges(finite_values, PSI_BINS)
                psi_ratio = _histogram_ratio(finite_values, psi_edges)
                scale = max(float(np.std(finite_values)), 1e-6)
            else:
                psi_edges = np.array([])
                psi_ratio = np.array([])
                scale = 1e-6
            self._reference_stats[column] = _ReferenceStats(
                sorted_finite_values=np.sort(finite_values),
                scale=scale,
                psi_edges=psi_edges,
                psi_ratio=psi_ratio,
                non_finite_rate=1.0 - len(finite_values) / max(1, len(values)),
            )

    def _measure_feature(
        self,
        column: str,
        current: pd.DataFrame,
    ) -> tuple[dict[str, float], float, bool, str | None]:
        reference = self._reference_stats[column]
        current_values = current[column].to_numpy(dtype=float)
        finite_current = current_values[np.isfinite(current_values)]
        non_finite_rate = 1.0 - len(finite_current) / max(1, len(current_values))
        non_finite_delta = abs(non_finite_rate - reference.non_finite_rate)
        if len(finite_current) < self.config.min_valid_feature_values:
            metrics = {
                "psi": 0.0,
                "ks": 0.0,
                "wasserstein_normalized": 0.0,
                "non_finite_rate": round(non_finite_rate, 6),
                "non_finite_rate_delta": round(non_finite_delta, 6),
                "severity": 0.0,
            }
            return metrics, 0.0, False, f"{column} 有效值不足"
        if len(reference.sorted_finite_values) < self.config.min_valid_feature_values:
            metrics = {
                "psi": 0.0,
                "ks": 0.0,
                "wasserstein_normalized": 0.0,
                "non_finite_rate": round(non_finite_rate, 6),
                "non_finite_rate_delta": round(non_finite_delta, 6),
                "severity": 0.0,
            }
            return metrics, 0.0, False, f"{column} 参考有效值不足"

        psi = _population_stability_from_stats(reference, finite_current)
        ks = _ks_statistic(reference.sorted_finite_values, finite_current)
        wasserstein = float(
            wasserstein_distance(reference.sorted_finite_values, finite_current) / reference.scale
        )
        missing_severity = non_finite_delta / self.config.non_finite_rate_threshold
        severity = max(
            psi / self.config.psi_threshold,
            ks / self.config.ks_threshold,
            wasserstein / WASSERSTEIN_SEVERITY_SCALE,
            missing_severity,
        )
        is_drift = (
            psi >= self.config.psi_threshold
            or ks >= self.config.ks_threshold
            or non_finite_delta >= self.config.non_finite_rate_threshold
        )
        metrics = {
            "psi": round(psi, 6),
            "ks": round(ks, 6),
            "wasserstein_normalized": round(wasserstein, 6),
            "non_finite_rate": round(non_finite_rate, 6),
            "non_finite_rate_delta": round(non_finite_delta, 6),
            "severity": round(severity, 6),
        }
        return metrics, severity, is_drift, None

    def detect(self, current: pd.DataFrame) -> FeatureDriftResult:
        if list(current.columns) != list(self.reference.columns):
            raise ValueError("当前窗口特征列与参考数据不一致")

        per_feature: dict[str, dict[str, float]] = {}
        severities: list[tuple[str, float]] = []
        issues: list[str] = []
        drifted_count = 0
        for column in self.reference.columns:
            metrics, severity, is_drift, issue = self._measure_feature(column, current)
            per_feature[column] = metrics
            severities.append((column, severity))
            drifted_count += int(is_drift)
            if issue:
                issues.append(issue)

        severities.sort(key=lambda item: item[1], reverse=True)
        top_count = min(TOP_FEATURE_COUNT, len(severities))
        overall = float(np.mean([value for _, value in severities[:top_count]]))
        minimum_features = max(
            1,
            math.ceil(len(severities) * MIN_DRIFTED_FEATURE_RATIO),
        )
        is_drift = drifted_count >= minimum_features or overall >= OVERALL_DRIFT_THRESHOLD
        return FeatureDriftResult(
            overall_score=round(overall, 6),
            is_drift=is_drift,
            per_feature=per_feature,
            top_features=[name for name, _ in severities[:top_count]],
            data_quality_valid=not issues,
            data_quality_issues=issues,
        )

    def update_reference(self, current: pd.DataFrame) -> None:
        rate = self.config.reference_update_rate
        if rate <= 0:
            return

        target_size = max(len(self.reference), len(current))
        old_count = min(len(self.reference), int(round(target_size * (1.0 - rate))))
        new_count = min(len(current), target_size - old_count)
        old = (
            self.reference.sample(n=old_count, random_state=17)
            if old_count
            else self.reference.iloc[:0]
        )
        new = current.sample(n=new_count, random_state=29) if new_count else current.iloc[:0]
        self.reference = pd.concat([old, new], ignore_index=True)
        self._refresh_reference_stats()

    def snapshot_state(self) -> dict[str, Any]:
        return {"reference": self.reference.copy(deep=True)}

    def restore_state(self, state: dict[str, Any]) -> None:
        reference = state.get("reference")
        if not isinstance(reference, pd.DataFrame):
            raise ValueError("特征漂移快照缺少参考数据")
        self.reference = reference.copy(deep=True).reset_index(drop=True)
        self._refresh_reference_stats()
