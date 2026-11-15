from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..domain import PerformanceDriftResult


class PageHinkley:
    def __init__(self, delta: float = 0.005, threshold: float = 0.12) -> None:
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.cumulative = 0.0
        self.minimum = 0.0

    def update(self, value: float) -> bool:
        self.count += 1
        self.mean += (value - self.mean) / self.count
        self.cumulative += value - self.mean - self.delta
        self.minimum = min(self.minimum, self.cumulative)
        return (self.cumulative - self.minimum) > self.threshold


class PerformanceDriftDetector:
    def __init__(self, baseline_score: float, drop_threshold: float, window: int = 3) -> None:
        self.baseline_score = baseline_score
        self.drop_threshold = drop_threshold
        self.scores: deque[float] = deque(maxlen=window)
        # Page-Hinkley 喂"相对基线的性能下降量"（准确率与负 RMSE 都是越大越好，所以
        # ``baseline - score`` 对两者都表示"变差了多少"）。delta 与阈值随 drop_threshold
        # 缩放，从而对 [0,1] 的分类分数与负 RMSE 回归分数两种尺度都成立。
        self.page_hinkley = PageHinkley(
            delta=max(drop_threshold * 0.05, 1e-4),
            threshold=max(drop_threshold * 2.0, 0.10),
        )

    def update(self, score: float) -> PerformanceDriftResult:
        self.scores.append(float(score))
        rolling = float(np.mean(self.scores))
        drop = max(0.0, self.baseline_score - rolling)
        ph_alarm = self.page_hinkley.update(self.baseline_score - float(score))
        enough_history = len(self.scores) == self.scores.maxlen
        is_drift = enough_history and (drop >= self.drop_threshold or ph_alarm)
        return PerformanceDriftResult(
            current_score=round(float(score), 6),
            rolling_score=round(rolling, 6),
            baseline_score=round(self.baseline_score, 6),
            score_drop=round(drop, 6),
            page_hinkley_alarm=ph_alarm,
            is_drift=is_drift,
        )

    def set_baseline(self, score: float) -> None:
        self.baseline_score = float(score)
        self.scores.clear()
        self.page_hinkley.reset()

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "baseline_score": self.baseline_score,
            "scores": list(self.scores),
            "window": self.scores.maxlen,
            "page_hinkley": {
                "count": self.page_hinkley.count,
                "mean": self.page_hinkley.mean,
                "cumulative": self.page_hinkley.cumulative,
                "minimum": self.page_hinkley.minimum,
            },
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        self.baseline_score = float(state["baseline_score"])
        self.scores = deque(
            (float(value) for value in state.get("scores", [])),
            maxlen=int(state.get("window") or 3),
        )
        page_state = state.get("page_hinkley", {})
        self.page_hinkley.count = int(page_state.get("count", 0))
        self.page_hinkley.mean = float(page_state.get("mean", 0.0))
        self.page_hinkley.cumulative = float(page_state.get("cumulative", 0.0))
        self.page_hinkley.minimum = float(page_state.get("minimum", 0.0))
