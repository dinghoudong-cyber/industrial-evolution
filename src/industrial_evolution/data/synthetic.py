from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from ..config import StreamConfig
from ..domain import DataWindow


class SyntheticDriftStream:
    """生成包含稳定、渐进漂移和突发漂移阶段的二分类数据流。"""

    def __init__(self, config: StreamConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed
        self._base_weights = np.linspace(1.4, -0.8, config.n_features)
        self._feature_names = [f"sensor_{i:02d}" for i in range(config.n_features)]

    def _rng(self, offset: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + offset * 7919)

    def _sample(self, n_samples: int, window_index: int) -> tuple[pd.DataFrame, np.ndarray, str]:
        rng = self._rng(window_index + 1)
        X = rng.normal(0.0, 1.0, size=(n_samples, self.config.n_features))
        weights = self._base_weights.copy()
        bias = -0.1
        phase = "stable"

        if window_index >= self.config.gradual_drift_start:
            span = max(1, self.config.abrupt_drift_start - self.config.gradual_drift_start)
            strength = min(1.0, (window_index - self.config.gradual_drift_start + 1) / span)
            X[:, 0] += 0.9 * strength
            X[:, 1] *= 1.0 + 0.5 * strength
            weights[0] -= 2.0 * strength
            weights[1] += 1.5 * strength
            bias += 0.45 * strength
            phase = "gradual"

        if window_index >= self.config.abrupt_drift_start:
            X[:, 0] += 1.4
            X[:, 2] -= 1.0
            weights[0] = -1.8
            weights[1] = 1.5
            weights[2] = -1.6
            bias = -0.65
            phase = "abrupt"

        logits = X @ weights + bias + rng.normal(0.0, 0.45, size=n_samples)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        y = rng.binomial(1, probabilities).astype(np.int64)

        if self.config.label_noise > 0:
            flips = rng.random(n_samples) < self.config.label_noise
            y[flips] = 1 - y[flips]

        return pd.DataFrame(X, columns=self._feature_names), y, phase

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        X, y, _ = self._sample(self.config.initial_samples, -1)
        return X, y

    def __iter__(self) -> Iterator[DataWindow]:
        for index in range(self.config.n_windows):
            X, y, phase = self._sample(self.config.window_size, index)
            start = self.config.initial_samples + index * self.config.window_size
            sample_ids = np.arange(start, start + self.config.window_size, dtype=np.int64)
            yield DataWindow(
                index=index,
                X=X,
                y=y,
                phase=phase,
                sample_ids=sample_ids,
            )
