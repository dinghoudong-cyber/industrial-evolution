from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from ..config import TrainingConfig
from ..memory.replay_buffer import ReplayBuffer
from ..models.sklearn_adapter import SklearnModelAdapter


@dataclass(slots=True)
class TrainingOutcome:
    model: SklearnModelAdapter
    seconds: float
    training_samples: int
    reference_X: pd.DataFrame
    training_sample_ids: np.ndarray


class CandidateTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(
        self,
        champion: SklearnModelAdapter,
        recent_X: pd.DataFrame,
        recent_y: np.ndarray,
        replay: ReplayBuffer,
        recent_sample_ids: np.ndarray | None = None,
        *,
        exclude_ids: set[object] | None = None,
    ) -> TrainingOutcome:
        if self.config.recent_fraction >= 1.0:
            replay_count = 0
        else:
            replay_count = int(
                len(recent_y) * (1.0 - self.config.recent_fraction) / self.config.recent_fraction
            )
        replay_count = min(replay_count, self.config.max_replay_samples)
        replay_X, replay_y, replay_ids = replay.sample_with_ids(
            replay_count,
            exclude_ids=exclude_ids,
        )
        if recent_sample_ids is None:
            recent_ids = np.arange(len(recent_y), dtype=object)
        else:
            recent_ids = np.asarray(recent_sample_ids, dtype=object)
        training_ids = np.concatenate([replay_ids, recent_ids])
        if exclude_ids and set(training_ids.tolist()) & exclude_ids:
            raise RuntimeError("候选训练数据与历史守护集重叠")

        training_X = pd.concat([replay_X, recent_X.reset_index(drop=True)], ignore_index=True)
        training_y = np.concatenate([replay_y, np.asarray(recent_y)])
        candidate = champion.new_unfitted()
        started = perf_counter()
        candidate.fit(training_X, training_y)
        elapsed = perf_counter() - started
        return TrainingOutcome(
            model=candidate,
            seconds=elapsed,
            training_samples=len(training_y),
            reference_X=training_X,
            training_sample_ids=training_ids,
        )
