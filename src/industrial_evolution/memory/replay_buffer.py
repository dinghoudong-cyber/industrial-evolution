from __future__ import annotations

import numpy as np
import pandas as pd


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42, task_type: str = "classification") -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.task_type = task_type
        self.X = pd.DataFrame()
        self.y = np.array([])
        self.sample_ids = np.array([], dtype=object)

    def __len__(self) -> int:
        return len(self.y)

    def update(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_ids: np.ndarray | None = None,
    ) -> None:
        incoming_X = X.reset_index(drop=True)
        incoming_y = np.asarray(y)
        if sample_ids is None:
            start = len(self.sample_ids)
            incoming_ids = np.arange(start, start + len(incoming_y), dtype=object)
        else:
            incoming_ids = np.asarray(sample_ids, dtype=object)
        if len(incoming_X) != len(incoming_y) or len(incoming_y) != len(incoming_ids):
            raise ValueError("回放数据、标签和样本ID长度不一致")
        if len(set(incoming_ids.tolist())) != len(incoming_ids):
            raise ValueError("同一批回放数据包含重复样本ID")

        if self.X.empty:
            combined_X = incoming_X
            combined_y = incoming_y
            combined_ids = incoming_ids
        else:
            existing = set(self.sample_ids.tolist())
            keep = np.array([sample_id not in existing for sample_id in incoming_ids])
            incoming_X = incoming_X.loc[keep].reset_index(drop=True)
            incoming_y = incoming_y[keep]
            incoming_ids = incoming_ids[keep]
            combined_X = pd.concat([self.X, incoming_X], ignore_index=True)
            combined_y = np.concatenate([self.y, incoming_y])
            combined_ids = np.concatenate([self.sample_ids, incoming_ids])

        if len(combined_y) <= self.capacity:
            self.X = combined_X.reset_index(drop=True)
            self.y = combined_y.copy()
            self.sample_ids = combined_ids.copy()
            return

        if self.task_type == "classification":
            indices = self._stratified_indices(combined_y)
        else:
            indices = np.sort(self.rng.choice(len(combined_y), self.capacity, replace=False))
        self.X = combined_X.iloc[indices].reset_index(drop=True)
        self.y = combined_y[indices]
        self.sample_ids = combined_ids[indices]

    def _stratified_indices(self, y: np.ndarray) -> np.ndarray:
        selected: list[int] = []
        classes, counts = np.unique(y, return_counts=True)
        remaining = self.capacity
        for position, (label, count) in enumerate(zip(classes, counts, strict=True)):
            class_indices = np.flatnonzero(y == label)
            if position == len(classes) - 1:
                quota = min(len(class_indices), remaining)
            else:
                quota = min(
                    len(class_indices),
                    max(1, int(round(self.capacity * count / len(y)))),
                )
                remaining -= quota
            chosen = self.rng.choice(class_indices, quota, replace=False)
            selected.extend(chosen.tolist())

        if len(selected) < self.capacity:
            available = np.setdiff1d(np.arange(len(y)), np.asarray(selected), assume_unique=False)
            extra = self.rng.choice(available, self.capacity - len(selected), replace=False)
            selected.extend(extra.tolist())
        return np.sort(np.asarray(selected[: self.capacity], dtype=int))

    def sample(self, n_samples: int) -> tuple[pd.DataFrame, np.ndarray]:
        X, y, _ = self.sample_with_ids(n_samples)
        return X, y

    def sample_with_ids(
        self,
        n_samples: int,
        *,
        exclude_ids: set[object] | None = None,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        excluded = exclude_ids or set()
        available = np.flatnonzero(
            np.array([sample_id not in excluded for sample_id in self.sample_ids])
        )
        if len(available) == 0 or n_samples <= 0:
            return (
                self.X.iloc[:0].copy(),
                self.y[:0].copy(),
                self.sample_ids[:0].copy(),
            )
        n_samples = min(n_samples, len(available))
        selected = np.sort(self.rng.choice(available, n_samples, replace=False))
        return (
            self.X.iloc[selected].reset_index(drop=True),
            self.y[selected].copy(),
            self.sample_ids[selected].copy(),
        )
