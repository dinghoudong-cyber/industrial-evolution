from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .domain import DataWindow, Decision, PolicyContext


class StreamSource(Protocol):
    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]: ...

    def __iter__(self) -> Iterator[DataWindow]: ...


class ModelAdapter(Protocol):
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ModelAdapter: ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None: ...

    def score(self, X: pd.DataFrame, y: np.ndarray) -> float: ...

    def new_unfitted(self) -> ModelAdapter: ...


class UpdatePolicy(Protocol):
    def decide(self, context: PolicyContext) -> Decision: ...

    def mark_update(self, window_index: int) -> None: ...

    def snapshot_state(self) -> dict[str, Any]: ...

    def restore_state(self, state: dict[str, Any]) -> None: ...


class TrackerMirror(Protocol):
    def log_event(self, event_type: str, payload: dict[str, Any]) -> None: ...

    def log_metrics(self, row: dict[str, Any]) -> None: ...

    def log_hypothesis(self, record: dict[str, Any]) -> None: ...

    def close(self, summary: dict[str, Any]) -> None: ...


class ExperimentTracker(TrackerMirror, Protocol):
    run_id: str
    run_dir: Path
