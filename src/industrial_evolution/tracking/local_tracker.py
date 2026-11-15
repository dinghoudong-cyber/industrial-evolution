from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class LocalTracker:
    def __init__(self, run_root: str | Path, run_id: str | None = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
        self.run_dir = Path(run_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.events_path = self.run_dir / "events.jsonl"
        self.metrics_path = self.run_dir / "metrics.csv"
        self._metric_fields: tuple[str, ...] | None = None

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": _jsonable(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_metrics(self, row: dict[str, Any]) -> None:
        serialized = _jsonable(row)
        if self._metric_fields is None:
            self._metric_fields = tuple(serialized)

        write_header = not self.metrics_path.exists()
        encoding = "utf-8-sig" if write_header else "utf-8"
        with self.metrics_path.open("a", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self._metric_fields,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerow(serialized)

    def log_hypothesis(self, record: dict[str, Any]) -> None:
        self.log_event("hypothesis_recorded", record)

    def close(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(_jsonable(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log_event("run_completed", summary)
