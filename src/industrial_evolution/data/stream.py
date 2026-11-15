from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain import DataWindow

MIN_WINDOW_SIZE = 10
DEFAULT_CHUNK_ROWS = 10_000


class CsvWindowStream:
    """按 CSV 原始行顺序分块产生窗口，避免时间泄漏和全量内存占用。"""

    def __init__(
        self,
        path: str | Path,
        target_column: str,
        initial_size: int,
        window_size: int,
        *,
        drop_columns: list[str] | None = None,
        task_type: str = "classification",
        chunk_rows: int = DEFAULT_CHUNK_ROWS,
    ) -> None:
        self.path = Path(path)
        self.target_column = target_column
        self.initial_size = initial_size
        self.window_size = window_size
        self.drop_columns = drop_columns or []
        self.task_type = task_type
        self.chunk_rows = max(window_size, chunk_rows)
        self._validate_arguments()
        self._feature_columns, self.total_rows = self._preflight()
        self.full_windows = (self.total_rows - self.initial_size) // self.window_size
        self.dropped_tail_rows = (self.total_rows - self.initial_size) % self.window_size
        if self.full_windows < 1:
            required = self.initial_size + self.window_size
            raise ValueError(
                f"CSV 没有可用的完整流式窗口：共 {self.total_rows} 行，至少需要 {required} 行"
            )

    def _validate_arguments(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"CSV 文件不存在：{self.path}")
        if self.initial_size < 2:
            raise ValueError("初始样本数至少为 2")
        if self.window_size < MIN_WINDOW_SIZE:
            raise ValueError(f"窗口大小至少为 {MIN_WINDOW_SIZE}")

    def _preflight(self) -> tuple[tuple[str, ...], int]:
        header = pd.read_csv(self.path, nrows=0)
        if self.target_column not in header.columns:
            raise ValueError(f"目标列不存在：{self.target_column}")
        missing = [column for column in self.drop_columns if column not in header.columns]
        if missing:
            raise ValueError(f"需要删除的列不存在：{missing}")
        excluded = {self.target_column, *self.drop_columns}
        features = tuple(column for column in header.columns if column not in excluded)
        if not features:
            raise ValueError("CSV 中没有可用于训练的特征列")

        total_rows = 0
        class_values: set[object] = set()
        for chunk in pd.read_csv(self.path, chunksize=self.chunk_rows):
            total_rows += len(chunk)
            self._validate_chunk(chunk, features, class_values)
        if total_rows <= self.initial_size:
            raise ValueError("CSV 数据不足以同时构造初始训练集和流式窗口")
        if self.task_type == "classification" and len(class_values) < 2:
            raise ValueError("分类目标列至少需要两个类别")
        return features, total_rows

    def _validate_chunk(
        self,
        chunk: pd.DataFrame,
        features: tuple[str, ...],
        class_values: set[object],
    ) -> None:
        nonnumeric: list[str] = []
        for column in features:
            try:
                converted = pd.to_numeric(chunk[column], errors="raise")
            except (TypeError, ValueError):
                nonnumeric.append(column)
                continue
            values = converted.to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError(f"特征列包含无穷值：{column}")
        if nonnumeric:
            raise ValueError(f"CSV 适配器只接受数值特征，需处理：{nonnumeric}")

        target = chunk[self.target_column]
        if target.isna().any():
            raise ValueError(f"目标列包含缺失值：{self.target_column}")
        if self.task_type == "regression":
            try:
                values = pd.to_numeric(target, errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("回归目标列必须是数值") from exc
            if not np.isfinite(values).all():
                raise ValueError("回归目标列包含非有限值")
        else:
            class_values.update(target.astype(str).unique().tolist())

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = frame.loc[:, self._feature_columns].apply(pd.to_numeric, errors="raise")
        y = frame[self.target_column].to_numpy()
        if self.task_type == "regression":
            y = y.astype(float)
        else:
            y = frame[self.target_column].astype(str).to_numpy()
        return X.reset_index(drop=True), y

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        parts: list[pd.DataFrame] = []
        remaining = self.initial_size
        for chunk in pd.read_csv(self.path, chunksize=self.chunk_rows):
            take = min(remaining, len(chunk))
            if take:
                parts.append(chunk.iloc[:take])
                remaining -= take
            if remaining == 0:
                break
        return self._xy(pd.concat(parts, ignore_index=True))

    def __iter__(self) -> Iterator[DataWindow]:
        skipped = 0
        buffer = pd.DataFrame()
        window_index = 0
        global_start = self.initial_size
        for chunk in pd.read_csv(self.path, chunksize=self.chunk_rows):
            if skipped < self.initial_size:
                drop = min(self.initial_size - skipped, len(chunk))
                chunk = chunk.iloc[drop:]
                skipped += drop
            if chunk.empty:
                continue
            buffer = pd.concat([buffer, chunk], ignore_index=True)
            while len(buffer) >= self.window_size:
                current = buffer.iloc[: self.window_size].copy()
                buffer = buffer.iloc[self.window_size :].reset_index(drop=True)
                X, y = self._xy(current)
                sample_ids = np.arange(
                    global_start,
                    global_start + len(current),
                    dtype=np.int64,
                )
                global_start += len(current)
                yield DataWindow(
                    index=window_index,
                    X=X,
                    y=y,
                    phase="csv",
                    sample_ids=sample_ids,
                )
                window_index += 1
