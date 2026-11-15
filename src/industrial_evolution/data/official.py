"""官方评测数据包的接入层。

赛题数据包的结构在发布前是未知的，因此这里不猜目录约定，而是按「清单优先、命令行可覆盖、
缺失即报错并列出已尝试的候选」三原则工作：

1. 包根目录若存在 ``manifest.json`` / ``manifest.yaml``，按其中字段解析；
2. 否则按常见命名推断（``initial.csv|train.csv`` + ``stream.csv|test.csv``）；
3. 命令行参数优先级最高，用于清单字段名与实际不符时临时纠正；
4. 无法确定时抛错，并把「找过哪些文件名、哪些列」写进错误信息，避免静默跑错。

产物语义与 ``CsvWindowStream`` 一致：初始文件是带标签的冷启动集，流文件按原始行顺序切窗，
绝不打乱顺序（时间顺序即防泄漏边界）。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..domain import DataWindow

MANIFEST_NAMES = ("manifest.json", "pack.json", "manifest.yaml", "pack.yaml")
INITIAL_CANDIDATES = ("initial.csv", "train.csv", "initial_train.csv", "warmup.csv")
STREAM_CANDIDATES = ("stream.csv", "test.csv", "eval.csv", "holdout.csv", "stream_test.csv")
MIN_WINDOW_SIZE = 10
DEFAULT_CHUNK_ROWS = 10_000


class OfficialPackError(ValueError):
    """数据包结构与配置不符时抛出，错误信息用于直接定位问题。"""


@dataclass(frozen=True, slots=True)
class OfficialPackSpec:
    pack_root: Path
    initial_file: Path
    stream_file: Path
    target_column: str
    initial_size: int
    window_size: int
    task_type: str = "classification"
    drop_columns: tuple[str, ...] = ()
    id_column: str | None = None
    timestamp_column: str | None = None
    source: str = "inferred"

    def describe(self) -> dict[str, Any]:
        return {
            "pack_root": str(self.pack_root),
            "initial_file": str(self.initial_file),
            "stream_file": str(self.stream_file),
            "target_column": self.target_column,
            "initial_size": self.initial_size,
            "window_size": self.window_size,
            "task_type": self.task_type,
            "drop_columns": list(self.drop_columns),
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "resolved_from": self.source,
        }


def _read_manifest(pack_root: Path, manifest: str | None) -> tuple[dict[str, Any], str | None]:
    if manifest is not None:
        path = Path(manifest)
        if not path.is_absolute():
            path = pack_root / path
        if not path.is_file():
            raise OfficialPackError(f"指定的数据包清单不存在：{path}")
        return _parse_manifest_file(path), str(path)

    for name in MANIFEST_NAMES:
        candidate = pack_root / name
        if candidate.is_file():
            return _parse_manifest_file(candidate), str(candidate)
    return {}, None


def _parse_manifest_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise OfficialPackError(f"数据包清单解析失败：{path}（{type(exc).__name__}）") from exc
    if not isinstance(raw, dict):
        raise OfficialPackError(f"数据包清单顶层必须是映射：{path}")
    return raw


def _first_existing(pack_root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = pack_root / name
        if candidate.is_file():
            return candidate
    return None


def _resolve_file(
    pack_root: Path,
    value: str | None,
    names: tuple[str, ...],
    label: str,
) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = pack_root / path
        if not path.is_file():
            raise OfficialPackError(f"{label}文件不存在：{path}")
        return path
    found = _first_existing(pack_root, names)
    if found is None:
        raise OfficialPackError(
            f"无法定位{label}文件：{pack_root} 下没有 {', '.join(names)}；"
            "请用命令行参数指定，或在清单中写清楚"
        )
    return found


def _resolve_columns(raw: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "target_column": ("target_column", "label", "target"),
        "id_column": ("id_column", "id", "sample_id"),
        "timestamp_column": ("timestamp_column", "timestamp", "time_column"),
    }
    resolved: dict[str, Any] = {}
    for key, aliases in mapping.items():
        if overrides.get(key):
            resolved[key] = overrides[key]
            continue
        value = next(
            (raw[alias] for alias in aliases if isinstance(raw.get(alias), str) and raw[alias]),
            None,
        )
        resolved[key] = value
    drop = overrides.get("drop_columns") or raw.get("drop_columns") or raw.get("ignore_columns")
    if isinstance(drop, str):
        drop = [drop]
    resolved["drop_columns"] = tuple(str(item) for item in (drop or ()))
    return resolved


def load_pack_spec(
    pack_root: str | Path,
    *,
    target_column: str | None = None,
    initial_file: str | None = None,
    stream_file: str | None = None,
    initial_size: int | None = None,
    window_size: int | None = None,
    task_type: str | None = None,
    drop_columns: list[str] | None = None,
    id_column: str | None = None,
    timestamp_column: str | None = None,
    manifest: str | None = None,
    default_initial_size: int | None = None,
    default_window_size: int | None = None,
) -> OfficialPackSpec:
    """解析数据包结构：命令行参数 > 清单 > 配置兜底。"""
    root = Path(pack_root).expanduser().resolve()
    if not root.is_dir():
        raise OfficialPackError(f"数据包目录不存在：{root}")

    raw, manifest_path = _read_manifest(root, manifest)
    columns = _resolve_columns(
        raw,
        {
            "target_column": target_column,
            "id_column": id_column,
            "timestamp_column": timestamp_column,
            "drop_columns": drop_columns,
        },
    )

    resolved_target = columns["target_column"]
    if not resolved_target:
        raise OfficialPackError(
            f"数据包未声明目标列：在 {root} 的清单里写 target_column，或用 --target 指定标签列名"
        )

    resolved_initial = _resolve_file(
        root,
        initial_file or raw.get("initial_file") or raw.get("train_file"),
        INITIAL_CANDIDATES,
        "初始训练集",
    )
    resolved_stream = _resolve_file(
        root,
        stream_file or raw.get("stream_file") or raw.get("test_file"),
        STREAM_CANDIDATES,
        "评测流",
    )

    initial_size = int(initial_size or raw.get("initial_size") or default_initial_size or 0)
    window_size = int(window_size or raw.get("window_size") or default_window_size or 0)
    if initial_size <= 0:
        raise OfficialPackError("初始训练集行数未知：请在命令行用 --initial-size 指定")
    if window_size < MIN_WINDOW_SIZE:
        raise OfficialPackError(f"窗口大小至少为 {MIN_WINDOW_SIZE}，请用 --window-size 指定")

    resolved_task = task_type or raw.get("task_type") or "classification"
    if resolved_task not in {"classification", "regression"}:
        raise OfficialPackError(f"task_type 只能是 classification 或 regression：{resolved_task}")

    return OfficialPackSpec(
        pack_root=root,
        initial_file=resolved_initial,
        stream_file=resolved_stream,
        target_column=str(resolved_target),
        initial_size=initial_size,
        window_size=window_size,
        task_type=str(resolved_task),
        drop_columns=columns["drop_columns"],
        id_column=columns["id_column"],
        timestamp_column=columns["timestamp_column"],
        source="manifest" if manifest_path else "inferred",
    )


class OfficialPackStream:
    """把官方数据包包装成 ``StreamSource``：初始集冷启动，流文件按序切窗。"""

    def __init__(self, spec: OfficialPackSpec, *, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> None:
        self.spec = spec
        self.chunk_rows = max(spec.window_size, chunk_rows)
        self._feature_columns = self._preflight()

    def _excluded_columns(self) -> set[str]:
        excluded = {self.spec.target_column, *self.spec.drop_columns}
        if self.spec.id_column:
            excluded.add(self.spec.id_column)
        if self.spec.timestamp_column:
            excluded.add(self.spec.timestamp_column)
        return excluded

    def _preflight(self) -> tuple[str, ...]:
        initial_header = pd.read_csv(self.spec.initial_file, nrows=0)
        stream_header = pd.read_csv(self.spec.stream_file, nrows=0)
        excluded = self._excluded_columns()

        if self.spec.target_column not in stream_header.columns:
            raise OfficialPackError(
                f"评测流 {self.spec.stream_file.name} 缺少目标列 {self.spec.target_column}；"
                f"实际列：{list(stream_header.columns)}"
            )
        if self.spec.target_column not in initial_header.columns:
            raise OfficialPackError(
                f"初始训练集 {self.spec.initial_file.name} 缺少目标列 {self.spec.target_column}"
            )
        missing = [column for column in excluded if column not in stream_header.columns]
        if missing:
            raise OfficialPackError(f"数据包缺少声明要排除的列：{missing}")

        features = tuple(column for column in stream_header.columns if column not in excluded)
        if not features:
            raise OfficialPackError("数据包中没有可用于训练的特征列")
        unknown_in_initial = [column for column in features if column not in initial_header.columns]
        if unknown_in_initial:
            raise OfficialPackError(
                f"初始训练集缺少特征列：{unknown_in_initial}；特征集合以评测流为准"
            )

        self.initial_file_rows = self._scan(self.spec.initial_file, features)
        self.total_stream_rows = self._scan(self.spec.stream_file, features)
        if self.initial_file_rows < self.spec.initial_size:
            raise OfficialPackError(
                f"初始训练集只有 {self.initial_file_rows} 行，少于声明的 "
                f"initial_size={self.spec.initial_size}"
            )
        # initial_size 与 CsvWindowStream 语义一致：只有前 initial_size 行是冷启动集，
        # 多出来的行既不参与训练也不进流，且必须显式暴露，避免静默丢数据。
        self.initial_rows = self.spec.initial_size
        self.initial_rows_ignored = self.initial_file_rows - self.spec.initial_size
        if self.initial_rows < 2:
            raise OfficialPackError("初始训练集至少需要两行带标签样本")
        if self.total_stream_rows < self.spec.window_size:
            raise OfficialPackError(
                f"评测流只有 {self.total_stream_rows} 行，不足一个 {self.spec.window_size} 行的窗口"
            )
        if self.spec.id_column:
            self._assert_ids_disjoint()

        self.full_windows = self.total_stream_rows // self.spec.window_size
        self.dropped_tail_rows = self.total_stream_rows % self.spec.window_size
        return features

    def _assert_ids_disjoint(self) -> None:
        """初始集与评测流的样本ID必须全局唯一，否则守护集会与训练集重叠。"""
        assert self.spec.id_column is not None
        initial_ids = set(
            pd.read_csv(
                self.spec.initial_file,
                usecols=[self.spec.id_column],
                nrows=self.spec.initial_size,
            )[self.spec.id_column].tolist()
        )
        for chunk in pd.read_csv(
            self.spec.stream_file,
            usecols=[self.spec.id_column],
            chunksize=self.chunk_rows,
        ):
            overlap = initial_ids.intersection(chunk[self.spec.id_column].tolist())
            if overlap:
                sample = sorted(str(item) for item in overlap)[:5]
                raise OfficialPackError(
                    "初始训练集与评测流存在重复样本ID（示例："
                    f"{', '.join(sample)}）；数据包的样本ID必须全局唯一，"
                    "否则历史守护集与训练集会重叠"
                )

    def _scan(self, path: Path, features: tuple[str, ...]) -> int:
        rows = 0
        for chunk in pd.read_csv(path, chunksize=self.chunk_rows):
            rows += len(chunk)
            self._validate_chunk(chunk, features, path)
        return rows

    def _validate_chunk(self, chunk: pd.DataFrame, features: tuple[str, ...], path: Path) -> None:
        nonnumeric: list[str] = []
        for column in features:
            try:
                values = pd.to_numeric(chunk[column], errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError):
                nonnumeric.append(column)
                continue
            if np.isinf(values).any():
                raise OfficialPackError(f"{path.name} 的特征列包含无穷值：{column}")
        if nonnumeric:
            raise OfficialPackError(
                f"{path.name} 含非数值特征列，需在清单的 drop_columns 中排除：{nonnumeric}"
            )

        target = chunk[self.spec.target_column]
        if target.isna().any():
            raise OfficialPackError(f"{path.name} 的目标列包含缺失值：{self.spec.target_column}")
        if self.spec.task_type == "regression":
            values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise OfficialPackError(f"{path.name} 的回归目标列包含非有限值")

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def _xy(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        X = frame.loc[:, self._feature_columns].apply(pd.to_numeric, errors="raise")
        y = frame[self.spec.target_column]
        values = (
            y.astype(float).to_numpy()
            if self.spec.task_type == "regression"
            else y.astype(str).to_numpy()
        )
        return X.reset_index(drop=True), values

    def _sample_ids(self, frame: pd.DataFrame, start: int) -> np.ndarray:
        if self.spec.id_column:
            return frame[self.spec.id_column].to_numpy()
        return np.arange(start, start + len(frame), dtype=np.int64)

    def initial_data(self) -> tuple[pd.DataFrame, np.ndarray]:
        frame = pd.read_csv(self.spec.initial_file, nrows=self.spec.initial_size)
        return self._xy(frame)

    def __iter__(self) -> Iterator[DataWindow]:
        next_id = self.initial_rows
        window_index = 0
        buffer = pd.DataFrame()
        for chunk in pd.read_csv(self.spec.stream_file, chunksize=self.chunk_rows):
            if chunk.empty:
                continue
            buffer = chunk if buffer.empty else pd.concat([buffer, chunk], ignore_index=True)
            while len(buffer) >= self.spec.window_size:
                current = buffer.iloc[: self.spec.window_size].copy()
                buffer = buffer.iloc[self.spec.window_size :].reset_index(drop=True)
                X, y = self._xy(current)
                yield DataWindow(
                    index=window_index,
                    X=X,
                    y=y,
                    phase="official",
                    sample_ids=self._sample_ids(current, next_id),
                )
                next_id += len(current)
                window_index += 1
