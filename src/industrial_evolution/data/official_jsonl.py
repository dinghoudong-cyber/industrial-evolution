from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_DATASET_BYTES = 128 * 1024 * 1024
PROFILE_SPECS = {
    "A": {"task": "classification", "features": 128, "group_fields": ("source_batch",)},
    "B": {"task": "regression", "features": 34, "group_fields": ("bearing",)},
    "C": {"task": "regression", "features": 24, "group_fields": ("source_subset", "unit")},
}
_FORBIDDEN_INPUT_KEYS = {
    "answer",
    "label",
    "target",
    "rul",
    "truerul",
    "normalizedrul",
    "remainingusefullife",
    "totalsnapshots",
    "snapshot",
    "cycle",
    "unit",
    "bearing",
    "engine",
}


@dataclass(frozen=True, slots=True)
class OfficialJsonlDataset:
    path: Path
    ids: tuple[str, ...]
    X: pd.DataFrame
    y: np.ndarray
    feature_paths: tuple[str, ...]
    groups: tuple[tuple[Any, ...], ...]
    sha256: str


def _reject_constant(_: str) -> None:
    raise ValueError("JSON 数值不能是 NaN 或 Infinity")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 对象包含重复字段：{key}")
        result[key] = value
    return result


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.lower())


def flatten_inputs(inputs: Any) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """按官方字段路径规则将嵌套 numeric inputs 展平；不读取 meta/answer。"""
    paths: list[str] = []
    values: list[float] = []

    def visit(value: Any, path: list[tuple[str, Any]], depth: int) -> None:
        if depth > 16:
            raise ValueError("输入嵌套超过 16 层")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or _normalized_key(key) in _FORBIDDEN_INPUT_KEYS:
                    raise ValueError(f"输入包含目标或实体派生字段：{key}")
                visit(value[key], [*path, ("key", key)], depth + 1)
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, [*path, ("index", index)], depth + 1)
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("inputs 只能包含有限数值特征")
        if not math.isfinite(value):
            raise ValueError("inputs 不能包含 NaN 或 Infinity")
        paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
        values.append(float(value))

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ValueError("特征数量必须在 1..4096 之间")
    return tuple(paths), tuple(values)


def read_official_jsonl(
    path: str | Path,
    profile: str,
    *,
    classes: list[str] | None = None,
    expected_paths: tuple[str, ...] | list[str] | None = None,
) -> OfficialJsonlDataset:
    """Load one explicitly selected public JSONL file for an A/B/C profile."""
    if profile not in PROFILE_SPECS:
        raise ValueError(f"不支持的官方 profile：{profile}")
    spec = PROFILE_SPECS[profile]
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("数据必须是非符号链接的普通 JSONL 文件且不超过 128 MiB")
    source = requested.resolve()
    if not source.is_file() or source.stat().st_size > MAX_DATASET_BYTES:
        raise ValueError("数据必须是非符号链接的普通 JSONL 文件且不超过 128 MiB")
    if spec["task"] == "classification" and (not classes or len(set(classes)) != len(classes)):
        raise ValueError("分类 profile 必须提供 checkpoint 中唯一的类别顺序")

    ids: list[str] = []
    features: list[tuple[float, ...]] = []
    targets: list[int | float] = []
    paths: tuple[str, ...] | None = tuple(expected_paths) if expected_paths is not None else None
    groups: list[tuple[Any, ...]] = []
    seen: set[str] = set()
    digest = hashlib.sha256()

    with source.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(
                    raw_line.decode("utf-8"),
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{source.name}:{line_number}: JSONL 行解析失败：{exc}") from exc
            if not isinstance(row, dict) or not {"id", "inputs", "answer", "meta"} <= row.keys():
                raise ValueError(
                    f"{source.name}:{line_number}: 必须含 id、inputs、answer、meta 字段"
                )

            identity = row["id"]
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError(f"{source.name}:{line_number}: id 必须是非空且唯一的字符串")
            meta = row["meta"]
            if not isinstance(meta, dict):
                raise ValueError(f"{source.name}:{line_number}: meta 必须是对象")
            group_fields = spec["group_fields"]
            if any(field not in meta for field in group_fields):
                raise ValueError(
                    f"{source.name}:{line_number}: profile {profile} 的 meta 缺少分组字段 "
                    f"{group_fields}"
                )
            group = tuple(meta[field] for field in group_fields)
            if any(isinstance(value, (dict, list)) or value is None for value in group):
                raise ValueError(f"{source.name}:{line_number}: 实体分组字段格式无效")

            row_paths, row_values = flatten_inputs(row["inputs"])
            if len(row_paths) != spec["features"]:
                raise ValueError(
                    f"{source.name}:{line_number}: profile {profile} 期望 "
                    f"{spec['features']} 个特征，实际 {len(row_paths)} 个"
                )
            if paths is None:
                paths = row_paths
            elif row_paths != paths:
                raise ValueError(f"{source.name}:{line_number}: 特征路径或顺序与数据契约不一致")

            answer = row["answer"]
            if spec["task"] == "classification":
                if not isinstance(answer, str) or answer not in classes:
                    raise ValueError(
                        f"{source.name}:{line_number}: 分类答案不在 checkpoint 类别表中"
                    )
                target: int | float = classes.index(answer)
            else:
                if isinstance(answer, bool) or not isinstance(answer, (int, float)):
                    raise ValueError(f"{source.name}:{line_number}: 回归答案必须是数值")
                if not math.isfinite(answer) or answer < 0:
                    raise ValueError(f"{source.name}:{line_number}: RUL 必须是有限非负数")
                target = float(answer)

            seen.add(identity)
            ids.append(identity)
            features.append(row_values)
            targets.append(target)
            groups.append(group)

    if not ids or paths is None:
        raise ValueError(f"{source.name}: 数据集为空")
    target_dtype = np.int64 if spec["task"] == "classification" else np.float64
    return OfficialJsonlDataset(
        path=source,
        ids=tuple(ids),
        X=pd.DataFrame(np.asarray(features, dtype=np.float64), columns=paths),
        y=np.asarray(targets, dtype=target_dtype),
        feature_paths=paths,
        groups=tuple(groups),
        sha256=f"sha256:{digest.hexdigest()}",
    )


def validate_official_pair(
    train: OfficialJsonlDataset,
    validation: OfficialJsonlDataset,
    profile: str,
) -> None:
    """Reject sample/entity overlap between public training and validation sets."""
    if set(train.ids) & set(validation.ids):
        raise ValueError("公开训练集与验证集 sample id 重叠")
    if set(train.groups) & set(validation.groups):
        raise ValueError(f"公开训练集与验证集 {profile} 实体/批次分组重叠")
    if train.feature_paths != validation.feature_paths:
        raise ValueError("公开训练集与验证集特征路径或顺序不一致")
