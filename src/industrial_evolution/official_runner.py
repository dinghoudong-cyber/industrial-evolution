from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .data.official_jsonl import (
    PROFILE_SPECS,
    OfficialJsonlDataset,
    read_official_jsonl,
    validate_official_pair,
)
from .interfaces import ExperimentTracker
from .models.official_adapter import OfficialBaselineAdapter
from .tracking.local_tracker import LocalTracker


class OfficialRunError(ValueError):
    """Official public-package contract failed validation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_pack_metadata(
    pack_root: Path, profile: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    info_path = pack_root / "baseline-info.json"
    config_path = pack_root / "baselines" / "configs" / f"{profile}.json"
    training_path = pack_root / "training.json"
    checkpoint_path = pack_root / "checkpoint.npz"
    for required in (info_path, config_path, training_path, checkpoint_path):
        if required.is_symlink() or not required.is_file():
            raise OfficialRunError(f"官方包缺少必要普通文件：{required}")

    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialRunError(f"官方包元数据解析失败：{exc}") from exc

    if not all(isinstance(value, dict) for value in (info, config, training)):
        raise OfficialRunError("官方包元数据必须是 JSON 对象")
    if info.get("dataset_key") != profile or config.get("profile") != profile:
        raise OfficialRunError(f"profile {profile} 与官方包元数据不一致")
    if config.get("task") != PROFILE_SPECS[profile]["task"]:
        raise OfficialRunError(f"profile {profile} 的官方任务类型不符合预期")
    if checkpoint_path.stat().st_size > 16 * 1024 * 1024:
        raise OfficialRunError("官方 checkpoint 超过 16 MiB 安全上限")
    actual_hash = _sha256_file(checkpoint_path)
    expected_hash = info.get("checkpoint_sha256")
    if expected_hash != actual_hash or training.get("checkpoint_sha256") != actual_hash:
        raise OfficialRunError("官方 checkpoint 摘要与 baseline-info.json 或 training.json 不匹配")
    if training.get("profile") != profile or training.get("task") != config["task"]:
        raise OfficialRunError("training.json 与 profile 配置不一致")
    return info, config, training


def _primary_metric(profile: str) -> tuple[str, bool]:
    return ("macro_f1", True) if profile == "A" else ("rmse", False)


def _beats_baseline(profile: str, candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    metric, higher_is_better = _primary_metric(profile)
    return (
        candidate[metric] >= baseline[metric]
        if higher_is_better
        else candidate[metric] <= baseline[metric]
    )


def _metric_row(
    stage: str, metrics: dict[str, float], training_seconds: float = 0.0
) -> dict[str, Any]:
    return {
        "stage": stage,
        "macro_f1": metrics.get("macro_f1"),
        "accuracy": metrics.get("accuracy"),
        "rmse": metrics.get("rmse"),
        "mae": metrics.get("mae"),
        "training_seconds": training_seconds,
    }


def _group_summary(data: OfficialJsonlDataset) -> dict[str, int]:
    return {"samples": len(data.ids), "groups": len(set(data.groups))}


def evaluate_public_package(
    *,
    pack_root: str | Path,
    profile: str,
    train_path: str | Path,
    validation_path: str | Path,
    run_root: str | Path,
    epochs: int = 0,
    label_provenance_path: str | Path | None = None,
    tracker: ExperimentTracker | None = None,
) -> dict[str, Any]:
    """Evaluate a static official checkpoint; optionally test a warm-start candidate.

    This is a local public-data experiment, not a competition SDK entrypoint.
    """
    if profile not in PROFILE_SPECS:
        raise OfficialRunError(f"不支持的官方 profile：{profile}")
    if type(epochs) is not int or epochs < 0 or epochs > 10000:
        raise OfficialRunError("epochs 必须在 0..10000；0 表示只评估原始 checkpoint")

    started_at = time.time()
    wall_started = time.perf_counter()
    tracker = tracker or LocalTracker(run_root)
    summary: dict[str, Any] = {
        "run_id": tracker.run_id,
        "run_dir": str(tracker.run_dir),
        "purpose": "public_package_local_evaluation",
        "official_scoring_ready": False,
        "profile": profile,
        "candidate_status": "not_requested",
        "network_used": False,
        "llm_calls": 0,
        "tokens_used": 0,
    }
    try:
        root = Path(pack_root).expanduser().resolve()
        if not root.is_dir():
            raise OfficialRunError(f"官方包目录不存在：{root}")
        info, profile_config, training_report = _load_pack_metadata(root, profile)
        checkpoint_path = root / "checkpoint.npz"
        model = OfficialBaselineAdapter.from_pack(root, profile, epochs=max(1, epochs))
        model.validate_profile()
        if model.task != profile_config["task"]:
            raise OfficialRunError("checkpoint task 与 profile 配置不一致")
        if model.classes != profile_config.get("classes", []):
            raise OfficialRunError("checkpoint 类别顺序与 profile 配置不一致")

        train = read_official_jsonl(
            train_path,
            profile,
            classes=model.classes,
            expected_paths=json.loads(model.meta_json)["paths"],
        )
        validation = read_official_jsonl(
            validation_path,
            profile,
            classes=model.classes,
            expected_paths=json.loads(model.meta_json)["paths"],
        )
        validate_official_pair(train, validation, profile)

        baseline_metrics = model.official_metric(validation.X, validation.y)
        summary.update(
            {
                "started_at_unix": started_at,
                "package_id": info.get("id"),
                "label_status": info.get("label_status"),
                "package_training_report": {
                    "purpose": training_report.get("purpose"),
                    "official_scoring_ready": training_report.get("official_scoring_ready"),
                    "label_provenance": training_report.get("label_provenance"),
                },
                "profile_config_sha256": _sha256_file(
                    root / "baselines" / "configs" / f"{profile}.json"
                ),
                "baseline_info_sha256": _sha256_file(root / "baseline-info.json"),
                "baseline_checkpoint_sha256": _sha256_file(checkpoint_path),
                "train": {"path": str(train.path), "sha256": train.sha256, **_group_summary(train)},
                "validation": {
                    "path": str(validation.path),
                    "sha256": validation.sha256,
                    **_group_summary(validation),
                },
                "feature_count": len(train.feature_paths),
                "feature_paths": list(train.feature_paths),
                "task": model.task,
                "classes": model.classes,
                "baseline_metrics": baseline_metrics,
                "metric_units": "score"
                if profile == "A"
                else profile_config.get("target_unit", "cycles"),
                "training_epochs_requested": epochs,
                "training_seconds": 0.0,
                "candidate_status": "not_requested" if epochs == 0 else "pending",
            }
        )
        tracker.log_event("public_baseline_evaluated", summary.copy())
        tracker.log_metrics(_metric_row("baseline", baseline_metrics))

        if epochs:
            if profile == "B" and label_provenance_path is None:
                raise OfficialRunError(
                    "B profile 续训必须显式提供 --label-provenance JSON（status 与 method）；"
                    "静态基线评估不需要此参数"
                )
            provenance: dict[str, Any] | None = None
            if label_provenance_path is not None:
                requested_provenance = Path(label_provenance_path).expanduser()
                if requested_provenance.is_symlink():
                    raise OfficialRunError("label provenance 必须是普通文件")
                provenance_path = requested_provenance.resolve()
                if not provenance_path.is_file():
                    raise OfficialRunError("label provenance 必须是普通文件")
                declared = json.loads(provenance_path.read_text(encoding="utf-8"))
                if (
                    not isinstance(declared, dict)
                    or declared.get("status") not in {"draft", "approved"}
                    or not isinstance(declared.get("method"), str)
                    or not declared["method"].strip()
                ):
                    raise OfficialRunError(
                        "label provenance 需要 status=draft|approved 和非空 method"
                    )
                provenance = {
                    "path": str(provenance_path),
                    "sha256": _sha256_file(provenance_path),
                    "declared_status": declared["status"],
                    "method": declared["method"],
                    "verified_by_platform": False,
                }

            candidate = model.new_unfitted()
            train_started = time.perf_counter()
            candidate.fit(train.X, train.y)
            training_seconds = time.perf_counter() - train_started
            candidate_metrics = candidate.official_metric(validation.X, validation.y)
            accepted = _beats_baseline(profile, candidate_metrics, baseline_metrics)
            status = "accepted" if accepted else "rejected_below_static_baseline"
            candidate_checkpoint = None
            candidate_hash = None
            if accepted:
                candidate_checkpoint = tracker.run_dir / "candidate-checkpoint.npz"
                candidate_hash = candidate.save_checkpoint(candidate_checkpoint)

            summary.update(
                {
                    "candidate_status": status,
                    "candidate_metrics": candidate_metrics,
                    "candidate_checkpoint": str(candidate_checkpoint)
                    if candidate_checkpoint
                    else None,
                    "candidate_checkpoint_sha256": candidate_hash,
                    "parent_checkpoint_sha256": summary["baseline_checkpoint_sha256"],
                    "training_seconds": training_seconds,
                    "label_provenance": provenance,
                }
            )
            tracker.log_event(
                "public_candidate_evaluated",
                {
                    "status": status,
                    "metrics": candidate_metrics,
                    "primary_metric": _primary_metric(profile)[0],
                    "baseline_checkpoint_sha256": summary["baseline_checkpoint_sha256"],
                    "candidate_checkpoint_sha256": candidate_hash,
                    "training_seconds": training_seconds,
                    "label_provenance": provenance,
                },
            )
            tracker.log_metrics(_metric_row("candidate", candidate_metrics, training_seconds))

        summary["total_wall_seconds"] = time.perf_counter() - wall_started
        summary["run_status"] = "complete"
        summary["note"] = "本地公开数据评估，不代表隐藏测试、平台 SDK 或正式比赛评测结果。"
        tracker.close(summary)
        return summary
    except Exception as exc:
        summary.update(
            {
                "run_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "total_wall_seconds": time.perf_counter() - wall_started,
            }
        )
        tracker.close(summary)
        raise


def predict_jsonl(
    *,
    pack_root: str | Path,
    profile: str,
    checkpoint: str | Path | None = None,
    stdin: Any = None,
    stdout: Any = None,
) -> None:
    """Serve the package reference stdin/stdout JSONL protocol without labels or training."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    root = Path(pack_root).expanduser().resolve()
    _load_pack_metadata(root, profile)
    model = (
        OfficialBaselineAdapter.from_pack(root, profile)
        if checkpoint is None
        else OfficialBaselineAdapter(root, profile, checkpoint)
    )
    model.validate_profile()
    paths = tuple(json.loads(model.meta_json)["paths"])

    for line_number, line in enumerate(stdin, start=1):
        if not line.strip():
            continue
        request = json.loads(
            line, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant
        )
        if (
            not isinstance(request, dict)
            or set(request) - {"task", "inputs"}
            or "inputs" not in request
        ):
            raise ValueError(f"第 {line_number} 行请求只能包含 task 和 inputs")
        if request.get("task", model.task) != model.task:
            raise ValueError(f"第 {line_number} 行 task 与 checkpoint 不匹配")
        row_paths, values = _flatten_request(request["inputs"])
        if row_paths != paths:
            raise ValueError(f"第 {line_number} 行特征路径/顺序与 checkpoint 不匹配")
        X = pd.DataFrame([values], columns=paths)
        if model.task == "classification":
            probabilities = model.predict_proba(X)
            output = {
                "outputs": [
                    {"label": label, "score": float(score)}
                    for label, score in zip(model.classes, probabilities[0], strict=True)
                ]
            }
        else:
            output = {"outputs": {"value": float(model.predict(X)[0])}}
        print(json.dumps(output, ensure_ascii=False, allow_nan=False), file=stdout, flush=True)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 请求有重复字段：{key}")
        result[key] = value
    return result


def _reject_json_constant(_: str) -> None:
    raise ValueError("JSON 请求不允许 NaN 或 Infinity")


def _flatten_request(inputs: Any) -> tuple[tuple[str, ...], tuple[float, ...]]:
    from .data.official_jsonl import flatten_inputs

    return flatten_inputs(inputs)
