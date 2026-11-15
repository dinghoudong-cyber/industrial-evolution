from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .budget import BudgetExceededError
from .config import AppConfig, load_config
from .data.official import OfficialPackSpec, OfficialPackStream, load_pack_spec
from .data.real_datasets import RUL_CLIP, CmapssStream, GasStream, XjtuBearingStream
from .data.stream import CsvWindowStream
from .data.synthetic import SyntheticDriftStream
from .interfaces import ExperimentTracker, StreamSource
from .llm.factory import build_provider, dependency_status
from .llm.openai_provider import LLMProviderError
from .llm.provider import LLMDecisionRequest
from .official_runner import evaluate_public_package, predict_jsonl
from .orchestrator import EvolutionOrchestrator
from .tracking.local_tracker import LocalTracker
from .tracking.swanlab_tracker import CompositeTracker, SwanLabTracker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_LLM_CONFIG = PROJECT_ROOT / "configs" / "llm-local.yaml"
DEFAULT_CMAPSS_CONFIG = PROJECT_ROOT / "configs" / "cmapss.yaml"
DEFAULT_GAS_CONFIG = PROJECT_ROOT / "configs" / "gas.yaml"


def _build_tracker(config: AppConfig) -> ExperimentTracker:
    local = LocalTracker(config.tracking.run_root)
    tracking = config.tracking
    if not tracking.enable_swanlab or tracking.swanlab_mode == "disabled":
        return CompositeTracker(local)

    log_dir = tracking.swanlab_log_dir or (Path(tracking.run_root) / ".swanlab")
    try:
        mirror = SwanLabTracker(
            project=tracking.swanlab_project,
            config=config.model_dump(mode="json"),
            run_name=local.run_id,
            mode=tracking.swanlab_mode,
            workspace=tracking.swanlab_workspace,
            log_dir=log_dir,
            tags=tracking.swanlab_tags,
            group=config.task_type,
            artifact_dir=local.run_dir,
            upload_artifacts=tracking.swanlab_upload_artifacts,
            artifact_max_mb=tracking.swanlab_artifact_max_mb,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        local.log_event(
            "swanlab_disabled",
            {"reason": reason, "mode": tracking.swanlab_mode},
        )
        print(f"[swanlab] 初始化失败，本次仅写本地记录：{reason}", file=sys.stderr, flush=True)
        return CompositeTracker(local)

    local.log_event(
        "swanlab_attached",
        {
            "project": tracking.swanlab_project,
            "mode": mirror.mode,
            "requested_mode": tracking.swanlab_mode,
            "mode_note": mirror.mode_note,
            "run_name": local.run_id,
            "url": mirror.url,
            "log_dir": str(log_dir),
            "upload_artifacts": tracking.swanlab_upload_artifacts,
        },
    )
    if mirror.mode_note:
        print(f"[swanlab] {mirror.mode_note}", file=sys.stderr, flush=True)
    elif mirror.url:
        print(f"[swanlab] 看板：{mirror.url}", flush=True)
    else:
        print(f"[swanlab] 记录模式 {mirror.mode}，落盘目录：{log_dir}", flush=True)
    return CompositeTracker(local, [mirror])


def _build_official_spec(args: argparse.Namespace, config: AppConfig) -> OfficialPackSpec:
    return load_pack_spec(
        args.pack,
        target_column=args.target,
        initial_file=args.initial_file,
        stream_file=args.stream_file,
        initial_size=args.initial_size,
        window_size=args.window_size,
        task_type=config.task_type,
        drop_columns=args.drop_column,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        manifest=args.manifest,
        default_initial_size=config.stream.initial_samples,
        default_window_size=config.stream.window_size,
    )


def _build_stream(args: argparse.Namespace, config: AppConfig) -> StreamSource:
    if args.command == "demo":
        return SyntheticDriftStream(config.stream, seed=config.seed)
    if args.command == "run-official":
        return OfficialPackStream(_build_official_spec(args, config))
    if args.command == "run-cmapss":
        return CmapssStream(
            data_dir=args.data_dir,
            fd=args.fd,
            initial_size=args.initial_size or config.stream.initial_samples,
            window_size=args.window_size or config.stream.window_size,
            rul_clip=args.rul_clip,
            use_rolling=not args.no_rolling,
            rolling_window=args.rolling_window,
        )
    if args.command == "run-gas":
        return GasStream(
            data_dir=args.data_dir,
            initial_size=args.initial_size or config.stream.initial_samples,
            window_size=args.window_size or config.stream.window_size,
        )
    if args.command == "run-xjtu":
        return XjtuBearingStream(
            data_dir=args.data_dir,
            condition=args.condition,
            bearing=args.bearing,
            initial_size=args.initial_size or config.stream.initial_samples,
            window_size=args.window_size or config.stream.window_size,
        )
    return CsvWindowStream(
        path=args.csv,
        target_column=args.target,
        initial_size=args.initial_size,
        window_size=args.window_size or config.stream.window_size,
        drop_columns=args.drop_column,
        task_type=config.task_type,
    )


def _run(config: AppConfig, stream: StreamSource) -> dict[str, Any]:
    tracker = _build_tracker(config)
    try:
        result = EvolutionOrchestrator(config, stream, tracker).run()
    except Exception as exc:
        failure = {
            "run_id": tracker.run_id,
            "run_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        tracker.close(failure)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nRun directory: {tracker.run_dir}")
    return result


def _run_llm_smoke_test(config: AppConfig) -> None:
    if not config.llm.enabled:
        raise ValueError("当前配置未启用LLM")
    provider = build_provider(config.llm)
    response = provider.decide(
        LLMDecisionRequest(
            window_index=1,
            drift_kind="mixed",
            feature_drift_score=1.5,
            top_features=("sensor_00", "sensor_01"),
            current_score=0.62,
            rolling_score=0.64,
            baseline_score=0.74,
            performance_drop=0.10,
            label_shift=0.13,
            remaining_updates=5,
            remaining_training_seconds=119.0,
            remaining_labels=4000,
            remaining_tokens=config.budget.max_tokens,
            requested_labels=140,
            rule_reason="持续混合漂移且规则策略建议训练候选",
        )
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "model": response.model,
                "action": response.action.value,
                "confidence": response.confidence,
                "reason": response.reason,
                "usage": asdict(response.usage),
                "latency_seconds": round(response.latency_seconds, 6),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _run_env_check(config: AppConfig, config_path: Path) -> dict[str, Any]:
    """离线评测前的环境自检：依赖、端点、输出目录。"""
    parsed = urlparse(config.llm.base_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    reachable: bool | str
    try:
        with socket.create_connection((host, port), timeout=3.0):
            reachable = True
    except OSError as exc:
        reachable = f"{type(exc).__name__}: {exc}"

    run_root = Path(config.tracking.run_root)
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        writable: bool | str = os.access(run_root, os.W_OK)
    except OSError as exc:
        writable = f"{type(exc).__name__}: {exc}"

    report = {
        "python": sys.version.split()[0],
        "config": str(config_path),
        "dependencies": dependency_status(),
        "llm_enabled": config.llm.enabled,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "llm_base_url": config.llm.base_url,
        "llm_api_key_env": config.llm.api_key_env,
        "llm_api_key_present": bool(os.getenv(config.llm.api_key_env)),
        "endpoint_reachable": reachable,
        "run_root": str(run_root),
        "run_root_writable": writable,
        "swanlab_enabled": config.tracking.enable_swanlab,
        "swanlab_mode": config.tracking.swanlab_mode,
        "swanlab_project": config.tracking.swanlab_project,
        "swanlab_api_key_present": bool(os.getenv("SWANLAB_API_KEY")),
    }
    return report


def _apply_llm_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    overrides = {
        key: value
        for key, value in {
            "provider": getattr(args, "provider", None),
            "base_url": getattr(args, "base_url", None),
            "model": getattr(args, "model", None),
            "api_key_env": getattr(args, "api_key_env", None),
            "timeout_seconds": getattr(args, "timeout", None),
        }.items()
        if value is not None
    }
    if overrides:
        merged = {**config.llm.model_dump(), **overrides}
        config.llm = type(config.llm).model_validate(merged)


def _run_swanlab_check(config: AppConfig, probe_online: bool) -> dict[str, Any]:
    """记录通道自检：包在不在、密钥有没有、真写一条能不能成功。

    默认用 ``offline`` 探针（不联网、不占端口、跑完删临时目录），确认 SDK 与磁盘写入
    正常；``--online`` 才会真的连云端，用来确认账号和密钥有效。
    """
    tracking = config.tracking
    log_dir = Path(tracking.swanlab_log_dir or Path(tracking.run_root) / ".swanlab")
    report: dict[str, Any] = {
        "installed": False,
        "version": None,
        "project": tracking.swanlab_project,
        "mode": tracking.swanlab_mode,
        "enable_swanlab": tracking.enable_swanlab,
        "api_key_present": bool(os.getenv("SWANLAB_API_KEY")),
        "log_dir": str(log_dir),
        "probe_mode": "online" if probe_online else "offline",
        "probe": "skipped",
        "url": None,
    }
    try:
        import swanlab
    except ImportError as exc:
        report["probe"] = f"missing: {type(exc).__name__}: {exc}"
        return report

    report["installed"] = True
    report["version"] = getattr(swanlab, "__version__", "unknown")
    probe_dir = Path(tempfile.mkdtemp(prefix="swanlab-probe-"))
    try:
        run = swanlab.init(
            project=tracking.swanlab_project,
            name="record-self-check",
            mode="online" if probe_online else "offline",
            log_dir=str(probe_dir),
        )
        run.log({"self_check/scalar": 1.0}, step=0)
        run.log_text(
            key="判断链/自检",
            data="记录通道自检：能看到这条文本，说明文本层可用。",
            step=0,
        )
        try:
            report["url"] = str(run.url)
        except Exception:
            report["url"] = None
        run.finish()
        report["offline_files"] = sorted(
            str(path.relative_to(probe_dir)) for path in probe_dir.rglob("*") if path.is_file()
        )
    except Exception as exc:
        report["probe"] = f"{type(exc).__name__}: {exc}"
        return report
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)

    report["probe"] = "ok"
    return report


def _apply_tracking_overrides(config: AppConfig, args: argparse.Namespace) -> None:
    updates: dict[str, Any] = {}
    if getattr(args, "swanlab", None) is not None:
        updates["enable_swanlab"] = bool(args.swanlab)
    for key in ("swanlab_mode", "swanlab_project", "swanlab_workspace"):
        value = getattr(args, key, None)
        if value is not None:
            updates[key] = value
    if updates:
        merged = {**config.tracking.model_dump(), **updates}
        config.tracking = type(config.tracking).model_validate(merged)


def _add_tracking_overrides(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--swanlab",
        dest="swanlab",
        action="store_true",
        default=None,
        help="强制开启 SwanLab 镜像记录（等价于把 enable_swanlab 改成 true）",
    )
    sub.add_argument(
        "--no-swanlab",
        dest="swanlab",
        action="store_false",
        help="强制关闭 SwanLab 镜像记录",
    )
    sub.add_argument(
        "--swanlab-mode",
        choices=["online", "local", "offline", "disabled"],
        help="上传形态；评测机无外网时用 offline 或 local",
    )
    sub.add_argument("--swanlab-project", help="覆盖看板项目名")
    sub.add_argument("--swanlab-workspace", help="覆盖看板工作空间（团队名）")


def _add_llm_overrides(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--provider", choices=["openai_compatible"])
    sub.add_argument("--base-url", help="覆盖配置中的端点，例如 http://127.0.0.1:8000/v1")
    sub.add_argument("--model", help="覆盖配置中的模型名称")
    sub.add_argument("--api-key-env", help="覆盖读取密钥的环境变量名")
    sub.add_argument("--timeout", type=float, help="覆盖单次调用超时秒数")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="工业模型自进化框架")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="运行带漂移的合成数据演示")
    demo.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    demo.add_argument("--output", type=Path, help="覆盖运行结果目录")
    _add_llm_overrides(demo)
    _add_tracking_overrides(demo)

    csv_parser = subparsers.add_parser("run-csv", help="按原始行顺序运行 CSV 数据流")
    csv_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    csv_parser.add_argument("--csv", type=Path, required=True)
    csv_parser.add_argument("--target", required=True, help="目标列名称")
    csv_parser.add_argument("--initial-size", type=int, required=True)
    csv_parser.add_argument("--window-size", type=int, help="默认使用配置文件中的窗口大小")
    csv_parser.add_argument("--drop-column", action="append", default=[])
    csv_parser.add_argument("--output", type=Path, help="覆盖运行结果目录")
    _add_llm_overrides(csv_parser)
    _add_tracking_overrides(csv_parser)

    official = subparsers.add_parser(
        "run-official",
        help="在官方数据包上运行：清单优先，文件与列名可用参数覆盖",
    )
    official.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    official.add_argument("--pack", type=Path, required=True, help="数据包根目录")
    official.add_argument("--manifest", help="清单文件名，默认在包根目录自动查找")
    official.add_argument("--initial-file", help="初始训练集文件名，默认自动推断")
    official.add_argument("--stream-file", help="评测流文件名，默认自动推断")
    official.add_argument("--target", help="目标列名")
    official.add_argument("--id-column", help="样本ID列名，用于跨窗口去重与守护集隔离")
    official.add_argument("--timestamp-column", help="时间列名，仅用于声明排除，不参与训练")
    official.add_argument("--initial-size", type=int, help="初始训练集行数")
    official.add_argument("--window-size", type=int, help="窗口大小")
    official.add_argument("--drop-column", action="append", default=[], help="需要排除的列")
    official.add_argument("--print-spec", action="store_true", help="只打印解析结果，不运行")
    official.add_argument("--output", type=Path, help="覆盖运行结果目录")
    _add_llm_overrides(official)
    _add_tracking_overrides(official)

    cmapss = subparsers.add_parser(
        "run-cmapss",
        help="在 C-MAPSS 发动机退化数据上运行 RUL 回归",
    )
    cmapss.add_argument("--config", type=Path, default=DEFAULT_CMAPSS_CONFIG)
    cmapss.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="含 train_FD001~FD004.txt 的目录",
    )
    cmapss.add_argument("--fd", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    cmapss.add_argument("--initial-size", type=int, help="默认取配置 stream.initial_samples")
    cmapss.add_argument("--window-size", type=int, help="默认取配置 stream.window_size")
    cmapss.add_argument("--rul-clip", type=int, default=RUL_CLIP, help="RUL 截断上限")
    cmapss.add_argument(
        "--no-rolling",
        action="store_true",
        help="关闭滑动窗特征，仅用原始传感器值",
    )
    cmapss.add_argument("--rolling-window", type=int, default=20, help="滑动窗统计的窗口大小")
    cmapss.add_argument("--output", type=Path, help="覆盖运行结果目录")
    _add_llm_overrides(cmapss)
    _add_tracking_overrides(cmapss)

    gas = subparsers.add_parser(
        "run-gas",
        help="在 Gas 传感器漂移数据上运行 6 类气体分类",
    )
    gas.add_argument("--config", type=Path, default=DEFAULT_GAS_CONFIG)
    gas.add_argument("--data-dir", type=Path, required=True, help="含 Dataset/batch*.dat 的目录")
    gas.add_argument("--initial-size", type=int, help="默认取配置 stream.initial_samples")
    gas.add_argument("--window-size", type=int, help="默认取配置 stream.window_size")
    gas.add_argument("--output", type=Path, help="覆盖运行结果目录")
    _add_llm_overrides(gas)
    _add_tracking_overrides(gas)

    xjtu = subparsers.add_parser(
        "run-xjtu",
        help="在单个 XJTU-SY 轴承全寿命序列上运行离线 RUL 回归",
    )
    xjtu.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "xjtu.yaml")
    xjtu.add_argument("--data-dir", type=Path, required=True, help="XJTU-SY 数据根目录")
    xjtu.add_argument("--condition", required=True, choices=XjtuBearingStream.CONDITIONS)
    xjtu.add_argument("--bearing", required=True, help="轴承名称，例如 Bearing3_1")
    xjtu.add_argument("--initial-size", type=int, help="默认取配置 stream.initial_samples")
    xjtu.add_argument("--window-size", type=int, help="默认取配置 stream.window_size")
    xjtu.add_argument("--output", type=Path, help="覆盖运行结果目录")
    _add_llm_overrides(xjtu)
    _add_tracking_overrides(xjtu)

    official_eval = subparsers.add_parser(
        "official-evaluate",
        help="离线评估官方 CPU A/B/C 包公开数据，不代表隐藏测试评测",
    )
    official_eval.add_argument("--pack-root", type=Path, required=True, help="官方包根目录")
    official_eval.add_argument("--profile", choices=["A", "B", "C"], required=True)
    official_eval.add_argument("--train", type=Path, required=True, help="公开 train.jsonl")
    official_eval.add_argument(
        "--validation", type=Path, required=True, help="公开 validation.jsonl"
    )
    official_eval.add_argument("--epochs", type=int, default=0, help="候选续训轮数；0 只测静态基线")
    official_eval.add_argument(
        "--label-provenance", type=Path, help="B profile 续训必填的标签来源声明 JSON"
    )
    official_eval.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "official",
        help="本地实验记录根目录；每次运行使用新目录",
    )
    official_eval.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="实验与 SwanLab 记录配置"
    )
    _add_tracking_overrides(official_eval)

    official_predict = subparsers.add_parser(
        "official-predict",
        help="按官方包参考协议从 stdin 读取 JSONL 并向 stdout 输出预测",
    )
    official_predict.add_argument("--pack-root", type=Path, required=True, help="官方包根目录")
    official_predict.add_argument("--profile", choices=["A", "B", "C"], required=True)
    official_predict.add_argument(
        "--checkpoint", type=Path, help="可选候选 checkpoint；默认包内基线"
    )

    smoke = subparsers.add_parser(
        "llm-smoke-test",
        help="对配置的策略模型端点发起一次真实决策调用",
    )
    smoke.add_argument("--config", type=Path, default=DEFAULT_LLM_CONFIG)
    _add_llm_overrides(smoke)

    env_check = subparsers.add_parser(
        "env-check",
        help="离线评测前的环境自检（依赖、端点连通性、输出目录）",
    )
    env_check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    env_check.add_argument("--strict", action="store_true", help="自检不通过时以非零码退出")
    _add_tracking_overrides(env_check)

    swanlab_check = subparsers.add_parser(
        "swanlab-check",
        help="SwanLab 记录通道自检：包、密钥、真正写一条标量与文本",
    )
    swanlab_check.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    swanlab_check.add_argument(
        "--online",
        action="store_true",
        help="真连云端探针（需已 swanlab login 或设置 SWANLAB_API_KEY），默认走离线探针",
    )
    swanlab_check.add_argument("--strict", action="store_true", help="探针失败时以非零码退出")
    _add_tracking_overrides(swanlab_check)
    return parser


def _env_check_passed(report: dict[str, Any], strict: bool) -> bool:
    if not strict:
        return True
    if report["run_root_writable"] is not True:
        return False
    if report["swanlab_enabled"] and report["swanlab_mode"] == "online":
        dependency = report["dependencies"].get("swanlab")
        if not isinstance(dependency, str) or dependency.startswith("missing"):
            return False
        if not report["swanlab_api_key_present"]:
            return False
    if report["llm_enabled"]:
        dependency = report["dependencies"].get("openai")
        if not isinstance(dependency, str) or dependency.startswith("missing"):
            return False
        return report["endpoint_reachable"] is True
    return True


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "official-evaluate":
        try:
            config = load_config(args.config)
            config.tracking.run_root = args.run_root.resolve()
            _apply_tracking_overrides(config, args)
            tracker = _build_tracker(config)
            result = evaluate_public_package(
                pack_root=args.pack_root,
                profile=args.profile,
                train_path=args.train,
                validation_path=args.validation,
                run_root=args.run_root,
                epochs=args.epochs,
                label_provenance_path=args.label_provenance,
                tracker=tracker,
            )
        except Exception as exc:
            print(f"官方公开包本地评估失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if args.command == "official-predict":
        try:
            predict_jsonl(
                pack_root=args.pack_root,
                profile=args.profile,
                checkpoint=args.checkpoint,
            )
        except Exception as exc:
            print(f"官方参考预测协议失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        return 0

    try:
        config = load_config(args.config)
        _apply_llm_overrides(config, args)
        _apply_tracking_overrides(config, args)
        if args.command == "llm-smoke-test":
            _run_llm_smoke_test(config)
            return 0
        if args.command == "swanlab-check":
            report = _run_swanlab_check(config, args.online)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not args.strict:
                return 0
            return 0 if report["probe"] == "ok" else 1
        if args.command == "env-check":
            report = _run_env_check(config, args.config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if _env_check_passed(report, args.strict) else 1
        if args.command == "run-official" and args.print_spec:
            spec = _build_official_spec(args, config)
            print(json.dumps(spec.describe(), ensure_ascii=False, indent=2))
            return 0
        if getattr(args, "output", None):
            config.tracking.run_root = args.output.resolve()
        stream = _build_stream(args, config)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError, LLMProviderError) as exc:
        parser.error(str(exc))

    try:
        result = _run(config, stream)
    except (BudgetExceededError, ValueError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    return 0 if result.get("run_status") in {"complete", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
