"""扩展性验证：在「数据规模 × 漂移强度 × 漂移时机 × 特征维度」上跑闭环与静态对照。

用途：为技术报告和线下答辩提供可复核的扩展性/资源效率证据。
运行：.venv\\Scripts\\python.exe tools\\扩展性验证实验.py
产物：runs/scalability/实验结果.json、实验结果.csv，以及每组独立的 run 目录。

边界：全部使用合成漂移流与自建 sklearn 模型，不是官方 A/B/C 数据、不是平台成绩。
静态对照是「本框架关掉更新」的自建对照组，不是官方静态基线。
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

from industrial_evolution.config import load_config
from industrial_evolution.data.synthetic import SyntheticDriftStream
from industrial_evolution.orchestrator import EvolutionOrchestrator
from industrial_evolution.tracking.local_tracker import LocalTracker
from industrial_evolution.tracking.swanlab_tracker import CompositeTracker

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "scalability"
RUN_ROOT = OUT_DIR / "runs"
BASE_CONFIG = ROOT / "configs" / "tuned.yaml"

# 名称, 覆盖的 stream 参数。所有 abrupt_drift_start 必须 < n_windows。
SCENARIOS: list[tuple[str, dict[str, float | int]]] = [
    ("基准", {}),
    ("小规模", {"window_size": 100, "n_windows": 14, "gradual_drift_start": 5, "abrupt_drift_start": 10}),
    ("大规模", {"window_size": 400, "n_windows": 24}),
    ("低标签噪声", {"label_noise": 0.02}),
    ("高标签噪声", {"label_noise": 0.20}),
    ("早漂移", {"gradual_drift_start": 3, "abrupt_drift_start": 8}),
    ("晚漂移", {"gradual_drift_start": 10, "abrupt_drift_start": 15}),
    ("低特征维度", {"n_features": 4}),
    ("高特征维度", {"n_features": 16}),
    ("长稳定期", {"n_windows": 24, "gradual_drift_start": 12, "abrupt_drift_start": 18}),
]


def _working_set_mib() -> float | None:
    """返回当前进程工作集（MiB）；非 Windows 返回 None，不猜数。"""
    if platform.system() != "Windows":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        getter = kernel32.K32GetProcessMemoryInfo
        getter.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        getter.restype = wintypes.BOOL
        if not getter(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return round(counters.WorkingSetSize / 1024 / 1024, 2)
    except Exception:
        return None


def _model_bytes(run_dir: Path) -> int:
    models = run_dir / "models"
    if not models.is_dir():
        return 0
    return sum(path.stat().st_size for path in models.glob("*.joblib"))


def _run_once(name: str, overrides: dict[str, float | int], static: bool) -> dict:
    config = load_config(BASE_CONFIG)
    for key, value in overrides.items():
        setattr(config.stream, key, value)
    config.tracking.run_root = RUN_ROOT
    config.tracking.enable_swanlab = False
    config.tracking.emit_batch_decisions = False
    config.budget.max_updates = 0 if static else max(6, config.stream.n_windows // 3)
    config.budget.min_updates = 0 if static else 3
    config.budget.label_budget_mode = "count_only"

    local = LocalTracker(RUN_ROOT)
    tracker = CompositeTracker(local)
    orchestrator = EvolutionOrchestrator(
        config, SyntheticDriftStream(config.stream, config.seed), tracker
    )
    started = perf_counter()
    result = orchestrator.run()
    wall = perf_counter() - started

    return {
        "scenario": name,
        "mode": "静态对照" if static else "自进化",
        "initial_samples": config.stream.initial_samples,
        "window_size": config.stream.window_size,
        "n_windows": config.stream.n_windows,
        "n_features": config.stream.n_features,
        "label_noise": config.stream.label_noise,
        "gradual_drift_start": config.stream.gradual_drift_start,
        "abrupt_drift_start": config.stream.abrupt_drift_start,
        "samples_total": config.stream.initial_samples
        + config.stream.n_windows * config.stream.window_size,
        "mean_score": result["mean_score"],
        "final_score": result["final_score"],
        "initial_score": result["initial_score"],
        "updates_attempted": result["updates_attempted"],
        "updates_deployed": result["updates_deployed"],
        "updates_rejected": result["updates_rejected"],
        "rollbacks": result["rollbacks"],
        "failures": result["failures"],
        "skipped_windows": result["skipped_windows"],
        "run_status": result["run_status"],
        "min_updates_met": result["min_updates_met"],
        "wall_seconds": round(wall, 4),
        "training_seconds": result["budget"]["training_seconds_used"],
        "labels_used": result["budget"]["labels_used"],
        "working_set_mib": _working_set_mib(),
        "model_bytes": _model_bytes(local.run_dir),
        "run_id": result["run_id"],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, overrides in SCENARIOS:
        evolving = _run_once(name, overrides, static=False)
        static = _run_once(name, overrides, static=True)
        evolving["static_mean_score"] = static["mean_score"]
        evolving["mean_gain"] = round(evolving["mean_score"] - static["mean_score"], 6)
        evolving["final_gain"] = round(evolving["final_score"] - static["final_score"], 6)
        static["static_mean_score"] = static["mean_score"]
        static["mean_gain"] = 0.0
        static["final_gain"] = 0.0
        rows.extend([evolving, static])
        print(
            f"{name:<8} 自进化 mean={evolving['mean_score']:.4f} "
            f"静态={static['mean_score']:.4f} 增益={evolving['mean_gain']:+.4f} "
            f"迭代={evolving['updates_attempted']} 墙钟={evolving['wall_seconds']:.2f}s "
            f"标签={evolving['labels_used']} 跳过={evolving['skipped_windows']}",
            flush=True,
        )

    payload = {
        "purpose": "scalability_validation",
        "data_source": "synthetic_drift_stream",
        "note": "合成流与自建静态对照，不是官方数据、不是平台成绩；工作集为单进程累计读数。",
        "config": str(BASE_CONFIG.relative_to(ROOT)),
        "scenarios": [name for name, _ in SCENARIOS],
        "rows": rows,
    }
    (OUT_DIR / "实验结果.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = list(rows[0])
    with (OUT_DIR / "实验结果.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n产物：{OUT_DIR / '实验结果.json'}")
    print(f"      {OUT_DIR / '实验结果.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
