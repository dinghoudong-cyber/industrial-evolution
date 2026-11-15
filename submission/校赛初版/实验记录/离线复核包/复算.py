"""离线复核包自检：从 events.jsonl / decisions.jsonl 复算决策链、上下线次数与版本链。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe "submission\\校赛初版\\实验记录\\离线复核包\\复算.py"

自动选择包内最后一个「闭环迭代-run-*」目录，不依赖网络、不读取模型权重、不修改任何文件。
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def main() -> int:
    run_dirs = sorted(PACKAGE_DIR.glob("闭环迭代-run-*"))
    if not run_dirs:
        print("未找到闭环迭代 run 目录")
        return 1
    run_dir = run_dirs[-1]
    print(f"复核 run：{run_dir.name}")

    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    registry_path = run_dir / "models" / "registry.json"
    decisions_path = run_dir / "decisions.jsonl"
    for path in (events_path, summary_path, registry_path, decisions_path):
        if not path.is_file():
            print(f"缺少文件：{path}")
            return 1

    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counter = collections.Counter(row["event_type"] for row in rows)
    print(f"事件总数：{len(rows)}")
    for name in (
        "drift_diagnosis",
        "experiment_history_read",
        "update_decision",
        "hypothesis_recorded",
        "candidate_evaluated",
        "model_deployed",
        "model_rolled_back",
    ):
        print(f"  {name}: {counter.get(name, 0)}")

    chains = [
        row["payload"]
        for row in rows
        if row["event_type"] == "hypothesis_recorded"
        and row["payload"]["decision"]["proposed_action"] == "retrain_with_replay"
    ]
    print(f"\n完整决策链（读历史→假设→实验→决策）：{len(chains)} 条")
    print(f"{'窗口':>4} {'历史引用':>8} {'近期增益':>12} {'历史回退':>12} {'结论':<10} 判定")
    for chain in chains:
        outcome = chain["outcome"]
        improvement = outcome.get("candidate_improvement")
        degradation = outcome.get("historical_drop")
        print(
            f"{chain['window']:>4} {len(chain['history_read']):>8} "
            f"{improvement if improvement is not None else float('nan'):>12.6f} "
            f"{degradation if degradation is not None else float('nan'):>12.6f} "
            f"{outcome['status']:<10} {chain['verdict']}"
        )
    refuted = sum(1 for chain in chains if chain["verdict"] == "refuted")
    print(f"\n被否定的假设（负结果保留）：{refuted} 条")

    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    deployed_batches = sum(1 for row in decisions if row["decision"] == "deploy")
    print(f"逐批次上线/不上线判断：{len(decisions)} 行，其中 deploy {deployed_batches} 行")
    print(f"判断值集合：{sorted({row['decision'] for row in decisions})}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    print(
        f"\n摘要：run_status={summary.get('run_status')} "
        f"windows={summary.get('windows_processed')} "
        f"updates_attempted={summary.get('updates_attempted')} "
        f"updates_deployed={summary.get('updates_deployed')} "
        f"mean_score={summary.get('mean_score')} "
        f"metric={summary.get('metric_name')}"
    )
    print(
        f"标签口径={summary.get('label_budget_mode')} "
        f"min_updates_met={summary.get('min_updates_met')} "
        f"vs_official_baseline={summary.get('vs_official_baseline')}"
    )
    print(f"版本链：{' → '.join(item['version'] for item in registry['versions'])}")
    print(f"权重随包分发：{registry.get('weights_included')}（应为 False）")
    print(f"源码 SHA-256：{summary.get('reproducibility', {}).get('source_sha256')}")

    ok = (
        len(chains) >= 3
        and registry.get("weights_included") is False
        and len(decisions) == summary.get("windows_processed")
        and deployed_batches == summary.get("updates_deployed")
    )
    print(f"\n满足赛题「至少三次迭代」：{'是' if len(chains) >= 3 else '否'}")
    print(f"整体自检：{'通过' if ok else '未通过'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
