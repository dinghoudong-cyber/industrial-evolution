#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 Harness 的决策事件流改造成 LLM 监督微调样本（规则策略蒸馏）。

为什么需要这一步
----------------
平台要求提交的是「持久化 Sampler 微调权重」，也就是一个能驱动 Harness 的
大语言模型。本项目 Harness 的策略层 `HybridPolicy` 在 LLM 开启时，会把
`LLMDecisionRequest` 序列化成 JSON 发给模型，模型必须回一个
`{"action","confidence","reason"}`。所以微调任务就是：教会模型做这个决策。

输入张量从哪来
--------------
`runs/*/events.jsonl` 里有两类事件正好构成一对：
  - `drift_diagnosis`：漂移诊断结果（含 per_feature 指标、performance、label_shift）
  - `update_decision`：规则策略对该窗口做出的决策（action + reason + 预算快照）
两者按 `window` 配对，即可还原出 15 个字段的 `LLMDecisionRequest`。

置信度怎么来（重要口径）
------------------------
规则策略本身不产出置信度（事件里 `confidence` 为 null）。这里用**信号强度做
单调映射**，作为校准目标，函数写在 `_confidence_for()` 里可逐项复核：

  - `no_update`：漂移总体分越低越确信（0.55 ~ 0.95）
  - `retrain_with_replay`：性能跌幅越大越确信（0.55 ~ 0.95）

这是蒸馏口径，不是凭空标注；若后续改用人工标注或 LLM 自标注，替换该函数即可。

用法
----
    python build_sft_data.py                      # 默认扫描 submission 与 runs
    python build_sft_data.py --out sft_data.jsonl
    python build_sft_data.py --min-samples 8      # 不足则报错（默认不报错）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

# 与 src/industrial_evolution/llm/openai_provider.py::_system_prompt 逐字一致。
# 训练与推理必须用同一段系统提示，否则微调后的模型在 Harness 里会跑偏。
SYSTEM_PROMPT = (
    "你是工业模型更新策略器。只能依据用户提供的结构化指标做决策。"
    "必须只输出一个JSON对象，不要输出Markdown、代码围栏、解释文字或额外字段。"
    "JSON必须严格符合这个示例结构："
    '{"action":"retrain_with_replay","confidence":0.85,'
    '"reason":"持续混合漂移且性能下降"}。'
    "action只能是no_update或retrain_with_replay；"
    "confidence必须为0到1之间的数字；reason必须是1到120字的中文字符串。"
    "不得建议绕过预算、冷却期、候选评估或回滚保护。"
)

ACTION_NO_UPDATE = "no_update"
ACTION_RETRAIN = "retrain_with_replay"
MAX_REASON_CHARS = 120


def _confidence_for(action: str, report: dict[str, Any]) -> float:
    """把规则决策的信号强度映射成 [0.55, 0.95] 的置信度（见模块说明）。"""
    if action == ACTION_NO_UPDATE:
        overall = float(report.get("feature", {}).get("overall_score") or 0.0)
        # 漂移分越低 → 越确信「不需要更新」
        return round(0.55 + 0.40 * (1.0 - min(max(overall, 0.0), 1.0)), 3)
    drop = float((report.get("performance") or {}).get("score_drop") or 0.0)
    # 跌幅越大 → 越确信「需要重训」
    return round(min(0.95, 0.55 + 2.0 * max(drop, 0.0)), 3)


def _remaining(budget: dict[str, Any], used_key: str, max_key: str) -> float:
    used = float(budget.get(used_key) or 0.0)
    cap = float(budget.get(max_key) or 0.0)
    return max(0.0, cap - used)


def _build_request(
    window: int,
    report: dict[str, Any],
    budget: dict[str, Any],
    rule_reason: str,
) -> dict[str, Any]:
    """还原 LLMDecisionRequest 的 15 个字段，顺序与 dataclass 一致。"""
    feature = report.get("feature") or {}
    performance = report.get("performance") or {}
    return {
        "window_index": window,
        "drift_kind": report.get("kind") or "none",
        "feature_drift_score": round(float(feature.get("overall_score") or 0.0), 6),
        "top_features": list(feature.get("top_features") or []),
        "current_score": round(float(performance.get("current_score") or 0.0), 6),
        "rolling_score": round(float(performance.get("rolling_score") or 0.0), 6),
        "baseline_score": round(float(performance.get("baseline_score") or 0.0), 6),
        "performance_drop": round(float(performance.get("score_drop") or 0.0), 6),
        "label_shift": round(float(report.get("label_shift") or 0.0), 6),
        "remaining_updates": int(_remaining(budget, "updates_used", "max_updates")),
        "remaining_training_seconds": round(
            _remaining(budget, "training_seconds_used", "max_training_seconds"), 3
        ),
        "remaining_labels": int(_remaining(budget, "labels_used", "max_labels")),
        "remaining_tokens": int(_remaining(budget, "tokens_used", "max_tokens")),
        # 规则策略在决策前不发标签请求，固定为 0，与 PolicyContext 一致
        "requested_labels": 0,
        "rule_reason": rule_reason,
    }


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _pairs_from_run(events_path: Path) -> list[tuple[dict, dict]]:
    """按 window 把 drift_diagnosis 与 update_decision 配对。"""
    diagnoses: dict[int, dict] = {}
    decisions: dict[int, dict] = {}
    for event in _iter_events(events_path):
        etype = event.get("event_type")
        payload = event.get("payload") or {}
        window = payload.get("window")
        if not isinstance(window, int):
            continue
        if etype == "drift_diagnosis" and payload.get("report"):
            diagnoses[window] = payload
        elif etype == "update_decision":
            decisions[window] = payload

    pairs: list[tuple[dict, dict]] = []
    for window in sorted(diagnoses.keys() & decisions.keys()):
        pairs.append((diagnoses[window], decisions[window]))
    return pairs


def _sample(diag: dict, dec: dict, source: str) -> dict[str, Any] | None:
    report = diag.get("report") or {}
    if not report:
        return None
    action = dec.get("action")
    if action not in (ACTION_NO_UPDATE, ACTION_RETRAIN):
        return None
    reason = str(dec.get("reason") or "").strip()
    if not reason:
        return None
    reason = reason[:MAX_REASON_CHARS]
    budget = dec.get("budget") or {}
    request = _build_request(diag["window"], report, budget, reason)
    response = {
        "action": action,
        "confidence": _confidence_for(action, report),
        "reason": reason,
    }
    return {
        "request": request,
        "response": response,
        "meta": {
            "source": source,
            "window": diag["window"],
            "phase": diag.get("phase"),
            "is_actionable": report.get("is_actionable"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="构造 Sampler 监督微调样本")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=[
            "submission/校赛初版/实验记录/离线复核包",
            "runs",
        ],
        help="要扫描的目录（相对项目根，或绝对路径）",
    )
    parser.add_argument("--out", default="pytrio-train/sft_data.jsonl")
    parser.add_argument("--min-samples", type=int, default=0)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    project_root = Path(__file__).resolve().parent.parent

    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in args.roots:
        base = Path(root)
        if not base.is_absolute():
            base = project_root / base
        if not base.exists():
            print(f"[跳过] 目录不存在: {base}")
            continue
        for events in sorted(base.rglob("events.jsonl")):
            run_name = events.parent.name
            for diag, dec in _pairs_from_run(events):
                item = _sample(diag, dec, run_name)
                if item is None:
                    continue
                # 同一 run 内 window 唯一，跨 run 用 run+window 去重
                key = f"{run_name}:{item['meta']['window']}"
                if key in seen:
                    continue
                seen.add(key)
                samples.append(item)

    if not samples:
        print("未从事件流中找到可用样本。请检查 --roots 或先跑一次 demo / csv。")
        return 1
    if args.min_samples and len(samples) < args.min_samples:
        print(f"样本数 {len(samples)} 少于要求 {args.min_samples}")
        return 1

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for item in samples:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for item in samples:
        counts[item["response"]["action"]] = counts.get(item["response"]["action"], 0) + 1
    print(f"已写出 {len(samples)} 条样本 -> {out_path}")
    print(f"动作分布: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
