#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 Sampler 权重「真能调用、输出合规」。

表单对 API Key 的要求是「须有权访问并调用所填权重或基础模型」，所以仅有一个
`trio://` 路径不够，必须证明三件事：

  1. `create_sampling_client(model_path=...)` 能加载这份权重；
  2. 用 Harness 实际会发的 prompt 采样，模型能返回文本；
  3. 返回文本能按 Harness 的 schema 解析（action / confidence / reason）。

第 3 点最关键。微调前模型有概率输出 Markdown 围栏或多余字段，那样在
`openai_provider.py::_parse` 里会被判为不合法而整条降级回规则策略。
本脚本用留出样本统计「合规率」和「与规则策略一致率」，两个数字都写进报告。

用法
----
    set PYTRIO_API_KEY=xxxxxx
    python verify_sampler.py --base-model Qwen/Qwen3.8-27B \\
        --model-path 'trio://.../sampler_weights/sampler-r1' --samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = Path(__file__).resolve().parent / "sft_data.jsonl"
OUTPUTS = Path(__file__).resolve().parent / "outputs"

# 与 src/industrial_evolution/llm/openai_provider.py::_system_prompt 逐字一致
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

ACTIONS = {"no_update", "retrain_with_replay"}

# 原始 tokenizer 采样会把结束符解码回字面量，HTTP 接口不会。
# 这里按与线上一致的口径剥掉，否则会把正确输出误判成「非法 JSON」。
SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|end|>", "</s>", "<eos>")


def _strip_special(text: str, tokenizer: Any = None) -> str:
    out = text
    extra = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    for token in (*SPECIAL_TOKENS, extra):
        if token:
            out = out.replace(token, "")
    return out.strip()


def _die(message: str) -> None:
    print(f"[失败] {message}")
    raise SystemExit(1)


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except (AttributeError, TypeError):
        return list(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def _parse(raw: str) -> tuple[dict[str, Any] | None, str]:
    """按 Harness 的实际解析规则校验输出，返回 (解析结果, 失败原因)。

    与 openai_provider 的做法一致：先剥掉可能出现的代码围栏，再整体 json.loads。
    不做「从文本里抠 JSON」的宽松兜底，因为线上也不会那样兜底。
    """
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"非法JSON: {exc.msg}"
    if not isinstance(obj, dict):
        return None, f"顶层不是对象: {type(obj).__name__}"
    extra = set(obj) - {"action", "confidence", "reason"}
    if extra:
        return None, f"多余字段: {sorted(extra)}"
    action = obj.get("action")
    if action not in ACTIONS:
        return None, f"action 非法: {action!r}"
    conf = obj.get("confidence")
    if not isinstance(conf, (int, float)) or isinstance(conf, bool):
        return None, f"confidence 非数字: {conf!r}"
    if not 0.0 <= float(conf) <= 1.0:
        return None, f"confidence 越界: {conf}"
    reason = obj.get("reason")
    if not isinstance(reason, str) or not 1 <= len(reason) <= 120:
        return None, f"reason 长度非法: {len(reason) if isinstance(reason, str) else type(reason)}"
    return obj, ""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="验证 Sampler 权重可调用且输出合规")
    parser.add_argument("--base-model", required=True, help="基座模型 ID")
    parser.add_argument("--model-path", required=True, help="save_weights_for_sampler 返回的 trio:// 路径")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="留出样本来源")
    parser.add_argument("--samples", type=int, default=20, help="抽样条数")
    parser.add_argument("--seed", type=int, default=7, help="抽样种子")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心，便于复现")
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    if not os.environ.get("PYTRIO_API_KEY"):
        _die("未设置环境变量 PYTRIO_API_KEY")
    if not args.model_path.startswith("trio://"):
        _die(f"--model-path 需要 trio:// 路径，实际收到: {args.model_path}")
    if not args.data.exists():
        _die(f"数据文件不存在: {args.data}")

    samples: list[dict[str, Any]] = []
    with args.data.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        _die(f"数据为空: {args.data}")

    picked = random.Random(args.seed).sample(samples, min(args.samples, len(samples)))
    print(f"[数据] 从 {len(samples)} 条中抽取 {len(picked)} 条验证（seed={args.seed}）")

    try:
        import pytrio
    except ImportError:
        _die("未安装 pytrio SDK")

    try:
        from pytrio.model.tokenizer import get_tokenizer
    except Exception as exc:  # noqa: BLE001
        _die(f"导入 tokenizer 失败: {exc}")

    print("[客户端] 创建 ServiceClient…")
    try:
        client = pytrio.ServiceClient()
    except Exception as exc:  # noqa: BLE001
        _die(f"认证失败: {type(exc).__name__}: {exc}")

    print(f"[客户端] 加载 Sampler 权重: {args.model_path}")
    try:
        sampling = client.create_sampling_client(
            base_model=args.base_model, model_path=args.model_path
        )
    except Exception as exc:  # noqa: BLE001
        _die(f"加载权重失败（路径是否过期/类型是否 Sampler？）: {exc}")

    try:
        tokenizer = get_tokenizer(args.base_model)
    except Exception as exc:  # noqa: BLE001
        _die(f"加载 tokenizer 失败: {exc}")

    params = pytrio.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stop=["\n\n"],
    )

    valid = 0
    agree = 0
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(picked, 1):
        request_json = json.dumps(
            sample["request"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        prompt_text = f"{SYSTEM_PROMPT}\n{request_json}\n"
        prompt_ids = _encode(tokenizer, prompt_text, add_special_tokens=True)
        try:
            response = sampling.sample(
                pytrio.ModelInput.from_ints(prompt_ids),
                num_samples=1,
                sampling_params=params,
            ).result()
            raw = (response.sequences[0].text or "").strip()
        except Exception as exc:  # noqa: BLE001
            raw = ""
            print(f"  [{index:02d}] 采样异常: {type(exc).__name__}: {exc}")

        cleaned = _strip_special(raw, tokenizer)
        obj, why = _parse(cleaned)
        expected = sample["response"]["action"]
        ok = obj is not None
        hit = ok and obj.get("action") == expected
        valid += int(ok)
        agree += int(hit)

        mark = "OK " if hit else ("合规不一致" if ok else "不合规")
        print(f"  [{index:02d}] {mark}  期望={expected}  实际={obj.get('action') if ok else '—'}")
        if not ok and why:
            print(f"        原因: {why}  原文前 100 字: {cleaned[:100]!r}")

        rows.append(
            {
                "index": index,
                "window_index": sample["request"].get("window_index"),
                "expected_action": expected,
                "raw_output": raw,
                "cleaned_output": cleaned,
                "parsed": obj,
                "valid": ok,
                "agree": hit,
                "error": why or None,
            }
        )

    total = len(picked)
    print("")
    print("=" * 72)
    print(f"权重路径    : {args.model_path}")
    print(f"基座模型    : {args.base_model}")
    print(f"抽样条数    : {total}")
    print(f"输出合规率  : {valid}/{total} = {valid / total:.3f}")
    print(f"与规则一致率: {agree}/{total} = {agree / total:.3f}")
    print("=" * 72)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"verify-{args.model_path.rstrip('/').rsplit('/', 1)[-1]}.json"
    out.write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "base_model": args.base_model,
                "samples": total,
                "seed": args.seed,
                "temperature": args.temperature,
                "valid_rate": round(valid / total, 6),
                "agree_rate": round(agree / total, 6),
                "details": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"明细已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
