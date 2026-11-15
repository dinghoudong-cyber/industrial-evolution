#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTRiO 冒烟训练：验证「认证 -> LoRA 训练 -> 保存 Sampler 权重 -> 采样」全链路。

为什么先做冒烟
--------------
平台开放测试入口后，真正要先确认的不是模型效果，而是这条链路能不能走通：

  1. `PYTRIO_API_KEY` 有效、账号额度可用；
  2. 能创建 LoRA 训练客户端，能提交 forward_backward + optim_step；
  3. `save_weights_for_sampler()` 能返回 `trio://` 路径。

第 3 点最关键：平台提交表单只认 **Sampler 类型** 的持久化权重。
`save_state()` 产出的是 Train 类型，只能续训、不能推理；临时采样权重、
以及工业模型产物（`checkpoint.npz` / sklearn 模型）都不能用来填那个字段。

跑通这一步之后，再用同样的脚本加 `--steps` 和正式数据做真正的微调。

用法
----
    set PYTRIO_API_KEY=xxxxxx            # PowerShell: $env:PYTRIO_API_KEY="xxxxxx"
    python smoke_train.py --list-models  # 先看平台支持哪些基座模型
    python smoke_train.py --base-model <模型ID> --steps 1
    python smoke_train.py --base-model <模型ID> --steps 20 --data sft_data.jsonl --swanlab

产物
----
- 控制台打印 `trio://...` 完整路径（直接复制进提交表单，不要自己拼接）
- `pytrio-train/outputs/<时间戳>-sampler.json` 记录路径、基座模型、步数，便于回填表单
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = Path(__file__).resolve().parent / "outputs"

# 数据文件缺失时的最小样本，保证冒烟测试不依赖任何前置产物
BUILTIN_SAMPLES: list[dict[str, Any]] = [
    {
        "request": {
            "window_index": 7,
            "drift_kind": "concept_suspected",
            "feature_drift_score": 0.774662,
            "top_features": ["sensor_00", "sensor_01", "sensor_05"],
            "current_score": 0.611111,
            "rolling_score": 0.639482,
            "baseline_score": 0.734093,
            "performance_drop": 0.122982,
            "label_shift": 0.148321,
            "remaining_updates": 4,
            "remaining_training_seconds": 96.4,
            "remaining_labels": 3400,
            "remaining_tokens": 0,
            "requested_labels": 0,
            "rule_reason": "疑似概念漂移且性能持续下降",
        },
        "response": {
            "action": "retrain_with_replay",
            "confidence": 0.796,
            "reason": "疑似概念漂移且性能持续下降",
        },
    },
    {
        "request": {
            "window_index": 2,
            "drift_kind": "none",
            "feature_drift_score": 0.476785,
            "top_features": ["sensor_02", "sensor_00", "sensor_07"],
            "current_score": 0.669782,
            "rolling_score": 0.669782,
            "baseline_score": 0.734093,
            "performance_drop": 0.064311,
            "label_shift": 0.030833,
            "remaining_updates": 6,
            "remaining_training_seconds": 119.81,
            "remaining_labels": 4200,
            "remaining_tokens": 0,
            "requested_labels": 0,
            "rule_reason": "当前窗口无可执行漂移",
        },
        "response": {
            "action": "no_update",
            "confidence": 0.759,
            "reason": "当前窗口无可执行漂移",
        },
    },
]


def _die(message: str) -> None:
    print(f"[失败] {message}")
    raise SystemExit(1)


def _load_pytrio():
    try:
        import pytrio  # noqa: PLC0415 - 延迟导入，未安装时给可读提示
    except ImportError:
        _die(
            "未安装 pytrio SDK。请先执行：\n"
            "    <你的环境>/python -m pip install pytrio\n"
            "注意：它会带入 transformers / modelscope，建议装在独立虚拟环境，"
            "不要装进交付镜像所用的环境。"
        )
    return pytrio


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    """兼容 HF 与 ModelScope 两种 tokenizer 接口。"""
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except (AttributeError, TypeError):
        encoded = tokenizer(text, add_special_tokens=add_special_tokens)
        return list(encoded["input_ids"])


def _build_datum(pytrio: Any, tokenizer: Any, sample: dict[str, Any]) -> Any:
    """把一条样本编码成 cross_entropy 损失可用的 Datum。

    prompt 部分权重置 0、回复部分置 1，让损失只作用于模型该学的输出。
    """
    from pytrio.types import TensorData  # noqa: PLC0415 - 仅此处需要

    request_json = json.dumps(
        sample["request"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    reply_json = json.dumps(
        sample["response"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    prompt_ids = _encode(tokenizer, f"{SYSTEM_PROMPT}\n{request_json}\n", add_special_tokens=True)
    eos = getattr(tokenizer, "eos_token", None) or ""
    reply_ids = _encode(tokenizer, f"{reply_json}{eos}", add_special_tokens=False)

    tokens = prompt_ids + reply_ids
    weights = [0.0] * len(prompt_ids) + [1.0] * len(reply_ids)
    return pytrio.Datum(
        model_input=pytrio.ModelInput.from_ints(tokens),
        loss_fn_inputs={
            "target_tokens": TensorData(data=tokens, dtype="int64", shape=[len(tokens)]),
            "weights": TensorData(data=weights, dtype="float32", shape=[len(weights)]),
        },
    )


def _load_samples(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        print("[数据] 未指定 --data，使用内置 2 条样本做冒烟")
        return BUILTIN_SAMPLES
    if not path.exists():
        _die(f"数据文件不存在: {path}\n先运行 build_sft_data.py 生成。")
    samples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if not samples:
        _die(f"数据文件为空: {path}")
    print(f"[数据] 读取 {len(samples)} 条样本 <- {path}")
    return samples


def _maybe_swanlab(enabled: bool, config: dict[str, Any]):
    """SwanLab 不在 PyTRiO SDK 内，由脚本自行上报，用于回填「训练记录链接」。"""
    if not enabled:
        return None
    try:
        import swanlab  # noqa: PLC0415 - 可选依赖
    except ImportError:
        print("[SwanLab] 未安装 swanlab，跳过上报（pip install swanlab）")
        return None
    run = swanlab.init(
        project="industrial-evolution-sampler",
        experiment_name=f"pytrio-{config['base_model'].split('/')[-1]}-{config['steps']}step",
        config=config,
    )
    print(f"[SwanLab] run 已创建: {getattr(run, 'url', '（见控制台）')}")
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTRiO 冒烟训练 / 小规模微调")
    parser.add_argument("--base-model", help="基座模型 ID（用 --list-models 查询）")
    parser.add_argument("--list-models", action="store_true", help="只查询平台支持的模型后退出")
    parser.add_argument("--steps", type=int, default=1, help="优化步数，冒烟用 1")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank，范围 4~64")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1, help="每步样本数")
    parser.add_argument("--data", type=Path, default=None, help="sft_data.jsonl 路径")
    parser.add_argument("--weight-name", default="sampler", help="保存的权重名")
    parser.add_argument("--swanlab", action="store_true", help="上报 SwanLab 并打印 run 链接")
    parser.add_argument("--dry-run", action="store_true", help="只做编码校验，不调用训练接口")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.environ.get("PYTRIO_API_KEY"):
        _die(
            "未设置环境变量 PYTRIO_API_KEY。\n"
            '  PowerShell: $env:PYTRIO_API_KEY="你的key"\n'
            "  CMD       : set PYTRIO_API_KEY=你的key"
        )

    pytrio = _load_pytrio()
    samples = _load_samples(args.data)

    print("[认证] 创建 ServiceClient 并查询模型目录…")
    try:
        client = pytrio.ServiceClient()
        models = client.get_supported_models()
    except Exception as exc:  # noqa: BLE001 - 认证/额度类错误统一提示
        _die(f"认证或模型目录查询失败: {type(exc).__name__}: {exc}")

    print(f"[认证] 通过。平台支持 {len(models)} 个基座模型：")
    for name in models:
        print(f"    - {name}")

    if args.list_models:
        return 0
    if not args.base_model:
        _die("请用 --base-model 指定基座模型（可先 --list-models 查看）")
    if models and args.base_model not in models:
        print(f"[警告] {args.base_model} 不在返回的模型目录中，仍尝试创建训练客户端")

    print(f"[训练] 创建 LoRA 训练客户端 base_model={args.base_model} rank={args.rank}")
    try:
        training = client.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    except Exception as exc:  # noqa: BLE001
        _die(f"创建训练客户端失败（常见原因：额度不足或模型无权限）: {exc}")

    try:
        from pytrio.model.tokenizer import get_tokenizer  # noqa: PLC0415

        tokenizer = get_tokenizer(args.base_model)
        print("[Tokenizer] 已加载")
    except Exception as exc:  # noqa: BLE001
        _die(f"加载 tokenizer 失败（检查 modelscope/transformers 是否安装、能否联网）: {exc}")

    print("[编码] 构造训练样本…")
    datums = [_build_datum(pytrio, tokenizer, s) for s in samples]
    token_lens = [len(d.model_input.chunks[0].tokens) for d in datums]
    print(f"[编码] {len(datums)} 条，token 长度 min={min(token_lens)} max={max(token_lens)}")

    if args.dry_run:
        print("[dry-run] 编码与配置校验通过，未调用训练接口。")
        return 0

    config = {
        "base_model": args.base_model,
        "rank": args.rank,
        "learning_rate": args.lr,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "samples": len(samples),
    }
    swan_run = _maybe_swanlab(args.swanlab, config)
    adam = pytrio.AdamParams(learning_rate=args.lr)

    batch = max(1, args.batch_size)
    started = time.time()
    for step in range(args.steps):
        window = [datums[(step * batch + i) % len(datums)] for i in range(batch)]
        fwd = training.forward_backward(window, loss_fn="cross_entropy", auto_shift=True).result()
        opt = training.optim_step(adam).result()
        loss = getattr(fwd, "loss", None)
        loss_text = f"{float(loss):.4f}" if isinstance(loss, (int, float)) else "n/a"
        print(f"[step {step + 1}/{args.steps}] loss={loss_text}  optimizer={type(opt).__name__}")
        if swan_run is not None and isinstance(loss, (int, float)):
            swan_run.log({"train/loss": float(loss), "step": step + 1})

    elapsed = time.time() - started
    print(f"[训练] 完成 {args.steps} 步，用时 {elapsed:.1f}s")

    print("[保存] 调用 save_weights_for_sampler（Sampler 类型，可推理、可提交）…")
    try:
        saved = training.save_weights_for_sampler(name=args.weight_name).result()
    except Exception as exc:  # noqa: BLE001
        _die(f"保存 Sampler 权重失败: {exc}")

    weight_path = getattr(saved, "path", None)
    if not weight_path:
        _die(f"返回结构中没有 path 字段，实际为: {saved!r}")

    print("")
    print("=" * 72)
    print("提交表单「完整 Sampler 权重路径」填这一串（原样复制，不要改写）：")
    print(weight_path)
    print("=" * 72)
    print(f"所属基座模型: {getattr(saved, 'model', args.base_model)}")
    size = getattr(saved, "size", None)
    if size:
        print(f"权重大小    : {size:,} bytes")

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    record = {
        "sampler_path": weight_path,
        "base_model": getattr(saved, "model", args.base_model),
        "size": size,
        "steps": args.steps,
        "rank": args.rank,
        "learning_rate": args.lr,
        "samples": len(samples),
        "data": str(args.data) if args.data else "builtin",
        "swanlab_url": getattr(swan_run, "url", None) if swan_run else None,
        "elapsed_seconds": round(elapsed, 2),
        "created_at": stamp,
    }
    out = OUTPUTS / f"{stamp}-sampler.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"记录已写入 {out}")

    if swan_run is not None:
        try:
            swan_run.finish()
        except Exception:  # noqa: BLE001 - 收尾失败不影响主流程
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
