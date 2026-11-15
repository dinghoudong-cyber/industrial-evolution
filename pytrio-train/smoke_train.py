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
import random
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


def _maybe_swanlab(enabled: bool, config: dict[str, Any], run_tag: str = ""):
    """SwanLab 不在 PyTRiO SDK 内，由脚本自行上报，用于回填「训练记录链接」。

    默认写进已经设为公开的项目 ``industrial_evolution``，保证评审能直接打开。
    若新建项目，务必先在云端把项目改为公开，否则链接对外不可见。
    """
    if not enabled:
        return None
    try:
        import swanlab  # noqa: PLC0415 - 可选依赖
    except ImportError:
        print("[SwanLab] 未安装 swanlab，跳过上报（pip install swanlab）")
        return None

    project = os.environ.get("SWANLAB_PROJECT", "industrial_evolution")
    alias = config.get("base_model", "model").split("/")[-1]
    name = os.environ.get("SWANLAB_RUN_NAME") or f"pytrio-{alias}-{run_tag or 'run'}"
    mode = os.environ.get("SWANLAB_MODE", "online")
    try:
        run = swanlab.init(
            project=project,
            experiment_name=name,
            config=config,
            mode=mode,
        )
    except BaseException as exc:  # noqa: BLE001 - 上报失败绝不影响训练主流程
        print(f"[SwanLab] 初始化失败，跳过上报：{type(exc).__name__}: {exc}")
        return None
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
    parser.add_argument(
        "--save-state",
        default=None,
        help="额外保存 Train 类型断点名（供下一轮 --resume 续训，不能填进提交表单）",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="从 save_state 产出的 trio:// 断点续训（此时 --base-model 与 --rank 由断点决定）",
    )
    parser.add_argument("--seed", type=int, default=20261115, help="样本打乱种子，保证可复现")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="样本起始偏移，配合 seed 让每轮使用不同片段，避免重复同一批数据",
    )
    parser.add_argument("--epochs", type=float, default=1.0, help="按样本量换算步数时的轮数")
    parser.add_argument("--swanlab", action="store_true", help="上报 SwanLab 并打印 run 链接")
    parser.add_argument("--dry-run", action="store_true", help="只做编码校验，不调用训练接口")
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="只训练不保存 Sampler 权重（测速/探测用，避免生成大权重占用额度）",
    )
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

    if args.resume:
        if not args.resume.startswith("trio://"):
            _die(f"--resume 需要 save_state 返回的 trio:// 路径，实际收到: {args.resume}")
        print(f"[训练] 从断点续训: {args.resume}")
        try:
            training = client.create_training_client_from_state(args.resume)
        except Exception as exc:  # noqa: BLE001
            _die(f"从断点恢复训练客户端失败（断点是否过期？）: {exc}")
        base_model = getattr(training, "model_id", None) or args.base_model
        if not base_model:
            _die("断点未返回 model_id，请额外用 --base-model 指定基座模型")
        print(f"[训练] 续训客户端就绪，基座={base_model}（rank 由断点决定）")
    else:
        if not args.base_model:
            _die("请用 --base-model 指定基座模型（可先 --list-models 查看）")
        if models and args.base_model not in models:
            print(f"[警告] {args.base_model} 不在返回的模型目录中，仍尝试创建训练客户端")
        print(f"[训练] 创建 LoRA 训练客户端 base_model={args.base_model} rank={args.rank}")
        try:
            training = client.create_lora_training_client(
                base_model=args.base_model, rank=args.rank
            )
        except Exception as exc:  # noqa: BLE001
            _die(f"创建训练客户端失败（常见原因：额度不足或模型无权限）: {exc}")
        base_model = args.base_model

    try:
        from pytrio.model.tokenizer import get_tokenizer  # noqa: PLC0415

        tokenizer = get_tokenizer(base_model)
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
        "base_model": base_model,
        "rank": args.rank,
        "learning_rate": args.lr,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "samples": len(samples),
        "seed": args.seed,
        "offset": args.offset,
        "resume_from": args.resume,
    }
    swan_run = _maybe_swanlab(args.swanlab, config, run_tag=args.weight_name)
    adam = pytrio.AdamParams(learning_rate=args.lr)

    # 固定种子打乱，再按 offset 轮转起点：每轮喂不同片段，同时保证可复现
    order = list(range(len(datums)))
    random.Random(args.seed).shuffle(order)
    if args.offset:
        k = args.offset % len(order)
        order = order[k:] + order[:k]
    datums = [datums[i] for i in order]

    batch = max(1, args.batch_size)
    started = time.time()
    losses: list[float] = []
    for step in range(args.steps):
        window = [datums[(step * batch + i) % len(datums)] for i in range(batch)]
        fwd = training.forward_backward(window, loss_fn="cross_entropy", auto_shift=True).result()
        training.optim_step(adam).result()
        # cross_entropy 的标量损失在 metrics 里，逐样本值在 loss_fn_outputs
        metrics = getattr(fwd, "metrics", None) or {}
        loss = metrics.get("loss") if isinstance(metrics, dict) else None
        if loss is None:
            for key, value in (metrics.items() if isinstance(metrics, dict) else ()):
                if "loss" in str(key).lower() and isinstance(value, (int, float)):
                    loss = value
                    break
        if isinstance(loss, (int, float)):
            losses.append(float(loss))
            loss_text = f"{float(loss):.4f}"
        else:
            loss_text = f"n/a {list(metrics)[:3]}"
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"[step {step + 1}/{args.steps}] loss={loss_text}")
        if swan_run is not None and isinstance(loss, (int, float)):
            swan_run.log({"train/loss": float(loss), "step": step + 1})

    elapsed = time.time() - started
    per_step = elapsed / max(1, args.steps)
    if losses:
        summary = (
            f"平均 loss={sum(losses) / len(losses):.4f}"
            f"（首 {losses[0]:.4f} → 末 {losses[-1]:.4f}）"
        )
    else:
        summary = "loss 未返回"
    print(
        f"[训练] 完成 {args.steps} 步，用时 {elapsed:.1f}s"
        f"（{per_step:.1f}s/步），{summary}"
    )

    state_path = None
    if args.save_state:
        print(f"[断点] save_state(name={args.save_state})（Train 类型，仅供续训）…")
        try:
            state = training.save_state(args.save_state).result()
        except Exception as exc:  # noqa: BLE001
            _die(f"保存训练断点失败: {exc}")
        state_path = getattr(state, "path", None)
        if not state_path:
            _die(f"断点返回结构中没有 path 字段，实际为: {state!r}")
        print(f"[断点] 已保存: {state_path}")

    if args.no_save:
        print("[保存] 已指定 --no-save，跳过 Sampler 权重保存。")
        if swan_run is not None:
            try:
                swan_run.finish()
            except Exception:  # noqa: BLE001
                pass
        return 0

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
        "train_state_path": state_path,
        "resume_from": args.resume,
        "base_model": getattr(saved, "model", None) or base_model,
        "size": size,
        "steps": args.steps,
        "rank": args.rank,
        "learning_rate": args.lr,
        "samples": len(samples),
        "seed": args.seed,
        "offset": args.offset,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "loss_mean": round(sum(losses) / len(losses), 6) if losses else None,
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
