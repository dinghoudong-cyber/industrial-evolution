#!/usr/bin/env bash
# 统一运行入口：评测方可以直接 `docker run <image> <mode>`，无需关心内部的 python 调用。
#
#   demo    合成漂移数据演示（默认）
#   csv     在给定 CSV 上按原始行顺序运行
#   smoke   对配置的策略模型端点做一次真实决策调用
#   check   环境自检（依赖 / 端点连通性 / 输出目录可写）
#   record  记录通道自检（SwanLab 包 / 凭证 / 真正写一条标量与文本）
#   cli     其余参数原样透传给 industrial_evolution.cli
#
# 环境变量：
#   IE_CONFIG        配置文件路径，默认 /app/configs/tuned.yaml
#   IE_RUN_ROOT      输出根目录，默认 /app/runs
#   IE_CSV           csv 模式的数据文件路径
#   IE_TARGET        csv 模式的目标列名
#   IE_INITIAL_SIZE  csv 模式的初始训练样本数，默认 600
#   IE_WINDOW_SIZE   csv 模式的窗口大小，默认 200
#   IE_SWANLAB       设为 1 时开启 SwanLab 镜像记录
#   IE_SWANLAB_MODE  online / local / offline / disabled，默认取配置文件
#   SWANLAB_API_KEY  云端记录密钥；只设 IE_SWANLAB=1 而不给密钥会自动切换为 offline
set -euo pipefail

CONFIG="${IE_CONFIG:-/app/configs/tuned.yaml}"
OUTPUT="${IE_RUN_ROOT:-/app/runs}"
MODE="${1:-demo}"
if [ "$#" -gt 0 ]; then shift; fi

# 记录开关统一从这里透传，评测方只需设置环境变量，不必改 YAML。
RECORD_ARGS=()
if [ "${IE_SWANLAB:-0}" = "1" ]; then
  RECORD_ARGS+=(--swanlab)
elif [ "${IE_SWANLAB:-}" = "0" ]; then
  RECORD_ARGS+=(--no-swanlab)
fi
if [ -n "${IE_SWANLAB_MODE:-}" ]; then
  RECORD_ARGS+=(--swanlab-mode "$IE_SWANLAB_MODE")
fi

case "$MODE" in
  demo)
    exec python -m industrial_evolution.cli demo \
      --config "$CONFIG" --output "$OUTPUT" "${RECORD_ARGS[@]}" "$@"
    ;;
  csv)
    : "${IE_CSV:?csv 模式需要设置 IE_CSV}"
    : "${IE_TARGET:?csv 模式需要设置 IE_TARGET}"
    exec python -m industrial_evolution.cli run-csv \
      --config "$CONFIG" \
      --csv "$IE_CSV" \
      --target "$IE_TARGET" \
      --initial-size "${IE_INITIAL_SIZE:-600}" \
      --window-size "${IE_WINDOW_SIZE:-200}" \
      --output "$OUTPUT" \
      "${RECORD_ARGS[@]}" \
      "$@"
    ;;
  smoke)
    exec python -m industrial_evolution.cli llm-smoke-test \
      --config "${IE_CONFIG:-/app/configs/llm-local.yaml}" "$@"
    ;;
  check)
    exec python -m industrial_evolution.cli env-check --config "$CONFIG" "$@"
    ;;
  record)
    exec python -m industrial_evolution.cli swanlab-check --config "$CONFIG" "${RECORD_ARGS[@]}" "$@"
    ;;
  *)
    exec python -m industrial_evolution.cli "$MODE" "$@"
    ;;
esac
