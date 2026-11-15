from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ENV_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SECRET_ENV_SUFFIXES = ("API_KEY", "TOKEN")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})
_PRIVATE_SUFFIXES = (".local", ".internal", ".lan", ".localhost")


def _is_private_host(host: str) -> bool:
    """判断主机是否落在本机或内网，用于放行断网环境下的本地推理服务。"""
    normalized = host.strip("[]").lower()
    if normalized in _LOCAL_HOSTS or normalized.endswith(_PRIVATE_SUFFIXES):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StreamConfig(StrictModel):
    initial_samples: int = Field(ge=100)
    window_size: int = Field(ge=30)
    n_windows: int = Field(ge=1)
    n_features: int = Field(ge=2)
    gradual_drift_start: int = Field(ge=0)
    abrupt_drift_start: int = Field(ge=1)
    label_noise: float = Field(ge=0.0, lt=0.5)

    @model_validator(mode="after")
    def validate_drift_order(self) -> StreamConfig:
        if self.gradual_drift_start >= self.abrupt_drift_start:
            raise ValueError("gradual_drift_start 必须小于 abrupt_drift_start")
        if self.abrupt_drift_start >= self.n_windows:
            raise ValueError("abrupt_drift_start 必须小于 n_windows")
        return self


class MonitoringConfig(StrictModel):
    psi_threshold: float = Field(gt=0)
    ks_threshold: float = Field(gt=0, lt=1)
    performance_drop_threshold: float = Field(gt=0)
    label_shift_threshold: float = Field(gt=0, lt=1)
    consecutive_windows: int = Field(ge=1)
    reference_update_rate: float = Field(ge=0.0, le=1.0)
    min_valid_feature_values: int = Field(default=5, ge=2)
    non_finite_rate_threshold: float = Field(default=0.10, gt=0.0, le=1.0)


class PolicyConfig(StrictModel):
    cooldown_windows: int = Field(ge=0)
    rollback_patience: int = Field(ge=1)
    rollback_drop: float = Field(gt=0)


class TrainingConfig(StrictModel):
    replay_capacity: int = Field(ge=100)
    recent_fraction: float = Field(gt=0.0, le=1.0)
    max_replay_samples: int = Field(ge=0)
    holdout_fraction: float = Field(gt=0.05, lt=0.5)


class ValidationConfig(StrictModel):
    min_improvement: float = Field(ge=0.0)
    max_historical_drop: float = Field(ge=0.0)
    guard_samples: int = Field(ge=0)
    # 主指标可配置：赛题官方 A 用 Macro-F1、B/C 用 RMSE，不能一律用 balanced_accuracy。
    # auto = 分类用 balanced_accuracy、回归用 negative_rmse（保持历史行为）。
    score_metric: Literal["auto", "balanced_accuracy", "macro_f1", "negative_rmse"] = "auto"
    official_baseline_score: float | None = None
    official_metric_name: str | None = None


class BudgetConfig(StrictModel):
    max_updates: int = Field(ge=0)
    max_training_seconds: float = Field(gt=0)
    max_labels: int = Field(ge=0)
    max_tokens: int = Field(ge=0)
    # 按批次提供标签的运行模式下，标签配额按 count_only 模式记账、
    # 与窗口推进解耦；enforce 模式保留给需要模拟配额的本地实验。
    label_budget_mode: Literal["count_only", "enforce"] = "count_only"
    # 赛题要求至少三次尝试迭代。这里做显式校验与告警，不伪造迭代动作。
    min_updates: int = Field(default=3, ge=0)


class TrackingConfig(StrictModel):
    """实验记录配置。

    ``enable_swanlab`` 打开后，SwanLab 只做镜像：本地 ``runs/<run_id>/`` 永远是主记录，
    SwanLab 上传失败不影响主流程，只会补一条 ``tracker_mirror_failure`` 事件。

    ``swanlab_mode`` 决定上传形态，评测环境无外网时必须显式选一个非 online 模式：

    - ``online``：上传到云端看板（需 ``swanlab login`` 或 ``SWANLAB_API_KEY``）；
    - ``local``：只在本机启动/写入本地看板，不出网；
    - ``offline``：写入本地目录，之后可 ``swanlab sync`` 补传；
    - ``disabled``：等价于不初始化，用于确认无网络又不想看到告警时。
    """

    run_root: Path
    enable_swanlab: bool = False
    swanlab_project: str = "industrial_evolution"
    swanlab_mode: Literal["online", "local", "offline", "disabled"] = "online"
    swanlab_workspace: str | None = None
    swanlab_log_dir: Path | None = None
    swanlab_tags: list[str] = Field(default_factory=list)
    swanlab_upload_artifacts: bool = True
    swanlab_artifact_max_mb: float = Field(default=64.0, gt=0)
    # 逐批次「上线 / 不上线」判断始终落盘到 run 目录的 decisions.jsonl；
    # 设为 true 时额外打印一行 JSON 到 stdout，供在线批次握手协议对接。
    emit_batch_decisions: bool = False


class LLMConfig(StrictModel):
    """可选策略模型接入配置。

    仅支持 ``openai_compatible`` 一种形态：任意 OpenAI 兼容端点，用于评测环境中由
    本机或内网推理服务（例如 ``http://127.0.0.1:8000/v1``）承载的模型。系统不提供
    任何公网模型服务实现；默认 ``enabled=false``，全部决策由确定性规则完成。

    安全边界：公网地址必须走 https，``allowed_hosts`` 可进一步锁定主机名，且
    ``api_key_env`` 只能指向以 ``API_KEY`` / ``TOKEN`` 结尾的大写环境变量，避免配置
    被改成读取任意环境变量做外发。
    """

    enabled: bool = False
    provider: Literal["openai_compatible"] = "openai_compatible"
    model: str = Field(default="local-model", min_length=1)
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key_env: str = "LOCAL_LLM_API_KEY"
    api_key_required: bool = True
    api_key_placeholder: str = Field(default="EMPTY", min_length=1)
    json_mode: bool = True
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    allowed_hosts: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=5)
    max_output_tokens: int = Field(default=160, ge=32, le=2048)
    max_tokens_per_call: int = Field(default=800, ge=64)
    max_calls: int = Field(default=6, ge=0)
    min_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    call_cooldown_windows: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_endpoint_and_secret_name(self) -> LLMConfig:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("llm.base_url 必须是带主机名的 http(s) 地址")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("llm.base_url 不允许包含账号、查询串或 URL 片段")

        host = parsed.hostname.lower()
        if self.allowed_hosts and host not in {item.lower() for item in self.allowed_hosts}:
            raise ValueError("llm.base_url 的主机不在 llm.allowed_hosts 白名单内")
        if parsed.scheme == "http" and not _is_private_host(host):
            raise ValueError("llm.base_url 对公网地址必须使用 https")

        if not _ENV_NAME_PATTERN.match(self.api_key_env):
            raise ValueError("llm.api_key_env 必须是大写环境变量名")
        if not self.api_key_env.endswith(_SECRET_ENV_SUFFIXES):
            raise ValueError("llm.api_key_env 必须以 API_KEY 或 TOKEN 结尾")
        return self

    @property
    def endpoint_label(self) -> str:
        return "OpenAI兼容端点"


class AppConfig(StrictModel):
    seed: int = 42
    task_type: Literal["classification", "regression"] = "classification"
    stream: StreamConfig
    monitoring: MonitoringConfig
    policy: PolicyConfig
    training: TrainingConfig
    validation: ValidationConfig
    budget: BudgetConfig
    tracking: TrackingConfig
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("配置文件顶层必须是YAML映射")
    return AppConfig.model_validate(raw)
