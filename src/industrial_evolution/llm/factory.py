from __future__ import annotations

from typing import Any

from ..config import LLMConfig
from .openai_provider import OpenAICompatibleProvider
from .provider import PolicyLLMProvider


def build_provider(config: LLMConfig, *, client: Any | None = None) -> PolicyLLMProvider:
    """构造受控策略客户端；模型服务限定为本机或内网的 OpenAI 兼容端点。"""
    return OpenAICompatibleProvider(config, client=client)


def dependency_status() -> dict[str, str | bool | None]:
    """返回运行依赖与端点配置的自检结果，供容器与评测脚本快速定位环境问题。"""
    status: dict[str, str | bool | None] = {}
    for name in ("openai", "swanlab"):
        try:
            module = __import__(name)
        except ImportError as exc:
            status[name] = f"missing ({type(exc).__name__})"
        else:
            status[name] = getattr(module, "__version__", "unknown")
    return status
