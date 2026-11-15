from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from ..config import LLMConfig
from ..domain import Action, LLMUsage
from .provider import LLMDecisionRequest, LLMDecisionResponse
from .schemas import LLMDecisionPayload

JSON_MODE_FALLBACK_STATUS = frozenset({400, 404, 422})


class LLMProviderError(RuntimeError):
    """调用策略大模型失败，但不暴露凭据或请求正文。"""

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        usage: LLMUsage | None = None,
        latency_seconds: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.usage = usage or LLMUsage()
        self.latency_seconds = latency_seconds


class OpenAICompatibleProvider:
    """任意 OpenAI 兼容端点的受控策略客户端。

    兼容本机或内网的 vLLM、sglang 等推理服务：只要端点提供
    ``POST {base_url}/chat/completions``，就能在断网评测环境里驱动决策。
    """

    def __init__(self, config: LLMConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client
        self.json_mode_disabled = False

    @property
    def label(self) -> str:
        return self.config.endpoint_label

    def _client_or_build(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            if self.config.api_key_required:
                raise LLMProviderError(f"环境变量 {self.config.api_key_env} 未设置")
            # vLLM / sglang 默认不校验鉴权，但仍要求一个非空占位值。
            api_key = self.config.api_key_placeholder
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderError(
                "未安装 openai 客户端，请安装 requirements-extras.txt 中的运行依赖"
            ) from exc
        return OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
        )

    def decide(self, request: LLMDecisionRequest) -> LLMDecisionResponse:
        client = self._client_or_build()
        started = perf_counter()
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        request.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            ],
            "max_tokens": self.config.max_output_tokens,
            "temperature": self.config.temperature,
        }
        try:
            response = self._create(client, payload)
        except Exception as exc:
            latency = perf_counter() - started
            raise LLMProviderError(
                f"{self.label} API调用失败（{type(exc).__name__}）",
                latency_seconds=latency,
            ) from exc

        latency = perf_counter() - started
        model = getattr(response, "model", None) or self.config.model
        usage = self._usage_from_response(response)
        finish_reason = self._finish_reason(response)
        content = self._content_text(response)
        if not content:
            raise self._response_error(
                f"{self.label}响应正文为空",
                model,
                usage,
                latency,
                finish_reason,
            )

        decoded = self._decode_json(content)
        if decoded is None:
            raise self._response_error(
                f"{self.label}返回的内容不是有效JSON",
                model,
                usage,
                latency,
                finish_reason,
            )

        try:
            decision = LLMDecisionPayload.model_validate(decoded)
        except ValidationError as exc:
            raise self._response_error(
                f"{self.label}返回的JSON不符合策略白名单",
                model,
                usage,
                latency,
                finish_reason,
            ) from exc

        return LLMDecisionResponse(
            action=Action(decision.action),
            confidence=decision.confidence,
            reason=decision.reason,
            model=model,
            usage=usage,
            latency_seconds=latency,
        )

    def _create(self, client: Any, payload: dict[str, Any]) -> Any:
        """优先用 response_format 强约束 JSON；端点不支持时降级为纯提示词约束。"""
        if not self.config.json_mode or self.json_mode_disabled:
            return client.chat.completions.create(**payload)
        try:
            return client.chat.completions.create(
                response_format={"type": "json_object"},
                **payload,
            )
        except Exception as exc:
            if not self._rejects_json_mode(exc):
                raise
            self.json_mode_disabled = True
            return client.chat.completions.create(**payload)

    @staticmethod
    def _rejects_json_mode(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status in JSON_MODE_FALLBACK_STATUS:
            return True
        return "response_format" in str(exc).lower()

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是工业模型更新策略器。只能依据用户提供的结构化指标做决策。"
            "必须只输出一个JSON对象，不要输出Markdown、代码围栏、解释文字或额外字段。"
            "JSON必须严格符合这个示例结构："
            '{"action":"retrain_with_replay","confidence":0.85,'
            '"reason":"持续混合漂移且性能下降"}。'
            "action只能是no_update或retrain_with_replay；"
            "confidence必须为0到1之间的数字；reason必须是1到120字的中文字符串。"
            "不得建议绕过预算、冷却期、候选评估或回滚保护。"
        )

    @staticmethod
    def _content_text(response: Any) -> str:
        """兼容不同端点的返回形态：字符串、分段列表、仅推理字段。"""
        try:
            message = response.choices[0].message
        except (AttributeError, IndexError, TypeError):
            return ""
        content = getattr(message, "content", None)
        if isinstance(content, list):
            parts = [
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            content = "".join(parts)
        if content:
            return str(content).strip()
        # 部分推理模型只填 reasoning_content，正文留空。
        reasoning = getattr(message, "reasoning_content", None)
        return str(reasoning).strip() if reasoning else ""

    @staticmethod
    def _decode_json(text: str) -> Any | None:
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                return None
        return None

    @staticmethod
    def _finish_reason(response: Any) -> str:
        try:
            return str(response.choices[0].finish_reason or "unknown")
        except (AttributeError, IndexError, TypeError):
            return "unknown"

    @staticmethod
    def _response_error(
        message: str,
        model: str,
        usage: LLMUsage,
        latency_seconds: float,
        finish_reason: str,
    ) -> LLMProviderError:
        diagnostic = (
            f"{message}（finish_reason={finish_reason}，total_tokens={usage.total_tokens}）"
        )
        return LLMProviderError(
            diagnostic,
            model=model,
            usage=usage,
            latency_seconds=latency_seconds,
        )

    @staticmethod
    def _usage_from_response(response: Any) -> LLMUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return LLMUsage()

        input_tokens = int(
            getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
        )
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        if total_tokens == 0:
            total_tokens = input_tokens + output_tokens
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
        )
