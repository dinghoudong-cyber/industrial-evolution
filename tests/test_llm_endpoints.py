from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from industrial_evolution.config import LLMConfig
from industrial_evolution.llm.factory import build_provider
from industrial_evolution.llm.openai_provider import LLMProviderError, OpenAICompatibleProvider
from industrial_evolution.llm.provider import LLMDecisionRequest


def local_config(**overrides) -> LLMConfig:
    base = {
        "enabled": True,
        "provider": "openai_compatible",
        "model": "LOCAL_MODEL_ID",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_env": "LOCAL_LLM_API_KEY",
        "api_key_required": False,
    }
    base.update(overrides)
    return LLMConfig(**base)


def request() -> LLMDecisionRequest:
    return LLMDecisionRequest(
        window_index=3,
        drift_kind="mixed",
        feature_drift_score=1.8,
        top_features=("sensor_00",),
        current_score=0.61,
        rolling_score=0.63,
        baseline_score=0.74,
        performance_drop=0.11,
        label_shift=0.14,
        remaining_updates=4,
        remaining_training_seconds=100.0,
        remaining_labels=3000,
        remaining_tokens=4000,
        requested_labels=140,
        rule_reason="持续漂移",
    )


class StubCompletions:
    def __init__(self, *, reject_json_mode: bool = False) -> None:
        self.reject_json_mode = reject_json_mode
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject_json_mode and "response_format" in kwargs:
            error = RuntimeError("response_format is not supported by this endpoint")
            error.status_code = 400
            raise error
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "action": "retrain_with_replay",
                                "confidence": 0.9,
                                "reason": "特征漂移伴随性能下降",
                            },
                            ensure_ascii=False,
                        )
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


class StubClient:
    def __init__(self, **kwargs) -> None:
        self.completions = StubCompletions(**kwargs)
        self.chat = SimpleNamespace(completions=self.completions)


def test_openai_compatible_endpoint_allows_loopback_http():
    config = local_config()

    assert config.base_url == "http://127.0.0.1:8000/v1"
    assert config.endpoint_label == "OpenAI兼容端点"
    assert isinstance(build_provider(config), OpenAICompatibleProvider)


def test_public_endpoint_must_use_https():
    with pytest.raises(ValueError, match="必须使用 https"):
        local_config(base_url="http://inference.example.com/v1")
    # https 公网端点允许，供云端评测环境使用
    assert local_config(base_url="https://inference.example.com/v1").base_url.startswith("https://")


def test_host_allowlist_is_enforced_when_configured():
    with pytest.raises(ValueError, match="白名单"):
        local_config(base_url="http://127.0.0.1:8000/v1", allowed_hosts=["gpu-node-1.internal"])
    assert local_config(
        base_url="http://gpu-node-1.internal:8000/v1",
        allowed_hosts=["gpu-node-1.internal"],
    )


def test_secret_env_name_must_look_like_a_key_or_token():
    with pytest.raises(ValueError, match="API_KEY 或 TOKEN"):
        local_config(api_key_env="OTHER_SECRET")
    with pytest.raises(ValueError, match="大写环境变量名"):
        local_config(api_key_env="local_llm_api_key")
    assert local_config(api_key_env="LOCAL_API_TOKEN").api_key_env == "LOCAL_API_TOKEN"


def test_missing_key_is_tolerated_when_endpoint_requires_no_auth(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(local_config())

    client = provider._client_or_build()

    assert getattr(client, "api_key", "") == "EMPTY"


def test_required_key_still_fails_fast(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(local_config(api_key_required=True))

    with pytest.raises(LLMProviderError, match="LOCAL_LLM_API_KEY 未设置"):
        provider.decide(request())


def test_provider_falls_back_when_endpoint_rejects_json_mode():
    client = StubClient(reject_json_mode=True)
    provider = OpenAICompatibleProvider(local_config(), client=client)

    response = provider.decide(request())

    assert response.action.value == "retrain_with_replay"
    assert provider.json_mode_disabled is True
    assert "response_format" in client.completions.calls[0]
    assert "response_format" not in client.completions.calls[1]


def test_provider_parses_json_wrapped_in_reasoning_text():
    class ChattyCompletions(StubCompletions):
        def create(self, **kwargs):
            response = super().create(**kwargs)
            text = response.choices[0].message.content
            response.choices[0].message.content = f"我先分析一下。{text} 以上。"
            return response

    client = StubClient()
    client.chat = SimpleNamespace(completions=ChattyCompletions())

    decision = OpenAICompatibleProvider(local_config(), client=client).decide(request())

    assert decision.confidence == 0.9


def test_factory_returns_local_openai_compatible_client():
    provider = build_provider(LLMConfig())

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.label == "OpenAI兼容端点"
