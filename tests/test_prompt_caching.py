from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from anthropic import APIStatusError
from pydantic import BaseModel

from mormi_api.llm import ClaudeGateway, ModelUnavailableError
from mormi_api.settings import Settings


class CacheRequest(BaseModel):
    child_utterance: str


class CacheResponse(BaseModel):
    category: str


class RecordingMessages:
    def __init__(self, *, reject_cache_once: bool = False) -> None:
        self.reject_cache_once = reject_cache_once
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.requests.append(kwargs)
        if self.reject_cache_once and len(self.requests) == 1:
            response = httpx.Response(
                400,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            )
            raise APIStatusError(
                "invalid cache control",
                response=response,
                body={"error": {"message": "cache_control is not supported"}},
            )
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(
                    type="text",
                    text=CacheResponse(category="ok").model_dump_json(),
                )
            ],
            usage=SimpleNamespace(
                input_tokens=40,
                cache_creation_input_tokens=1_200,
                cache_read_input_tokens=0,
                output_tokens=8,
            ),
        )


def gateway_with(messages: RecordingMessages, **settings: Any) -> ClaudeGateway:
    gateway = ClaudeGateway(
        Settings(
            _env_file=None,
            anthropic_api_key=None,
            **settings,
        )
    )
    gateway.client = SimpleNamespace(messages=messages)  # type: ignore[assignment]
    return gateway


async def request(
    gateway: ClaudeGateway,
    *,
    stage: str = "understanding_v2",
    child_utterance: str = "600원이야",
) -> CacheResponse:
    return await gateway._request_dialogue_v2_structured(
        stage=stage,
        model="claude-sonnet-4-6",
        system="stable system instructions",
        request=CacheRequest(child_utterance=child_utterance),
        response_model=CacheResponse,
        max_tokens=32,
        temperature=0,
    )


@pytest.mark.asyncio
async def test_disabled_prompt_cache_preserves_original_provider_payload() -> None:
    messages = RecordingMessages()
    gateway = gateway_with(messages, prompt_caching_enabled=False)

    assert await request(gateway) == CacheResponse(category="ok")

    sent = messages.requests[0]
    assert sent["system"] == "stable system instructions"
    assert json.loads(sent["messages"][0]["content"]) == {
        "child_utterance": "600원이야"
    }


@pytest.mark.asyncio
async def test_enabled_prompt_cache_marks_only_static_system_prefix() -> None:
    messages = RecordingMessages()
    gateway = gateway_with(
        messages,
        prompt_caching_enabled=True,
        prompt_cache_ttl="1h",
        prompt_cache_stages=frozenset({"understanding_v2"}),
    )

    await request(gateway, child_utterance="낸 돈에서 쿠키 값을 빼")

    sent = messages.requests[0]
    assert sent["system"] == [
        {
            "type": "text",
            "text": "stable system instructions",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    dynamic_message = sent["messages"][0]
    assert dynamic_message["role"] == "user"
    assert "cache_control" not in dynamic_message
    assert json.loads(dynamic_message["content"])["child_utterance"] == (
        "낸 돈에서 쿠키 값을 빼"
    )


@pytest.mark.asyncio
async def test_prompt_cache_stage_allowlist_leaves_other_roles_unchanged() -> None:
    messages = RecordingMessages()
    gateway = gateway_with(
        messages,
        prompt_caching_enabled=True,
        prompt_cache_stages=frozenset({"speaker_v2"}),
    )

    await request(gateway, stage="understanding_v2")

    assert messages.requests[0]["system"] == "stable system instructions"


@pytest.mark.asyncio
async def test_cache_metadata_rejection_retries_once_without_cache() -> None:
    messages = RecordingMessages(reject_cache_once=True)
    gateway = gateway_with(messages, prompt_caching_enabled=True)

    assert await request(gateway) == CacheResponse(category="ok")

    assert len(messages.requests) == 2
    assert isinstance(messages.requests[0]["system"], list)
    assert messages.requests[1]["system"] == "stable system instructions"


@pytest.mark.asyncio
async def test_non_cache_bad_request_does_not_retry() -> None:
    class RejectingMessages(RecordingMessages):
        async def create(self, **kwargs: Any) -> object:
            self.requests.append(kwargs)
            response = httpx.Response(
                400,
                request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            )
            raise APIStatusError(
                "invalid request",
                response=response,
                body={"error": {"message": "temperature is invalid"}},
            )

    messages = RejectingMessages()
    gateway = gateway_with(messages, prompt_caching_enabled=True)

    with pytest.raises(ModelUnavailableError):
        await request(gateway)

    assert len(messages.requests) == 1


@pytest.mark.asyncio
async def test_cache_usage_is_logged_without_prompt_content(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages = RecordingMessages()
    gateway = gateway_with(messages, prompt_caching_enabled=True)
    llm_logger = logging.getLogger("mormi_api.llm")
    monkeypatch.setattr(llm_logger, "disabled", False)
    monkeypatch.setattr(llm_logger, "propagate", True)

    with caplog.at_level(logging.INFO, logger="mormi_api.llm"):
        await request(gateway, child_utterance="로그에 남으면 안 되는 아이 발화")

    log_text = caplog.text
    assert "cache_status=write" in log_text
    assert "cache_write_tokens=1200" in log_text
    assert "cache_read_tokens=0" in log_text
    assert "로그에 남으면 안 되는 아이 발화" not in log_text


def test_prompt_cache_settings_accept_json_stage_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MORMI_PROMPT_CACHE_STAGES", '["understanding_v2", "speaker_v2"]')

    settings = Settings(_env_file=None)

    assert settings.prompt_cache_stages == frozenset({"understanding_v2", "speaker_v2"})
