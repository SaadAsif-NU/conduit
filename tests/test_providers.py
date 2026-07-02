from __future__ import annotations

import httpx
import pytest
import respx

from conduit.errors import ProviderError, ProviderTimeout
from conduit.providers import EchoProvider, OpenAIProvider
from conduit.types import ChatRequest, Message


def _req(model: str = "echo", content: str = "hello there") -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content=content)])


async def test_echo_returns_last_user_message():
    provider = EchoProvider()
    resp = await provider.complete(_req(content="ping"))
    assert resp.text == "ping"
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.total_tokens == resp.usage.prompt_tokens + resp.usage.completion_tokens


async def test_echo_prefix():
    provider = EchoProvider(prefix="[bot] ")
    resp = await provider.complete(_req(content="hi"))
    assert resp.text == "[bot] hi"


async def test_echo_fail_times_then_succeed():
    provider = EchoProvider(fail_times=2)
    for _ in range(2):
        with pytest.raises(ProviderError):
            await provider.complete(_req())
    # third call succeeds
    assert (await provider.complete(_req())).text == "hello there"


_OPENAI_BODY = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi back"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


@respx.mock
async def test_openai_success_parses_response():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_OPENAI_BODY)
    )
    provider = OpenAIProvider("openai", api_key="sk-test")
    resp = await provider.complete(_req(model="gpt-4o"))
    assert resp.text == "hi back"
    assert resp.usage.total_tokens == 7
    await provider.aclose()


@respx.mock
async def test_openai_429_is_retryable_error():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(429, text="slow down")
    )
    provider = OpenAIProvider("openai", api_key="sk-test")
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(_req(model="gpt-4o"))
    assert excinfo.value.retryable is True
    await provider.aclose()


@respx.mock
async def test_openai_400_is_not_retryable():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(400, text="bad request")
    )
    provider = OpenAIProvider("openai", api_key="sk-test")
    with pytest.raises(ProviderError) as excinfo:
        await provider.complete(_req(model="gpt-4o"))
    assert excinfo.value.retryable is False
    await provider.aclose()


@respx.mock
async def test_openai_timeout_maps_to_provider_timeout():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    provider = OpenAIProvider("openai", api_key="sk-test")
    with pytest.raises(ProviderTimeout):
        await provider.complete(_req(model="gpt-4o"))
    await provider.aclose()
