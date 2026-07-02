"""A deterministic, offline provider.

EchoProvider needs no API key and no network: it replies with the last user
message (optionally transformed) and computes realistic token usage via the
estimator. That makes the entire gateway (routing, cost accounting, caching, and
the HTTP surface) exercisable in tests and demos at zero cost.

It can also be told to fail, which is how the router's retry/fallback behaviour
is tested deterministically.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..errors import ProviderError
from ..tokens import count_message_tokens, count_tokens
from ..types import ChatRequest, ChatResponse, Usage
from .base import Provider


class EchoProvider(Provider):
    """Echoes the conversation back as the assistant reply.

    Args:
        name: Provider id.
        prefix: Prepended to the echoed content.
        latency: Optional simulated latency in seconds.
        fail_times: Fail this many times (with a retryable error) before
            succeeding, used to test retries. ``-1`` means always fail.
    """

    def __init__(
        self,
        name: str = "echo",
        *,
        prefix: str = "",
        latency: float = 0.0,
        fail_times: int = 0,
    ) -> None:
        self.name = name
        self._prefix = prefix
        self._latency = latency
        self._remaining_failures = fail_times
        self.calls = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self._latency:
            await asyncio.sleep(self._latency)
        if self._remaining_failures != 0:
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
            raise ProviderError("simulated failure", provider=self.name, retryable=True)

        content = self._reply(request)
        usage = Usage.of(
            prompt=count_message_tokens(request.prompt_text()),
            completion=count_tokens(content),
        )
        return ChatResponse.single(model=request.model, content=content, usage=usage)

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        self.calls += 1
        if self._latency:
            await asyncio.sleep(self._latency)
        if self._remaining_failures != 0:
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
            raise ProviderError("simulated failure", provider=self.name, retryable=True)
        words = self._reply(request).split()
        for i, word in enumerate(words):
            yield word if i == 0 else f" {word}"

    def _reply(self, request: ChatRequest) -> str:
        last_user = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
        return f"{self._prefix}{last_user}"
