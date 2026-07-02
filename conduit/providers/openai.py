"""OpenAI-compatible upstream adapter.

Works against the OpenAI API or any server that speaks the same
``/chat/completions`` shape (vLLM, Together, Groq, a local llama.cpp server, …).
HTTP and transport failures are mapped onto the typed error hierarchy so the
router can retry or fail over intelligently.
"""

from __future__ import annotations

import httpx

from ..errors import ProviderError, ProviderTimeout
from ..types import ChatRequest, ChatResponse
from .base import Provider

# Status codes worth retrying / failing over on (transient by nature).
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenAIProvider(Provider):
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        # An injected client lets tests supply a mock transport (no network).
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def complete(self, request: ChatRequest) -> ChatResponse:
        payload = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=self._headers
            )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(str(exc), provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc), provider=self.name, retryable=True) from exc

        if resp.status_code >= 400:
            raise ProviderError(
                f"upstream returned {resp.status_code}: {resp.text[:200]}",
                provider=self.name,
                retryable=resp.status_code in _RETRYABLE_STATUS,
            )

        # The response already matches our OpenAI-compatible schema.
        return ChatResponse.model_validate(resp.json())

    async def aclose(self) -> None:
        await self._client.aclose()
