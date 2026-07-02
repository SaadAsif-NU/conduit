"""OpenAI-compatible request/response models, plus internal metadata types.

The chat request and response deliberately mirror the OpenAI schema field-for-
field: that shape *is* Conduit's public contract, so any OpenAI SDK works
unchanged. Gateway-specific bookkeeping (which provider served it, what it cost,
whether it was cached) is kept out of the response body and carried separately in
:class:`RequestOutcome`, surfaced via ``X-Conduit-*`` headers and the ledger.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    """An OpenAI-compatible chat completion request."""

    model: str
    messages: list[Message] = Field(..., min_length=1)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    # Opaque per-caller identifier, used for per-key rate limiting / attribution.
    user: str | None = None

    def prompt_text(self) -> str:
        """Flatten the conversation for token counting / cache keying."""
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def of(cls, prompt: int, completion: int) -> Usage:
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    """An OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=_new_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage

    @classmethod
    def single(cls, model: str, content: str, usage: Usage) -> ChatResponse:
        """Convenience constructor for a one-choice assistant reply."""
        return cls(
            model=model,
            choices=[Choice(message=Message(role="assistant", content=content))],
            usage=usage,
        )

    @property
    def text(self) -> str:
        return self.choices[0].message.content if self.choices else ""


class RequestOutcome(BaseModel):
    """Gateway-side metadata about how a request was served (for the ledger)."""

    response: ChatResponse
    provider: str
    model: str
    cost_usd: float
    latency_ms: float
    cached: bool = False
    attempts: int = 1

    def headers(self) -> dict[str, str]:
        """The ``X-Conduit-*`` headers echoed to the client."""
        return {
            "X-Conduit-Provider": self.provider,
            "X-Conduit-Cost-USD": f"{self.cost_usd:.6f}",
            "X-Conduit-Cached": "true" if self.cached else "false",
            "X-Conduit-Attempts": str(self.attempts),
        }


# Arbitrary provider-configuration blob (kept loose on purpose).
ProviderConfig = dict[str, Any]
