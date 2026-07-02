"""The provider abstraction.

A provider is anything that can turn a :class:`~conduit.types.ChatRequest` into a
:class:`~conduit.types.ChatResponse`. The router depends only on this interface,
so adding an upstream (Anthropic, a local server, a mock) never touches routing,
accounting, or the HTTP layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..types import ChatRequest, ChatResponse


class Provider(ABC):
    """An upstream LLM backend."""

    #: Unique provider id, used in routing and the ledger.
    name: str

    @abstractmethod
    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Produce a completion, or raise a :class:`~conduit.errors.ProviderError`.

        Implementations must translate transport/HTTP failures into the typed
        error hierarchy so the router can decide whether to retry or fail over.
        """

    async def stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Yield the completion as a sequence of text deltas.

        The default implementation calls :meth:`complete` and yields the whole
        text as a single chunk, so every provider streams even if its backend
        does not. Providers with a real streaming API override this.
        """
        response = await self.complete(request)
        yield response.text

    async def aclose(self) -> None:
        """Release any resources (network clients). Default: no-op."""
        return None
