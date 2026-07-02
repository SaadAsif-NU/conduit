"""Response cache interface and request fingerprinting.

A cache maps a chat request to a previously produced response. Two lookup
strategies are provided: exact (identical request) and semantic (a
near-duplicate prompt). Both share the same tiny interface, so the router treats
them uniformly and a disabled cache is just a no-op implementation.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod

from ..types import ChatRequest, ChatResponse


def request_fingerprint(request: ChatRequest) -> str:
    """Stable content hash of the parameters that determine the output.

    The ``user`` field is intentionally excluded so identical prompts from
    different callers share a cache entry.
    """
    payload = {
        "model": request.model,
        "messages": [(m.role, m.content) for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache(ABC):
    """Maps a request to a cached response, or ``None`` on a miss."""

    @abstractmethod
    def get(self, request: ChatRequest) -> ChatResponse | None: ...

    @abstractmethod
    def put(self, request: ChatRequest, response: ChatResponse) -> None: ...

    def close(self) -> None:
        return None


class NullCache(ResponseCache):
    """Caching disabled: every lookup misses."""

    def get(self, request: ChatRequest) -> ChatResponse | None:
        return None

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        return None
