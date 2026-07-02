"""The top-level handle that wires everything together.

`Gateway` is what the HTTP server (and any embedding application) talks to. It
builds the providers, ledger and router from a :class:`~conduit.config.GatewayConfig`
and exposes a tiny surface: complete a request, list models, read usage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from .cache.base import NullCache, ResponseCache
from .cache.exact import ExactCache
from .cache.semantic import SemanticCache
from .config import GatewayConfig
from .ledger import UsageLedger
from .providers.registry import build_providers
from .ratelimit import build_rate_limiter
from .router import Router
from .types import ChatRequest, RequestOutcome


def _build_cache(config: GatewayConfig) -> ResponseCache:
    mode = config.cache.mode
    if mode == "exact":
        config.data_dir.mkdir(parents=True, exist_ok=True)
        return ExactCache(config.data_dir / "cache.db", ttl=config.cache.ttl)
    if mode == "semantic":
        return SemanticCache(threshold=config.cache.threshold)
    if mode == "none":
        return NullCache()
    raise ValueError(f"unknown cache mode {mode!r}; expected none/exact/semantic")


class Gateway:
    """Owns the configured providers, the ledger, and the router."""

    def __init__(
        self, config: GatewayConfig | None = None, *, ledger: UsageLedger | None = None
    ) -> None:
        self.config = config or GatewayConfig.from_env()
        if ledger is None:
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            ledger = UsageLedger(self.config.data_dir / "conduit.db")
        self.ledger = ledger
        self.providers = build_providers(self.config)
        self.router = Router(
            self.config,
            self.providers,
            self.ledger,
            rate_limiter=build_rate_limiter(self.config.rate_limit),
            cache=_build_cache(self.config),
        )

    async def complete(
        self, request: ChatRequest, *, client_key: str = "anonymous"
    ) -> RequestOutcome:
        return await self.router.complete(request, client_key=client_key)

    def stream(self, request: ChatRequest, *, client_key: str = "anonymous") -> AsyncIterator[str]:
        return self.router.stream(request, client_key=client_key)

    @property
    def models(self) -> list[str]:
        return self.router.models

    def usage(self) -> dict[str, object]:
        return self.ledger.summary()

    def recent(self, limit: int = 20) -> list[dict[str, object]]:
        return self.ledger.recent(limit)

    async def aclose(self) -> None:
        await self.router.aclose()
        self.ledger.close()
