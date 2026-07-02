"""The top-level handle that wires everything together.

`Gateway` is what the HTTP server (and any embedding application) talks to. It
builds the providers, ledger and router from a :class:`~conduit.config.GatewayConfig`
and exposes a tiny surface: complete a request, list models, read usage.
"""

from __future__ import annotations

from .config import GatewayConfig
from .ledger import UsageLedger
from .providers.registry import build_providers
from .router import Router
from .types import ChatRequest, RequestOutcome


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
        self.router = Router(self.config, self.providers, self.ledger)

    async def complete(self, request: ChatRequest) -> RequestOutcome:
        return await self.router.complete(request)

    @property
    def models(self) -> list[str]:
        return self.router.models

    def usage(self) -> dict[str, object]:
        return self.ledger.summary()

    async def aclose(self) -> None:
        await self.router.aclose()
        self.ledger.close()
