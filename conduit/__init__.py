"""Conduit, a self-hostable LLM gateway.

Programmatic use mirrors the HTTP surface::

    import asyncio
    from conduit import Gateway, ChatRequest, Message

    gateway = Gateway()  # offline echo provider by default
    req = ChatRequest(model="echo", messages=[Message(role="user", content="hi")])
    outcome = asyncio.run(gateway.complete(req))
    print(outcome.response.text, outcome.cost_usd)
"""

from __future__ import annotations

from .config import GatewayConfig, ProviderSpec
from .cost import ModelPrice, compute_cost
from .errors import (
    AllProvidersFailedError,
    ConduitError,
    ProviderError,
    ProviderTimeout,
    RateLimitedError,
    UnknownModelError,
)
from .gateway import Gateway
from .ledger import LedgerEntry, UsageLedger
from .providers import EchoProvider, OpenAIProvider, Provider
from .router import Router
from .types import ChatRequest, ChatResponse, Choice, Message, RequestOutcome, Usage

__version__ = "0.1.0"

__all__ = [
    "AllProvidersFailedError",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "ConduitError",
    "EchoProvider",
    "Gateway",
    "GatewayConfig",
    "LedgerEntry",
    "Message",
    "ModelPrice",
    "OpenAIProvider",
    "Provider",
    "ProviderError",
    "ProviderSpec",
    "ProviderTimeout",
    "RateLimitedError",
    "RequestOutcome",
    "Router",
    "UnknownModelError",
    "Usage",
    "UsageLedger",
    "compute_cost",
    "__version__",
]
