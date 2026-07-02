"""Gateway configuration.

A :class:`GatewayConfig` describes the providers Conduit fronts, how models route
to them (with fallback order), and pricing. It can be built programmatically, or
resolved from the environment via :meth:`GatewayConfig.from_env` — which always
includes the offline ``echo`` provider and adds real ones when their keys are
present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .cost import DEFAULT_PRICING, ModelPrice
from .types import ProviderConfig


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One configured upstream.

    Attributes:
        name: Unique id used in routing and the ledger (e.g. ``"openai"``).
        type: Adapter type — ``"echo"`` or ``"openai"``.
        models: Model ids this provider serves.
        options: Adapter options (``api_key``, ``base_url``, ``timeout`` …).
    """

    name: str
    type: str
    models: tuple[str, ...]
    options: ProviderConfig = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    providers: tuple[ProviderSpec, ...]
    pricing: dict[str, ModelPrice] = field(default_factory=lambda: dict(DEFAULT_PRICING))
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CONDUIT_DATA_DIR", "./data"))
    )
    # If a real model's providers all fail, fall back to the echo provider so
    # development never hard-stops. Disable for strict production accounting.
    fallback_to_echo: bool = True

    def routes(self) -> dict[str, list[str]]:
        """Build ``model -> ordered provider names`` (the fallback chain).

        A model served by several providers chains them in config order; if
        ``fallback_to_echo`` is set and an echo provider exists, it is appended
        to every model as a last resort.
        """
        echo_names = [p.name for p in self.providers if p.type == "echo"]
        routes: dict[str, list[str]] = {}
        for provider in self.providers:
            for model in provider.models:
                routes.setdefault(model, []).append(provider.name)
        if self.fallback_to_echo and echo_names:
            for chain in routes.values():
                for echo in echo_names:
                    if echo not in chain:
                        chain.append(echo)
        return routes

    @classmethod
    def default(cls) -> GatewayConfig:
        """Offline config: a single echo provider serving ``echo``."""
        return cls(providers=(ProviderSpec(name="echo", type="echo", models=("echo",)),))

    @classmethod
    def from_env(cls) -> GatewayConfig:
        """Resolve providers from the environment.

        Always includes the ``echo`` provider; adds an OpenAI-compatible
        provider when ``OPENAI_API_KEY`` is set (honouring ``OPENAI_BASE_URL``).
        """
        providers: list[ProviderSpec] = [ProviderSpec(name="echo", type="echo", models=("echo",))]
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            providers.insert(
                0,
                ProviderSpec(
                    name="openai",
                    type="openai",
                    models=("gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"),
                    options={
                        "api_key": openai_key,
                        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    },
                ),
            )
        return cls(providers=tuple(providers))
