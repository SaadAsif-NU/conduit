"""Build provider instances from configuration."""

from __future__ import annotations

from ..config import GatewayConfig, ProviderSpec
from .base import Provider
from .echo import EchoProvider
from .openai import OpenAIProvider


def build_provider(spec: ProviderSpec) -> Provider:
    if spec.type == "echo":
        return EchoProvider(name=spec.name, **spec.options)
    if spec.type == "openai":
        return OpenAIProvider(name=spec.name, **spec.options)
    raise ValueError(f"unknown provider type {spec.type!r}")


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    """Instantiate every configured provider, keyed by name."""
    return {spec.name: build_provider(spec) for spec in config.providers}
