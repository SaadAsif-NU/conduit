"""Model pricing and cost computation.

Prices are expressed per 1K tokens, matching how providers publish them. The
table below is a reasonable default; deployments can override it via config. The
``echo`` provider is free, which keeps offline demos and tests at zero cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Usage


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per 1,000 tokens."""

    input_per_1k: float
    output_per_1k: float


# Indicative public prices (USD / 1K tokens). Not authoritative, override in
# config for real accounting.
DEFAULT_PRICING: dict[str, ModelPrice] = {
    "echo": ModelPrice(0.0, 0.0),
    "gpt-4o": ModelPrice(0.0025, 0.01),
    "gpt-4o-mini": ModelPrice(0.00015, 0.0006),
    "gpt-3.5-turbo": ModelPrice(0.0005, 0.0015),
}


def compute_cost(model: str, usage: Usage, pricing: dict[str, ModelPrice]) -> float:
    """Cost in USD for ``usage`` under ``model``'s price (0 if model unpriced)."""
    price = pricing.get(model)
    if price is None:
        return 0.0
    return (
        usage.prompt_tokens / 1000.0 * price.input_per_1k
        + usage.completion_tokens / 1000.0 * price.output_per_1k
    )
