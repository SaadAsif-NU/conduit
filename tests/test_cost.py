from __future__ import annotations

from conduit.cost import DEFAULT_PRICING, ModelPrice, compute_cost
from conduit.types import Usage


def test_priced_model():
    pricing = {"m": ModelPrice(input_per_1k=1.0, output_per_1k=2.0)}
    cost = compute_cost("m", Usage.of(prompt=1000, completion=2000), pricing)
    assert cost == 1.0 * 1 + 2.0 * 2  # 5.0


def test_unpriced_model_is_free():
    assert compute_cost("unknown", Usage.of(1000, 1000), {}) == 0.0


def test_echo_is_free():
    cost = compute_cost("echo", Usage.of(1000, 1000), DEFAULT_PRICING)
    assert cost == 0.0


def test_zero_usage():
    assert compute_cost("gpt-4o", Usage.of(0, 0), DEFAULT_PRICING) == 0.0
