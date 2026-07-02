from __future__ import annotations

import pytest

from conduit import ChatRequest, Gateway, GatewayConfig, Message, ProviderSpec, UsageLedger
from conduit.config import CacheConfig


def _config(tmp_path, **kw) -> GatewayConfig:
    return GatewayConfig(
        providers=(ProviderSpec(name="echo", type="echo", models=("echo",)),),
        data_dir=tmp_path,
        **kw,
    )


def _req() -> ChatRequest:
    return ChatRequest(model="echo", messages=[Message(role="user", content="hi there")])


async def test_gateway_exact_cache_roundtrip(tmp_path):
    gw = Gateway(_config(tmp_path, cache=CacheConfig(mode="exact")), ledger=UsageLedger(":memory:"))
    first = await gw.complete(_req())
    second = await gw.complete(_req())
    assert first.cached is False
    assert second.cached is True
    assert (tmp_path / "cache.db").exists()  # exact cache persisted its dir
    await gw.aclose()


async def test_gateway_recent_delegates_to_ledger(tmp_path):
    gw = Gateway(_config(tmp_path), ledger=UsageLedger(":memory:"))
    await gw.complete(_req())
    recent = gw.recent(limit=5)
    assert len(recent) == 1 and recent[0]["provider"] == "echo"
    await gw.aclose()


def test_unknown_cache_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="cache mode"):
        Gateway(_config(tmp_path, cache=CacheConfig(mode="bogus")), ledger=UsageLedger(":memory:"))
