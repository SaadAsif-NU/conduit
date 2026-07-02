"""A tour of Conduit's Python API, fully offline.

Demonstrates the five things the gateway does: complete a request, fail over to a
backup provider, serve a repeat from cache, stream, and report usage. It uses the
echo provider, so there is nothing to install and no key required.

Run with:  python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from conduit import ChatRequest, Gateway, GatewayConfig, Message, ProviderSpec, UsageLedger
from conduit.config import CacheConfig


def _req(model: str, content: str) -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content=content)])


async def main() -> None:
    # 'demo' routes to a provider that always fails, then falls back to echo.
    config = GatewayConfig(
        providers=(
            ProviderSpec(name="flaky", type="echo", models=("demo",), options={"fail_times": -1}),
            ProviderSpec(name="echo", type="echo", models=("echo",)),
        ),
        cache=CacheConfig(mode="semantic"),  # in-memory; no files written
        fallback_to_echo=True,
    )
    gateway = Gateway(config, ledger=UsageLedger(":memory:"))

    # 1. A basic completion, priced into the ledger.
    out = await gateway.complete(_req("echo", "Hello, gateway!"))
    print(f"1. basic     -> {out.response.text!r}  provider={out.provider}  ${out.cost_usd:.4f}")

    # 2. Fallback: the primary provider fails, so echo answers instead.
    out = await gateway.complete(_req("demo", "Who answers me?"))
    print(f"2. fallback  -> provider={out.provider}  attempts={out.attempts}")

    # 3. Caching: an identical request is served for free from cache.
    await gateway.complete(_req("echo", "cache this please"))
    out = await gateway.complete(_req("echo", "cache this please"))
    print(f"3. cache     -> cached={out.cached}  provider={out.provider}")

    # 4. Streaming: deltas arrive one chunk at a time.
    print("4. stream    -> ", end="")
    async for delta in gateway.stream(_req("echo", "streamed word by word")):
        print(delta, end="", flush=True)
    print()

    # 5. Usage: aggregate spend and per-model breakdown.
    usage = gateway.usage()
    print(f"5. usage     -> requests={usage['requests']}  cache_hits={usage['cache_hits']}")

    await gateway.aclose()


if __name__ == "__main__":
    asyncio.run(main())
