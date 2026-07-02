"""Call a running Conduit server over HTTP, including streaming.

Start the gateway first, in another terminal:

    conduit serve            # listens on http://localhost:8080

then run:  python examples/http_client.py

Conduit speaks the OpenAI wire format, so the official OpenAI SDK works too, just
point its base_url at http://localhost:8080/v1. This example uses httpx (already
a dependency) so it needs nothing extra.
"""

from __future__ import annotations

import asyncio
import json

import httpx

BASE_URL = "http://localhost:8080"
HEADERS = {"Authorization": "Bearer demo-key", "content-type": "application/json"}


async def non_streaming(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/v1/chat/completions",
        headers=HEADERS,
        json={"model": "echo", "messages": [{"role": "user", "content": "hello over http"}]},
    )
    resp.raise_for_status()
    body = resp.json()
    print("reply    :", body["choices"][0]["message"]["content"])
    print("provider :", resp.headers.get("x-conduit-provider"))
    print("cost usd :", resp.headers.get("x-conduit-cost-usd"))
    print("req id   :", resp.headers.get("x-request-id"))


async def streaming(client: httpx.AsyncClient) -> None:
    print("stream   : ", end="")
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "echo",
            "messages": [{"role": "user", "content": "stream this response"}],
            "stream": True,
        },
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0]["delta"].get("content")
            if delta:
                print(delta, end="", flush=True)
    print()


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10.0) as client:
        await non_streaming(client)
        await streaming(client)
        usage = (await client.get("/usage")).json()
        print("usage    :", usage)


if __name__ == "__main__":
    asyncio.run(main())
