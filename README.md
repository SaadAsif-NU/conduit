# Conduit

**A self-hostable LLM gateway. One OpenAI-compatible endpoint in front of every provider, with retries, fallback, rate limiting, semantic caching, and cost tracking. Pure Python, no external services.**

[![CI](https://github.com/SaadAsif-NU/conduit/actions/workflows/ci.yml/badge.svg)](https://github.com/SaadAsif-NU/conduit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20to%203.13-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Conduit sits between your application and the LLM providers you call. You point your existing OpenAI SDK at Conduit, and it handles the operational concerns that every team eventually rebuilds by hand: retrying transient failures, failing over to a backup provider, throttling noisy clients, caching repeated prompts, and tracking exactly what every request cost.

It runs anywhere Python runs. All state (the usage ledger, the cache) lives in a single embedded SQLite database. No Redis, no Postgres, no managed control plane.

> **Why this exists.** "I integrated the OpenAI API" is table stakes. The interesting engineering is everything *around* the call: the reliability, cost, and multi-tenancy concerns of running LLMs in production. Conduit implements that layer from scratch, so the mechanics are visible instead of hidden behind a SaaS dashboard.

---

## Features

- 🔌 **Drop-in OpenAI compatibility.** A standard `/v1/chat/completions` endpoint. Point any OpenAI SDK at it by changing the base URL.
- 🔀 **Routing, retries and fallback.** Map models to an ordered list of providers; transient errors retry with backoff, hard failures fail over to the next provider.
- 🧪 **Offline by default.** A deterministic `echo` provider means the gateway, examples, and the whole test suite run with no API keys and no network.
- 💸 **Cost and usage tracking.** Every request (tokens, cost, latency, provider, cache status) is written to an embedded SQLite ledger, queryable via `/usage`.
- 🚦 **Rate limiting.** Per-API-key token-bucket limits that allow bursts and refill steadily, returning a `429` when a client is over budget.
- 🧠 **Response caching.** An exact cache (byte-identical prompts, SQLite-backed) and a semantic cache (near-duplicate prompts) that serve repeated requests for free.
- 🌊 **Streaming.** Server-sent-events streaming, OpenAI-compatible, including replay of cached responses as a stream.
- 🔎 **Observability.** A request id on every response, one structured JSON log line per request (provider, cost, latency, cache status), and a `/usage` plus `/usage/recent` activity feed.
- ✅ **Tested and typed.** Async `pytest`, `mypy`-clean, `ruff`-clean, CI on Python 3.10 to 3.13.

## Architecture

```
                     ┌───────────────────────────────────────────────┐
   OpenAI SDK  ─────▶│  FastAPI  /v1/chat/completions                 │
   (any client)      └───────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────▼───────────────────────┐
                     │  Router                                        │
                     │  rate-limit ▶ cache ▶ provider(+retry/fallback)│
                     │           ▶ cost ▶ usage ledger                │
                     └───────┬───────────────────────────┬───────────┘
                             │                           │
                   ┌─────────▼─────────┐        ┌────────▼─────────┐
                   │  Providers        │        │  Storage (SQLite)│
                   │  echo / openai    │        │  ledger · cache  │
                   └───────────────────┘        └──────────────────┘
```

Each provider implements one small async interface, so adding a new upstream (Anthropic, a local model server, and so on) never touches the router, the server, or the accounting.

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Run the gateway (it boots with an offline `echo` provider, so no keys are needed):

```bash
conduit serve      # or: uvicorn conduit.server.app:app --reload --port 8080
```

Call it with plain HTTP:

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "hello"}]}'
```

Or, with the **official OpenAI SDK**, just change the base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-local")
resp = client.chat.completions.create(
    model="echo",
    messages=[{"role": "user", "content": "hello"}],
)
print(resp.choices[0].message.content)
```

Inspect what you've spent:

```bash
curl -s localhost:8080/usage | jq
```

## Configuring real providers

Providers are configured from the environment. Add an OpenAI-compatible upstream with a key, and route models to it with fallback:

```bash
export OPENAI_API_KEY=sk-...
```

Conduit then exposes those models on the same endpoint, with the echo provider available as an always-on local fallback for development.

## Rate limiting, caching, and streaming

Rate limiting and caching are configured from the environment:

```bash
# Token bucket: 60 requests/minute per API key, bursts of up to 20.
export CONDUIT_RATE_LIMIT_RPM=60
export CONDUIT_RATE_LIMIT_BURST=20

# Cache identical requests (also try "semantic" for near-duplicate prompts).
export CONDUIT_CACHE=exact
export CONDUIT_CACHE_TTL=3600
```

The rate-limit identity is the bearer token from the `Authorization` header, so
each API key gets its own bucket. Cache hits are served for free and marked with
`X-Conduit-Cached: true` and a `cache` provider in the usage ledger.

Stream a response with server-sent events by setting `stream: true`:

```bash
curl -N localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model": "echo", "messages": [{"role": "user", "content": "stream me"}], "stream": true}'
```

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"stream"}}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" me"}}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

## Observability

Every response carries an `X-Request-ID` (an inbound one is honoured, so a trace
id can flow across services), and every request emits one structured JSON log
line:

```json
{"level":"INFO","event":"http_request","request_id":"req_9f2c...","method":"POST","path":"/v1/chat/completions","status":200,"latency_ms":12.4,"provider":"echo","cost_usd":"0.000000","cached":"false"}
```

Read aggregate spend at `/usage`, or a live feed of the most recent requests at
`/usage/recent`:

```bash
curl -s localhost:8080/usage | jq
curl -s 'localhost:8080/usage/recent?limit=10' | jq
```

## Examples

- [`examples/quickstart.py`](examples/quickstart.py) tours the Python API offline: completion, fallback, caching, streaming, and usage.
- [`examples/http_client.py`](examples/http_client.py) calls a running gateway over HTTP, including streaming.

## Roadmap

Built in deliberate, reviewable increments:

- [x] OpenAI-compatible chat endpoint plus typed request/response models
- [x] Provider abstraction, offline `echo` provider, OpenAI-compatible adapter
- [x] Router with retries and provider fallback
- [x] Cost tracking, SQLite usage ledger, and `/usage`
- [x] **Token-bucket rate limiting** per API key
- [x] **Exact and semantic response caching**
- [x] **SSE streaming** responses
- [x] **Observability**: request ids, structured JSON logs, and a live usage feed

That completes the roadmap. See [`docs/architecture.md`](docs/architecture.md) for the full design write-up and [`examples/`](examples/) for runnable demos.

## Development

```bash
make dev        # install with dev deps
make test       # run the async test suite
make cov        # tests + coverage
make lint       # ruff
make typecheck  # mypy
make serve      # run the gateway locally
```

## License

[MIT](LICENSE) © Saad Asif
