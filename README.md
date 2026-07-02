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
- 🚦 **Rate limiting** *(Day 2, see [Roadmap](#roadmap)).* Token-bucket limits per API key and model.
- 🧠 **Semantic caching** *(Day 2).* Serve cached responses for semantically equivalent prompts.
- 🌊 **Streaming** *(Day 2).* Server-sent-events streaming, OpenAI-compatible.
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
                   │  echo / openai /… │        │  ledger · cache  │
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

## Roadmap

Built in deliberate, reviewable increments:

- [x] OpenAI-compatible chat endpoint plus typed request/response models
- [x] Provider abstraction, offline `echo` provider, OpenAI-compatible adapter
- [x] Router with retries and provider fallback
- [x] Cost tracking, SQLite usage ledger, and `/usage`
- [ ] **Token-bucket rate limiting** per API key and model
- [ ] **Semantic and exact response caching**
- [ ] **SSE streaming** responses
- [ ] Observability: structured request logs, metrics, a usage summary

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
