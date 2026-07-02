# Architecture

Conduit is a thin, reliable layer between your application and the LLM providers
it calls. This document explains the request lifecycle, the components, and the
decisions behind them.

## Request lifecycle

A single chat request flows through a fixed pipeline. The ordering is deliberate:
cheap rejections happen before expensive work, and nothing hits a paid upstream
until it has to.

```
  client
    │  POST /v1/chat/completions  (OpenAI-compatible)
    ▼
  request-id middleware ── assigns X-Request-ID, times the call, logs it
    │
    ▼
  route handler ── validates the body, derives the client key from the token
    │
    ▼
  Gateway ─▶ Router
                │  1. rate limit    (reject over-budget clients: 429)
                │  2. cache         (exact / semantic hit -> return, free)
                │  3. provider chain (primary, then fallbacks)
                │       └─ retry retryable failures with backoff
                │  4. cost          (price the usage)
                │  5. cache put + ledger record
                ▼
             ChatResponse  (+ X-Conduit-Provider / Cost / Cached headers)
```

Streaming follows the same pipeline; the first chunk is pulled eagerly so that a
pre-stream error (rate limit, unknown model, total failure) becomes a proper HTTP
status instead of a broken 200 stream. A cache hit is replayed as a chunked
stream.

## Components

| Component | Responsibility |
|---|---|
| `types` | OpenAI-compatible request/response models; the public contract. |
| `errors` | Typed hierarchy separating *retryable* failures from *fatal* ones, each with an HTTP status. |
| `providers` | One small async `Provider` interface. `EchoProvider` (offline), `OpenAIProvider` (any OpenAI-compatible upstream). |
| `ratelimit` | Per-key token-bucket limiter. |
| `cache` | `ExactCache` (SQLite, fingerprint key) and `SemanticCache` (prompt-similarity), behind one interface. |
| `cost` / `ledger` | Per-1K pricing, and an append-only SQLite usage log with summaries. |
| `router` | Orchestrates the pipeline above. |
| `gateway` | Assembles config into providers + limiter + cache + ledger + router. |
| `server` | FastAPI: the OpenAI-compatible endpoint, error translation, request-id middleware, `/usage`. |

The dependency direction is strict and one-way: the router depends on the
provider/cache/limiter *interfaces*, never on concrete implementations, so each
is swappable in isolation.

## Key design decisions

| Decision | Why |
|---|---|
| **OpenAI wire format is the contract.** | Any OpenAI SDK works by changing one base URL; no client rewrite. |
| **Retryable vs. fatal errors are typed.** | The router retries timeouts/429s/5xx and fails over, but never wastes attempts on a bad request or unknown model. |
| **Rate limit, then cache, then provider.** | Reject abusive clients before doing work; serve repeats for free before paying an upstream. |
| **Fingerprint excludes the caller.** | Identical prompts from different API keys share a cache entry. |
| **Fallback only before the first streamed chunk.** | Once bytes are flowing to the client we are committed; switching providers mid-stream would corrupt the response. |
| **Offline echo provider.** | The whole gateway, including cost accounting and caching, runs and is tested with no keys and no network. |
| **Everything embedded.** | The ledger and exact cache are SQLite files; there is no Redis or Postgres to operate. |

## Observability

Every request emits one structured (JSON) log line with a request id, method,
path, status, latency, and, for chat requests, the provider, cost, and cache
status pulled from the `X-Conduit-*` headers. An inbound `X-Request-ID` is
honoured so a trace id can flow across services. The `/usage` and
`/usage/recent` endpoints expose aggregate spend and a live activity feed from
the same ledger.

## Extension points

- **New provider** (Anthropic, Bedrock, a local server): implement the
  `Provider` interface and register a type in the provider registry. Routing,
  retries, accounting and streaming all work unchanged.
- **New cache** (Redis, a real embedding model for semantic lookup): implement
  the `ResponseCache` interface.
- **New limiter** (sliding window, distributed): implement the `RateLimiter`
  interface.
