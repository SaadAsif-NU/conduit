"""The router, Conduit's core.

Given a request, it enforces the rate limit, checks the response cache, and only
then resolves the model's provider chain and tries each provider in turn. Within
a provider it retries *retryable* failures with exponential backoff; when a
provider is exhausted it fails over to the next one. On success it prices the
usage, stores it in the cache, and writes it to the ledger; if the whole chain
fails it records the failure and raises.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from .cache.base import NullCache, ResponseCache
from .config import GatewayConfig
from .cost import compute_cost
from .errors import AllProvidersFailedError, ConduitError, UnknownModelError
from .ledger import LedgerEntry, UsageLedger
from .providers.base import Provider
from .ratelimit import NullRateLimiter, RateLimiter
from .tokens import count_message_tokens, count_tokens
from .types import ChatRequest, ChatResponse, RequestOutcome, Usage


def _chunk_words(text: str) -> AsyncIterator[str]:
    async def gen() -> AsyncIterator[str]:
        for i, word in enumerate(text.split()):
            yield word if i == 0 else f" {word}"

    return gen()


class Router:
    """Routes chat requests to providers with retries and fallback."""

    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        ledger: UsageLedger,
        *,
        rate_limiter: RateLimiter | None = None,
        cache: ResponseCache | None = None,
        max_retries: int = 2,
        backoff_base: float = 0.05,
    ) -> None:
        self._config = config
        self._providers = providers
        self._routes = config.routes()
        self._pricing = config.pricing
        self._ledger = ledger
        self._rate_limiter = rate_limiter or NullRateLimiter()
        self._cache = cache or NullCache()
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    @property
    def models(self) -> list[str]:
        return sorted(self._routes)

    def _backoff(self, attempt: int) -> float:
        return self._backoff_base * (2**attempt)

    async def complete(
        self, request: ChatRequest, *, client_key: str = "anonymous"
    ) -> RequestOutcome:
        # Order matters: reject over-budget clients before doing any work, then
        # try the cache, and only then pay for an upstream call.
        self._rate_limiter.check(client_key)

        chain = self._routes.get(request.model)
        if not chain:
            raise UnknownModelError(f"no provider serves model {request.model!r}")

        started = time.perf_counter()

        cached = self._cache.get(request)
        if cached is not None:
            outcome = RequestOutcome(
                response=cached,
                provider="cache",
                model=request.model,
                cost_usd=0.0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cached=True,
                attempts=0,
            )
            self._ledger.record_outcome(outcome)
            return outcome

        failures: list[ConduitError] = []
        attempts = 0

        for provider_name in chain:
            provider = self._providers[provider_name]
            for attempt in range(self._max_retries + 1):
                attempts += 1
                call_start = time.perf_counter()
                try:
                    response = await provider.complete(request)
                except ConduitError as exc:
                    failures.append(exc)
                    if exc.retryable and attempt < self._max_retries:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    break  # non-retryable, or retries exhausted → next provider
                latency_ms = (time.perf_counter() - call_start) * 1000.0
                cost = compute_cost(request.model, response.usage, self._pricing)
                self._cache.put(request, response)
                outcome = RequestOutcome(
                    response=response,
                    provider=provider_name,
                    model=request.model,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    cached=False,
                    attempts=attempts,
                )
                self._ledger.record_outcome(outcome)
                return outcome

        # Whole chain exhausted, record the failure and surface it.
        self._ledger.record(
            LedgerEntry(
                request_id="-",
                model=request.model,
                provider=chain[-1],
                prompt_tokens=count_message_tokens(request.prompt_text()),
                completion_tokens=0,
                cost_usd=0.0,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                cached=False,
                status="error",
            )
        )
        raise AllProvidersFailedError(request.model, failures)

    async def stream(
        self, request: ChatRequest, *, client_key: str = "anonymous"
    ) -> AsyncIterator[str]:
        """Stream a completion as text deltas.

        Applies the same rate-limit and cache checks as :meth:`complete` (a cache
        hit is replayed as a chunked stream), then streams from the first
        provider in the chain that produces output. Fallback happens only before
        the first chunk; once bytes are flowing we are committed to that
        provider. Usage is priced and logged once the stream ends.
        """
        self._rate_limiter.check(client_key)

        chain = self._routes.get(request.model)
        if not chain:
            raise UnknownModelError(f"no provider serves model {request.model!r}")

        started = time.perf_counter()

        cached = self._cache.get(request)
        if cached is not None:
            async for chunk in _chunk_words(cached.text):
                yield chunk
            self._ledger.record_outcome(
                RequestOutcome(
                    response=cached,
                    provider="cache",
                    model=request.model,
                    cost_usd=0.0,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    cached=True,
                    attempts=0,
                )
            )
            return

        failures: list[ConduitError] = []
        for provider_name in chain:
            gen = self._providers[provider_name].stream(request)
            try:
                first = await gen.__anext__()
            except StopAsyncIteration:
                first = None  # empty but successful stream
            except ConduitError as exc:
                failures.append(exc)
                continue

            parts: list[str] = []
            if first is not None:
                parts.append(first)
                yield first
            async for delta in gen:
                parts.append(delta)
                yield delta

            content = "".join(parts)
            usage = Usage.of(count_message_tokens(request.prompt_text()), count_tokens(content))
            response = ChatResponse.single(request.model, content, usage)
            self._cache.put(request, response)
            self._ledger.record_outcome(
                RequestOutcome(
                    response=response,
                    provider=provider_name,
                    model=request.model,
                    cost_usd=compute_cost(request.model, usage, self._pricing),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    cached=False,
                    attempts=1,
                )
            )
            return

        raise AllProvidersFailedError(request.model, failures)

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
        self._cache.close()
