"""The router — Conduit's core.

Given a request, it resolves the model's provider chain and tries each provider
in turn. Within a provider it retries *retryable* failures with exponential
backoff; when a provider is exhausted it fails over to the next one. On success
it prices the usage and writes it to the ledger; if the whole chain fails it
records the failure and raises.

Rate limiting and caching are layered in front of this on Day 2 — the router
stays focused on "get a completion, reliably, and account for it".
"""

from __future__ import annotations

import asyncio
import time

from .config import GatewayConfig
from .cost import compute_cost
from .errors import AllProvidersFailedError, ConduitError, UnknownModelError
from .ledger import LedgerEntry, UsageLedger
from .providers.base import Provider
from .tokens import count_message_tokens
from .types import ChatRequest, RequestOutcome


class Router:
    """Routes chat requests to providers with retries and fallback."""

    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        ledger: UsageLedger,
        *,
        max_retries: int = 2,
        backoff_base: float = 0.05,
    ) -> None:
        self._config = config
        self._providers = providers
        self._routes = config.routes()
        self._pricing = config.pricing
        self._ledger = ledger
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    @property
    def models(self) -> list[str]:
        return sorted(self._routes)

    def _backoff(self, attempt: int) -> float:
        return self._backoff_base * (2**attempt)

    async def complete(self, request: ChatRequest) -> RequestOutcome:
        chain = self._routes.get(request.model)
        if not chain:
            raise UnknownModelError(f"no provider serves model {request.model!r}")

        started = time.perf_counter()
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

        # Whole chain exhausted — record the failure and surface it.
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

    async def aclose(self) -> None:
        for provider in self._providers.values():
            await provider.aclose()
