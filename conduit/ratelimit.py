"""Token-bucket rate limiting.

Each client (identified by API key) gets a bucket that refills at a steady rate
up to a burst capacity. A request costs one token; if the bucket is empty the
request is rejected with a 429. This smooths bursty traffic while still allowing
short spikes, which is the standard shape for API rate limiting.

The check is a single synchronous, non-awaiting operation, so under asyncio it is
effectively atomic (no interleaving mid-update) and needs no lock.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from .errors import RateLimitedError


@dataclass(frozen=True, slots=True)
class RateLimit:
    """Rate-limit policy.

    Attributes:
        requests_per_minute: Steady-state refill rate.
        burst: Bucket capacity (the largest instantaneous spike allowed).
    """

    requests_per_minute: float
    burst: int

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if self.burst < 1:
            raise ValueError("burst must be >= 1")


class RateLimiter:
    """Interface: raise :class:`RateLimitedError` when a client is over budget."""

    def check(self, key: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class NullRateLimiter(RateLimiter):
    """A limiter that never limits (rate limiting disabled)."""

    def check(self, key: str) -> None:
        return None


class TokenBucketLimiter(RateLimiter):
    """Per-key token bucket."""

    def __init__(self, limit: RateLimit, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._refill_per_sec = limit.requests_per_minute / 60.0
        self._capacity = float(limit.burst)
        self._clock = clock
        # key -> (tokens available, timestamp of last update)
        self._buckets: dict[str, tuple[float, float]] = {}

    def _current_tokens(self, key: str, now: float) -> float:
        tokens, last = self._buckets.get(key, (self._capacity, now))
        return min(self._capacity, tokens + (now - last) * self._refill_per_sec)

    def check(self, key: str) -> None:
        now = self._clock()
        tokens = self._current_tokens(key, now)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            retry_after = (1.0 - tokens) / self._refill_per_sec
            raise RateLimitedError(f"rate limit exceeded for {key!r}; retry in {retry_after:.1f}s")
        self._buckets[key] = (tokens - 1.0, now)


def build_rate_limiter(limit: RateLimit | None) -> RateLimiter:
    """Return a token-bucket limiter, or a no-op limiter when ``limit`` is None."""
    return NullRateLimiter() if limit is None else TokenBucketLimiter(limit)
