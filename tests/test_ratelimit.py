from __future__ import annotations

import pytest

from conduit.errors import RateLimitedError
from conduit.ratelimit import (
    NullRateLimiter,
    RateLimit,
    TokenBucketLimiter,
    build_rate_limiter,
)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_burst_then_block():
    clock = FakeClock()
    limiter = TokenBucketLimiter(RateLimit(requests_per_minute=60, burst=2), clock=clock)
    limiter.check("a")  # token 2 -> 1
    limiter.check("a")  # token 1 -> 0
    with pytest.raises(RateLimitedError):
        limiter.check("a")  # empty


def test_refills_over_time():
    clock = FakeClock()
    limiter = TokenBucketLimiter(RateLimit(requests_per_minute=60, burst=1), clock=clock)
    limiter.check("a")
    with pytest.raises(RateLimitedError):
        limiter.check("a")
    clock.advance(1.0)  # 60/min = 1/sec -> one token back
    limiter.check("a")  # ok again


def test_per_key_isolation():
    clock = FakeClock()
    limiter = TokenBucketLimiter(RateLimit(requests_per_minute=60, burst=1), clock=clock)
    limiter.check("a")
    limiter.check("b")  # b has its own bucket
    with pytest.raises(RateLimitedError):
        limiter.check("a")


def test_capacity_caps_refill():
    clock = FakeClock()
    limiter = TokenBucketLimiter(RateLimit(requests_per_minute=60, burst=2), clock=clock)
    clock.advance(100.0)  # would refill 100 tokens, but capacity is 2
    limiter.check("a")
    limiter.check("a")
    with pytest.raises(RateLimitedError):
        limiter.check("a")


def test_null_limiter_never_blocks():
    limiter = NullRateLimiter()
    for _ in range(1000):
        limiter.check("a")


def test_build_rate_limiter():
    assert isinstance(build_rate_limiter(None), NullRateLimiter)
    assert isinstance(build_rate_limiter(RateLimit(60, 5)), TokenBucketLimiter)


def test_invalid_rate_limit():
    with pytest.raises(ValueError):
        RateLimit(requests_per_minute=0, burst=1)
    with pytest.raises(ValueError):
        RateLimit(requests_per_minute=60, burst=0)
