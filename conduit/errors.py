"""Typed error hierarchy.

The router distinguishes *retryable* failures (timeouts, 429s, 5xx, worth
trying again or failing over) from *fatal* ones (a malformed request, an unknown
model, retrying is pointless). Every error also carries an HTTP ``status_code``
so the server can translate it to an OpenAI-style error response directly.
"""

from __future__ import annotations


class ConduitError(Exception):
    """Base class for all gateway errors."""

    status_code: int = 500
    #: Whether the router should retry / fall over on this error.
    retryable: bool = False

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownModelError(ConduitError):
    """The requested model is not routed to any provider."""

    status_code = 404
    retryable = False


class ProviderError(ConduitError):
    """An upstream provider failed. Retryable unless told otherwise."""

    status_code = 502
    retryable = True

    def __init__(self, message: str, *, provider: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ProviderTimeout(ProviderError):
    """An upstream provider timed out."""

    status_code = 504

    def __init__(self, message: str, *, provider: str) -> None:
        super().__init__(message, provider=provider, retryable=True)


class RateLimitedError(ConduitError):
    """A local (Conduit-enforced) rate limit was exceeded."""

    status_code = 429
    retryable = False


class AllProvidersFailedError(ConduitError):
    """Every provider in the fallback chain failed."""

    status_code = 502
    retryable = False

    def __init__(self, model: str, failures: list[ConduitError]) -> None:
        detail = "; ".join(f"{type(e).__name__}: {e}" for e in failures) or "no providers"
        super().__init__(f"all providers failed for model {model!r}: {detail}")
        self.failures = failures
