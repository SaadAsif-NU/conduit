"""Dependency-injection wiring for the server.

A single process-wide :class:`~conduit.gateway.Gateway` is built lazily from the
environment and shared across requests. Tests override ``get_gateway`` to inject
an isolated, in-memory gateway.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from ..gateway import Gateway

_gateway: Gateway | None = None


def get_gateway() -> Gateway:
    global _gateway
    if _gateway is None:
        _gateway = Gateway()
    return _gateway


async def close_gateway() -> None:
    """Close the process gateway if one was created (called on shutdown).

    Deliberately does *not* lazily create a gateway just to close it, so tests
    that override the dependency never spin up the real (file-backed) one.
    """
    global _gateway
    if _gateway is not None:
        await _gateway.aclose()
        _gateway = None


GatewayDep = Annotated[Gateway, Depends(get_gateway)]
