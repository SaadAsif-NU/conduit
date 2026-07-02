from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conduit.config import GatewayConfig, ProviderSpec
from conduit.gateway import Gateway
from conduit.ledger import UsageLedger
from conduit.server.app import app
from conduit.server.deps import get_gateway


@pytest.fixture
def client():
    """A TestClient backed by an isolated, in-memory, offline gateway.

    The config exposes the echo provider under both ``echo`` and a priced
    ``gpt-4o`` route (echo serves it via fallback), so cost accounting is
    exercised without any network.
    """
    config = GatewayConfig(
        providers=(ProviderSpec(name="echo", type="echo", models=("echo", "gpt-4o")),),
    )
    gateway = Gateway(config, ledger=UsageLedger(":memory:"))
    app.dependency_overrides[get_gateway] = lambda: gateway
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    gateway.ledger.close()
