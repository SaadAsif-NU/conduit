from __future__ import annotations

import pytest

from conduit.config import GatewayConfig, ProviderSpec
from conduit.errors import AllProvidersFailedError, ProviderError, UnknownModelError
from conduit.ledger import UsageLedger
from conduit.providers import EchoProvider, build_providers
from conduit.providers.base import Provider
from conduit.router import Router
from conduit.types import ChatRequest, ChatResponse, Message


def _req(model: str, content: str = "hi") -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content=content)])


class CountingFailer(Provider):
    """A provider that always fails, counting calls — for retry/fallback tests."""

    def __init__(self, name: str, *, retryable: bool) -> None:
        self.name = name
        self._retryable = retryable
        self.calls = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        raise ProviderError("boom", provider=self.name, retryable=self._retryable)


def _two_provider_config() -> GatewayConfig:
    # model "m" routes to "primary" then falls back to "echo".
    return GatewayConfig(
        providers=(
            ProviderSpec(name="primary", type="echo", models=("m",)),
            ProviderSpec(name="echo", type="echo", models=("echo",)),
        ),
        fallback_to_echo=True,
    )


async def test_routes_and_records():
    config = GatewayConfig.default()
    ledger = UsageLedger()
    router = Router(config, build_providers(config), ledger, backoff_base=0.0)
    outcome = await router.complete(_req("echo", "ping"))
    assert outcome.provider == "echo"
    assert outcome.response.text == "ping"
    assert ledger.count() == 1


async def test_unknown_model_raises():
    config = GatewayConfig.default()
    router = Router(config, build_providers(config), UsageLedger())
    with pytest.raises(UnknownModelError):
        await router.complete(_req("does-not-exist"))


async def test_retries_then_succeeds():
    config = GatewayConfig(
        providers=(ProviderSpec(name="p", type="echo", models=("m",), options={"fail_times": 2}),),
        fallback_to_echo=False,
    )
    ledger = UsageLedger()
    router = Router(config, build_providers(config), ledger, max_retries=2, backoff_base=0.0)
    outcome = await router.complete(_req("m"))
    assert outcome.attempts == 3  # two failures + one success
    assert outcome.provider == "p"


async def test_fails_over_to_next_provider():
    config = _two_provider_config()
    providers = {"primary": CountingFailer("primary", retryable=True), "echo": EchoProvider("echo")}
    ledger = UsageLedger()
    router = Router(config, providers, ledger, max_retries=2, backoff_base=0.0)
    outcome = await router.complete(_req("m"))
    assert providers["primary"].calls == 3  # retried max_retries+1 times
    assert outcome.provider == "echo"  # then failed over


async def test_non_retryable_error_is_not_retried():
    config = _two_provider_config()
    primary = CountingFailer("primary", retryable=False)
    providers = {"primary": primary, "echo": EchoProvider("echo")}
    router = Router(config, providers, UsageLedger(), max_retries=2, backoff_base=0.0)
    outcome = await router.complete(_req("m"))
    assert primary.calls == 1  # fatal error → straight to fallback
    assert outcome.provider == "echo"


async def test_all_providers_fail_raises_and_logs_error():
    config = GatewayConfig(
        providers=(ProviderSpec(name="p", type="echo", models=("m",), options={"fail_times": -1}),),
        fallback_to_echo=False,
    )
    ledger = UsageLedger()
    router = Router(config, build_providers(config), ledger, max_retries=1, backoff_base=0.0)
    with pytest.raises(AllProvidersFailedError):
        await router.complete(_req("m"))
    # the failure is logged, but excluded from the (successful) usage summary
    assert ledger.count() == 1
    assert ledger.summary()["requests"] == 0
