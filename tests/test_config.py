from __future__ import annotations

from conduit.config import GatewayConfig, ProviderSpec


def test_default_is_echo_only():
    config = GatewayConfig.default()
    assert [p.name for p in config.providers] == ["echo"]
    assert config.routes() == {"echo": ["echo"]}


def test_from_env_without_key_is_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GatewayConfig.from_env()
    assert [p.type for p in config.providers] == ["echo"]


def test_from_env_adds_openai_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = GatewayConfig.from_env()
    names = [p.name for p in config.providers]
    assert "openai" in names and "echo" in names
    # openai models fall back to echo as a last resort by default
    assert config.routes()["gpt-4o"] == ["openai", "echo"]


def test_routes_chain_multiple_providers_in_order():
    config = GatewayConfig(
        providers=(
            ProviderSpec(name="a", type="echo", models=("shared",)),
            ProviderSpec(name="b", type="echo", models=("shared",)),
        ),
        fallback_to_echo=False,
    )
    assert config.routes()["shared"] == ["a", "b"]


def test_fallback_to_echo_can_be_disabled():
    config = GatewayConfig(
        providers=(
            ProviderSpec(name="primary", type="openai", models=("m",), options={}),
            ProviderSpec(name="echo", type="echo", models=("echo",)),
        ),
        fallback_to_echo=False,
    )
    assert config.routes()["m"] == ["primary"]
