from __future__ import annotations


def _chat(client, model="echo", content="hello", **extra):
    body = {"model": model, "messages": [{"role": "user", "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_completion_is_openai_shaped(client):
    resp = _chat(client, content="ping")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "ping"
    assert body["usage"]["total_tokens"] > 0


def test_chat_completion_sets_conduit_headers(client):
    resp = _chat(client)
    assert resp.headers["x-conduit-provider"] == "echo"
    assert resp.headers["x-conduit-cached"] == "false"
    assert "x-conduit-cost-usd" in resp.headers


def test_unknown_model_returns_404_openai_error(client):
    resp = _chat(client, model="gpt-9-turbo")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_streaming_rejected_for_now(client):
    resp = _chat(client, stream=True)
    assert resp.status_code == 400
    assert "streaming" in resp.json()["error"]["message"].lower()


def test_list_models(client):
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert {m["id"] for m in models["data"]} == {"echo", "gpt-4o"}


def test_usage_accumulates(client):
    _chat(client, model="gpt-4o", content="one")
    _chat(client, model="gpt-4o", content="two")
    usage = client.get("/usage").json()
    assert usage["requests"] == 2
    assert any(m["model"] == "gpt-4o" for m in usage["by_model"])


def test_invalid_request_body_is_422(client):
    # missing 'messages'
    resp = client.post("/v1/chat/completions", json={"model": "echo"})
    assert resp.status_code == 422
