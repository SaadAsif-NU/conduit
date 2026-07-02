from __future__ import annotations

from conduit.cache import ExactCache, NullCache, SemanticCache, request_fingerprint
from conduit.types import ChatRequest, ChatResponse, Message, Usage


def _req(content: str, model: str = "echo") -> ChatRequest:
    return ChatRequest(model=model, messages=[Message(role="user", content=content)])


def _resp(text: str, model: str = "echo") -> ChatResponse:
    return ChatResponse.single(model, text, Usage.of(3, 2))


def test_fingerprint_is_stable_and_content_sensitive():
    assert request_fingerprint(_req("hello")) == request_fingerprint(_req("hello"))
    assert request_fingerprint(_req("hello")) != request_fingerprint(_req("world"))
    # user field must not affect the key
    a = ChatRequest(model="echo", messages=[Message(role="user", content="hi")], user="alice")
    b = ChatRequest(model="echo", messages=[Message(role="user", content="hi")], user="bob")
    assert request_fingerprint(a) == request_fingerprint(b)


def test_null_cache_always_misses():
    cache = NullCache()
    cache.put(_req("hi"), _resp("hi"))
    assert cache.get(_req("hi")) is None


def test_exact_cache_hit_and_miss():
    cache = ExactCache()
    assert cache.get(_req("hi")) is None
    cache.put(_req("hi"), _resp("cached answer"))
    hit = cache.get(_req("hi"))
    assert hit is not None and hit.text == "cached answer"
    assert cache.get(_req("different")) is None
    cache.close()


def test_exact_cache_respects_ttl():
    cache = ExactCache(ttl=60.0)
    cache.put(_req("hi"), _resp("answer"))
    # force the stored entry to look old, then it should be evicted on read
    cache._conn.execute("UPDATE response_cache SET created = 0")
    cache._conn.commit()
    assert cache.get(_req("hi")) is None
    cache.close()


def test_semantic_cache_hits_near_duplicate():
    cache = SemanticCache(threshold=0.6)
    cache.put(_req("machine learning is fun"), _resp("ml answer"))
    # shares 3 of 4 tokens -> above threshold
    hit = cache.get(_req("machine learning is great"))
    assert hit is not None and hit.text == "ml answer"


def test_semantic_cache_misses_unrelated():
    cache = SemanticCache(threshold=0.6)
    cache.put(_req("machine learning is fun"), _resp("ml answer"))
    assert cache.get(_req("cooking pasta tonight")) is None


def test_semantic_cache_is_per_model():
    cache = SemanticCache(threshold=0.6)
    cache.put(_req("hello world", model="gpt-4o"), _resp("a", model="gpt-4o"))
    assert cache.get(_req("hello world", model="echo")) is None
    assert cache.get(_req("hello world", model="gpt-4o")) is not None
