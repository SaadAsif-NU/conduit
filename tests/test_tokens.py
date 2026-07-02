from __future__ import annotations

from conduit.tokens import count_message_tokens, count_tokens


def test_empty_is_zero():
    assert count_tokens("") == 0


def test_monotonic_in_length():
    short = count_tokens("hello")
    long = count_tokens("hello there, this is a considerably longer sentence")
    assert long > short


def test_message_tokens_add_overhead():
    assert count_message_tokens("hello") > count_tokens("hello")


def test_reasonable_magnitude():
    # ~9 words; a sane estimator should land in a believable range, not 1 or 1000.
    n = count_tokens("the quick brown fox jumps over the lazy dog")
    assert 5 <= n <= 20
