"""Lightweight token estimation.

A production gateway would count tokens with the model's real tokenizer
(``tiktoken`` for OpenAI, etc.). To stay dependency-free and offline, Conduit
ships a fast heuristic that approximates GPT-style BPE counts well enough for
cost estimation: roughly one token per ~4 characters, with a per-word floor so
short, punctuation-heavy text isn't undercounted.

The estimator is centralised here so swapping in a real tokenizer later is a
one-function change.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``."""
    if not text:
        return 0
    words = _WORD_RE.findall(text)
    # Blend a word count with a character-based estimate; BPE tends to land
    # between "one token per word" and "one token per 4 chars".
    char_estimate = len(text) / 4.0
    return max(len(words), round(char_estimate))


def count_message_tokens(prompt: str) -> int:
    """Prompt-side token count (adds a small per-request overhead, like the
    role/formatting tokens real chat models include)."""
    return count_tokens(prompt) + 3
