"""Semantic response cache.

Exact caching only fires on byte-identical prompts. A semantic cache also serves
prompts that *mean* the same thing ("What's the capital of France?" vs "what is
france's capital"). It embeds each prompt and, on lookup, returns the cached
response whose prompt is within a cosine-similarity threshold.

To stay dependency-free it uses signed feature hashing into a sparse, unit-length
vector and computes cosine similarity directly over the sparse representation.
This is lexical rather than deeply semantic (it keys on shared words, not learned
meaning), but it demonstrates the mechanism end to end and swaps trivially for a
real embedding model. The store is in memory and scoped per model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict, deque

from ..types import ChatRequest, ChatResponse
from .base import ResponseCache

_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

# A sparse, unit-length embedding: bucket -> weight.
SparseVector = dict[int, float]


class SemanticCache(ResponseCache):
    """In-memory cache keyed by prompt similarity.

    Args:
        threshold: Minimum cosine similarity (in ``[0, 1]``) for a hit.
        dim: Number of hash buckets.
        max_entries_per_model: Bound on stored prompts per model (FIFO eviction).
    """

    def __init__(
        self, *, threshold: float = 0.92, dim: int = 1024, max_entries_per_model: int = 512
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self._threshold = threshold
        self._dim = dim
        self._max = max_entries_per_model
        self._store: dict[str, deque[tuple[SparseVector, ChatResponse]]] = defaultdict(
            lambda: deque(maxlen=self._max)
        )

    def _embed(self, text: str) -> SparseVector:
        buckets: SparseVector = defaultdict(float)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            sign = 1.0 if (value >> 63) & 1 else -1.0
            buckets[value % self._dim] += sign
        norm = math.sqrt(sum(w * w for w in buckets.values()))
        if norm == 0.0:
            return {}
        return {k: w / norm for k, w in buckets.items()}

    @staticmethod
    def _cosine(a: SparseVector, b: SparseVector) -> float:
        # Both vectors are unit-length, so the dot product is the cosine.
        if len(a) > len(b):
            a, b = b, a
        return sum(weight * b.get(bucket, 0.0) for bucket, weight in a.items())

    def get(self, request: ChatRequest) -> ChatResponse | None:
        query = self._embed(request.prompt_text())
        if not query:
            return None
        best_sim = 0.0
        best_response: ChatResponse | None = None
        for vector, response in self._store.get(request.model, ()):
            sim = self._cosine(query, vector)
            if sim > best_sim:
                best_sim, best_response = sim, response
        return best_response if best_sim >= self._threshold else None

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        vector = self._embed(request.prompt_text())
        if vector:
            self._store[request.model].append((vector, response))
