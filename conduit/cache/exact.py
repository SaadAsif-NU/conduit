"""Exact-match response cache, backed by SQLite.

Keyed by a content fingerprint of the request, so a byte-identical prompt (same
model, messages, temperature, max_tokens) is served from cache. Durable and
process-restart safe, with an optional TTL.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from ..types import ChatRequest, ChatResponse
from .base import ResponseCache, request_fingerprint

_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_cache (
    key      TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created  REAL NOT NULL
);
"""


class ExactCache(ResponseCache):
    """SQLite-backed exact response cache.

    Args:
        path: SQLite file path, or ``":memory:"`` for an ephemeral cache.
        ttl: Optional entry lifetime in seconds. ``None`` means no expiry.
    """

    def __init__(self, path: str | Path = ":memory:", *, ttl: float | None = None) -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ttl = ttl
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def get(self, request: ChatRequest) -> ChatResponse | None:
        key = request_fingerprint(request)
        with self._lock:
            row = self._conn.execute(
                "SELECT response, created FROM response_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if self._ttl is not None and time.time() - row["created"] > self._ttl:
                self._conn.execute("DELETE FROM response_cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
        return ChatResponse.model_validate_json(row["response"])

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        key = request_fingerprint(request)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO response_cache (key, response, created) VALUES (?, ?, ?)",
                (key, response.model_dump_json(), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
