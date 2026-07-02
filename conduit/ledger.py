"""SQLite usage & cost ledger.

Every completed request, success or failure, lands here: tokens, cost,
latency, which provider served it, and whether it was a cache hit. That single
append-only table is enough to answer "what did we spend, on what, and how fast
was it", which the ``/usage`` endpoint surfaces. Embedded SQLite keeps it
zero-dependency and durable.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .types import RequestOutcome

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id      TEXT NOT NULL,
    created         REAL NOT NULL,
    model           TEXT NOT NULL,
    provider        TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd        REAL NOT NULL,
    latency_ms      REAL NOT NULL,
    cached          INTEGER NOT NULL,
    status          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_created ON usage(created);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model);
"""


@dataclass(slots=True)
class LedgerEntry:
    request_id: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    cached: bool
    status: str = "ok"
    created: float | None = None


class UsageLedger:
    """Thread-safe append-only usage log."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def record(self, entry: LedgerEntry) -> None:
        created = entry.created if entry.created is not None else time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage (request_id, created, model, provider, prompt_tokens, "
                "completion_tokens, cost_usd, latency_ms, cached, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.request_id,
                    created,
                    entry.model,
                    entry.provider,
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    entry.cost_usd,
                    entry.latency_ms,
                    int(entry.cached),
                    entry.status,
                ),
            )
            self._conn.commit()

    def record_outcome(self, outcome: RequestOutcome) -> None:
        usage = outcome.response.usage
        self.record(
            LedgerEntry(
                request_id=outcome.response.id,
                model=outcome.model,
                provider=outcome.provider,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=outcome.cost_usd,
                latency_ms=outcome.latency_ms,
                cached=outcome.cached,
                status="ok",
            )
        )

    def summary(self) -> dict[str, object]:
        """Aggregate totals plus a per-model breakdown."""
        with self._lock:
            totals = self._conn.execute(
                "SELECT COUNT(*) AS requests, "
                "COALESCE(SUM(cost_usd), 0) AS cost_usd, "
                "COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens, "
                "COALESCE(SUM(completion_tokens), 0) AS completion_tokens, "
                "COALESCE(SUM(cached), 0) AS cache_hits "
                "FROM usage WHERE status = 'ok'"
            ).fetchone()
            by_model = self._conn.execute(
                "SELECT model, COUNT(*) AS requests, COALESCE(SUM(cost_usd), 0) AS cost_usd, "
                "COALESCE(SUM(prompt_tokens + completion_tokens), 0) AS tokens "
                "FROM usage WHERE status = 'ok' GROUP BY model ORDER BY cost_usd DESC"
            ).fetchall()
        return {
            "requests": totals["requests"],
            "cost_usd": round(totals["cost_usd"], 6),
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "cache_hits": totals["cache_hits"],
            "by_model": [
                {
                    "model": r["model"],
                    "requests": r["requests"],
                    "cost_usd": round(r["cost_usd"], 6),
                    "tokens": r["tokens"],
                }
                for r in by_model
            ],
        }

    def recent(self, limit: int = 20) -> list[dict[str, object]]:
        """The most recent requests (newest first) for a live activity view."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT request_id, created, model, provider, prompt_tokens, completion_tokens, "
                "cost_usd, latency_ms, cached, status FROM usage ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "request_id": r["request_id"],
                "created": r["created"],
                "model": r["model"],
                "provider": r["provider"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
                "cost_usd": round(r["cost_usd"], 6),
                "latency_ms": round(r["latency_ms"], 2),
                "cached": bool(r["cached"]),
                "status": r["status"],
            }
            for r in rows
        ]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) AS n FROM usage").fetchone()["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
