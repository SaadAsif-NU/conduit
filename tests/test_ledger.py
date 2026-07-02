from __future__ import annotations

from conduit.ledger import LedgerEntry, UsageLedger
from conduit.types import ChatResponse, RequestOutcome, Usage


def _entry(model="gpt-4o", cost=0.01, cached=False, status="ok") -> LedgerEntry:
    return LedgerEntry(
        request_id="req-1",
        model=model,
        provider="openai",
        prompt_tokens=10,
        completion_tokens=5,
        cost_usd=cost,
        latency_ms=12.3,
        cached=cached,
        status=status,
    )


def test_record_and_count():
    ledger = UsageLedger()
    ledger.record(_entry())
    assert ledger.count() == 1


def test_summary_aggregates():
    ledger = UsageLedger()
    ledger.record(_entry(model="gpt-4o", cost=0.01))
    ledger.record(_entry(model="gpt-4o", cost=0.02))
    ledger.record(_entry(model="gpt-4o-mini", cost=0.001))
    summary = ledger.summary()
    assert summary["requests"] == 3
    assert round(summary["cost_usd"], 4) == 0.031
    models = {m["model"]: m for m in summary["by_model"]}
    assert models["gpt-4o"]["requests"] == 2
    assert round(models["gpt-4o"]["cost_usd"], 4) == 0.03


def test_errors_excluded_from_cost_summary():
    ledger = UsageLedger()
    ledger.record(_entry(cost=0.05, status="ok"))
    ledger.record(_entry(cost=0.0, status="error"))
    summary = ledger.summary()
    assert summary["requests"] == 1  # only successful requests counted


def test_cache_hits_counted():
    ledger = UsageLedger()
    ledger.record(_entry(cached=True))
    ledger.record(_entry(cached=False))
    assert ledger.summary()["cache_hits"] == 1


def test_record_outcome():
    ledger = UsageLedger()
    outcome = RequestOutcome(
        response=ChatResponse.single("gpt-4o", "hi", Usage.of(10, 3)),
        provider="openai",
        model="gpt-4o",
        cost_usd=0.02,
        latency_ms=8.0,
    )
    ledger.record_outcome(outcome)
    summary = ledger.summary()
    assert summary["requests"] == 1
    assert summary["completion_tokens"] == 3
