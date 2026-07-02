from __future__ import annotations

import json
import logging

from conduit.observability import JsonFormatter, configure_logging, get_logger, log_event


def test_json_formatter_emits_valid_json_with_fields():
    record = logging.LogRecord("conduit", logging.INFO, "", 0, "hello", None, None)
    record.fields = {"event": "http_request", "status": 200}
    out = JsonFormatter().format(record)
    parsed = json.loads(out)
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["event"] == "http_request"
    assert parsed["status"] == 200


def test_configure_logging_is_idempotent():
    configure_logging("DEBUG")
    configure_logging("INFO")
    logger = get_logger()
    assert len(logger.handlers) == 1  # not duplicated
    assert logger.level == logging.INFO


def test_log_event_carries_structured_fields():
    configure_logging("INFO")
    logger = get_logger()
    captured: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger.addHandler(Capture())
    log_event("http_request", status=200, path="/v1/chat/completions")
    logger.handlers = [h for h in logger.handlers if not isinstance(h, Capture)]

    assert any(getattr(r, "fields", {}).get("path") == "/v1/chat/completions" for r in captured)
