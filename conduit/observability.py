"""Structured logging.

Emits one JSON object per log line so records are machine-parseable by any log
pipeline (Loki, CloudWatch, Datadog) without a parsing rule. Kept dependency
free: it is a thin ``logging.Formatter`` plus a couple of helpers. The HTTP
access log is produced by the request-id middleware in the server.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

LOGGER_NAME = "conduit"


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the ``conduit`` logger (idempotent)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = [handler]
    logger.setLevel(level.upper())
    logger.propagate = False


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, **fields: Any) -> None:
    """Emit a structured record: the message is ``event`` and ``fields`` are
    merged into the JSON output."""
    get_logger().info(event, extra={"fields": {"event": event, **fields}})
