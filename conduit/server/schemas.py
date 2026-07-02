"""HTTP-only response models.

The chat request/response reuse :mod:`conduit.types` directly (they already are
the OpenAI wire format). These extra models cover the endpoints that have no
OpenAI-request equivalent: the model list, errors, and the usage summary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "conduit"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorBody(BaseModel):
    message: str
    type: str
    code: int


class ErrorResponse(BaseModel):
    """OpenAI-style error envelope."""

    error: ErrorBody


def error_response(message: str, type_: str, code: int) -> dict[str, Any]:
    return {"error": {"message": message, "type": type_, "code": code}}
