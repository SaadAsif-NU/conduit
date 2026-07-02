"""FastAPI application exposing the gateway.

The chat endpoint is OpenAI-compatible, so pointing an OpenAI SDK at this server
Just Works. Domain errors are translated into OpenAI-style error envelopes, and
gateway metadata (provider, cost, cache status) rides back on ``X-Conduit-*``
headers.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..errors import ConduitError
from ..gateway import Gateway
from ..types import ChatRequest, ChatResponse
from .deps import GatewayDep, close_gateway
from .schemas import ErrorResponse, ModelCard, ModelList, error_response


def _client_key(authorization: str | None) -> str:
    """Derive the rate-limit identity from a Bearer token (or 'anonymous')."""
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or "anonymous"
    return "anonymous"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield
    # Close provider network clients / the ledger on shutdown.
    await close_gateway()


app = FastAPI(
    title="Conduit",
    version=__version__,
    summary="A self-hostable, OpenAI-compatible LLM gateway.",
    lifespan=lifespan,
)


@app.exception_handler(ConduitError)
async def _conduit_error_handler(request: Request, exc: ConduitError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(exc.message, type_=_error_type(exc), code=exc.status_code),
    )


def _error_type(exc: ConduitError) -> str:
    # Map internal exceptions to OpenAI-ish error type strings.
    return {
        404: "invalid_request_error",
        429: "rate_limit_error",
    }.get(exc.status_code, "upstream_error")


@app.get("/health", tags=["meta"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post(
    "/v1/chat/completions",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 502: {"model": ErrorResponse}},
    tags=["chat"],
)
async def chat_completions(
    body: ChatRequest,
    response: Response,
    gateway: GatewayDep,
    authorization: Annotated[str | None, Header()] = None,
) -> ChatResponse | Response:
    client_key = _client_key(authorization)
    if body.stream:
        return await _stream_chat(gateway, body, client_key)
    outcome = await gateway.complete(body, client_key=client_key)
    response.headers.update(outcome.headers())
    return outcome.response


async def _stream_chat(gateway: Gateway, body: ChatRequest, client_key: str) -> Response:
    """Serve an OpenAI-compatible SSE stream.

    The first chunk is pulled eagerly so that errors raised before any output
    (rate limits, unknown model, total provider failure) become proper HTTP
    error responses instead of a 200 stream that fails halfway.
    """
    agen = gateway.stream(body, client_key=client_key)
    try:
        first: str | None = await agen.__anext__()
    except StopAsyncIteration:
        first = None
    except ConduitError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc.message, _error_type(exc), exc.status_code),
        )

    stream_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    async def event_gen() -> AsyncIterator[str]:
        if first is not None:
            yield _sse_delta(stream_id, created, body.model, first)
        async for delta in agen:
            yield _sse_delta(stream_id, created, body.model, delta)
        yield _sse_stop(stream_id, created, body.model)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse_delta(stream_id: str, created: int, model: str, content: str) -> str:
    chunk = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _sse_stop(stream_id: str, created: int, model: str) -> str:
    chunk = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


@app.get("/v1/models", response_model=ModelList, tags=["chat"])
async def list_models(gateway: GatewayDep) -> ModelList:
    return ModelList(data=[ModelCard(id=model) for model in gateway.models])


@app.get("/usage", tags=["meta"])
async def usage(gateway: GatewayDep) -> dict[str, object]:
    return gateway.usage()
