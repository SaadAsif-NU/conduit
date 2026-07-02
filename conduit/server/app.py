"""FastAPI application exposing the gateway.

The chat endpoint is OpenAI-compatible, so pointing an OpenAI SDK at this server
Just Works. Domain errors are translated into OpenAI-style error envelopes, and
gateway metadata (provider, cost, cache status) rides back on ``X-Conduit-*``
headers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .. import __version__
from ..errors import ConduitError
from ..types import ChatRequest, ChatResponse
from .deps import GatewayDep, close_gateway
from .schemas import ErrorResponse, ModelCard, ModelList, error_response


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
    body: ChatRequest, response: Response, gateway: GatewayDep
) -> ChatResponse | JSONResponse:
    if body.stream:
        # SSE streaming lands in a later version; be explicit rather than silently
        # returning a non-streamed body.
        return JSONResponse(
            status_code=400,
            content=error_response(
                "streaming is not supported yet; set stream=false",
                "invalid_request_error",
                400,
            ),
        )
    outcome = await gateway.complete(body)
    response.headers.update(outcome.headers())
    return outcome.response


@app.get("/v1/models", response_model=ModelList, tags=["chat"])
async def list_models(gateway: GatewayDep) -> ModelList:
    return ModelList(data=[ModelCard(id=model) for model in gateway.models])


@app.get("/usage", tags=["meta"])
async def usage(gateway: GatewayDep) -> dict[str, object]:
    return gateway.usage()
