# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY conduit ./conduit
RUN pip install --upgrade pip && pip install .

RUN useradd --create-home --uid 1000 conduit \
    && mkdir -p /data && chown -R conduit:conduit /data /app
USER conduit

ENV CONDUIT_DATA_DIR=/data
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "conduit.server.app:app", "--host", "0.0.0.0", "--port", "8080"]
