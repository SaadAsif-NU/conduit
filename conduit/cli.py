"""Command-line entrypoint."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="conduit", description="Conduit LLM gateway")
    parser.add_argument("--version", action="version", version=f"conduit {__version__}")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Run the gateway HTTP server")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("conduit.server.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
