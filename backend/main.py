"""Entry point for the voicebox backend.

Imports the configured FastAPI app and provides a ``python -m backend.main``
entry point for development.
"""

import argparse

from .startup import run_server


def __getattr__(name: str):
    """Keep ``uvicorn backend.main:app`` compatible without eager ML imports."""
    if name == "app":
        from .app import app  # lazy: heavy import

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: list[str] | None = None) -> int:
    """Validate configuration and ownership before loading the backend."""
    parser = argparse.ArgumentParser(description="voicebox backend server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (prefer loopback behind an HTTPS reverse proxy)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for database, profiles, and generated audio",
    )
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Require the requested address instead of reusing this data folder's backend on another local port",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return run_server(host=args.host, port=args.port, data_dir=args.data_dir, strict_port=args.strict_port)


if __name__ == "__main__":
    raise SystemExit(main())
