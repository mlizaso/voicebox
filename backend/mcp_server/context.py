"""Per-request client identity for MCP calls.

MCP clients identify themselves via an ``X-Voicebox-Client-Id`` HTTP header
(direct-HTTP clients set it in their MCP config; the stdio shim forwards it
from the ``VOICEBOX_CLIENT_ID`` env var). Middleware copies the value into a
ContextVar so tool implementations can read it without plumbing the request
object through every service call.
"""

import asyncio
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from ..api_security import TRUSTED_LOCAL_REQUEST_SCOPE_KEY

logger = logging.getLogger(__name__)

# Strong refs to in-flight stamp tasks so asyncio.create_task results
# don't get garbage-collected mid-flight (cf. asyncio.create_task docs).
_pending_stamps: set[asyncio.Task] = set()

CLIENT_ID_HEADER = "X-Voicebox-Client-Id"

# Tool handlers read this to apply per-client voice bindings.
current_client_id: ContextVar[str | None] = ContextVar("current_client_id", default=None)

# Remote address of the in-flight request, retained for request diagnostics.
current_remote_addr: ContextVar[str | None] = ContextVar("current_remote_addr", default=None)

# Security-boundary locality of the in-flight request. Unlike the socket peer,
# this stays false for an authenticated remote caller behind a loopback proxy.
current_request_is_trusted_local: ContextVar[bool] = ContextVar("current_request_is_trusted_local", default=False)


def request_is_loopback() -> bool:
    """True only for a security-verified local peer and local Host authority.

    The outer API security middleware publishes this decision after validating
    both fields. Missing boundary context fails closed, which prevents a
    loopback reverse proxy from granting remote callers host-filesystem access.
    """
    return current_request_is_trusted_local.get()


# Endpoints that consume X-Voicebox-Client-Id for its MCP-semantic
# meaning (per-client profile resolution + per-client default_personality).
# These are the paths where a stamp into last_seen_at is accurate.
# Unrelated REST traffic that happens to set the header is intentionally
# ignored so the Settings UI's "last heard from" column only reflects
# calls that actually acted on the client's bindings.
#
# - /mcp — FastMCP tool calls (voicebox.speak, voicebox.transcribe, …)
#   and the /mcp/bindings admin surface. The admin surface is never
#   called with the header in practice (the frontend manages bindings
#   over plain REST), so the `startswith("/mcp")` match doesn't cause
#   false stamps.
# - /speak — REST mirror of voicebox.speak for non-MCP agents (shell
#   scripts, ACP, A2A). Uses the same per-client binding lookup, so its
#   callers belong in the last-seen list too.
_STAMPED_PATH_PREFIXES: tuple[str, ...] = ("/mcp", "/speak")


class ClientIdMiddleware(BaseHTTPMiddleware):
    """Copy X-Voicebox-Client-Id into a ContextVar and stamp last_seen_at
    for requests that act on the caller's MCP bindings."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        client_id = request.headers.get(CLIENT_ID_HEADER)
        remote_addr = request.client.host if request.client else None
        trusted_local = request.scope.get(TRUSTED_LOCAL_REQUEST_SCOPE_KEY) is True
        client_token = current_client_id.set(client_id)
        addr_token = current_remote_addr.set(remote_addr)
        local_token = current_request_is_trusted_local.set(trusted_local)
        try:
            response = await call_next(request)
        finally:
            current_client_id.reset(client_token)
            current_remote_addr.reset(addr_token)
            current_request_is_trusted_local.reset(local_token)

        if client_id and _is_stamped_path(request.url.path):
            _enqueue_stamp(client_id)
        return response


def _enqueue_stamp(client_id: str) -> None:
    """Fire-and-forget the SQLite write so it doesn't block the response.

    The stamp does sync SQLAlchemy I/O; running it inline on the event loop
    serialises every MCP request behind the SQLite write and starves SSE
    streams. ``asyncio.to_thread`` parks it on the default executor while
    the response goes back to the caller.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Middleware shouldn't run outside a loop, but if it ever does
        # (tests, weird wsgi shim), do the write inline rather than drop it.
        _stamp_last_seen(client_id)
        return
    from ..backends.mlx_tts_lifecycle import run_blocking_operation_cancellation_safe
    from ..services.task_queue import create_background_task

    operation = run_blocking_operation_cancellation_safe(
        _stamp_last_seen,
        client_id,
    )
    try:
        task = create_background_task(operation)
    except RuntimeError:
        # Shutdown has stopped accepting work. The response is already
        # complete and a last-seen stamp is advisory, so dropping this write
        # is safer than allowing it to outlive the data-root lock.
        return
    _pending_stamps.add(task)
    task.add_done_callback(_pending_stamps.discard)


def _is_stamped_path(path: str) -> bool:
    # Require a path boundary so a future ``/speakers`` or ``/mcpfoo``
    # route doesn't silently inherit the stamp from ``/speak`` / ``/mcp``.
    return any(path == p or path.startswith(p + "/") for p in _STAMPED_PATH_PREFIXES)


def _stamp_last_seen(client_id: str) -> None:
    """Update or create the MCPClientBinding row for this client_id."""
    try:
        from ..database import get_db
        from ..database.models import MCPClientBinding
    except Exception:
        return
    try:
        db = next(get_db())
    except Exception:
        return
    try:
        row = db.query(MCPClientBinding).filter(MCPClientBinding.client_id == client_id).first()
        if row is None:
            row = MCPClientBinding(client_id=client_id)
            db.add(row)
        row.last_seen_at = datetime.now(UTC)
        db.commit()
    except Exception:
        logger.debug("Could not stamp last_seen_at for %s", client_id, exc_info=True)
        db.rollback()
    finally:
        db.close()
