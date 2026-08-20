"""Hard request-body limits enforced before framework body parsing.

Starlette parses multipart bodies before a FastAPI endpoint runs.  Endpoint
read loops are still useful as a second line of defence, but cannot stop an
oversized file part from being spooled by the multipart parser first.  This
ASGI middleware bounds both declared and streaming request bodies at the
receive boundary.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .utils.disk_reservations import DiskSpaceReservation, DiskSpaceReservationError, reserve_disk_space

MIB = 1024 * 1024

# Multipart framing, filenames, and the small text fields accepted alongside a
# file need some room beyond each endpoint's file budget.  The file itself is
# still checked against its exact limit by the endpoint after parsing.
MULTIPART_OVERHEAD_BYTES = MIB

DEFAULT_REQUEST_BODY_MAX_BYTES = 2 * MIB
MCP_TRANSCRIBE_MAX_FILE_BYTES = 200 * MIB
MCP_TRANSCRIBE_MAX_BASE64_CHARS = 4 * ((MCP_TRANSCRIBE_MAX_FILE_BYTES + 2) // 3)
MCP_REQUEST_BODY_MAX_BYTES = MCP_TRANSCRIBE_MAX_BASE64_CHARS + MULTIPART_OVERHEAD_BYTES
LARGE_REQUEST_MAX_CONCURRENCY = 2
LARGE_MCP_REQUEST_MAX_CONCURRENCY = 1
LARGE_REQUEST_ADMISSION_THRESHOLD_BYTES = 5 * MIB
TEMP_STORAGE_RESERVE_BYTES = 1024 * MIB
ARCHIVE_IMPORT_MAX_FILE_BYTES = 516 * MIB

_EXACT_PATH_LIMITS: dict[tuple[str, str], int] = {
    ("POST", "/captures"): 100 * MIB + MULTIPART_OVERHEAD_BYTES,
    ("POST", "/generate/import"): 200 * MIB + MULTIPART_OVERHEAD_BYTES,
    ("POST", "/history/import"): ARCHIVE_IMPORT_MAX_FILE_BYTES + MULTIPART_OVERHEAD_BYTES,
    ("POST", "/profiles/import"): ARCHIVE_IMPORT_MAX_FILE_BYTES + MULTIPART_OVERHEAD_BYTES,
    ("POST", "/transcribe"): 100 * MIB + MULTIPART_OVERHEAD_BYTES,
}


def _normalized_path(scope: Scope) -> str:
    path = str(scope.get("path", ""))
    return path[:-1] if path != "/" and path.endswith("/") else path


def _is_large_multipart_route(scope: Scope) -> bool:
    method = str(scope.get("method", "GET")).upper()
    path = _normalized_path(scope)
    if (method, path) in _EXACT_PATH_LIMITS:
        return True
    if method != "POST" or not path.startswith("/profiles/"):
        return False
    tail = path.removeprefix("/profiles/")
    return tail.count("/") == 1 and tail.endswith(("/samples", "/avatar"))


def _is_mcp_route(scope: Scope) -> bool:
    return str(scope.get("method", "GET")).upper() == "POST" and _normalized_path(scope) == "/mcp"


class _RequestBodyTooLargeError(Exception):
    """Private control-flow exception raised only by the wrapped receive."""


def request_body_limit(scope: Scope) -> int:
    """Return the tightest safe wire-body limit for an HTTP request."""
    method = str(scope.get("method", "GET")).upper()
    normalized_path = _normalized_path(scope)

    exact = _EXACT_PATH_LIMITS.get((method, normalized_path))
    if exact is not None:
        return exact

    if method == "POST" and normalized_path.startswith("/profiles/"):
        tail = normalized_path.removeprefix("/profiles/")
        if tail.endswith("/samples") and tail.count("/") == 1:
            return 50 * MIB + MULTIPART_OVERHEAD_BYTES
        if tail.endswith("/avatar") and tail.count("/") == 1:
            return 5 * MIB + MULTIPART_OVERHEAD_BYTES

    # MCP JSON can legitimately contain a bounded 200 MiB audio file encoded
    # as base64.  Both /mcp and /mcp/ reach the mounted Streamable HTTP app.
    if method == "POST" and normalized_path == "/mcp":
        return MCP_REQUEST_BODY_MAX_BYTES

    return DEFAULT_REQUEST_BODY_MAX_BYTES


def _header_values(scope: Scope, name: bytes) -> list[bytes]:
    return [value for key, value in scope.get("headers", ()) if key.lower() == name]


def _declared_content_length(scope: Scope) -> int | None:
    values = _header_values(scope, b"content-length")
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("multiple Content-Length headers")
    try:
        raw = values[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid Content-Length header") from exc
    # Header values larger than an unsigned 64-bit decimal are invalid for any
    # supported request and must not reach Python's bounded decimal parser.
    if not raw or len(raw) > 20 or not raw.isdecimal():
        raise ValueError("invalid Content-Length header")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("invalid Content-Length header") from exc
    if value < 0:
        raise ValueError("invalid Content-Length header")
    return value


class RequestBodyLimitMiddleware:
    """Reject oversized HTTP bodies before Starlette can parse or spool them."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        limit_resolver: Callable[[Scope], int] = request_body_limit,
        exact_path_limits: Mapping[tuple[str, str], int] | None = None,
        default_max_bytes: int | None = None,
        max_concurrent_large_requests: int = LARGE_REQUEST_MAX_CONCURRENCY,
        max_concurrent_large_mcp_requests: int = LARGE_MCP_REQUEST_MAX_CONCURRENCY,
        admission_threshold_bytes: int = LARGE_REQUEST_ADMISSION_THRESHOLD_BYTES,
        temp_storage_reserve_bytes: int = TEMP_STORAGE_RESERVE_BYTES,
    ) -> None:
        self.app = app
        self.limit_resolver = limit_resolver
        self.exact_path_limits = dict(exact_path_limits or {})
        self.default_max_bytes = default_max_bytes
        self.max_concurrent_large_requests = max_concurrent_large_requests
        self.max_concurrent_large_mcp_requests = max_concurrent_large_mcp_requests
        self.admission_threshold_bytes = admission_threshold_bytes
        self.temp_storage_reserve_bytes = temp_storage_reserve_bytes
        self._admission_lock = asyncio.Lock()
        self._active_large_requests = 0
        self._active_large_mcp_requests = 0
        self._reserved_temp_bytes = 0
        if default_max_bytes is not None and default_max_bytes <= 0:
            raise ValueError("request-body limit must be positive")
        if any(limit <= 0 for limit in self.exact_path_limits.values()):
            raise ValueError("request-body limits must be positive")
        if max_concurrent_large_requests <= 0:
            raise ValueError("large-request concurrency must be positive")
        if max_concurrent_large_mcp_requests <= 0:
            raise ValueError("large MCP request concurrency must be positive")
        if admission_threshold_bytes < 0 or temp_storage_reserve_bytes < 0:
            raise ValueError("large-request storage bounds must not be negative")

    def _limit(self, scope: Scope) -> int:
        method_path = (
            str(scope.get("method", "GET")).upper(),
            str(scope.get("path", "")),
        )
        if method_path in self.exact_path_limits:
            return self.exact_path_limits[method_path]
        if self.default_max_bytes is not None:
            return self.default_max_bytes
        limit = self.limit_resolver(scope)
        if limit <= 0:
            raise RuntimeError("request-body resolver returned a non-positive limit")
        return limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit(scope)
        try:
            declared = _declared_content_length(scope)
        except ValueError:
            await self._reject(send, 400, "Invalid Content-Length header")
            return
        if declared is not None and declared > limit:
            await self._reject(send, 413, f"Request body is too large (max {limit} bytes)")
            return

        reservation = declared if declared is not None else limit
        admitted_multipart = False
        admitted_mcp = False
        temp_reservation = 0
        disk_reservation: DiskSpaceReservation | None = None
        custom_large_route = (
            str(scope.get("method", "GET")).upper(),
            str(scope.get("path", "")),
        ) in self.exact_path_limits
        large_multipart_request = (
            limit > self.admission_threshold_bytes
            and not _is_mcp_route(scope)
            and (custom_large_route or _is_large_multipart_route(scope))
        )
        # Small, declared MCP control calls stay fully concurrent. A request
        # that can carry a multi-hundred-MiB base64 transcription is admitted
        # separately from disk-spooled multipart uploads because its dominant
        # risk is the parsed JSON string plus decoded bytes in process memory.
        large_mcp_request = (
            _is_mcp_route(scope)
            and limit > self.admission_threshold_bytes
            and (declared is None or declared > self.admission_threshold_bytes)
        )
        if large_multipart_request or large_mcp_request:
            async with self._admission_lock:
                if large_multipart_request and (self._active_large_requests >= self.max_concurrent_large_requests):
                    await self._reject(
                        send,
                        429,
                        "Too many large request bodies are already in flight",
                        extra_headers=[(b"retry-after", b"1")],
                    )
                    return
                if large_mcp_request and (self._active_large_mcp_requests >= self.max_concurrent_large_mcp_requests):
                    await self._reject(
                        send,
                        429,
                        "Too many large MCP request bodies are already in flight",
                        extra_headers=[(b"retry-after", b"1")],
                    )
                    return
                if large_multipart_request:
                    # Starlette owns the parsed UploadFile spool while the
                    # endpoint makes its independently bounded working copy.
                    # Reserve both simultaneous files, not merely wire bytes.
                    temp_reservation = 2 * reservation
                    try:
                        disk_reservation = reserve_disk_space(
                            Path(tempfile.gettempdir()),
                            temp_reservation,
                            min_free_bytes=self.temp_storage_reserve_bytes,
                        )
                    except DiskSpaceReservationError:
                        await self._reject(
                            send,
                            507,
                            "Insufficient temporary storage for request body",
                        )
                        return
                    self._active_large_requests += 1
                    self._reserved_temp_bytes += temp_reservation
                    admitted_multipart = True
                if large_mcp_request:
                    self._active_large_mcp_requests += 1
                    admitted_mcp = True

        try:
            await self._run_bounded_request(
                scope,
                receive,
                send,
                limit,
                declared_content_length=declared,
            )
        finally:
            if admitted_multipart or admitted_mcp:
                async with self._admission_lock:
                    if admitted_multipart:
                        self._active_large_requests -= 1
                        self._reserved_temp_bytes -= temp_reservation
                        if disk_reservation is not None:
                            disk_reservation.release()
                    if admitted_mcp:
                        self._active_large_mcp_requests -= 1
                    if (
                        self._active_large_requests < 0
                        or self._active_large_mcp_requests < 0
                        or self._reserved_temp_bytes < 0
                    ):
                        raise RuntimeError("large-request admission accounting underflow")

    async def _run_bounded_request(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        limit: int,
        *,
        declared_content_length: int | None,
    ) -> None:
        received = 0
        invalid_content_length = False
        oversized = False

        async def bounded_receive() -> Message:
            nonlocal invalid_content_length, oversized, received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                received += len(body)
                if declared_content_length is not None and received > declared_content_length:
                    # The admission path may use a declared length to decide a
                    # request is small. Enforce that HTTP framing invariant at
                    # this boundary rather than trusting a broken ASGI server
                    # or proxy to preserve it.
                    invalid_content_length = True
                    raise _RequestBodyTooLargeError
                if received > limit:
                    oversized = True
                    raise _RequestBodyTooLargeError
            return message

        async def bounded_send(message: Message) -> None:
            # FastAPI deliberately converts arbitrary failures during body
            # parsing into a generic 400 response.  Once this receive wrapper
            # has proved the wire body oversized, suppress that parser response
            # so the boundary can return the correct 413 below.
            if not oversized and not invalid_content_length:
                await send(message)

        with suppress(_RequestBodyTooLargeError):
            await self.app(scope, bounded_receive, bounded_send)
        if invalid_content_length:
            await self._reject(send, 400, "Request body exceeds declared Content-Length")
        elif oversized:
            await self._reject(send, 413, f"Request body is too large (max {limit} bytes)")

    @staticmethod
    async def _reject(
        send: Send,
        status_code: int,
        detail: str,
        *,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                    *(extra_headers or []),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
