"""Local and remote API request-boundary protections.

The desktop backend is intentionally usable by originless CLI clients, while
browser traffic must prove both an expected Host and an allowed Origin.  This
blocks DNS-rebinding and simple-request CSRF. Credential-free access requires
both a loopback ASGI peer and a local Host authority; all other requests need
an explicitly configured remote bearer capability over a secure transport.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
from base64 import urlsafe_b64encode
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie
from urllib.parse import SplitResult, urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

DEFAULT_BROWSER_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:17493",
    "http://127.0.0.1:17493",
    "tauri://localhost",
    "https://tauri.localhost",
    "http://tauri.localhost",
)
DEFAULT_TRUSTED_HOSTS = (
    "localhost",
    "127.0.0.1",
    "[::1]",
    "tauri.localhost",
    # Starlette's conventional in-process test authority. It is an
    # unqualified local name, not an internet DNS suffix.
    "testserver",
)
REMOTE_API_TOKEN_ENV = "VOICEBOX_REMOTE_API_TOKEN"
REMOTE_API_TOKEN_MIN_BYTES = 32
REMOTE_API_TOKEN_MAX_BYTES = 512
REMOTE_SESSION_COOKIE = "__Host-voicebox_remote_session"
INSECURE_REMOTE_SESSION_COOKIE = "voicebox_remote_session"
ALLOW_INSECURE_REMOTE_HTTP_ENV = "VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP"
TRUSTED_LOCAL_REQUEST_SCOPE_KEY = "voicebox.trusted_local_request"
_REMOTE_SESSION_CONTEXT = b"voicebox remote session cookie v1"


@dataclass(frozen=True)
class _Authority:
    hostname: str
    port: int | None


def _split_csv_environment(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _normalized_hostname(hostname: str | None) -> str:
    if not hostname:
        raise ValueError("missing hostname")
    normalized = hostname.rstrip(".").lower()
    if not normalized or any(character.isspace() for character in normalized):
        raise ValueError("invalid hostname")
    return normalized


def _parse_host_authority(value: str) -> _Authority:
    if not value or any(character in value for character in "/?#@,"):
        raise ValueError("invalid Host header")
    try:
        parsed = urlsplit(f"http://{value}")
        hostname = _normalized_hostname(parsed.hostname)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid Host header") from exc
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("invalid Host header")
    return _Authority(hostname=hostname, port=port)


def _parse_origin(value: str) -> tuple[SplitResult, _Authority]:
    if not value or value == "null" or "," in value or any(character.isspace() for character in value):
        raise ValueError("invalid Origin header")
    try:
        parsed = urlsplit(value)
        hostname = _normalized_hostname(parsed.hostname)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid Origin header") from exc
    if (
        parsed.scheme not in {"http", "https", "tauri"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("invalid Origin header")
    return parsed, _Authority(hostname=hostname, port=port)


def configured_browser_origins() -> list[str]:
    """Return exact CORS/CSRF origins, rejecting wildcard configuration."""
    origins = list(DEFAULT_BROWSER_ORIGINS)
    for value in _split_csv_environment("VOICEBOX_CORS_ORIGINS"):
        if value == "*":
            raise ValueError("VOICEBOX_CORS_ORIGINS must list explicit origins")
        parsed, _authority = _parse_origin(value)
        if parsed.path == "/":
            value = value[:-1]
        origins.append(value)
    return list(dict.fromkeys(origins))


def configured_trusted_hosts() -> tuple[_Authority, ...]:
    """Return exact trusted authorities for the local API."""
    configured = _split_csv_environment("VOICEBOX_TRUSTED_HOSTS")
    if "*" in configured:
        raise ValueError("VOICEBOX_TRUSTED_HOSTS must list explicit hosts")

    values = [*DEFAULT_TRUSTED_HOSTS, *configured]
    return tuple(dict.fromkeys(_parse_host_authority(value) for value in values))


def configured_remote_api_token() -> bytes | None:
    """Read and validate the opt-in capability used by non-loopback peers."""
    value = os.environ.get(REMOTE_API_TOKEN_ENV)
    if value is None or value == "":
        return None
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{REMOTE_API_TOKEN_ENV} must contain only ASCII characters") from exc
    if not REMOTE_API_TOKEN_MIN_BYTES <= len(encoded) <= REMOTE_API_TOKEN_MAX_BYTES:
        raise ValueError(
            f"{REMOTE_API_TOKEN_ENV} must be between "
            f"{REMOTE_API_TOKEN_MIN_BYTES} and {REMOTE_API_TOKEN_MAX_BYTES} ASCII characters"
        )
    if any(character <= 0x20 or character == 0x7F for character in encoded):
        raise ValueError(f"{REMOTE_API_TOKEN_ENV} must not contain whitespace or control characters")
    if any(not (chr(character).isalnum() or chr(character) in "-._~") for character in encoded):
        raise ValueError(f"{REMOTE_API_TOKEN_ENV} must use URL-safe ASCII characters")
    return encoded


def insecure_remote_http_allowed() -> bool:
    """Return whether the operator explicitly accepted plaintext remote auth."""
    value = os.environ.get(ALLOW_INSECURE_REMOTE_HTTP_ENV)
    if value in {None, "", "0"}:
        return False
    if value == "1":
        return True
    raise ValueError(f"{ALLOW_INSECURE_REMOTE_HTTP_ENV} must be either 0 or 1")


def _default_port(scheme: str) -> int | None:
    if scheme == "http":
        return 80
    if scheme == "https":
        return 443
    return None


def _origin_is_same_authority(
    parsed_origin: SplitResult,
    origin: _Authority,
    host: _Authority,
    request_scheme: str,
) -> bool:
    normalized_request_scheme = {"ws": "http", "wss": "https"}.get(request_scheme, request_scheme)
    if parsed_origin.scheme != normalized_request_scheme or origin.hostname != host.hostname:
        return False
    origin_port = origin.port or _default_port(parsed_origin.scheme)
    host_port = host.port or _default_port(normalized_request_scheme)
    return origin_port == host_port


def _headers(scope: Scope, name: bytes) -> list[str]:
    return [value.decode("latin-1") for header_name, value in scope.get("headers", ()) if header_name.lower() == name]


def _client_is_loopback(scope: Scope) -> bool:
    client = scope.get("client")
    if not client:
        return False
    client_host = str(client[0])
    # Starlette's in-process TestClient has no network peer. Keep its explicit
    # testserver convention useful without trusting any HTTP forwarding header.
    if client_host == "testclient":
        return True
    try:
        return ipaddress.ip_address(client_host).is_loopback
    except ValueError:
        return False


def _host_is_local(host: _Authority) -> bool:
    if host.hostname in {"localhost", "tauri.localhost", "testserver"}:
        return True
    try:
        return ipaddress.ip_address(host.hostname).is_loopback
    except ValueError:
        return False


def _presented_bearer_token(scope: Scope) -> bytes | None:
    values = _headers(scope, b"authorization")
    if not values:
        return None
    if len(values) != 1:
        return b""
    scheme, separator, value = values[0].partition(" ")
    if scheme.lower() != "bearer" or separator != " " or not value or " " in value:
        return b""
    try:
        return value.encode("ascii")
    except UnicodeEncodeError:
        return b""


def _presented_session_cookie(scope: Scope) -> bytes | None:
    values = _headers(scope, b"cookie")
    if not values:
        return None
    if len(values) != 1:
        return b""
    cookies = SimpleCookie()
    try:
        cookies.load(values[0])
    except CookieError:
        return b""
    cookie_name = (
        REMOTE_SESSION_COOKIE
        if str(scope.get("scheme", "http")).lower() in {"https", "wss"}
        else INSECURE_REMOTE_SESSION_COOKIE
    )
    morsel = cookies.get(cookie_name)
    if morsel is None:
        return None
    try:
        return morsel.value.encode("ascii")
    except UnicodeEncodeError:
        return b""


def _remote_session_capability(token: bytes) -> bytes:
    """Derive a cookie-only capability without disclosing the API token."""
    digest = hmac.digest(token, _REMOTE_SESSION_CONTEXT, "sha256")
    return urlsafe_b64encode(digest).rstrip(b"=")


def _is_public_frontend_shell(scope: Scope) -> bool:
    """Allow only immutable UI code needed to enter a remote capability."""
    if str(scope.get("method", "")).upper() not in {"GET", "HEAD"}:
        return False
    path = str(scope.get("path", ""))
    return path == "/" or path.startswith("/assets/")


def _is_cors_preflight(scope: Scope) -> bool:
    return (
        str(scope.get("method", "")).upper() == "OPTIONS"
        and len(_headers(scope, b"origin")) == 1
        and len(_headers(scope, b"access-control-request-method")) == 1
    )


def _session_cookie_header(token: bytes, *, secure: bool) -> bytes:
    cookie_name = REMOTE_SESSION_COOKIE if secure else INSECURE_REMOTE_SESSION_COOKIE
    attributes = [
        f"{cookie_name}={token.decode('ascii')}",
        "Path=/",
        "HttpOnly",
        "SameSite=None" if secure else "SameSite=Strict",
    ]
    if secure:
        attributes.append("Secure")
    return "; ".join(attributes).encode("ascii")


class LocalAPISecurityMiddleware:
    """Enforce authority, browser-origin, and remote-peer authentication."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        trusted_hosts: tuple[_Authority, ...] | None = None,
        allowed_origins: tuple[str, ...] | None = None,
    ) -> None:
        self.app = app
        self.trusted_hosts = configured_trusted_hosts() if trusted_hosts is None else trusted_hosts
        origin_values = configured_browser_origins() if allowed_origins is None else list(allowed_origins)
        self.allowed_origins = frozenset(origin_values)
        self.remote_api_token = configured_remote_api_token()
        self.allow_insecure_remote_http = insecure_remote_http_allowed()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        host_headers = _headers(scope, b"host")
        if len(host_headers) != 1:
            await self._reject(scope, send, 400, "Invalid Host header")
            return
        try:
            host = _parse_host_authority(host_headers[0])
        except ValueError:
            await self._reject(scope, send, 400, "Invalid Host header")
            return
        if not _host_is_local(host) and not any(
            candidate.hostname == host.hostname and (candidate.port is None or candidate.port == host.port)
            for candidate in self.trusted_hosts
        ):
            await self._reject(scope, send, 400, "Untrusted Host header")
            return

        origin_headers = _headers(scope, b"origin")
        if len(origin_headers) > 1:
            await self._reject(scope, send, 403, "Invalid Origin header")
            return
        if origin_headers:
            origin_value = origin_headers[0]
            try:
                parsed_origin, origin = _parse_origin(origin_value)
            except ValueError:
                await self._reject(scope, send, 403, "Untrusted Origin")
                return
            same_authority = _origin_is_same_authority(
                parsed_origin,
                origin,
                host,
                str(scope.get("scheme", "http")),
            )
            if origin_value.rstrip("/") not in self.allowed_origins and not same_authority:
                await self._reject(scope, send, 403, "Untrusted Origin")
                return
        elif any(value.lower() == "cross-site" for value in _headers(scope, b"sec-fetch-site")):
            await self._reject(scope, send, 403, "Cross-site browser request rejected")
            return

        authenticated_by_bearer = False
        trusted_local_request = _client_is_loopback(scope) and _host_is_local(host)
        # Inner application code must not re-derive locality from the socket
        # peer alone: a documented same-host reverse proxy also appears as
        # 127.0.0.1. Publish the already-verified peer+Host decision through
        # the server-owned ASGI scope, never through a client header.
        scope[TRUSTED_LOCAL_REQUEST_SCOPE_KEY] = trusted_local_request
        remote_request = not trusted_local_request
        cors_preflight = _is_cors_preflight(scope)
        remote_api_request = remote_request and not _is_public_frontend_shell(scope)
        if remote_api_request:
            if self.remote_api_token is None:
                await self._reject(scope, send, 403, "Remote API access is disabled")
                return
            secure_transport = str(scope.get("scheme", "http")).lower() in {"https", "wss"}
            if not secure_transport and not self.allow_insecure_remote_http:
                await self._reject(
                    scope,
                    send,
                    426,
                    "Remote API authentication requires HTTPS",
                    extra_headers=[(b"upgrade", b"TLS/1.2")],
                )
                return
        if remote_api_request and not cors_preflight:
            bearer = _presented_bearer_token(scope)
            if bearer is not None:
                authenticated_by_bearer = hmac.compare_digest(bearer, self.remote_api_token)
                authenticated = authenticated_by_bearer
            else:
                cookie = _presented_session_cookie(scope)
                authenticated = cookie is not None and hmac.compare_digest(
                    cookie,
                    _remote_session_capability(self.remote_api_token),
                )
            if not authenticated:
                await self._reject(
                    scope,
                    send,
                    401,
                    "Remote authentication required",
                    extra_headers=[(b"www-authenticate", b'Bearer realm="Voicebox remote API"')],
                )
                return

        if authenticated_by_bearer and scope["type"] == "http":
            cookie_header = _session_cookie_header(
                _remote_session_capability(self.remote_api_token),
                secure=str(scope.get("scheme", "http")) == "https",
            )

            async def send_with_session_cookie(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", ()))
                    headers.append((b"set-cookie", cookie_header))
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_with_session_cookie)
            return

        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(
        scope: Scope,
        send: Send,
        status_code: int,
        detail: str,
        *,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            *(extra_headers or []),
        ]
        await send(
            {
                "type": "http.response.start",
                "status": status_code,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})
