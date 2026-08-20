"""Model-free regressions for the local HTTP request boundary."""

import json

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from backend.api_security import (
    LocalAPISecurityMiddleware,
    configured_browser_origins,
)
from backend.mcp_server.context import ClientIdMiddleware, request_is_loopback

REMOTE_TOKEN = "a-secure-random-remote-token-that-is-long-enough"


@pytest.fixture(autouse=True)
def _clear_remote_security_environment(monkeypatch):
    for name in (
        "VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP",
        "VOICEBOX_CORS_ORIGINS",
        "VOICEBOX_REMOTE_API_TOKEN",
        "VOICEBOX_TRUSTED_HOSTS",
    ):
        monkeypatch.delenv(name, raising=False)


def _build_app() -> tuple[FastAPI, dict[str, int]]:
    app = FastAPI()
    calls = {"mutations": 0}
    app.add_middleware(
        CORSMiddleware,
        allow_origins=configured_browser_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LocalAPISecurityMiddleware)

    @app.get("/private")
    async def private():
        return {"secret": "local"}

    @app.post("/shutdown")
    async def shutdown():
        calls["mutations"] += 1
        return {"ok": True}

    return app, calls


async def _call_asgi(
    app,
    *,
    client_host: str,
    method: str = "POST",
    host: str = "localhost:17493",
    path: str = "/shutdown",
    scheme: str = "http",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict, list[tuple[bytes, bytes]]]:
    request_headers = [(b"host", host.encode()), *(headers or [])]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": scheme,
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": request_headers,
        "client": (client_host, 54321),
        "server": ("127.0.0.1", 17493),
    }
    received = False

    async def receive():
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    try:
        decoded = json.loads(body or b"{}")
    except json.JSONDecodeError:
        decoded = {"raw": body.decode("utf-8", errors="replace")}
    return start["status"], decoded, start["headers"]


def test_dns_rebinding_host_is_rejected_before_route_runs():
    app, _calls = _build_app()
    with TestClient(app) as client:
        response = client.get("/private", headers={"Host": "attacker.example"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Untrusted Host header"}


def test_hostile_origin_cannot_call_simple_shutdown_post():
    app, calls = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/shutdown",
            headers={"Host": "localhost:17493", "Origin": "https://attacker.example"},
        )
    assert response.status_code == 403
    assert calls["mutations"] == 0


def test_originless_cli_post_remains_supported():
    app, calls = _build_app()
    with TestClient(app) as client:
        response = client.post("/shutdown", headers={"Host": "127.0.0.1:17493"})
    assert response.status_code == 200
    assert calls["mutations"] == 1


@pytest.mark.asyncio
async def test_remote_peer_cannot_bypass_authentication_with_forged_localhost(monkeypatch):
    monkeypatch.delenv("VOICEBOX_REMOTE_API_TOKEN", raising=False)
    app, calls = _build_app()

    status, body, _headers = await _call_asgi(app, client_host="192.0.2.44")

    assert status == 403
    assert body == {"detail": "Remote API access is disabled"}
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_loopback_reverse_proxy_with_remote_authority_still_requires_bearer(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.example:17493")
    app, calls = _build_app()

    status, body, _headers = await _call_asgi(
        app,
        client_host="127.0.0.1",
        host="voicebox.example:17493",
        scheme="https",
    )

    assert status == 401
    assert body == {"detail": "Remote authentication required"}
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_authenticated_loopback_proxy_does_not_gain_local_mcp_filesystem_trust(
    monkeypatch,
):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.example:17493")
    app = FastAPI()
    app.add_middleware(ClientIdMiddleware)
    app.add_middleware(LocalAPISecurityMiddleware)

    @app.get("/local-context")
    async def local_context():
        return {"trusted_local": request_is_loopback()}

    status, body, _headers = await _call_asgi(
        app,
        client_host="127.0.0.1",
        method="GET",
        host="voicebox.example:17493",
        path="/local-context",
        scheme="https",
        headers=[(b"authorization", f"Bearer {REMOTE_TOKEN}".encode())],
    )

    assert status == 200
    assert body == {"trusted_local": False}

    local_status, local_body, _local_headers = await _call_asgi(
        app,
        client_host="127.0.0.1",
        method="GET",
        host="localhost:17493",
        path="/local-context",
    )

    assert local_status == 200
    assert local_body == {"trusted_local": True}


@pytest.mark.asyncio
async def test_remote_peer_needs_correct_bearer_even_for_trusted_host(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    app, calls = _build_app()

    status, body, _headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        host="voicebox.lan:17493",
        scheme="https",
        headers=[(b"authorization", b"Bearer incorrect-token")],
    )

    assert status == 401
    assert body == {"detail": "Remote authentication required"}
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_remote_peer_with_trusted_host_and_bearer_is_allowed(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    app, calls = _build_app()

    status, body, response_headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        host="voicebox.lan:17493",
        scheme="https",
        headers=[(b"authorization", f"Bearer {REMOTE_TOKEN}".encode())],
    )

    assert status == 200
    assert body == {"ok": True}
    session_headers = [value for name, value in response_headers if name.lower() == b"set-cookie"]
    assert len(session_headers) == 1
    assert session_headers[0].startswith(b"__Host-voicebox_remote_session=")
    assert REMOTE_TOKEN.encode() not in session_headers[0]
    assert b"HttpOnly" in session_headers[0]
    assert b"SameSite=None" in session_headers[0]
    assert b"Secure" in session_headers[0]
    assert calls["mutations"] == 1


@pytest.mark.asyncio
async def test_remote_bearer_is_rejected_over_plain_http_by_default(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    app, calls = _build_app()

    status, body, response_headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        host="voicebox.lan:17493",
        headers=[
            (b"authorization", f"Bearer {REMOTE_TOKEN}".encode()),
            # Middleware trusts the ASGI server's normalized scheme, never a
            # forwarding header supplied directly by a network client.
            (b"x-forwarded-proto", b"https"),
        ],
    )

    assert status == 426
    assert body == {"detail": "Remote API authentication requires HTTPS"}
    assert (b"upgrade", b"TLS/1.2") in response_headers
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_plain_http_remote_bearer_requires_explicit_insecure_opt_in(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    monkeypatch.setenv("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP", "1")
    app, calls = _build_app()

    status, body, response_headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        host="voicebox.lan:17493",
        headers=[(b"authorization", f"Bearer {REMOTE_TOKEN}".encode())],
    )

    assert status == 200
    assert body == {"ok": True}
    session_header = next(value for name, value in response_headers if name == b"set-cookie")
    assert b"SameSite=Strict" in session_header
    assert b"Secure" not in session_header
    assert calls["mutations"] == 1


@pytest.mark.asyncio
async def test_remote_cors_preflight_remains_available_without_bearer(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    monkeypatch.setenv("VOICEBOX_CORS_ORIGINS", "https://reader.example")
    app, calls = _build_app()

    status, _body, _headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        method="OPTIONS",
        host="voicebox.lan:17493",
        scheme="https",
        headers=[
            (b"origin", b"https://reader.example"),
            (b"access-control-request-method", b"POST"),
            (b"access-control-request-headers", b"authorization"),
        ],
    )

    assert status == 200
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_plain_http_cors_preflight_fails_before_browser_can_send_bearer(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    monkeypatch.setenv("VOICEBOX_CORS_ORIGINS", "https://reader.example")
    app, calls = _build_app()

    status, body, _headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        method="OPTIONS",
        host="voicebox.lan:17493",
        headers=[
            (b"origin", b"https://reader.example"),
            (b"access-control-request-method", b"POST"),
            (b"access-control-request-headers", b"authorization"),
        ],
    )

    assert status == 426
    assert body == {"detail": "Remote API authentication requires HTTPS"}
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_originless_remote_options_is_not_treated_as_cors_preflight(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    app, calls = _build_app()

    status, body, _headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        method="OPTIONS",
        host="voicebox.lan:17493",
        scheme="https",
    )

    assert status == 401
    assert body == {"detail": "Remote authentication required"}
    assert calls["mutations"] == 0


@pytest.mark.asyncio
async def test_remote_session_cookie_authorizes_follow_up_resource_request(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    app, calls = _build_app()

    initial_status, _initial_body, initial_headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        host="voicebox.lan:17493",
        scheme="https",
        headers=[(b"authorization", f"Bearer {REMOTE_TOKEN}".encode())],
    )
    assert initial_status == 200
    set_cookie = next(value for name, value in initial_headers if name == b"set-cookie")
    cookie = set_cookie.split(b";", 1)[0]

    status, body, _headers = await _call_asgi(
        app,
        client_host="192.0.2.44",
        host="voicebox.lan:17493",
        scheme="https",
        headers=[(b"cookie", cookie)],
    )

    assert status == 200
    assert body == {"ok": True}
    assert calls["mutations"] == 2


def test_remote_token_configuration_rejects_short_secrets(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "short")
    app, _calls = _build_app()
    with pytest.raises(ValueError, match="between 32 and 512"), TestClient(app) as client:
        client.get("/private")


def test_insecure_remote_http_configuration_rejects_ambiguous_values(monkeypatch):
    monkeypatch.setenv("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP", "true")
    app, _calls = _build_app()
    with pytest.raises(ValueError, match="must be either 0 or 1"), TestClient(app) as client:
        client.get("/private")


def test_originless_cross_site_browser_request_is_rejected():
    app, calls = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/shutdown",
            headers={
                "Host": "localhost:17493",
                "Sec-Fetch-Site": "cross-site",
            },
        )
    assert response.status_code == 403
    assert calls["mutations"] == 0


def test_same_authority_origin_is_allowed_on_dynamic_backend_port():
    app, calls = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/shutdown",
            headers={
                "Host": "localhost:17494",
                "Origin": "http://localhost:17494",
            },
        )
    assert response.status_code == 200
    assert calls["mutations"] == 1


def test_same_host_with_different_origin_scheme_is_rejected():
    app, calls = _build_app()
    with TestClient(app, base_url="https://localhost") as client:
        response = client.post(
            "/shutdown",
            headers={
                "Host": "localhost",
                "Origin": "http://localhost:443",
            },
        )
    assert response.status_code == 403
    assert calls["mutations"] == 0


def test_explicit_remote_host_and_browser_origin_opt_in(monkeypatch):
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", REMOTE_TOKEN)
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "voicebox.lan:17493")
    monkeypatch.setenv("VOICEBOX_CORS_ORIGINS", "https://reader.example")
    app, calls = _build_app()
    with TestClient(app, base_url="https://voicebox.lan:17493") as client:
        response = client.post(
            "/shutdown",
            headers={
                "Host": "voicebox.lan:17493",
                "Origin": "https://reader.example",
                "Authorization": f"Bearer {REMOTE_TOKEN}",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://reader.example"
    assert calls["mutations"] == 1


def test_wildcard_remote_configuration_is_refused(monkeypatch):
    monkeypatch.setenv("VOICEBOX_TRUSTED_HOSTS", "*")
    app, _calls = _build_app()
    with pytest.raises(ValueError, match="must list explicit hosts"), TestClient(app) as client:
        client.get("/private")


def test_voicebox_app_wraps_shutdown_and_watchdog_routes_with_boundary():
    from backend.app import app as voicebox_app

    assert voicebox_app.user_middleware[0].cls is LocalAPISecurityMiddleware
    route_paths = {
        getattr(route, "path", None)
        for included in voicebox_app.routes
        for route in getattr(getattr(included, "original_router", None), "routes", ())
    }
    assert "/shutdown" in route_paths
    assert "/watchdog/disable" in route_paths
