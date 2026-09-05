"""Browser pairing reaches the state-protected callback without weakening CSRF checks."""

from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.api_security import LocalAPISecurityMiddleware
from backend.database import Base, CloudSettings, get_db
from backend.routes.cloud import router
from backend.services import cloud

NAVIGATION_HEADERS = {
    "Sec-Fetch-Site": "cross-site",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Dest": "document",
}


@pytest.fixture
def pairing_client(monkeypatch):
    monkeypatch.setattr(cloud, "_pending", {})
    monkeypatch.setattr(cloud.webbrowser, "open", lambda _url: True)
    monkeypatch.setattr(cloud.config, "get_cloud_web_url", lambda: "https://cloud.example")
    monkeypatch.setattr(cloud.config, "get_cloud_api_url", lambda: "https://api.example")
    exchanges = []

    def respond(request):
        exchanges.append(request.url.path)
        if request.url.path == "/api/connect/exchange":
            return httpx.Response(200, json={"key": "voicebox_test_key", "label": "Desktop"})
        assert request.headers["authorization"] == "Bearer voicebox_test_key"
        return httpx.Response(200, json={"data": {"userId": "user"}})

    async_client = httpx.AsyncClient
    monkeypatch.setattr(
        cloud.httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    app = FastAPI()
    app.add_middleware(LocalAPISecurityMiddleware)
    app.include_router(router)
    try:
        with Session(engine) as db:
            app.dependency_overrides[get_db] = lambda: db
            with TestClient(app) as client:
                yield client, db, exchanges
    finally:
        engine.dispose()


def test_cloud_redirect_exchanges_once_and_stores_the_verified_key(pairing_client):
    client, db, exchanges = pairing_client
    started = client.post("/cloud/login/start")
    assert started.status_code == 200
    authorize_url = started.json()["authorize_url"]
    state = parse_qs(urlsplit(authorize_url).query)["state"][0]

    response = client.get("/cloud/callback", params={"state": state, "code": "code"}, headers=NAVIGATION_HEADERS)

    assert response.status_code == 200
    row = db.get(CloudSettings, 1)
    assert row.api_key == "voicebox_test_key"
    assert row.account_user_id == "user"
    assert exchanges == ["/api/connect/exchange", "/v1/account/me"]

    replay = client.get("/cloud/callback", params={"state": state, "code": "code"}, headers=NAVIGATION_HEADERS)
    assert replay.status_code == 400
    assert len(exchanges) == 2


@pytest.mark.parametrize("state", ["", "unissued", "expired"])
def test_cloud_redirect_rejects_invalid_state_without_exchanging(pairing_client, state):
    client, db, exchanges = pairing_client
    cloud._pending["expired"] = 0

    response = client.get("/cloud/callback", params={"state": state, "code": "code"}, headers=NAVIGATION_HEADERS)

    assert response.status_code == 400
    assert exchanges == []
    assert db.get(CloudSettings, 1) is None


@pytest.mark.parametrize(
    ("method", "path", "headers"),
    [
        ("POST", "/cloud/login/start", NAVIGATION_HEADERS),
        ("GET", "/cloud/status", NAVIGATION_HEADERS),
        ("GET", "/cloud/callback", {**NAVIGATION_HEADERS, "Sec-Fetch-Mode": "cors"}),
        ("GET", "/cloud/callback", {**NAVIGATION_HEADERS, "Origin": "https://attacker.example"}),
    ],
)
def test_callback_exception_does_not_allow_other_cross_site_requests(pairing_client, method, path, headers):
    client, db, exchanges = pairing_client

    response = client.request(method, path, headers=headers)

    assert response.status_code == 403
    assert exchanges == []
    assert db.get(CloudSettings, 1) is None
