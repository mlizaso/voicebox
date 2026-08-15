"""Regression coverage for version metadata on history detail responses."""

import importlib
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.database import Base, Generation, GenerationVersion, VoiceProfile, get_db


def _load_history_router():
    """Import the isolated router without constructing the production app recursively."""
    installed_app_shim = "backend.app" not in sys.modules
    route_module = None
    if installed_app_shim:
        app_shim = types.ModuleType("backend.app")
        app_shim.safe_content_disposition = lambda disposition, filename: (  # type: ignore[attr-defined]
            f'{disposition}; filename="{filename}"'
        )
        sys.modules["backend.app"] = app_shim
    try:
        route_module = importlib.import_module("backend.routes.history")
        return route_module.router
    finally:
        if installed_app_shim:
            del sys.modules["backend.app"]
            sys.modules.pop("backend.routes.history", None)
            routes_package = sys.modules.get("backend.routes")
            if route_module is not None and getattr(routes_package, "history", None) is route_module:
                delattr(routes_package, "history")


@pytest.fixture
def client(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'history.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = testing_session_local()
    session.add(VoiceProfile(id="profile-1", name="Test Profile"))
    session.add_all(
        [
            Generation(
                id="generation-1",
                profile_id="profile-1",
                text="Versioned generation",
                language="en",
                audio_path="generations/processed.wav",
                status="completed",
            ),
            Generation(
                id="generation-without-versions",
                profile_id="profile-1",
                text="Unversioned generation",
                language="en",
                audio_path="generations/plain.wav",
                status="completed",
            ),
        ]
    )
    created = datetime(2026, 1, 2, 3, 4, 5)
    session.add_all(
        [
            GenerationVersion(
                id="version-original",
                generation_id="generation-1",
                label="original",
                audio_path="generations/original.wav",
                effects_chain=None,
                is_default=False,
                created_at=created,
            ),
            GenerationVersion(
                id="version-processed",
                generation_id="generation-1",
                label="processed",
                audio_path="generations/processed.wav",
                effects_chain='[{"type":"gain","params":{"gain_db":-1.0}}]',
                is_default=True,
                created_at=created + timedelta(seconds=1),
            ),
        ]
    )
    session.commit()
    session.close()

    app = FastAPI()
    app.include_router(_load_history_router())

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client


def test_history_detail_includes_ordered_versions_and_active_id(client):
    response = client.get("/history/generation-1")

    assert response.status_code == 200
    body = response.json()
    assert [version["id"] for version in body["versions"]] == [
        "version-original",
        "version-processed",
    ]
    assert [version["label"] for version in body["versions"]] == [
        "original",
        "processed",
    ]
    assert body["versions"][1]["effects_chain"] == [
        {
            "type": "gain",
            "enabled": True,
            "params": {"gain_db": -1.0},
        }
    ]
    assert body["active_version_id"] == "version-processed"


def test_history_detail_without_versions_keeps_optional_fields_empty(client):
    response = client.get("/history/generation-without-versions")

    assert response.status_code == 200
    assert response.json()["versions"] is None
    assert response.json()["active_version_id"] is None
