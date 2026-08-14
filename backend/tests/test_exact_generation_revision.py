"""Server-side enforcement for race-free exact TTS generation."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from backend import models
from backend.routes import generations


def _request(revision: str | None) -> models.GenerationRequest:
    return models.GenerationRequest(
        profile_id="profile-id",
        text="Texto de prueba.",
        language="es",
        engine="qwen",
        model_size="1.7B",
        tts_implementation_revision=revision,
    )


def test_exact_routes_are_distinct_from_legacy_generation_routes():
    paths = {route.path for route in generations.router.routes}

    assert "/generate" in paths
    assert "/generate/exact" in paths
    assert "/generate/stream" in paths
    assert "/generate/stream/exact" in paths


def test_legacy_request_without_revision_remains_accepted():
    generations._require_tts_implementation_revision(_request(None))


def test_mismatched_revision_creates_no_generation_or_queue_job(monkeypatch):
    get_task_manager = Mock(side_effect=AssertionError("must reject before creating a task"))
    get_profile = AsyncMock(side_effect=AssertionError("must reject before reading a profile"))
    create_generation = AsyncMock(side_effect=AssertionError("must not create history"))
    enqueue_generation = Mock(side_effect=AssertionError("must not enqueue generation"))

    monkeypatch.setattr(generations, "get_task_manager", get_task_manager)
    monkeypatch.setattr(generations.profiles, "get_profile", get_profile)
    monkeypatch.setattr(generations.history, "create_generation", create_generation)
    monkeypatch.setattr(generations, "enqueue_generation", enqueue_generation)
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "running-revision",
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech(_request("saved-revision"), db=object()))

    assert raised.value.status_code == 409
    assert "revision mismatch" in raised.value.detail.lower()
    get_task_manager.assert_not_called()
    get_profile.assert_not_awaited()
    create_generation.assert_not_awaited()
    enqueue_generation.assert_not_called()


def test_exact_route_requires_revision_before_dispatch(monkeypatch):
    dispatch = AsyncMock(side_effect=AssertionError("must not dispatch without revision"))
    monkeypatch.setattr(generations, "generate_speech", dispatch)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_exact(_request(None), db=object()))

    assert raised.value.status_code == 422
    dispatch.assert_not_awaited()


def test_exact_route_dispatches_matching_revision(monkeypatch):
    sentinel = object()
    dispatch = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(generations, "generate_speech", dispatch)
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "saved-revision",
    )
    request = _request("saved-revision")
    db = object()

    result = asyncio.run(generations.generate_speech_exact(request, db=db))

    assert result is sentinel
    dispatch.assert_awaited_once_with(request, db)


def test_exact_stream_rejects_mismatch_before_dispatch(monkeypatch):
    dispatch = AsyncMock(side_effect=AssertionError("must not start streaming"))
    monkeypatch.setattr(generations, "stream_speech", dispatch)
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "running-revision",
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.stream_speech_exact(_request("saved-revision"), db=object()))

    assert raised.value.status_code == 409
    dispatch.assert_not_awaited()
