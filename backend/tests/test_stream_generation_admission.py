"""Foreground generation shares bounded queue and private response storage."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from backend import backends, config, models
from backend.backends import mlx_tts_lifecycle
from backend.routes import generations
from backend.services import effects_processing
from backend.utils import chunked_tts


def _request() -> models.GenerationRequest:
    return models.GenerationRequest(
        profile_id="profile",
        text="Bounded foreground speech.",
        engine="qwen",
        model_size="1.7B",
        seed=42,
    )


def _install_generation_fakes(monkeypatch: pytest.MonkeyPatch):
    profile = SimpleNamespace(
        id="profile",
        voice_type="cloned",
        effects_chain=None,
        default_engine="qwen",
        preset_engine=None,
    )

    async def get_profile(_profile_id, _db):
        return profile

    async def ensure_model(_engine, _model_size):
        return None

    @asynccontextmanager
    async def loaded_backend(_engine, _model_size):
        yield object()

    async def cancellation_safe(_backend, operation):
        return await operation

    async def create_prompt(_profile_id, _db, *, engine):
        assert engine == "qwen"
        return {"prompt": "voice"}

    def get_db():
        yield object()

    async def generate_chunked(*_args, **_kwargs):
        return np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24_000

    monkeypatch.setattr(generations.profiles, "get_profile", get_profile)
    monkeypatch.setattr(generations.profiles, "validate_profile_engine", lambda *_args: None)
    monkeypatch.setattr(generations.profiles, "create_voice_prompt_for_profile", create_prompt)
    monkeypatch.setattr(generations, "get_db", get_db)
    monkeypatch.setattr(backends, "ensure_model_cached_or_raise", ensure_model)
    monkeypatch.setattr(backends, "engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr(backends, "engine_retries_runaway", lambda _engine: False)
    monkeypatch.setattr(mlx_tts_lifecycle, "loaded_tts_backend_for_request", loaded_backend)
    monkeypatch.setattr(mlx_tts_lifecycle, "run_tts_operation_cancellation_safe", cancellation_safe)
    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", generate_chunked)


@pytest.mark.asyncio
async def test_stream_generation_runs_one_queue_job_and_returns_cleanup_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_generation_fakes(monkeypatch)
    response_path = tmp_path / "speech.wav"
    response_path.write_bytes(b"RIFF-private-response")
    cleaned = False
    queued_ids = []

    class GeneratedFile:
        path = response_path

        def cleanup(self):
            nonlocal cleaned
            if cleaned:
                return
            cleaned = True
            response_path.unlink(missing_ok=True)

    async def create_response_file(audio, sample_rate, effects_chain, normalize):
        assert audio.dtype == np.float32
        assert sample_rate == 24_000
        assert effects_chain == []
        assert normalize is True
        return GeneratedFile()

    async def run_queued(generation_id, coro, *, discard_result):
        assert callable(discard_result)
        queued_ids.append(generation_id)
        return await coro

    monkeypatch.setattr(
        effects_processing,
        "create_generated_audio_response_file",
        create_response_file,
    )
    monkeypatch.setattr(generations, "run_queued_generation", run_queued)
    request_db_closed = False

    class RequestDatabase:
        def close(self):
            nonlocal request_db_closed
            request_db_closed = True

    response = await generations._stream_speech_impl(_request(), RequestDatabase(), exact=False)

    assert len(queued_ids) == 1
    assert queued_ids[0].startswith("stream-")
    assert request_db_closed is True
    assert response.path == response_path
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await response(
        {"type": "http", "method": "GET", "path": "/generate/stream", "headers": []},
        receive,
        send,
    )

    assert b"".join(message.get("body", b"") for message in sent) == b"RIFF-private-response"
    assert cleaned is True
    assert not response_path.exists()


@pytest.mark.asyncio
async def test_stream_generation_rejects_full_queue_before_model_work(monkeypatch: pytest.MonkeyPatch):
    _install_generation_fakes(monkeypatch)

    async def reject(_generation_id, coro, *, discard_result):
        assert callable(discard_result)
        coro.close()
        raise generations.GenerationQueueFullError("full")

    monkeypatch.setattr(generations, "run_queued_generation", reject)

    with pytest.raises(HTTPException) as raised:
        await generations._stream_speech_impl(_request(), object(), exact=False)

    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "1"}


@pytest.mark.asyncio
async def test_stream_generation_reports_zero_frame_model_result_as_client_error(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_generation_fakes(monkeypatch)

    async def reject_empty_audio(*_args, **_kwargs):
        raise chunked_tts.GeneratedAudioEmptyError("TTS returned no audio frames for text '[…].'")

    async def run_queued(_generation_id, coro, *, discard_result):
        assert callable(discard_result)
        return await coro

    async def effects_must_not_run(*_args, **_kwargs):
        raise AssertionError("zero-frame audio must be rejected before effects processing")

    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", reject_empty_audio)
    monkeypatch.setattr(generations, "run_queued_generation", run_queued)
    monkeypatch.setattr(
        effects_processing,
        "create_generated_audio_response_file",
        effects_must_not_run,
    )

    with pytest.raises(HTTPException) as raised:
        await generations._stream_speech_impl(_request(), object(), exact=False)

    assert raised.value.status_code == 400
    assert raised.value.detail == "TTS returned no audio frames for text '[…].'"


@pytest.mark.asyncio
async def test_stream_generation_releases_disk_audio_when_model_context_exit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _install_generation_fakes(monkeypatch)
    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    config.initialize_data_permissions()
    accumulator = chunked_tts._DiskBackedChunkAccumulator(24_000, 0)
    accumulator.append(np.ones(2400, dtype=np.float32))
    audio = accumulator.finish()
    temporary_file = audio._voicebox_temporary_file

    async def generate_chunked(*_args, **_kwargs):
        return audio, 24_000

    @asynccontextmanager
    async def failing_exit(_engine, _model_size):
        yield object()
        raise RuntimeError("model release failed")

    async def run_queued(_generation_id, coro, *, discard_result):
        assert callable(discard_result)
        return await coro

    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", generate_chunked)
    monkeypatch.setattr(mlx_tts_lifecycle, "loaded_tts_backend_for_request", failing_exit)
    monkeypatch.setattr(generations, "run_queued_generation", run_queued)

    class RequestDatabase:
        def close(self):
            return None

    with pytest.raises(RuntimeError, match="model release failed"):
        await generations._stream_speech_impl(_request(), RequestDatabase(), exact=False)

    assert temporary_file.closed


@pytest.mark.asyncio
async def test_status_stream_closes_every_database_session_before_yield(
    monkeypatch: pytest.MonkeyPatch,
):
    active_sessions = 0
    poll_sessions_closed = 0
    request_session_closed = False
    generation = SimpleNamespace(
        id="generation",
        status="generating",
        duration=None,
        error=None,
        source="manual",
    )

    class Query:
        def filter_by(self, **_filters):
            return self

        def first(self):
            return generation

    class PollDatabase:
        def query(self, _model):
            return Query()

    class RequestDatabase:
        def close(self):
            nonlocal request_session_closed
            request_session_closed = True

    def get_db():
        nonlocal active_sessions, poll_sessions_closed
        active_sessions += 1
        try:
            yield PollDatabase()
        finally:
            active_sessions -= 1
            poll_sessions_closed += 1

    monkeypatch.setattr(generations, "get_db", get_db)

    response = await generations.get_generation_status("generation", RequestDatabase())
    first_event = await anext(response.body_iterator)

    assert '"status": "generating"' in first_event
    assert request_session_closed is True
    assert active_sessions == 0
    assert poll_sessions_closed == 1
    await response.body_iterator.aclose()
