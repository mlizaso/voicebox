"""Accepted chunking and normalization settings survive persistence and replay."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base, Generation, VoiceProfile
from backend.database.migrations import _migrate_generations
from backend.routes import generations


@pytest.mark.asyncio
@pytest.mark.parametrize(("engine", "size"), [("qwen", "3B"), ("tada", "0.6B"), ("tada", "1.7B")])
@pytest.mark.parametrize("stream", [False, True])
async def test_invalid_engine_size_is_rejected_before_persistence_or_inference(monkeypatch, engine, size, stream):
    monkeypatch.setattr(
        generations.profiles,
        "get_profile",
        AsyncMock(return_value=SimpleNamespace(voice_type="cloned", effects_chain=None)),
    )
    request = models.GenerationRequest(profile_id="profile", text="Speech", engine=engine, model_size=size)
    endpoint = generations._stream_speech_impl if stream else generations._generate_speech_impl
    with pytest.raises(HTTPException) as raised:
        await endpoint(request, object(), exact=False)
    assert raised.value.status_code == 422
    assert "Unsupported model size" in raised.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(("tts_engine", "model_size"), [("qwen", "1.7B"), ("tada", "1B")])
async def test_generation_replay_keeps_accepted_settings(tmp_path, monkeypatch, normalize, tts_engine, model_size):
    engine = create_engine(f"sqlite:///{tmp_path / 'replay.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    observed = []

    async def record_generation(**kwargs):
        observed.append(kwargs)

    async def enqueue(**kwargs):
        await kwargs["generation_coro"]

    monkeypatch.setattr(generations, "run_generation", record_generation)
    monkeypatch.setattr(generations, "_enqueue_generation_or_restore", enqueue)
    monkeypatch.setattr(generations, "generation_job_is_active", lambda _id: False)
    monkeypatch.setattr(
        generations, "get_task_manager", lambda: SimpleNamespace(start_generation=lambda **_kwargs: None)
    )
    try:
        result = await generations.generate_speech(
            models.GenerationRequest(
                profile_id="profile",
                text="Speech",
                engine=tts_engine,
                max_chunk_chars=345,
                crossfade_ms=0,
                normalize=normalize,
            ),
            db,
        )
        for status, endpoint in (
            ("failed", generations.retry_generation),
            ("completed", generations.regenerate_generation),
        ):
            row = db.query(Generation).filter_by(id=result.id).one()
            row.status = status
            if tts_engine == "tada":
                # Pre-validation releases stored Qwen sizes while loading TADA 1B.
                row.model_size = "0.6B"
            db.commit()
            db.expire_all()
            await endpoint(result.id, db)
        assert len(observed) == 3
        for settings in observed:
            assert settings["max_chunk_chars"] == 345
            assert settings["crossfade_ms"] == 0
            assert settings["normalize"] is normalize
            assert settings["model_size"] == model_size
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_imported_audio_is_not_reinterpreted_as_tts(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'import.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        db.add(VoiceProfile(id="profile", name="Imported Audio"))
        db.add(Generation(id="import", profile_id="profile", text="clip", engine="import", status="completed"))
        db.commit()
        with pytest.raises(HTTPException) as raised:
            await generations.regenerate_generation("import", db)
        assert raised.value.status_code == 409
        assert "Imported audio" in raised.value.detail
        assert db.query(Generation).one().status == "completed"
    finally:
        db.close()
        engine.dispose()


def test_replay_settings_migration_is_idempotent_and_preserves_legacy_rows(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE generations (id TEXT PRIMARY KEY)"))
            connection.execute(text("INSERT INTO generations (id) VALUES ('legacy')"))
        for _ in range(2):
            _migrate_generations(engine, inspect(engine), {"generations"})
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT id, max_chunk_chars, crossfade_ms, normalize_audio FROM generations")
            ).one()
            assert tuple(row) == ("legacy", None, None, None)
    finally:
        engine.dispose()
