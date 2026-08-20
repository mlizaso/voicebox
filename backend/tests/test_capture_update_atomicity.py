"""Concurrency and acknowledgement regressions for capture transcript updates."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, Capture
from backend.services import captures
from backend.services.refinement import RefinementFlags


@pytest.fixture
def capture_db(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    captures_dir = data_root / "captures"
    captures_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "_data_dir", data_root)
    engine = create_engine(f"sqlite:///{tmp_path / 'captures.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    seed = factory()
    seed.add(
        Capture(
            id="capture-id",
            audio_path="captures/capture-id.wav",
            source="file",
            transcript_raw="old transcript",
            stt_model="base",
        )
    )
    seed.commit()
    seed.close()
    (captures_dir / "capture-id.wav").write_bytes(b"audio")
    try:
        yield engine, factory
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_refine_rejects_stale_transcript_and_waiters_release_db_connections(
    capture_db,
    monkeypatch,
):
    engine, factory = capture_db
    stt_started = asyncio.Event()
    stt_release = asyncio.Event()
    refine_started = asyncio.Event()
    refine_release = asyncio.Event()

    class Whisper:
        model_size = "base"

        async def transcribe(self, *_args, **_kwargs):
            stt_started.set()
            await stt_release.wait()
            return "new transcript"

    async def refine_transcript(text, _flags, *, model_size):
        del model_size
        assert text == "old transcript"
        refine_started.set()
        await refine_release.wait()
        return "refined old transcript", "0.6B"

    monkeypatch.setattr(captures, "get_whisper_model", lambda: Whisper())
    monkeypatch.setattr(captures, "refine_transcript", refine_transcript)

    retranscribe_db = factory()
    refine_db = factory()
    retranscribe = asyncio.create_task(
        captures.retranscribe_capture(
            "capture-id",
            "base",
            None,
            retranscribe_db,
        )
    )
    await stt_started.wait()
    refinement = asyncio.create_task(
        captures.refine_capture(
            "capture-id",
            RefinementFlags(),
            "0.6B",
            refine_db,
        )
    )
    await refine_started.wait()
    assert engine.pool.checkedout() == 0

    stt_release.set()
    await retranscribe
    refine_release.set()
    with pytest.raises(captures.CaptureTranscriptChangedError, match="changed"):
        await refinement

    verify = factory()
    try:
        row = verify.query(Capture).filter_by(id="capture-id").one()
        assert row.transcript_raw == "new transcript"
        assert row.transcript_refined is None
        assert row.llm_model is None
    finally:
        retranscribe_db.close()
        refine_db.close()
        verify.close()


@pytest.mark.asyncio
async def test_refine_returns_durable_success_when_commit_acknowledgement_raises(
    capture_db,
    monkeypatch,
):
    _engine, factory = capture_db

    async def refine_transcript(text, _flags, *, model_size):
        assert text == "old transcript"
        return "clean transcript", model_size or "0.6B"

    monkeypatch.setattr(captures, "refine_transcript", refine_transcript)
    db = factory()
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after durable commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    try:
        response = await captures.refine_capture(
            "capture-id",
            RefinementFlags(),
            "0.6B",
            db,
        )
    finally:
        db.close()

    assert response is not None
    assert response.transcript_raw == "old transcript"
    assert response.transcript_refined == "clean transcript"
    assert response.llm_model == "0.6B"
