"""Durability and cleanup regressions for capture creation."""

from __future__ import annotations

import asyncio
import io
import wave

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, Capture
from backend.services import captures, deletion_journal


class _Whisper:
    model_size = "small"

    async def transcribe(self, _path: str, _language: str | None, _model: str) -> str:
        return "transcript"


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(b"\0\0" * 2_400)
    return output.getvalue()


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    return data_dir, engine, sessionmaker(bind=engine)


def _create(db):
    return asyncio.run(
        captures.create_capture(
            audio_bytes=_wav_bytes(),
            filename="capture.wav",
            source="file",
            language="en",
            stt_model=None,
            db=db,
        )
    )


def test_create_capture_durably_publishes_completed_row(tmp_path, monkeypatch):
    data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())

    response = _create(db)

    row = db.query(Capture).filter_by(id=response.id).one()
    audio = config.resolve_storage_path(row.audio_path)
    assert audio is not None
    assert audio.read_bytes() == _wav_bytes()
    assert row.transcript_raw == "transcript"
    assert not list((data_dir / "captures").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_create_capture_streams_from_private_upload_path(tmp_path, monkeypatch):
    _data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())
    upload = tmp_path / "bounded-upload.wav"
    upload.write_bytes(_wav_bytes())

    response = asyncio.run(
        captures.create_capture(
            audio_bytes=upload,
            filename="capture.wav",
            source="file",
            language="en",
            stt_model=None,
            db=db,
        )
    )

    row = db.query(Capture).filter_by(id=response.id).one()
    audio = config.resolve_storage_path(row.audio_path)
    assert audio is not None
    assert audio.read_bytes() == _wav_bytes()
    assert upload.read_bytes() == _wav_bytes()
    db.close()


def test_startup_discards_capture_interrupted_during_population(tmp_path, monkeypatch):
    data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()

    def interrupt_population(path, _source, *, expected_stat):
        assert path.stat().st_ino == expected_stat.st_ino
        assert list(config.get_deletion_journal_dir().glob("*.json"))
        path.write_bytes(b"partial capture")
        raise RuntimeError("simulated hard-crash boundary")

    monkeypatch.setattr(captures, "_populate_private_capture", interrupt_population)
    monkeypatch.setattr(captures, "_reconcile_capture_audio", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="hard-crash boundary"):
        _create(db)

    assert len(list((data_dir / "captures").glob(".voicebox-delete-*"))) == 1
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1
    db.close()

    fresh = session_factory()
    report = deletion_journal.recover_interrupted_deletions(fresh)
    assert report.discarded == 1
    assert list((data_dir / "captures").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    fresh.close()


def test_create_capture_returns_committed_audio_after_ambiguous_error(
    tmp_path,
    monkeypatch,
):
    _data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    response = _create(db)

    row = db.query(Capture).one()
    assert response.id == row.id
    assert response.transcript_raw == "transcript"
    audio = config.resolve_storage_path(row.audio_path)
    assert audio is not None
    assert audio.read_bytes() == _wav_bytes()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_create_capture_returns_durable_response_when_refresh_fails(
    tmp_path,
    monkeypatch,
):
    _data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())
    monkeypatch.setattr(db, "refresh", lambda _row: (_ for _ in ()).throw(RuntimeError("refresh failed")))

    response = _create(db)

    row = db.query(Capture).one()
    assert response.id == row.id
    assert response.audio_path == row.audio_path
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_create_capture_journal_cleanup_failure_does_not_revoke_success(
    tmp_path,
    monkeypatch,
):
    _data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())
    real_finish = deletion_journal.finish_deletion_intent

    def fail_finish(_intent):
        raise RuntimeError("journal directory flush failed")

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", fail_finish)
    response = _create(db)

    assert db.query(Capture).filter_by(id=response.id).one().transcript_raw == "transcript"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", real_finish)
    report = deletion_journal.recover_interrupted_deletions(db)
    assert report.cleared == 1
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_create_capture_does_not_acknowledge_mismatched_durable_row(
    tmp_path,
    monkeypatch,
):
    _data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())
    real_commit = db.commit

    def commit_mutate_then_raise():
        real_commit()
        with session_factory() as concurrent:
            row = concurrent.query(Capture).one()
            row.transcript_raw = "changed after commit"
            concurrent.commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_mutate_then_raise)
    with pytest.raises(RuntimeError, match="connection failed after commit"):
        _create(db)

    assert db.query(Capture).one().transcript_raw == "changed after commit"
    db.close()


def test_create_capture_rollback_failure_uses_durable_ownership(
    tmp_path,
    monkeypatch,
):
    data_dir, _engine, session_factory = _database(tmp_path, monkeypatch)
    db = session_factory()
    monkeypatch.setattr(captures, "get_whisper_model", lambda: _Whisper())
    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("commit failed")))
    monkeypatch.setattr(db, "rollback", lambda: (_ for _ in ()).throw(RuntimeError("rollback failed")))

    with pytest.raises(RuntimeError, match="rollback failed"):
        _create(db)

    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()

    fresh = session_factory()
    report = deletion_journal.recover_interrupted_deletions(fresh)
    assert report.discarded == 0
    assert fresh.query(Capture).count() == 0
    assert list((data_dir / "captures").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    fresh.close()
