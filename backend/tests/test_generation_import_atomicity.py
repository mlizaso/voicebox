"""Durability and cleanup regressions for imported generation audio."""

from __future__ import annotations

import asyncio
import io
import threading
import wave

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, Generation
from backend.routes import generations


class _Upload:
    def __init__(self, payload: bytes, filename: str = "clip.wav") -> None:
        self.filename = filename
        self._payload = io.BytesIO(payload)

    async def read(self, size: int = -1) -> bytes:
        return self._payload.read(size)


def _wav_bytes(*, channels: int = 1, sample_rate: int = 24_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0\0" * channels * max(1, sample_rate // 10))
    return output.getvalue()


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    return data_dir, sessionmaker(bind=engine)()


def test_import_audio_removes_durable_file_when_row_creation_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    async def fail_create_generation(**_kwargs):
        raise RuntimeError("database write failed")

    monkeypatch.setattr(generations.history, "create_generation", fail_create_generation)

    with pytest.raises(RuntimeError, match="database write failed"):
        asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert db.query(Generation).count() == 0
    assert list((data_dir / "generations").iterdir()) == []
    db.close()


def test_import_audio_stream_limit_leaves_no_partial_file(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(generations, "IMPORT_AUDIO_MAX_BYTES", 4)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.import_audio(_Upload(b"12345"), db))

    assert raised.value.status_code == 413
    assert list((data_dir / "generations").iterdir()) == []
    db.close()


def test_import_audio_publishes_valid_file_and_completed_row(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    response = asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    row = db.query(Generation).filter_by(id=response.id).one()
    audio = config.resolve_storage_path(row.audio_path)
    assert audio is not None
    assert audio.is_file()
    assert audio.stat().st_size == len(_wav_bytes())
    assert row.status == "completed"
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_import_audio_probes_duration_off_loop_without_decoding_pcm(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    event_loop_thread = threading.get_ident()
    observed = {}

    def probe_duration(path):
        observed.update(path=path, thread=threading.get_ident())
        return 0.1, 1, 24_000

    monkeypatch.setattr(generations, "probe_audio_metadata", probe_duration)

    response = asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert response.duration == pytest.approx(0.1)
    assert observed["thread"] != event_loop_thread
    assert str(observed["path"]).endswith(".part.wav")
    db.close()


def test_import_audio_rejects_overlong_compressed_media_without_pcm_decode(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(generations, "PORTABLE_AUDIO_MAX_DURATION_SECONDS", 30)
    monkeypatch.setattr(
        generations,
        "probe_audio_metadata",
        lambda _path: (31.0, 1, 24_000),
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert raised.value.status_code == 413
    assert list((data_dir / "generations").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        (_wav_bytes(channels=9), "8-channel"),
        (_wav_bytes(sample_rate=192_001), "192000 Hz"),
    ],
)
def test_import_audio_rejects_media_story_and_effects_cannot_process(
    tmp_path,
    monkeypatch,
    payload,
    expected_detail,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.import_audio(_Upload(payload), db))

    assert raised.value.status_code == 413
    assert expected_detail in str(raised.value.detail)
    assert db.query(Generation).count() == 0
    assert list((data_dir / "generations").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_import_audio_returns_committed_row_after_outcome_ambiguous_error(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    real_create_generation = generations.history.create_generation

    async def commit_then_raise(**kwargs):
        await real_create_generation(**kwargs)
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(generations.history, "create_generation", commit_then_raise)

    response = asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    row = db.query(Generation).one()
    assert response.id == row.id
    assert response.text == "clip"
    assert response.engine == "import"
    audio = config.resolve_storage_path(row.audio_path)
    assert audio is not None
    assert audio.read_bytes() == _wav_bytes()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_import_audio_journal_cleanup_failure_does_not_revoke_success(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    real_finish = generations.deletion_journal.finish_deletion_intent

    def fail_finish(_intent):
        raise RuntimeError("journal directory flush failed")

    monkeypatch.setattr(
        generations.deletion_journal,
        "finish_deletion_intent",
        fail_finish,
    )
    response = asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert db.query(Generation).filter_by(id=response.id).one().status == "completed"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    monkeypatch.setattr(
        generations.deletion_journal,
        "finish_deletion_intent",
        real_finish,
    )
    report = generations.deletion_journal.recover_interrupted_deletions(db)
    assert report.cleared == 1
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    assert len(list((data_dir / "generations").glob("*.wav"))) == 1
    db.close()


def test_import_audio_does_not_acknowledge_mismatched_durable_row(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    real_create_generation = generations.history.create_generation

    async def commit_mutate_then_raise(**kwargs):
        response = await real_create_generation(**kwargs)
        with sessionmaker(bind=db.get_bind())() as concurrent:
            row = concurrent.query(Generation).filter_by(id=response.id).one()
            row.text = "changed after commit"
            concurrent.commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(
        generations.history,
        "create_generation",
        commit_mutate_then_raise,
    )
    with pytest.raises(RuntimeError, match="connection failed after commit"):
        asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert db.query(Generation).one().text == "changed after commit"
    db.close()


def test_import_audio_cleans_post_rename_failure_without_losing_intent(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    real_rename = generations.deletion_journal.rename_managed_entry

    def rename_then_raise(source, destination):
        real_rename(source, destination)
        raise OSError("directory fsync failed")

    monkeypatch.setattr(
        generations.deletion_journal,
        "rename_managed_entry",
        rename_then_raise,
    )

    with pytest.raises(OSError, match="directory fsync failed"):
        asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert db.query(Generation).count() == 0
    assert list((data_dir / "generations").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_import_audio_journals_empty_inode_before_streaming(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    class _FailingUpload(_Upload):
        async def read(self, size: int = -1) -> bytes:
            assert list(config.get_deletion_journal_dir().glob("*.json"))
            assert len(list((data_dir / "generations").glob(".voicebox-delete-*"))) == 1
            raise RuntimeError("stream interrupted")

    with pytest.raises(RuntimeError, match="stream interrupted"):
        asyncio.run(generations.import_audio(_FailingUpload(_wav_bytes()), db))

    assert list((data_dir / "generations").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_import_audio_uses_durable_owner_view_when_rollback_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    async def fail_create_generation(**_kwargs):
        raise RuntimeError("database write failed")

    def fail_rollback():
        raise RuntimeError("rollback failed")

    monkeypatch.setattr(generations.history, "create_generation", fail_create_generation)
    monkeypatch.setattr(db, "rollback", fail_rollback)

    with pytest.raises(RuntimeError, match="rollback failed"):
        asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    assert list((data_dir / "generations").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_import_audio_retains_intent_when_durable_ownership_is_unavailable(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    engine = db.get_bind()

    async def fail_create_generation(**_kwargs):
        raise RuntimeError("database write failed")

    def fail_rollback():
        raise RuntimeError("rollback failed")

    def fail_durable_session(_db):
        raise RuntimeError("durable database unavailable")

    monkeypatch.setattr(generations.history, "create_generation", fail_create_generation)
    monkeypatch.setattr(db, "rollback", fail_rollback)
    monkeypatch.setattr(
        generations.deletion_journal,
        "durable_reconciliation_session",
        fail_durable_session,
    )

    with pytest.raises(RuntimeError, match="rollback failed"):
        asyncio.run(generations.import_audio(_Upload(_wav_bytes()), db))

    published = list((data_dir / "generations").glob("*.wav"))
    assert len(published) == 1
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    db.close()
    recovery_db = sessionmaker(bind=engine)()
    monkeypatch.undo()
    monkeypatch.setattr(config, "_data_dir", data_dir)
    report = generations.deletion_journal.recover_interrupted_deletions(recovery_db)
    assert report.discarded == 1
    assert not published[0].exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    recovery_db.close()
