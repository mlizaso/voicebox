"""Race and durability regressions for effects-version publication."""

from __future__ import annotations

import asyncio
import os

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config, models
from backend.database import Base, Generation, GenerationVersion, VoiceProfile
from backend.routes import effects as effects_routes
from backend.services import deletion_journal, effects_processing, versions
from backend.utils import audio as audio_utils, effects as effects_utils


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    db = session_factory()
    source = data_dir / "generations" / "source.wav"
    source.write_bytes(b"source")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="text",
            language="en",
            audio_path="generations/source.wav",
            status="completed",
        )
    )
    db.commit()
    return data_dir, session_factory, db


def _patch_effects(monkeypatch, populate):
    async def render(_source_path, output_descriptor, _output_directory, _chain):
        assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1
        pending = next(config.get_generations_dir().glob(".voicebox-delete-effects-*.part.wav"))
        assert pending.stat().st_ino == os.fstat(output_descriptor).st_ino
        populate(output_descriptor)

    monkeypatch.setattr(effects_processing, "render_effects_to_descriptor", render)
    monkeypatch.setattr(effects_utils, "validate_effects_chain", lambda _chain: None)


def _request():
    return models.ApplyEffectsRequest(effects_chain=[], set_as_default=False)


def test_effects_wav_population_preserves_journaled_inode(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    source = tmp_path / "source.wav"
    audio_utils.save_audio(np.zeros(2_400, dtype=np.float32), str(source), 24_000)
    path = tmp_path / "pending.wav"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    expected_stat = path.stat()

    asyncio.run(effects_processing.render_effects_to_descriptor(source, descriptor, tmp_path, []))
    os.close(descriptor)

    assert path.stat().st_ino == expected_stat.st_ino
    audio, sample_rate = audio_utils.load_audio(str(path))
    assert sample_rate == 24_000
    assert len(audio) == 2_400


def test_apply_effects_rechecks_parent_after_final_await(tmp_path, monkeypatch):
    data_dir, session_factory, db = _database(tmp_path, monkeypatch)

    def save_then_delete(descriptor):
        os.write(descriptor, b"processed")
        other = session_factory()
        other.query(Generation).filter_by(id="generation").delete()
        other.commit()
        other.close()

    _patch_effects(monkeypatch, save_then_delete)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            effects_routes.apply_effects_to_generation(
                "generation",
                _request(),
                db,
            )
        )

    assert raised.value.status_code == 404
    assert db.query(GenerationVersion).count() == 0
    assert not list((data_dir / "generations").glob("generation_*.wav"))
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_apply_effects_preserves_committed_version_after_ambiguous_error(
    tmp_path,
    monkeypatch,
):
    _data_dir, _session_factory, db = _database(tmp_path, monkeypatch)

    def save_processed(descriptor):
        os.write(descriptor, b"processed")

    _patch_effects(monkeypatch, save_processed)
    real_create_version = versions.create_version

    def commit_then_raise(**kwargs):
        real_create_version(**kwargs)
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(versions, "create_version", commit_then_raise)

    with pytest.raises(RuntimeError, match="connection failed after commit"):
        asyncio.run(
            effects_routes.apply_effects_to_generation(
                "generation",
                _request(),
                db,
            )
        )

    row = db.query(GenerationVersion).one()
    processed = config.resolve_storage_path(row.audio_path)
    assert processed is not None
    assert processed.read_bytes() == b"processed"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_apply_effects_acknowledges_commit_when_journal_cleanup_fails(
    tmp_path,
    monkeypatch,
):
    _data_dir, _session_factory, db = _database(tmp_path, monkeypatch)

    def save_processed(descriptor):
        os.write(descriptor, b"processed")

    _patch_effects(monkeypatch, save_processed)
    real_finish = deletion_journal.finish_deletion_intent

    def fail_cleanup(_intent):
        raise OSError("journal directory flush failed")

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", fail_cleanup)

    version = asyncio.run(
        effects_routes.apply_effects_to_generation(
            "generation",
            _request(),
            db,
        )
    )

    row = db.query(GenerationVersion).one()
    processed = config.resolve_storage_path(row.audio_path)
    assert version.id == row.id
    assert processed is not None
    assert processed.read_bytes() == b"processed"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", real_finish)
    report = deletion_journal.recover_interrupted_deletions(db)
    assert report.cleared == 1
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    assert processed.read_bytes() == b"processed"
    db.close()


def test_apply_effects_reconciles_an_ambiguous_publish_rename(
    tmp_path,
    monkeypatch,
):
    data_dir, _session_factory, db = _database(tmp_path, monkeypatch)

    def save_processed(descriptor):
        os.write(descriptor, b"processed")

    _patch_effects(monkeypatch, save_processed)
    real_rename = deletion_journal.rename_managed_entry

    def rename_then_raise(source, destination):
        real_rename(source, destination)
        raise OSError("generation directory flush failed")

    monkeypatch.setattr(deletion_journal, "rename_managed_entry", rename_then_raise)

    with pytest.raises(OSError, match="generation directory flush failed"):
        asyncio.run(
            effects_routes.apply_effects_to_generation(
                "generation",
                _request(),
                db,
            )
        )

    assert db.query(GenerationVersion).count() == 0
    assert not list((data_dir / "generations").glob("generation_*.wav"))
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()
