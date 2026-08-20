"""History deletion must not race queued or executing generation work."""

import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

# Importing the application first lets its router registration finish before
# this test addresses the history module directly.
from backend import (
    app as _app,  # noqa: F401
    config,
)
from backend.database import Base, Generation, GenerationVersion, Story, StoryItem, VoiceProfile
from backend.routes import history as history_routes


def test_history_delete_rejects_member_still_owned_by_shared_batch(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'history-delete.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    row = Generation(
        id="batch-member",
        profile_id="profile",
        text="Already persisted but shared worker is still exiting",
        language="es",
        status="completed",
    )
    db.add(row)
    db.commit()
    shared_ids = {"batch-member", "batch-peer"}
    monkeypatch.setattr(
        history_routes,
        "generation_job_is_active",
        lambda generation_id: generation_id in shared_ids,
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(history_routes.delete_generation(row.id, db=db))

    assert raised.value.status_code == 409
    assert db.query(Generation).filter_by(id=row.id).one_or_none() is row
    db.close()


def test_history_delete_removes_committed_managed_audio_and_versions(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    generations_dir = data_dir / "generations"
    generations_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'history-delete.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    main_audio = generations_dir / "main.wav"
    version_audio = generations_dir / "version.wav"
    main_audio.write_bytes(b"main")
    version_audio.write_bytes(b"version")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Completed generation",
            language="es",
            audio_path="generations/main.wav",
            status="completed",
        )
    )
    db.add(
        GenerationVersion(
            id="version",
            generation_id="generation",
            label="original",
            audio_path="generations/version.wav",
        )
    )
    db.commit()

    response = asyncio.run(history_routes.delete_generation("generation", db=db))

    assert response == {"message": "Generation deleted successfully"}
    assert db.query(Generation).count() == 0
    assert db.query(GenerationVersion).count() == 0
    assert not main_audio.exists()
    assert not version_audio.exists()
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()


def test_history_delete_rejects_generation_used_by_story(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    generations_dir = data_dir / "generations"
    generations_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'history-delete.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    audio = generations_dir / "generation.wav"
    audio.write_bytes(b"story-owned audio")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Story narration",
            language="es",
            audio_path="generations/generation.wav",
            status="completed",
        )
    )
    db.add(Story(id="story", name="Book"))
    db.add(
        StoryItem(
            id="item",
            story_id="story",
            generation_id="generation",
            start_time_ms=1234,
            track=2,
            trim_start_ms=100,
            trim_end_ms=200,
        )
    )
    db.commit()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(history_routes.delete_generation("generation", db=db))

    assert raised.value.status_code == 409
    item = db.query(StoryItem).filter_by(id="item").one()
    assert (item.start_time_ms, item.track, item.trim_start_ms, item.trim_end_ms) == (1234, 2, 100, 200)
    assert db.query(Generation).filter_by(id="generation").one_or_none() is not None
    assert audio.read_bytes() == b"story-owned audio"
    db.close()
    engine.dispose()


def test_history_delete_restores_audio_even_when_rollback_fails(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    generations_dir = data_dir / "generations"
    generations_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'history-delete.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    audio = generations_dir / "generation.wav"
    audio.write_bytes(b"restore despite rollback failure")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Completed generation",
            language="es",
            audio_path="generations/generation.wav",
            status="completed",
        )
    )
    db.commit()
    real_rollback = db.rollback

    def fail_commit():
        raise OperationalError("COMMIT", {}, RuntimeError("deliberate commit failure"))

    def fail_rollback():
        raise RuntimeError("deliberate rollback failure")

    monkeypatch.setattr(db, "commit", fail_commit)
    monkeypatch.setattr(db, "rollback", fail_rollback)
    with pytest.raises(RuntimeError, match="rollback failure"):
        asyncio.run(history_routes.delete_generation("generation", db=db))

    assert audio.read_bytes() == b"restore despite rollback failure"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    real_rollback()
    assert db.query(Generation).filter_by(id="generation").one_or_none() is not None
    db.close()
