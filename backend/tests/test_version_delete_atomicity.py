"""Version deletion keeps database state and managed audio in one commit boundary."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from backend import (
    app as _app,  # noqa: F401
    config,
)
from backend.database import (
    Base,
    Generation,
    GenerationVersion,
    Story,
    StoryItem,
    VoiceProfile,
)
from backend.services import versions as version_service


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    generations_dir = data_dir / "generations"
    generations_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(
        f"sqlite:///{data_dir / 'versions.db'}",
        connect_args={"timeout": 0.01},
    )
    Base.metadata.create_all(engine)
    return generations_dir, engine, sessionmaker(bind=engine)()


def _add_generation(db, *, audio_path: str) -> None:
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Versioned audio",
            language="es",
            audio_path=audio_path,
            status="completed",
        )
    )


@pytest.mark.parametrize("reference_kind", ["story", "derived-version"])
def test_delete_version_refuses_durable_references(monkeypatch, tmp_path, reference_kind):
    generations_dir, _engine, db = _database(tmp_path, monkeypatch)
    source_audio = generations_dir / "source.wav"
    source_audio.write_bytes(b"source remains playable")
    replacement_audio = generations_dir / "replacement.wav"
    replacement_audio.write_bytes(b"replacement")
    _add_generation(db, audio_path="generations/replacement.wav")
    db.add_all(
        [
            GenerationVersion(
                id="source",
                generation_id="generation",
                label="source",
                audio_path="generations/source.wav",
            ),
            GenerationVersion(
                id="replacement",
                generation_id="generation",
                label="replacement",
                audio_path="generations/replacement.wav",
                is_default=True,
            ),
        ]
    )
    if reference_kind == "story":
        db.add(Story(id="story", name="Pinned story"))
        db.add(
            StoryItem(
                id="item",
                story_id="story",
                generation_id="generation",
                version_id="source",
            )
        )
    else:
        db.add(
            GenerationVersion(
                id="derived",
                generation_id="generation",
                label="derived",
                audio_path="generations/derived.wav",
                source_version_id="source",
            )
        )
    db.commit()

    with pytest.raises(version_service.VersionInUseError):
        version_service.delete_version("source", db)

    db.expire_all()
    assert db.query(GenerationVersion).filter_by(id="source").one() is not None
    assert source_audio.read_bytes() == b"source remains playable"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()


def test_default_version_delete_promotes_and_updates_in_one_commit(
    monkeypatch,
    tmp_path,
):
    generations_dir, _engine, db = _database(tmp_path, monkeypatch)
    old_audio = generations_dir / "old-default.wav"
    first_audio = generations_dir / "first.wav"
    second_audio = generations_dir / "second.wav"
    old_audio.write_bytes(b"old default")
    first_audio.write_bytes(b"first replacement")
    second_audio.write_bytes(b"second replacement")
    _add_generation(db, audio_path="generations/old-default.wav")
    created_at = datetime(2026, 1, 1)
    db.add_all(
        [
            GenerationVersion(
                id="old-default",
                generation_id="generation",
                label="old",
                audio_path="generations/old-default.wav",
                is_default=True,
                created_at=created_at,
            ),
            GenerationVersion(
                id="first",
                generation_id="generation",
                label="first",
                audio_path="generations/first.wav",
                created_at=created_at + timedelta(seconds=1),
            ),
            GenerationVersion(
                id="second",
                generation_id="generation",
                label="second",
                audio_path="generations/second.wav",
                # Imported databases may contain more than one default flag.
                is_default=True,
                created_at=created_at + timedelta(seconds=2),
            ),
        ]
    )
    db.commit()

    commit_calls = 0
    real_commit = db.commit

    def counted_commit():
        nonlocal commit_calls
        commit_calls += 1
        return real_commit()

    monkeypatch.setattr(db, "commit", counted_commit)

    assert version_service.delete_version("old-default", db) is True

    assert commit_calls == 1
    assert db.query(GenerationVersion).filter_by(id="old-default").one_or_none() is None
    first = db.query(GenerationVersion).filter_by(id="first").one()
    second = db.query(GenerationVersion).filter_by(id="second").one()
    assert first.is_default is True
    assert second.is_default is False
    assert db.query(Generation).filter_by(id="generation").one().audio_path == ("generations/first.wav")
    assert not old_audio.exists()
    assert first_audio.read_bytes() == b"first replacement"
    assert second_audio.read_bytes() == b"second replacement"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()


def test_default_version_delete_restores_everything_on_database_commit_failure(
    monkeypatch,
    tmp_path,
):
    generations_dir, engine, db = _database(tmp_path, monkeypatch)
    old_audio = generations_dir / "old-default.wav"
    replacement_audio = generations_dir / "replacement.wav"
    old_audio.write_bytes(b"survives failed commit")
    replacement_audio.write_bytes(b"replacement")
    _add_generation(db, audio_path="generations/old-default.wav")
    db.add_all(
        [
            GenerationVersion(
                id="old-default",
                generation_id="generation",
                label="old",
                audio_path="generations/old-default.wav",
                is_default=True,
            ),
            GenerationVersion(
                id="replacement",
                generation_id="generation",
                label="replacement",
                audio_path="generations/replacement.wav",
                is_default=False,
            ),
        ]
    )
    db.commit()

    def fail_database_commit(_connection):
        raise RuntimeError("deliberate database commit failure")

    event.listen(engine, "commit", fail_database_commit)
    try:
        with pytest.raises(RuntimeError, match="deliberate database commit failure"):
            version_service.delete_version("old-default", db)
    finally:
        event.remove(engine, "commit", fail_database_commit)

    db.expire_all()
    old = db.query(GenerationVersion).filter_by(id="old-default").one()
    replacement = db.query(GenerationVersion).filter_by(id="replacement").one()
    assert old.is_default is True
    assert replacement.is_default is False
    assert db.query(Generation).filter_by(id="generation").one().audio_path == ("generations/old-default.wav")
    assert old_audio.read_bytes() == b"survives failed commit"
    assert replacement_audio.read_bytes() == b"replacement"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()


def test_default_delete_preserves_canonical_alias_owned_by_surviving_version(
    monkeypatch,
    tmp_path,
):
    generations_dir, _engine, db = _database(tmp_path, monkeypatch)
    shared_audio = generations_dir / "shared.wav"
    shared_audio.write_bytes(b"shared audio survives")
    _add_generation(db, audio_path="data/generations/shared.wav")
    db.add_all(
        [
            GenerationVersion(
                id="legacy-default",
                generation_id="generation",
                label="legacy alias",
                audio_path="data/generations/shared.wav",
                is_default=True,
            ),
            GenerationVersion(
                id="current-survivor",
                generation_id="generation",
                label="current alias",
                audio_path="generations/shared.wav",
                is_default=False,
            ),
        ]
    )
    db.commit()

    assert version_service.delete_version("legacy-default", db) is True

    survivor = db.query(GenerationVersion).filter_by(id="current-survivor").one()
    assert survivor.is_default is True
    assert db.query(Generation).filter_by(id="generation").one().audio_path == ("generations/shared.wav")
    assert shared_audio.read_bytes() == b"shared audio survives"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()


def test_delete_all_versions_restores_rows_and_audio_when_sqlite_is_locked(
    monkeypatch,
    tmp_path,
):
    generations_dir, engine, db = _database(tmp_path, monkeypatch)
    first_audio = generations_dir / "first.wav"
    second_audio = generations_dir / "second.wav"
    first_audio.write_bytes(b"first survives lock")
    second_audio.write_bytes(b"second survives lock")
    _add_generation(db, audio_path="generations/main.wav")
    db.add_all(
        [
            GenerationVersion(
                id="first",
                generation_id="generation",
                label="first",
                audio_path="generations/first.wav",
            ),
            GenerationVersion(
                id="second",
                generation_id="generation",
                label="second",
                audio_path="generations/second.wav",
            ),
        ]
    )
    db.commit()

    locker = engine.raw_connection()
    try:
        locker.execute("BEGIN IMMEDIATE")
        with pytest.raises(OperationalError):
            version_service.delete_versions_for_generation("generation", db)
    finally:
        locker.rollback()
        locker.close()

    db.expire_all()
    assert {row.id for row in db.query(GenerationVersion).all()} == {"first", "second"}
    assert first_audio.read_bytes() == b"first survives lock"
    assert second_audio.read_bytes() == b"second survives lock"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()


def test_delete_all_versions_unlinks_only_managed_entries(monkeypatch, tmp_path):
    generations_dir, _engine, db = _database(tmp_path, monkeypatch)
    managed_audio = generations_dir / "managed.wav"
    managed_audio.write_bytes(b"managed")
    outside_target = tmp_path / "outside-target.wav"
    outside_target.write_bytes(b"symlink target remains")
    linked_audio = generations_dir / "linked.wav"
    linked_audio.symlink_to(outside_target)
    unmanaged_audio = tmp_path / "unmanaged.wav"
    unmanaged_audio.write_bytes(b"unmanaged remains")
    _add_generation(db, audio_path="generations/main.wav")
    db.add_all(
        [
            GenerationVersion(
                id="managed",
                generation_id="generation",
                label="managed",
                audio_path="generations/managed.wav",
            ),
            GenerationVersion(
                id="linked",
                generation_id="generation",
                label="linked",
                audio_path="generations/linked.wav",
            ),
            GenerationVersion(
                id="unmanaged",
                generation_id="generation",
                label="unmanaged",
                audio_path=str(unmanaged_audio),
            ),
        ]
    )
    db.commit()

    assert version_service.delete_versions_for_generation("generation", db) == 3

    assert db.query(GenerationVersion).count() == 0
    assert not managed_audio.exists()
    assert not linked_audio.exists()
    assert outside_target.read_bytes() == b"symlink target remains"
    assert unmanaged_audio.read_bytes() == b"unmanaged remains"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    db.close()
