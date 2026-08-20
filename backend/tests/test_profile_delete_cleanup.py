"""Profile deletion must not race inference or orphan private generation data."""

import asyncio

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from backend import (
    app as _app,  # noqa: F401
    config,
)
from backend.database import (
    AudioChannel,
    Base,
    Generation,
    GenerationVersion,
    MCPClientBinding,
    ProfileChannelMapping,
    ProfileSample,
    Story,
    StoryItem,
    VoiceProfile,
)
from backend.routes import profiles as profile_routes
from backend.services import profiles as profile_service, task_queue


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'profiles.db'}")
    Base.metadata.create_all(engine)
    return data_dir, engine, sessionmaker(bind=engine)()


def test_profile_delete_removes_bounded_generation_files_and_rows(
    monkeypatch,
    tmp_path,
):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    generations_dir = data_dir / "generations"
    profile_dir.mkdir(parents=True)
    generations_dir.mkdir()
    sample_audio = profile_dir / "sample.wav"
    main_audio = generations_dir / "main.wav"
    original_audio = generations_dir / "original.wav"
    processed_audio = generations_dir / "processed.wav"
    for path, payload in (
        (sample_audio, b"private voice biometrics"),
        (main_audio, b"main generation"),
        (original_audio, b"original generation"),
        (processed_audio, b"processed generation"),
    ):
        path.write_bytes(payload)

    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"not managed by Voicebox")
    linked_audio = generations_dir / "linked.wav"
    linked_audio.symlink_to(outside_audio)
    exact_snapshot = data_dir / "exact_voice_snapshots" / ("raw-" + "a" * 64)
    exact_snapshot.mkdir(parents=True)
    (exact_snapshot / "sample-0000.wav").write_bytes(b"shared immutable bytes")

    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="Transcript",
        )
    )
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Generated text",
            language="es",
            audio_path="generations/main.wav",
            status="completed",
        )
    )
    db.add_all(
        [
            GenerationVersion(
                id="original",
                generation_id="generation",
                label="original",
                audio_path="generations/original.wav",
            ),
            GenerationVersion(
                id="processed",
                generation_id="generation",
                label="processed",
                audio_path="generations/processed.wav",
                is_default=True,
            ),
            GenerationVersion(
                id="linked",
                generation_id="generation",
                label="untrusted-link",
                audio_path="generations/linked.wav",
            ),
            GenerationVersion(
                id="outside",
                generation_id="generation",
                label="unmanaged-path",
                audio_path=str(outside_audio),
            ),
        ]
    )
    db.add(Story(id="story", name="Book"))
    db.add(
        StoryItem(
            id="item",
            story_id="story",
            generation_id="generation",
            version_id="processed",
        )
    )
    db.add(AudioChannel(id="channel", name="Default"))
    db.add(ProfileChannelMapping(profile_id="profile", channel_id="channel"))
    db.add(MCPClientBinding(client_id="client", profile_id="profile"))
    db.commit()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(profile_routes.delete_profile("profile", db=db))

    assert raised.value.status_code == 409
    assert db.query(VoiceProfile).count() == 1
    assert db.query(Generation).count() == 1
    assert db.query(StoryItem).count() == 1
    assert main_audio.read_bytes() == b"main generation"
    assert processed_audio.read_bytes() == b"processed generation"

    db.query(StoryItem).delete()
    db.commit()
    response = asyncio.run(profile_routes.delete_profile("profile", db=db))

    assert "shared immutable voice snapshots are retained" in response["message"]
    assert db.query(VoiceProfile).count() == 0
    assert db.query(ProfileSample).count() == 0
    assert db.query(Generation).count() == 0
    assert db.query(GenerationVersion).count() == 0
    assert db.query(StoryItem).count() == 0
    assert db.query(ProfileChannelMapping).count() == 0
    assert db.query(MCPClientBinding).one().profile_id is None
    assert not profile_dir.exists()
    assert not main_audio.exists()
    assert not original_audio.exists()
    assert not processed_audio.exists()
    assert not linked_audio.exists()
    assert outside_audio.read_bytes() == b"not managed by Voicebox"
    assert exact_snapshot.is_dir()
    db.close()


@pytest.mark.parametrize(
    ("status", "queue_owned"),
    [("completed", True), ("generating", False)],
)
def test_profile_delete_rejects_active_or_shared_batch_generation(
    monkeypatch,
    tmp_path,
    status,
    queue_owned,
):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    generations_dir = data_dir / "generations"
    profile_dir.mkdir(parents=True)
    generations_dir.mkdir()
    sample_audio = profile_dir / "sample.wav"
    generated_audio = generations_dir / "active.wav"
    sample_audio.write_bytes(b"sample")
    generated_audio.write_bytes(b"generation in use")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="Transcript",
        )
    )
    db.add(
        Generation(
            id="batch-member",
            profile_id="profile",
            text="Active text",
            language="es",
            audio_path="generations/active.wav",
            status=status,
        )
    )
    db.commit()
    monkeypatch.setattr(
        task_queue,
        "generation_job_is_active",
        lambda generation_id: queue_owned and generation_id == "batch-member",
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(profile_routes.delete_profile("profile", db=db))

    assert raised.value.status_code == 409
    assert db.query(VoiceProfile).filter_by(id="profile").one_or_none() is not None
    assert db.query(Generation).filter_by(id="batch-member").one_or_none() is not None
    assert generated_audio.read_bytes() == b"generation in use"
    assert sample_audio.read_bytes() == b"sample"
    db.close()


def test_profile_delete_restores_audio_when_sqlite_transaction_is_locked(
    monkeypatch,
    tmp_path,
):
    data_dir, engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    generations_dir = data_dir / "generations"
    profile_dir.mkdir(parents=True)
    generations_dir.mkdir()
    sample_audio = profile_dir / "sample.wav"
    generated_audio = generations_dir / "locked.wav"
    version_audio = generations_dir / "locked-original.wav"
    sample_audio.write_bytes(b"sample survives rollback")
    generated_audio.write_bytes(b"generation survives rollback")
    version_audio.write_bytes(b"version survives rollback")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="Transcript",
        )
    )
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Locked text",
            language="es",
            audio_path="generations/locked.wav",
            status="completed",
        )
    )
    db.add(
        GenerationVersion(
            id="version",
            generation_id="generation",
            label="original",
            audio_path="generations/locked-original.wav",
        )
    )
    db.commit()

    locker = engine.raw_connection()
    try:
        locker.execute("BEGIN IMMEDIATE")
        with pytest.raises(OperationalError):
            asyncio.run(profile_routes.delete_profile("profile", db=db))
    finally:
        locker.rollback()
        locker.close()

    assert db.query(VoiceProfile).filter_by(id="profile").one_or_none() is not None
    assert db.query(Generation).filter_by(id="generation").one_or_none() is not None
    assert db.query(GenerationVersion).filter_by(id="version").one_or_none() is not None
    assert sample_audio.read_bytes() == b"sample survives rollback"
    assert generated_audio.read_bytes() == b"generation survives rollback"
    assert version_audio.read_bytes() == b"version survives rollback"
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    assert profile_dir.is_dir()
    db.close()


def test_profile_delete_restores_profile_and_audio_when_commit_fails(
    monkeypatch,
    tmp_path,
):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    generations_dir = data_dir / "generations"
    profile_dir.mkdir(parents=True)
    generations_dir.mkdir()
    sample_audio = profile_dir / "sample.wav"
    generated_audio = generations_dir / "commit.wav"
    sample_audio.write_bytes(b"sample survives failed commit")
    generated_audio.write_bytes(b"generation survives failed commit")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="Transcript",
        )
    )
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Commit failure",
            language="es",
            audio_path="generations/commit.wav",
            status="completed",
        )
    )
    db.commit()

    def fail_commit():
        raise OperationalError("COMMIT", {}, RuntimeError("deliberate failure"))

    monkeypatch.setattr(db, "commit", fail_commit)
    with pytest.raises(OperationalError):
        asyncio.run(profile_routes.delete_profile("profile", db=db))

    assert db.query(VoiceProfile).filter_by(id="profile").one_or_none() is not None
    assert db.query(Generation).filter_by(id="generation").one_or_none() is not None
    assert sample_audio.read_bytes() == b"sample survives failed commit"
    assert generated_audio.read_bytes() == b"generation survives failed commit"
    assert profile_dir.is_dir()
    assert not list(generations_dir.glob(".voicebox-delete-*.tmp"))
    assert not list(profile_dir.parent.glob(".voicebox-delete-*"))
    db.close()


def test_profile_delete_rejects_glob_capable_storage_id(monkeypatch, tmp_path):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir()
    unrelated_cache = cache_dir / "combined_unrelated_voice.wav"
    unrelated_cache.write_bytes(b"must remain")
    db.add(VoiceProfile(id="*", name="Unsafe legacy row", voice_type="cloned"))
    db.commit()

    with pytest.raises(ValueError, match="unsafe for managed storage cleanup"):
        asyncio.run(profile_service.delete_profile("*", db))

    assert db.query(VoiceProfile).filter_by(id="*").one_or_none() is not None
    assert unrelated_cache.read_bytes() == b"must remain"
    db.close()


def test_profile_sample_delete_restores_audio_when_commit_fails(
    monkeypatch,
    tmp_path,
):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir(parents=True)
    sample_audio = profile_dir / "sample.wav"
    sample_audio.write_bytes(b"voice sample survives failed commit")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="Transcript",
        )
    )
    db.commit()

    def fail_commit():
        raise OperationalError("COMMIT", {}, RuntimeError("deliberate failure"))

    monkeypatch.setattr(db, "commit", fail_commit)
    with pytest.raises(OperationalError):
        asyncio.run(profile_service.delete_profile_sample("sample", db))

    assert db.query(ProfileSample).filter_by(id="sample").one_or_none() is not None
    assert sample_audio.read_bytes() == b"voice sample survives failed commit"
    assert not list(profile_dir.glob(".voicebox-delete-sample-*.tmp"))
    db.close()


def test_profile_sample_delete_never_follows_or_removes_unmanaged_audio(
    monkeypatch,
    tmp_path,
):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir(parents=True)
    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"external voice data")
    linked_audio = profile_dir / "linked.wav"
    linked_audio.symlink_to(outside_audio)
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add_all(
        [
            ProfileSample(
                id="linked",
                profile_id="profile",
                ordinal=0,
                audio_path="profiles/profile/linked.wav",
                reference_text="Linked",
            ),
            ProfileSample(
                id="outside",
                profile_id="profile",
                ordinal=1,
                audio_path=str(outside_audio),
                reference_text="Outside",
            ),
        ]
    )
    db.commit()

    assert asyncio.run(profile_service.delete_profile_sample("linked", db)) is True
    assert not linked_audio.exists()
    assert outside_audio.read_bytes() == b"external voice data"
    assert asyncio.run(profile_service.delete_profile_sample("outside", db)) is True
    assert outside_audio.read_bytes() == b"external voice data"
    assert db.query(ProfileSample).count() == 0
    db.close()


@pytest.mark.parametrize("delete_kind", ["sample", "profile"])
def test_profile_deletions_restore_staged_audio_even_when_rollback_fails(
    monkeypatch,
    tmp_path,
    delete_kind,
):
    data_dir, _engine, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir(parents=True)
    sample_audio = profile_dir / "sample.wav"
    sample_audio.write_bytes(b"restore despite rollback failure")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="Transcript",
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
    operation = (
        profile_service.delete_profile_sample("sample", db)
        if delete_kind == "sample"
        else profile_service.delete_profile("profile", db)
    )
    with pytest.raises(RuntimeError, match="rollback failure"):
        asyncio.run(operation)

    assert sample_audio.read_bytes() == b"restore despite rollback failure"
    assert not list(profile_dir.glob(".voicebox-delete-sample-*.tmp"))
    assert not list(profile_dir.parent.glob(".voicebox-delete-profile-*"))
    real_rollback()
    assert db.query(ProfileSample).filter_by(id="sample").one_or_none() is not None
    assert db.query(VoiceProfile).filter_by(id="profile").one_or_none() is not None
    db.close()
