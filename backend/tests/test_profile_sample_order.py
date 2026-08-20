"""Deterministic ordering for multi-sample cloned voice profiles."""

import asyncio
import io
import json
import os
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, ProfileSample, VoiceProfile
from backend.database.migrations import run_migrations
from backend.services import export_import, profiles


def _write_profile_sample_payload(descriptor: int, payload: bytes = b"wav") -> None:
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.write(descriptor, payload)
    os.fsync(descriptor)


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def test_migration_freezes_legacy_row_insertion_order(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE profile_samples ("
                "id VARCHAR PRIMARY KEY, profile_id VARCHAR NOT NULL, "
                "audio_path VARCHAR NOT NULL, reference_text TEXT NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO profile_samples "
                "(id, profile_id, audio_path, reference_text) VALUES "
                "('z-id', 'profile-a', 'z.wav', 'first'), "
                "('x-id', 'profile-b', 'x.wav', 'other first'), "
                "('a-id', 'profile-a', 'a.wav', 'second'), "
                "('m-id', 'profile-a', 'm.wav', 'third'), "
                "('b-id', 'profile-b', 'b.wav', 'other second')"
            )
        )

    run_migrations(engine)

    ordinal_column = next(
        column for column in inspect(engine).get_columns("profile_samples") if column["name"] == "ordinal"
    )
    assert ordinal_column["nullable"] is False
    with engine.connect() as connection:
        rows = connection.execute(
            text("SELECT id, profile_id, ordinal FROM profile_samples ORDER BY profile_id, ordinal")
        ).fetchall()
    assert rows == [
        ("z-id", "profile-a", 0),
        ("a-id", "profile-a", 1),
        ("m-id", "profile-a", 2),
        ("x-id", "profile-b", 0),
        ("b-id", "profile-b", 1),
    ]

    # Re-running startup migrations must not renumber surviving samples.
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM profile_samples WHERE id = 'a-id'"))
    run_migrations(engine)
    with engine.connect() as connection:
        remaining = connection.execute(
            text("SELECT id, ordinal FROM profile_samples WHERE profile_id = 'profile-a' ORDER BY ordinal")
        ).fetchall()
    assert remaining == [("z-id", 0), ("m-id", 2)]

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO profile_samples "
                "(id, profile_id, ordinal, audio_path, reference_text) "
                "VALUES ('duplicate-order', 'profile-a', 2, 'd.wav', 'duplicate')"
            )
        )


@pytest.mark.asyncio
async def test_add_rechecks_sample_limit_after_audio_await(db, tmp_path, monkeypatch):
    profile_id = "profile-id"
    db.add(VoiceProfile(id=profile_id, name="Narrator"))
    profile_dir = tmp_path / "profiles" / profile_id
    profile_dir.mkdir(parents=True)
    shared_audio = profile_dir / "shared.wav"
    shared_audio.write_bytes(b"existing")
    for ordinal in range(profiles.EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES - 1):
        db.add(
            ProfileSample(
                id=f"existing-{ordinal}",
                profile_id=profile_id,
                ordinal=ordinal,
                audio_path=f"profiles/{profile_id}/shared.wav",
                reference_text="existing",
            )
        )
    db.commit()
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        lambda descriptor, _audio, _sample_rate, _intent: _write_profile_sample_payload(descriptor),
    )

    class FakeUUID(str):
        @property
        def hex(self) -> str:
            return str(self).replace("-", "")

    monkeypatch.setattr(
        profiles,
        "uuid",
        SimpleNamespace(uuid4=lambda: FakeUUID("new-id")),
    )

    original_rollback = db.rollback
    inserted = False

    def rollback_then_win_concurrent_slot():
        nonlocal inserted
        original_rollback()
        if inserted:
            return
        inserted = True
        concurrent = sessionmaker(bind=db.get_bind())()
        try:
            concurrent.add(
                ProfileSample(
                    id="concurrent-id",
                    profile_id=profile_id,
                    ordinal=profiles.EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES - 1,
                    audio_path=f"profiles/{profile_id}/shared.wav",
                    reference_text="concurrent",
                )
            )
            concurrent.commit()
        finally:
            concurrent.close()

    monkeypatch.setattr(db, "rollback", rollback_then_win_concurrent_slot)

    with pytest.raises(ValueError, match="at most 64"):
        await profiles.add_profile_sample(
            profile_id,
            str(tmp_path / "input.wav"),
            "new",
            db,
        )

    db.expire_all()
    assert db.query(ProfileSample).filter_by(profile_id=profile_id).count() == 64
    assert not (profile_dir / "new-id.wav").exists()


@pytest.mark.asyncio
async def test_add_and_list_samples_use_append_only_ordinals(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()

    class FakeUUID(str):
        @property
        def hex(self) -> str:
            return str(self).replace("-", "")

    sample_ids = iter(FakeUUID(value) for value in ("z-id", "a-id", "m-id", "delete-stage-id", "new-id"))
    monkeypatch.setattr(
        profiles,
        "uuid",
        SimpleNamespace(uuid4=lambda: next(sample_ids)),
    )
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        lambda descriptor, _audio, _sample_rate, _intent: _write_profile_sample_payload(descriptor),
    )
    monkeypatch.setattr(profiles, "clear_profile_cache", lambda _profile_id: None)

    created = []
    for text_value in ("first", "second", "third"):
        created.append(
            await profiles.add_profile_sample(
                "profile-id",
                str(tmp_path / "input.wav"),
                text_value,
                db,
            )
        )

    assert [sample.id for sample in created] == ["z-id", "a-id", "m-id"]
    assert [
        (sample.id, sample.ordinal) for sample in db.query(ProfileSample).order_by(ProfileSample.ordinal).all()
    ] == [("z-id", 0), ("a-id", 1), ("m-id", 2)]

    assert await profiles.delete_profile_sample("a-id", db) is True
    await profiles.add_profile_sample(
        "profile-id",
        str(tmp_path / "input.wav"),
        "fourth",
        db,
    )

    listed = await profiles.get_profile_samples("profile-id", db)
    assert [(sample.id, sample.reference_text) for sample in listed] == [
        ("z-id", "first"),
        ("m-id", "third"),
        ("new-id", "fourth"),
    ]
    assert [sample.ordinal for sample in db.query(ProfileSample).order_by(ProfileSample.ordinal).all()] == [0, 2, 3]


@pytest.mark.asyncio
async def test_add_retries_a_concurrent_ordinal_collision(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.add(
        ProfileSample(
            id="existing-id",
            profile_id="profile-id",
            ordinal=0,
            audio_path="profiles/profile-id/existing.wav",
            reference_text="existing",
        )
    )
    db.commit()

    attempted_ordinals = iter([0, 1])
    monkeypatch.setattr(
        profiles,
        "_next_profile_sample_ordinal",
        lambda _profile_id, _db: next(attempted_ordinals),
    )
    monkeypatch.setattr(
        profiles,
        "uuid",
        SimpleNamespace(uuid4=lambda: "new-id"),
    )
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    existing_dir = tmp_path / "profiles" / "profile-id"
    existing_dir.mkdir(parents=True)
    (existing_dir / "existing.wav").write_bytes(b"existing")
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        lambda descriptor, _audio, _sample_rate, _intent: _write_profile_sample_payload(descriptor),
    )
    monkeypatch.setattr(profiles, "clear_profile_cache", lambda _profile_id: None)

    created = await profiles.add_profile_sample(
        "profile-id",
        str(tmp_path / "input.wav"),
        "new",
        db,
    )

    assert created.id == "new-id"
    assert db.query(ProfileSample).filter_by(id="new-id").one().ordinal == 1


@pytest.mark.asyncio
async def test_add_revalidates_profile_after_audio_save_and_removes_orphan(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"
    session_factory = sessionmaker(bind=db.get_bind())

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )

    def save_then_delete_profile(descriptor, _audio, _sample_rate, _intent):
        _write_profile_sample_payload(descriptor)
        with session_factory() as concurrent_db:
            concurrent_profile = concurrent_db.query(VoiceProfile).filter_by(id="profile-id").one()
            concurrent_db.delete(concurrent_profile)
            concurrent_db.commit()

    monkeypatch.setattr(profiles, "_populate_profile_sample_audio", save_then_delete_profile)

    with pytest.raises(ValueError, match="Profile profile-id not found"):
        await profiles.add_profile_sample(
            "profile-id",
            str(tmp_path / "input.wav"),
            "new",
            db,
        )

    assert db.query(VoiceProfile).filter_by(id="profile-id").one_or_none() is None
    assert db.query(ProfileSample).count() == 0
    assert not (data_dir / "profiles" / "profile-id").exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_add_removes_unowned_audio_after_commit_failure(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        lambda descriptor, _audio, _sample_rate, _intent: _write_profile_sample_payload(descriptor),
    )
    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("commit failed")))

    with pytest.raises(RuntimeError, match="commit failed"):
        await profiles.add_profile_sample(
            "profile-id",
            str(tmp_path / "input.wav"),
            "new",
            db,
        )

    assert db.query(ProfileSample).count() == 0
    assert list((data_dir / "profiles" / "profile-id").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_add_returns_committed_sample_after_outcome_ambiguous_commit(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        lambda descriptor, _audio, _sample_rate, _intent: _write_profile_sample_payload(
            descriptor,
            b"committed wav",
        ),
    )
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)

    created = await profiles.add_profile_sample(
        "profile-id",
        str(tmp_path / "input.wav"),
        "new",
        db,
    )

    sample = db.query(ProfileSample).filter_by(id="new-id").one()
    assert created.id == sample.id
    assert db.query(ProfileSample).count() == 1
    audio_path = config.resolve_storage_path(sample.audio_path)
    assert audio_path is not None
    assert audio_path.read_bytes() == b"committed wav"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_add_returns_committed_sample_after_refresh_failure(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        lambda descriptor, _audio, _sample_rate, _intent: _write_profile_sample_payload(
            descriptor,
            b"committed wav",
        ),
    )
    monkeypatch.setattr(
        db,
        "refresh",
        lambda _sample: (_ for _ in ()).throw(RuntimeError("refresh failed after commit")),
    )

    created = await profiles.add_profile_sample(
        "profile-id",
        str(tmp_path / "input.wav"),
        "new",
        db,
    )

    sample = db.query(ProfileSample).filter_by(id="new-id").one()
    assert created.id == sample.id
    assert db.query(ProfileSample).count() == 1
    audio_path = config.resolve_storage_path(sample.audio_path)
    assert audio_path is not None
    assert audio_path.read_bytes() == b"committed wav"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_add_drains_cancelled_audio_save_before_cleanup(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"
    save_started = threading.Event()
    release_save = threading.Event()

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )

    def delayed_save(descriptor, _audio, _sample_rate, _intent):
        save_started.set()
        assert release_save.wait(timeout=5)
        _write_profile_sample_payload(descriptor, b"late wav")

    monkeypatch.setattr(profiles, "_populate_profile_sample_audio", delayed_save)
    task = asyncio.create_task(
        profiles.add_profile_sample(
            "profile-id",
            str(tmp_path / "input.wav"),
            "new",
            db,
        )
    )
    assert await asyncio.to_thread(save_started.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    release_save.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert db.query(ProfileSample).count() == 0
    assert list((data_dir / "profiles" / "profile-id").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_add_journals_empty_inode_before_encoding_private_audio(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"
    observed = {}

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(16, dtype=np.float32), 24000),
    )

    def fail_during_encode(descriptor, _audio, _sample_rate, intent):
        observed["journal_exists"] = (config.get_deletion_journal_dir() / intent.journal_name).is_file()
        observed["staged_size"] = os.fstat(descriptor).st_size
        raise RuntimeError("encoding interrupted")

    monkeypatch.setattr(
        profiles,
        "_populate_profile_sample_audio",
        fail_during_encode,
    )

    with pytest.raises(RuntimeError, match="encoding interrupted"):
        await profiles.add_profile_sample(
            "profile-id",
            str(tmp_path / "input.wav"),
            "new",
            db,
        )

    assert observed == {"journal_exists": True, "staged_size": 0}
    assert db.query(ProfileSample).count() == 0
    assert list((data_dir / "profiles" / "profile-id").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_add_encodes_valid_wav_into_the_journaled_inode(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator"))
    db.commit()
    data_dir = tmp_path / "data"

    monkeypatch.setattr(config, "_data_dir", data_dir)
    monkeypatch.setattr(profiles, "uuid", SimpleNamespace(uuid4=lambda: "new-id"))
    monkeypatch.setattr(
        profiles,
        "validate_and_load_reference_audio",
        lambda _path: (True, None, np.zeros(2400, dtype=np.float32), 24000),
    )
    monkeypatch.setattr(profiles, "clear_profile_cache", lambda _profile_id: None)

    created = await profiles.add_profile_sample(
        "profile-id",
        str(tmp_path / "input.wav"),
        "new",
        db,
    )

    stored = config.resolve_storage_path(created.audio_path)
    assert stored is not None
    info = sf.info(stored)
    assert info.samplerate == 24000
    assert info.channels == 1
    assert info.frames == 2400
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_voice_prompt_uses_ordinal_instead_of_sample_uuid(
    db,
    tmp_path,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile-id", name="Narrator", voice_type="cloned"))
    db.add_all(
        [
            ProfileSample(
                id="a-random-id",
                profile_id="profile-id",
                ordinal=1,
                audio_path="second.wav",
                reference_text="second",
            ),
            ProfileSample(
                id="z-random-id",
                profile_id="profile-id",
                ordinal=0,
                audio_path="first.wav",
                reference_text="first",
            ),
        ]
    )
    db.commit()
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    (tmp_path / "first.wav").write_bytes(b"first-reference")
    (tmp_path / "second.wav").write_bytes(b"second-reference")
    monkeypatch.setattr(profiles, "_get_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(profiles, "save_audio", lambda *_args: None)

    captured = {}

    class FakeBackend:
        async def combine_voice_prompts(self, audio_paths, reference_texts):
            captured["audio_paths"] = audio_paths
            captured["reference_texts"] = reference_texts
            return np.zeros(16, dtype=np.float32), "first second"

        async def create_voice_prompt(self, audio_path, reference_text, use_cache):
            return {
                "audio_path": audio_path,
                "reference_text": reference_text,
                "use_cache": use_cache,
            }, False

    import backend.backends

    monkeypatch.setattr(
        backend.backends,
        "get_tts_backend_for_engine",
        lambda _engine: FakeBackend(),
    )

    await profiles.create_voice_prompt_for_profile("profile-id", db)

    assert captured["reference_texts"] == ["first", "second"]
    assert [Path(path).name for path in captured["audio_paths"]] == [
        "first.wav",
        "second.wav",
    ]


@pytest.mark.asyncio
async def test_zip_roundtrip_uses_explicit_ordinal_not_uuid_or_json_key_order(
    db,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    sample_dir = tmp_path / "profiles" / "profile-id"
    sample_dir.mkdir(parents=True)
    (sample_dir / "z-id.wav").write_bytes(b"first audio")
    (sample_dir / "a-id.wav").write_bytes(b"second audio")

    db.add(VoiceProfile(id="profile-id", name="Narrator", voice_type="cloned"))
    db.add_all(
        [
            ProfileSample(
                id="a-id",
                profile_id="profile-id",
                ordinal=1,
                audio_path="profiles/profile-id/a-id.wav",
                reference_text="second",
            ),
            ProfileSample(
                id="z-id",
                profile_id="profile-id",
                ordinal=0,
                audio_path="profiles/profile-id/z-id.wav",
                reference_text="first",
            ),
        ]
    )
    db.commit()

    exported = await export_import.export_profile_to_zip("profile-id", db)
    try:
        with zipfile.ZipFile(exported.path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["sample_order"] == ["z-id.wav", "a-id.wav"]

            # Reverse the samples.json object members. A robust importer must use
            # the explicit sequence and not JSON object or UUID lexical order.
            reordered_buffer = io.BytesIO()
            with zipfile.ZipFile(reordered_buffer, "w", zipfile.ZIP_DEFLATED) as reordered:
                for info in archive.infolist():
                    payload = archive.read(info.filename)
                    if info.filename == "samples.json":
                        payload = json.dumps(
                            {
                                "a-id.wav": "second",
                                "z-id.wav": "first",
                            }
                        ).encode()
                    reordered.writestr(info, payload)
    finally:
        exported.cleanup()

    def fake_validate(audio_path):
        payload = Path(audio_path).read_bytes()
        marker = 1.0 if payload == b"first audio" else 2.0
        return True, None, np.array([marker], dtype=np.float32), 24_000

    def fake_save(audio, output_path, sample_rate):
        assert sample_rate == 24_000
        payload = b"first audio" if float(audio[0]) == 1.0 else b"second audio"
        Path(output_path).write_bytes(payload)

    monkeypatch.setattr(export_import, "validate_and_load_reference_audio", fake_validate)
    monkeypatch.setattr(export_import, "save_audio", fake_save)

    imported_profile = await export_import.import_profile_from_zip(
        reordered_buffer.getvalue(),
        db,
    )

    assert imported_profile.id != "profile-id"
    imported_rows = (
        db.query(ProfileSample).filter_by(profile_id=imported_profile.id).order_by(ProfileSample.ordinal).all()
    )
    imported_samples = [
        (
            sample.reference_text,
            config.resolve_storage_path(sample.audio_path).read_bytes(),
        )
        for sample in imported_rows
    ]
    assert imported_samples == [
        ("first", b"first audio"),
        ("second", b"second audio"),
    ]
