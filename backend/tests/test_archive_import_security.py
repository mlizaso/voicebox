"""Hostile archive and atomic publication regressions."""

import asyncio
import io
import json
import os
import random
import threading
import wave
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Import the application first so route modules do not enter through their
# safe_content_disposition dependency while app.py is only partially loaded.
import backend.app  # noqa: F401
from backend import config
from backend.database import Base, Generation, ProfileSample, VoiceProfile
from backend.routes import history as history_routes, profiles as profile_routes, transcription as transcription_routes
from backend.services import captures, deletion_journal, export_import
from backend.utils.upload_limits import UploadDurationLimitError


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'archive.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    root.mkdir(mode=0o700)
    monkeypatch.setattr(config, "_data_dir", root)
    monkeypatch.delenv("VOICEBOX_SHARED_GENERATIONS", raising=False)
    return root


def _zip_bytes(
    entries: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return output.getvalue()


def _profile_archive(samples: list[tuple[str, str, bytes]]) -> bytes:
    manifest = {
        "version": "1.0",
        "profile": {"name": "Imported narrator", "language": "en"},
        "sample_order": [filename for filename, _text, _audio in samples],
    }
    entries = [
        ("manifest.json", json.dumps(manifest).encode()),
        (
            "samples.json",
            json.dumps({filename: text for filename, text, _audio in samples}).encode(),
        ),
    ]
    entries.extend((f"samples/{filename}", audio) for filename, _text, audio in samples)
    return _zip_bytes(entries)


def _wav_bytes(
    *,
    duration: float = 2.0,
    sample_rate: int = 24_000,
    channels: int = 1,
    seed: int = 7,
) -> bytes:
    frame_count = int(duration * sample_rate)
    pcm = random.Random(seed).randbytes(frame_count * channels * 2)
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return output.getvalue()


def _generation_archive(*, duration: float = 2.0, audio: bytes | None = None) -> bytes:
    manifest = {
        "version": "1.0",
        "generation": {
            "text": "Imported speech",
            "language": "en",
            "duration": duration,
        },
        "profile": {"name": "Narrator"},
    }
    return _zip_bytes(
        [
            ("manifest.json", json.dumps(manifest).encode()),
            ("audio/import.wav", audio if audio is not None else _wav_bytes()),
        ]
    )


def test_profile_import_preflight_enforces_conditioning_contract(monkeypatch):
    assert export_import.PROFILE_ARCHIVE_MAX_SAMPLES == 64
    assert export_import.EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES == 64 * 1024 * 1024
    assert export_import.EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES == 128 * 1024 * 1024
    assert export_import.EXACT_VOICE_SNAPSHOT_MAX_TOTAL_REFERENCE_TEXT_BYTES == 512 * 1024

    oversized_text = _profile_archive([("voice.wav", "x" * 1001, b"audio")])
    with pytest.raises(ValueError, match="1000 characters"):
        export_import._inspect_profile_import(oversized_text)

    monkeypatch.setattr(export_import, "EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES", 5)
    oversized_audio = _profile_archive(
        [
            ("one.wav", "one", b"123"),
            ("two.wav", "two", b"456"),
        ]
    )
    with pytest.raises(ValueError, match="safe aggregate size"):
        export_import._inspect_profile_import(oversized_audio)


def test_import_storage_admission_is_shared_and_released(
    storage_root,
    monkeypatch,
):
    monkeypatch.setattr(
        export_import.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=export_import.ARCHIVE_IMPORT_MIN_FREE_BYTES + 150),
    )
    first = export_import._allocate_archive_import(100)
    try:
        with pytest.raises(export_import.ArchiveImportStorageError, match="Insufficient free space"):
            export_import._allocate_archive_import(60)
        with pytest.raises(export_import.ArchiveExportStorageError, match="Insufficient free space"):
            export_import._allocate_archive_export(60)
    finally:
        first.cleanup()

    second = export_import._allocate_archive_import(60)
    second.cleanup()
    assert export_import._active_archive_import_directories == {}


def test_import_concurrency_bound_releases_capacity(storage_root, monkeypatch):
    monkeypatch.setattr(export_import, "ARCHIVE_IMPORT_MAX_CONCURRENT", 1)
    first = export_import._allocate_archive_import(1)
    try:
        with pytest.raises(export_import.ArchiveImportBusyError, match="already in progress"):
            export_import._allocate_archive_import(1)
    finally:
        first.cleanup()

    replacement = export_import._allocate_archive_import(1)
    replacement.cleanup()
    assert export_import._active_archive_import_directories == {}


@pytest.mark.asyncio
async def test_generation_import_extraction_keeps_event_loop_responsive(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    started = threading.Event()
    release = threading.Event()

    def held_extract(_source, _plan, destination):
        started.set()
        assert release.wait(timeout=5)
        destination.write_bytes(_wav_bytes())

    monkeypatch.setattr(export_import, "_extract_generation_import", held_extract)
    task = asyncio.create_task(
        export_import.import_generation_from_zip(
            _generation_archive(),
            db,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)

    heartbeat = asyncio.Event()
    asyncio.get_running_loop().call_soon(heartbeat.set)
    await asyncio.wait_for(heartbeat.wait(), timeout=0.2)
    assert not task.done()

    release.set()
    result = await asyncio.wait_for(task, timeout=2)
    assert result["profile_id"] == "profile"
    assert export_import._active_archive_import_directories == {}


@pytest.mark.asyncio
async def test_generation_import_cancellation_drains_worker_before_cleanup(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()

    def held_extract(_source, _plan, _destination):
        started.set()
        assert release.wait(timeout=5)
        finished.set()

    monkeypatch.setattr(export_import, "_extract_generation_import", held_extract)
    task = asyncio.create_task(
        export_import.import_generation_from_zip(
            _generation_archive(),
            db,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
    task.cancel()
    await asyncio.sleep(0.05)

    assert not task.done()
    assert not finished.is_set()
    assert len(export_import._active_archive_import_directories) == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert finished.is_set()
    assert export_import._active_archive_import_directories == {}
    import_root = storage_root / "cache" / export_import.ARCHIVE_IMPORT_ROOT_NAME
    assert not import_root.exists() or list(import_root.iterdir()) == []
    assert db.query(Generation).count() == 0


@pytest.mark.asyncio
async def test_generation_import_cancellation_drains_journaled_publication(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    started = threading.Event()
    release = threading.Event()
    real_populate = export_import._populate_private_file

    def held_populate(source, destination, *, expected_stat):
        started.set()
        assert release.wait(timeout=5)
        real_populate(source, destination, expected_stat=expected_stat)

    monkeypatch.setattr(export_import, "_populate_private_file", held_populate)
    task = asyncio.create_task(
        export_import.import_generation_from_zip(
            _generation_archive(),
            db,
        )
    )
    await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)
    task.cancel()
    await asyncio.sleep(0.05)

    assert not task.done()
    assert list(config.get_deletion_journal_dir().glob("*.json"))
    assert len(export_import._active_archive_import_directories) == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert export_import._active_archive_import_directories == {}
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    assert not list((storage_root / "generations").glob("*.wav"))
    assert db.query(Generation).count() == 0


@pytest.mark.asyncio
async def test_generation_import_selects_manifest_default_version(db, storage_root):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    nondefault_audio = _wav_bytes(seed=11)
    default_audio = _wav_bytes(seed=19)
    manifest = {
        "version": "1.0",
        "generation": {
            "text": "Imported speech",
            "language": "en",
            "duration": 2.0,
        },
        "profile": {"name": "Narrator"},
        "versions": [
            {"filename": "a.wav", "is_default": False},
            {"filename": "z.wav", "is_default": True},
        ],
    }
    archive = _zip_bytes(
        [
            ("manifest.json", json.dumps(manifest).encode()),
            ("audio/a.wav", nondefault_audio),
            ("audio/z.wav", default_audio),
        ]
    )

    result = await export_import.import_generation_from_zip(archive, db)

    imported = db.query(Generation).filter_by(id=result["id"]).one()
    imported_path = config.resolve_storage_path(imported.audio_path)
    assert imported_path is not None
    assert imported_path.read_bytes() == default_audio


def _install_fake_profile_audio(monkeypatch):
    def fake_validate(path):
        if Path(path).read_bytes() == b"invalid":
            return False, "test-invalid", None, None
        return True, None, np.ones(48_000, dtype=np.float32), 24_000

    def fake_save(_audio, path, _sample_rate):
        Path(path).write_bytes(b"canonical wav")

    monkeypatch.setattr(export_import, "validate_and_load_reference_audio", fake_validate)
    monkeypatch.setattr(export_import, "save_audio", fake_save)


def test_zip_preflight_rejects_duplicate_member():
    with pytest.warns(UserWarning, match="Duplicate name"):
        payload = _zip_bytes([("manifest.json", b"{}"), ("manifest.json", b"{}")])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive, pytest.raises(ValueError, match="duplicate member"):
        export_import._validate_zip_archive(
            archive,
            max_members=10,
            max_total_bytes=1024,
            max_entry_bytes=1024,
        )


def test_zip_preflight_rejects_parent_traversal_member():
    payload = _zip_bytes([("../manifest.json", b"{}")])
    with (
        zipfile.ZipFile(io.BytesIO(payload)) as archive,
        pytest.raises(
            ValueError,
            match="unsafe member path",
        ),
    ):
        export_import._validate_zip_archive(
            archive,
            max_members=10,
            max_total_bytes=1024,
            max_entry_bytes=1024,
        )


@pytest.mark.skipif(os.name != "posix", reason="O_NOFOLLOW is a POSIX boundary")
def test_archive_source_symlink_is_refused_without_touching_target(tmp_path):
    payload = _generation_archive()
    target = tmp_path / "target.zip"
    target.write_bytes(payload)
    linked = tmp_path / "linked.zip"
    linked.symlink_to(target)

    with pytest.raises(ValueError, match="not a readable regular file"), export_import._open_zip_archive(linked):
        raise AssertionError("symlink archive was opened")
    assert target.read_bytes() == payload


def test_zip_preflight_enforces_member_entry_total_and_ratio_limits():
    payload = _zip_bytes([("one", b"12345"), ("two", b"67890")])
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with pytest.raises(ValueError, match="too many members"):
            export_import._validate_zip_archive(
                archive,
                max_members=1,
                max_total_bytes=100,
                max_entry_bytes=100,
            )
        with pytest.raises(ValueError, match="uncompressed size limit"):
            export_import._validate_zip_archive(
                archive,
                max_members=10,
                max_total_bytes=100,
                max_entry_bytes=4,
            )
        with pytest.raises(ValueError, match="total uncompressed size limit"):
            export_import._validate_zip_archive(
                archive,
                max_members=10,
                max_total_bytes=9,
                max_entry_bytes=100,
            )

    compressed = _zip_bytes([("bomb", b"0" * (128 * 1024))])
    with zipfile.ZipFile(io.BytesIO(compressed)) as archive, pytest.raises(ValueError, match="compression-ratio"):
        export_import._validate_zip_archive(
            archive,
            max_members=10,
            max_total_bytes=1024 * 1024,
            max_entry_bytes=1024 * 1024,
        )


@pytest.mark.asyncio
async def test_second_invalid_profile_sample_leaves_no_db_or_files(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)
    archive = _profile_archive(
        [
            ("first.wav", "first", b"valid"),
            ("second.wav", "second", b"invalid"),
        ]
    )

    with pytest.raises(ValueError, match=r"second\.wav"):
        await export_import.import_profile_from_zip(archive, db)

    assert db.query(VoiceProfile).count() == 0
    assert db.query(ProfileSample).count() == 0
    profiles_dir = storage_root / "profiles"
    assert not profiles_dir.exists() or list(profiles_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_profile_route_spools_path_into_real_service_and_cleans_it(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)
    archive = _profile_archive([("voice.wav", "spoken text", b"valid")])
    original_import = export_import.import_profile_from_zip
    observed_path = None

    async def checking_import(source, session):
        nonlocal observed_path
        assert isinstance(source, Path)
        assert source.is_file()
        observed_path = source
        return await original_import(source, session)

    monkeypatch.setattr(export_import, "import_profile_from_zip", checking_import)
    result = await profile_routes.import_profile(
        UploadFile(file=io.BytesIO(archive), filename="profile.zip"),
        db,
    )

    assert result.name == "Imported narrator"
    assert observed_path is not None
    assert not observed_path.exists()
    sample = db.query(ProfileSample).one()
    sample_path = config.resolve_storage_path(sample.audio_path)
    assert sample_path is not None
    assert sample_path.is_file()
    if os.name == "posix":
        assert sample_path.stat().st_mode & 0o777 == 0o600
        assert sample_path.parent.stat().st_mode & 0o777 == 0o700


@pytest.mark.asyncio
async def test_generation_route_spools_path_into_real_service_and_cleans_it(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    original_import = export_import.import_generation_from_zip
    observed_path = None

    async def checking_import(source, session):
        nonlocal observed_path
        assert isinstance(source, Path)
        assert source.is_file()
        observed_path = source
        return await original_import(source, session)

    monkeypatch.setattr(export_import, "import_generation_from_zip", checking_import)
    result = await history_routes.import_generation(
        UploadFile(
            file=io.BytesIO(_generation_archive()),
            filename="generation.zip",
        ),
        db,
    )

    assert result["profile_id"] == "profile"
    assert observed_path is not None
    assert not observed_path.exists()
    generation = db.query(Generation).one()
    audio_path = config.resolve_storage_path(generation.audio_path)
    assert audio_path is not None
    assert audio_path.is_file()
    assert generation.duration == pytest.approx(2.0)
    if os.name == "posix":
        assert audio_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_generation_import_rejects_invalid_audio_and_forged_duration(
    db,
    storage_root,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()

    with pytest.raises(ValueError, match="invalid WAV"):
        await export_import.import_generation_from_zip(
            _generation_archive(audio=b"not audio"),
            db,
        )
    with pytest.raises(ValueError, match="duration does not match"):
        await export_import.import_generation_from_zip(
            _generation_archive(duration=30.0),
            db,
        )

    assert db.query(Generation).count() == 0
    generations_dir = storage_root / "generations"
    assert not generations_dir.exists() or list(generations_dir.iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "audio",
    [
        _wav_bytes(duration=0.1, channels=9),
        _wav_bytes(duration=0.1, sample_rate=192_001),
    ],
)
async def test_generation_archive_rejects_audio_outside_story_effects_shape(
    db,
    storage_root,
    audio,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()

    with pytest.raises(ValueError, match="invalid WAV"):
        await export_import.import_generation_from_zip(
            _generation_archive(duration=0.1, audio=audio),
            db,
        )

    assert db.query(Generation).count() == 0
    generations_dir = storage_root / "generations"
    assert not generations_dir.exists() or list(generations_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_profile_postcommit_raise_returns_exact_durable_profile(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("injected post-commit failure")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    response = await export_import.import_profile_from_zip(
        _profile_archive([("voice.wav", "spoken text", b"valid")]),
        db,
    )

    fresh = Session(bind=db.get_bind())
    try:
        profile = fresh.query(VoiceProfile).one()
        sample = fresh.query(ProfileSample).one()
        assert response.id == profile.id
        assert response.name == "Imported narrator"
        assert response.sample_count == 1
        sample_path = config.resolve_storage_path(sample.audio_path)
        assert sample_path is not None
        assert sample_path.is_file()
        assert sample_path.parent.name == profile.id
    finally:
        fresh.close()


@pytest.mark.asyncio
async def test_generation_postcommit_raise_returns_exact_durable_result(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("injected post-commit failure")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    result = await export_import.import_generation_from_zip(
        _generation_archive(),
        db,
    )

    fresh = Session(bind=db.get_bind())
    try:
        generation = fresh.query(Generation).one()
        assert result["id"] == generation.id
        assert result["profile_id"] == "profile"
        assert result["text"] == "Imported speech"
        audio_path = config.resolve_storage_path(generation.audio_path)
        assert audio_path is not None
        assert audio_path.is_file()
    finally:
        fresh.close()


@pytest.mark.asyncio
async def test_profile_import_journal_cleanup_failure_does_not_revoke_success(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)
    real_finish = deletion_journal.finish_deletion_intent

    def fail_finish(_intent):
        raise RuntimeError("journal directory flush failed")

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", fail_finish)
    response = await export_import.import_profile_from_zip(
        _profile_archive([("voice.wav", "spoken text", b"valid")]),
        db,
    )

    assert db.query(VoiceProfile).filter_by(id=response.id).one().name == response.name
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", real_finish)
    report = deletion_journal.recover_interrupted_deletions(db)
    assert report.cleared == 1
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_generation_import_journal_cleanup_failure_does_not_revoke_success(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    real_finish = deletion_journal.finish_deletion_intent

    def fail_finish(_intent):
        raise RuntimeError("journal directory flush failed")

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", fail_finish)
    result = await export_import.import_generation_from_zip(
        _generation_archive(),
        db,
    )

    assert db.query(Generation).filter_by(id=result["id"]).one().text == result["text"]
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    monkeypatch.setattr(deletion_journal, "finish_deletion_intent", real_finish)
    report = deletion_journal.recover_interrupted_deletions(db)
    assert report.cleared == 1
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_profile_import_does_not_acknowledge_mismatched_durable_row(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)
    real_commit = db.commit

    def commit_mutate_then_raise():
        real_commit()
        with Session(bind=db.get_bind()) as concurrent:
            profile = concurrent.query(VoiceProfile).one()
            profile.description = "changed after commit"
            concurrent.commit()
        raise RuntimeError("injected post-commit failure")

    monkeypatch.setattr(db, "commit", commit_mutate_then_raise)
    with pytest.raises(RuntimeError, match="post-commit"):
        await export_import.import_profile_from_zip(
            _profile_archive([("voice.wav", "spoken text", b"valid")]),
            db,
        )

    assert db.query(VoiceProfile).one().description == "changed after commit"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_generation_import_does_not_acknowledge_mismatched_durable_row(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    real_commit = db.commit

    def commit_mutate_then_raise():
        real_commit()
        with Session(bind=db.get_bind()) as concurrent:
            generation = concurrent.query(Generation).one()
            generation.text = "changed after commit"
            concurrent.commit()
        raise RuntimeError("injected post-commit failure")

    monkeypatch.setattr(db, "commit", commit_mutate_then_raise)
    with pytest.raises(RuntimeError, match="post-commit"):
        await export_import.import_generation_from_zip(
            _generation_archive(),
            db,
        )

    assert db.query(Generation).one().text == "changed after commit"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1


@pytest.mark.asyncio
async def test_profile_indeterminate_commit_outcome_preserves_published_files(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)
    monkeypatch.setattr(
        db,
        "commit",
        lambda: (_ for _ in ()).throw(RuntimeError("indeterminate commit")),
    )
    monkeypatch.setattr(
        export_import,
        "_durable_imported_profile_response",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="indeterminate commit"):
        await export_import.import_profile_from_zip(
            _profile_archive([("voice.wav", "spoken text", b"valid")]),
            db,
        )

    published = list((storage_root / "profiles").glob("*/*.wav"))
    assert len(published) == 1
    assert published[0].read_bytes() == b"canonical wav"


@pytest.mark.asyncio
async def test_generation_indeterminate_commit_outcome_preserves_published_file(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()
    monkeypatch.setattr(
        db,
        "commit",
        lambda: (_ for _ in ()).throw(RuntimeError("indeterminate commit")),
    )
    monkeypatch.setattr(
        export_import,
        "_durable_imported_generation_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("database unavailable")),
    )

    with pytest.raises(RuntimeError, match="indeterminate commit"):
        await export_import.import_generation_from_zip(
            _generation_archive(),
            db,
        )

    published = list((storage_root / "generations").glob("*.wav"))
    assert len(published) == 1
    assert published[0].is_file()


@pytest.mark.asyncio
async def test_startup_recovery_discards_interrupted_profile_population(
    db,
    storage_root,
    monkeypatch,
):
    _install_fake_profile_audio(monkeypatch)

    def interrupt_managed_copy(_source, destination):
        assert destination.parent.name.startswith(".voicebox-delete-profile-import-")
        assert list(config.get_deletion_journal_dir().glob("*.json"))
        destination.write_bytes(b"partial profile audio")
        raise RuntimeError("simulated hard-crash boundary")

    monkeypatch.setattr(export_import, "_copy_private_file", interrupt_managed_copy)
    monkeypatch.setattr(export_import, "_reconcile_failed_publication", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="hard-crash boundary"):
        await export_import.import_profile_from_zip(
            _profile_archive([("voice.wav", "spoken text", b"valid")]),
            db,
        )

    assert len(list((storage_root / "profiles").glob(".voicebox-delete-*/*.wav"))) == 1
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    fresh = Session(bind=db.get_bind())
    try:
        report = deletion_journal.recover_interrupted_deletions(fresh)
    finally:
        fresh.close()

    assert report.discarded == 1
    assert not list((storage_root / "profiles").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_startup_recovery_discards_interrupted_generation_population(
    db,
    storage_root,
    monkeypatch,
):
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.commit()

    def interrupt_managed_copy(_source, destination, *, expected_stat):
        assert list(config.get_deletion_journal_dir().glob("*.json"))
        assert destination.stat().st_ino == expected_stat.st_ino
        destination.write_bytes(b"partial generation audio")
        raise RuntimeError("simulated hard-crash boundary")

    monkeypatch.setattr(export_import, "_populate_private_file", interrupt_managed_copy)
    monkeypatch.setattr(export_import, "_reconcile_failed_publication", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="hard-crash boundary"):
        await export_import.import_generation_from_zip(
            _generation_archive(),
            db,
        )

    assert len(list((storage_root / "generations").glob(".voicebox-delete-*.wav"))) == 1
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    fresh = Session(bind=db.get_bind())
    try:
        report = deletion_journal.recover_interrupted_deletions(fresh)
    finally:
        fresh.close()

    assert report.discarded == 1
    assert not list((storage_root / "generations").glob(".voicebox-delete-*.wav"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_transcription_decode_is_duration_bounded(monkeypatch):
    monkeypatch.setattr(
        transcription_routes,
        "AUDIO_UPLOAD_MAX_DURATION_SECONDS",
        1,
    )

    def oversized_decode(_path, *, sr, mono, duration):
        assert sr == 24_000
        assert mono is True
        assert duration == 2
        return np.zeros(48_001, dtype=np.float32), sr

    monkeypatch.setattr(transcription_routes.librosa, "load", oversized_decode)
    with pytest.raises(HTTPException) as caught:
        await transcription_routes.transcribe_audio(
            UploadFile(file=io.BytesIO(b"encoded audio"), filename="audio.mp3"),
        )
    assert caught.value.status_code == 413


@pytest.mark.asyncio
async def test_capture_decode_is_duration_bounded_and_cleans_pending_file(
    db,
    storage_root,
    monkeypatch,
):
    monkeypatch.setattr(captures, "AUDIO_UPLOAD_MAX_DURATION_SECONDS", 1)

    def oversized_decode(_path, *, sr, mono, duration):
        assert sr == 24_000
        assert mono is True
        assert duration == 2
        return np.zeros(48_001, dtype=np.float32), sr

    monkeypatch.setattr(captures.librosa, "load", oversized_decode)
    with pytest.raises(UploadDurationLimitError):
        await captures.create_capture(
            audio_bytes=b"encoded audio",
            filename="capture.mp3",
            source="file",
            language=None,
            stt_model=None,
            db=db,
        )
    captures_dir = storage_root / "captures"
    assert not captures_dir.exists() or list(captures_dir.iterdir()) == []
