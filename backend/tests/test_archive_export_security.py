"""Bounded, private, disk-backed archive export regressions."""

import asyncio
import os
import stat
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Establish the route module's safe-content-disposition dependency first.
import backend.app  # noqa: F401
from backend import config
from backend.database import Base, Generation, GenerationVersion, ProfileSample, VoiceProfile
from backend.request_limits import MULTIPART_OVERHEAD_BYTES, request_body_limit
from backend.routes import history as history_routes, profiles as profile_routes
from backend.services import export_import


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'exports.db'}")
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
    yield root
    for directory in list(export_import._active_archive_export_directories):
        export_import._cleanup_archive_export(directory)


def _add_profile_with_sample(db, storage_root: Path, *, payload: bytes = b"sample audio") -> VoiceProfile:
    profile_dir = storage_root / "profiles" / "profile-id"
    profile_dir.mkdir(parents=True, mode=0o700)
    sample_path = profile_dir / "sample-id.wav"
    sample_path.write_bytes(payload)
    profile = VoiceProfile(id="profile-id", name="Narrator", voice_type="cloned")
    db.add(profile)
    db.add(
        ProfileSample(
            id="sample-id",
            profile_id=profile.id,
            ordinal=0,
            audio_path="profiles/profile-id/sample-id.wav",
            reference_text="Reference transcript",
        )
    )
    db.commit()
    return profile


def test_archive_http_import_caps_cover_every_exporter_owned_archive():
    generation_max = export_import.GENERATION_ARCHIVE_MAX_TOTAL_BYTES + export_import.ARCHIVE_EXPORT_OVERHEAD_BYTES
    profile_max = export_import.PROFILE_ARCHIVE_MAX_TOTAL_BYTES + export_import.ARCHIVE_EXPORT_OVERHEAD_BYTES

    assert history_routes.GENERATION_ARCHIVE_MAX_BYTES == generation_max
    assert profile_routes.PROFILE_ARCHIVE_MAX_BYTES == profile_max
    assert request_body_limit({"method": "POST", "path": "/history/import"}) == (
        generation_max + MULTIPART_OVERHEAD_BYTES
    )
    assert request_body_limit({"method": "POST", "path": "/profiles/import"}) == (
        profile_max + MULTIPART_OVERHEAD_BYTES
    )


def _add_generation(db, storage_root: Path, *, with_version: bool = False) -> Generation:
    generations_dir = storage_root / "generations"
    generations_dir.mkdir(parents=True, mode=0o700)
    audio_path = generations_dir / "generation.wav"
    audio_path.write_bytes(b"generation audio")
    generation = Generation(
        id="generation-id",
        profile_id="profile-id",
        text="Generated speech",
        language="en",
        audio_path="generations/generation.wav",
        duration=1.0,
        created_at=datetime.utcnow(),
    )
    db.add(generation)
    if with_version:
        db.add(
            GenerationVersion(
                id="version-id",
                generation_id=generation.id,
                label="Original",
                audio_path="generations/generation.wav",
                is_default=True,
                created_at=datetime.utcnow(),
            )
        )
    db.commit()
    return generation


@pytest.mark.asyncio
async def test_profile_export_is_off_loop_private_disk_backed_and_cleanup(
    db,
    storage_root,
    monkeypatch,
):
    _add_profile_with_sample(db, storage_root)
    main_thread = threading.get_ident()
    writer_threads = []
    original_writer = export_import._write_archive_export

    def recording_writer(*args, **kwargs):
        writer_threads.append(threading.get_ident())
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(export_import, "_write_archive_export", recording_writer)
    archive_export = await export_import.export_profile_to_zip("profile-id", db)
    export_directory = archive_export.temporary_directory
    try:
        assert writer_threads
        assert writer_threads[0] != main_thread
        assert archive_export.path.is_file()
        assert stat.S_IMODE(archive_export.path.stat().st_mode) == 0o600
        assert stat.S_IMODE(export_directory.stat().st_mode) == 0o700
        with zipfile.ZipFile(archive_export.path) as archive:
            assert set(archive.namelist()) == {
                "manifest.json",
                "samples.json",
                "samples/sample-id.wav",
            }
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    finally:
        archive_export.cleanup()
    assert not export_directory.exists()


@pytest.mark.asyncio
async def test_generation_export_deduplicates_shared_version_audio(db, storage_root):
    _add_profile_with_sample(db, storage_root)
    generation = _add_generation(db, storage_root, with_version=True)
    db.add(
        GenerationVersion(
            id="second-version",
            generation_id=generation.id,
            label="Shared",
            audio_path="generations/generation.wav",
            is_default=False,
            created_at=datetime.utcnow(),
        )
    )
    db.commit()

    archive_export = await export_import.export_generation_to_zip(generation.id, db)
    try:
        with zipfile.ZipFile(archive_export.path) as archive:
            assert len([name for name in archive.namelist() if name.startswith("audio/")]) == 1
            assert archive.read("audio/generation.wav") == b"generation audio"
    finally:
        archive_export.cleanup()


@pytest.mark.asyncio
async def test_non_wav_generation_export_round_trips_without_reencoding(db, storage_root):
    _add_profile_with_sample(db, storage_root)
    generations_dir = storage_root / "generations"
    generations_dir.mkdir(parents=True, exist_ok=True)
    source_path = generations_dir / "imported.flac"
    sf.write(
        source_path,
        np.linspace(-0.25, 0.25, 24_000, dtype=np.float32),
        24_000,
        format="FLAC",
    )
    source_bytes = source_path.read_bytes()
    generation = Generation(
        id="flac-generation",
        profile_id="profile-id",
        text="Imported FLAC",
        language="en",
        audio_path="generations/imported.flac",
        duration=1.0,
        created_at=datetime.utcnow(),
    )
    db.add(generation)
    db.commit()

    audio_response = await history_routes.export_generation_audio(generation.id, db)
    assert audio_response.media_type in {"audio/flac", "audio/x-flac"}
    assert ".flac" in audio_response.headers["content-disposition"]

    archive_export = await export_import.export_generation_to_zip(generation.id, db)
    try:
        with zipfile.ZipFile(archive_export.path) as archive:
            assert "audio/imported.flac" in archive.namelist()
            assert archive.read("audio/imported.flac") == source_bytes

        result = await export_import.import_generation_from_zip(archive_export.path, db)
    finally:
        archive_export.cleanup()

    imported = db.query(Generation).filter_by(id=result["id"]).one()
    imported_path = config.resolve_storage_path(imported.audio_path)
    assert imported_path is not None
    assert imported_path.suffix == ".flac"
    assert imported_path.read_bytes() == source_bytes


@pytest.mark.asyncio
async def test_profile_export_rejects_final_symlink_without_following(db, storage_root):
    if not hasattr(os, "symlink"):
        pytest.skip("symbolic links are unavailable")
    profile_dir = storage_root / "profiles" / "profile-id"
    profile_dir.mkdir(parents=True, mode=0o700)
    outside = storage_root.parent / "outside.wav"
    outside.write_bytes(b"private outside bytes")
    (profile_dir / "sample-id.wav").symlink_to(outside)
    db.add(VoiceProfile(id="profile-id", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample-id",
            profile_id="profile-id",
            ordinal=0,
            audio_path="profiles/profile-id/sample-id.wav",
            reference_text="Reference transcript",
        )
    )
    db.commit()

    with pytest.raises(ValueError, match="unavailable or unsafe"):
        await export_import.export_profile_to_zip("profile-id", db)
    assert outside.read_bytes() == b"private outside bytes"


@pytest.mark.asyncio
async def test_export_fails_before_allocation_when_disk_reserve_would_be_crossed(
    db,
    storage_root,
    monkeypatch,
):
    _add_profile_with_sample(db, storage_root)
    monkeypatch.setattr(
        export_import.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=export_import.ARCHIVE_EXPORT_MIN_FREE_BYTES),
    )

    with pytest.raises(export_import.ArchiveExportStorageError, match="Insufficient free space"):
        await export_import.export_profile_to_zip("profile-id", db)
    assert export_import._active_archive_export_directories == {}


@pytest.mark.asyncio
async def test_export_enforces_matching_import_aggregate_limit(db, storage_root, monkeypatch):
    _add_profile_with_sample(db, storage_root, payload=b"12345678")
    monkeypatch.setattr(export_import, "PROFILE_ARCHIVE_MAX_TOTAL_BYTES", 8)

    with pytest.raises(export_import.ArchiveExportLimitError, match="total uncompressed"):
        await export_import.export_profile_to_zip("profile-id", db)
    assert export_import._active_archive_export_directories == {}


@pytest.mark.asyncio
async def test_cancelled_export_drains_writer_and_removes_private_archive(
    db,
    storage_root,
    monkeypatch,
):
    _add_profile_with_sample(db, storage_root)
    started = threading.Event()
    release = threading.Event()
    original_writer = export_import._write_archive_export

    def blocking_writer(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(export_import, "_write_archive_export", blocking_writer)
    task = asyncio.create_task(export_import.export_profile_to_zip("profile-id", db))
    while not started.is_set():
        await asyncio.sleep(0)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert export_import._active_archive_export_directories == {}
    export_root = storage_root / "cache" / export_import.ARCHIVE_EXPORT_ROOT_NAME
    assert not list(export_root.glob("job-*"))


@pytest.mark.asyncio
async def test_export_admission_bounds_builders_and_response_leases(
    db,
    storage_root,
    monkeypatch,
):
    _add_profile_with_sample(db, storage_root)
    monkeypatch.setattr(export_import, "ARCHIVE_EXPORT_MAX_ACTIVE", 1)
    started = threading.Event()
    release = threading.Event()
    original_writer = export_import._write_archive_export

    def blocking_writer(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(export_import, "_write_archive_export", blocking_writer)
    first_task = asyncio.create_task(export_import.export_profile_to_zip("profile-id", db))
    assert await asyncio.to_thread(started.wait, 2)

    with pytest.raises(export_import.ArchiveExportBusyError, match="already active"):
        await export_import.export_profile_to_zip("profile-id", db)

    release.set()
    first = await first_task
    try:
        # A completed but still-downloading response keeps the same bounded
        # lease; it cannot be bypassed by queuing another small export.
        with pytest.raises(export_import.ArchiveExportBusyError, match="already active"):
            await export_import.export_profile_to_zip("profile-id", db)
    finally:
        first.cleanup()

    second = await export_import.export_profile_to_zip("profile-id", db)
    second.cleanup()
    assert export_import._active_archive_export_directories == {}


@pytest.mark.asyncio
async def test_next_export_reclaims_private_archive_left_by_prior_process(db, storage_root):
    _add_profile_with_sample(db, storage_root)
    export_root = config.get_cache_dir() / export_import.ARCHIVE_EXPORT_ROOT_NAME
    export_root.mkdir(mode=0o700)
    stale = export_root / "job-stale"
    stale.mkdir(mode=0o700)
    (stale / "archive.voicebox.zip").write_bytes(b"private stale archive")

    archive_export = await export_import.export_profile_to_zip("profile-id", db)
    try:
        assert not stale.exists()
        assert archive_export.path.exists()
    finally:
        archive_export.cleanup()


@pytest.mark.asyncio
async def test_export_routes_return_file_response_with_background_cleanup(db, storage_root):
    _add_profile_with_sample(db, storage_root)
    generation = _add_generation(db, storage_root)
    generation_id = generation.id

    profile_response = await profile_routes.export_profile("profile-id", db)
    profile_path = Path(profile_response.path)
    assert profile_path.is_file()
    await profile_response.background()
    assert not profile_path.exists()

    generation_response = await history_routes.export_generation(generation_id, db)
    generation_path = Path(generation_response.path)
    assert generation_path.is_file()
    await generation_response.background()
    assert not generation_path.exists()


@pytest.mark.asyncio
async def test_export_routes_map_busy_admission_to_retryable_response(db, storage_root, monkeypatch):
    _add_profile_with_sample(db, storage_root)
    generation = _add_generation(db, storage_root)

    async def busy(*_args, **_kwargs):
        raise export_import.ArchiveExportBusyError("export capacity busy")

    monkeypatch.setattr(export_import, "export_profile_to_zip", busy)
    with pytest.raises(HTTPException) as profile_error:
        await profile_routes.export_profile("profile-id", db)
    assert profile_error.value.status_code == 429
    assert profile_error.value.headers == {"Retry-After": "1"}

    monkeypatch.setattr(export_import, "export_generation_to_zip", busy)
    with pytest.raises(HTTPException) as generation_error:
        await history_routes.export_generation(generation.id, db)
    assert generation_error.value.status_code == 429
    assert generation_error.value.headers == {"Retry-After": "1"}


@pytest.mark.asyncio
async def test_export_response_disconnect_releases_file_and_capacity(db, storage_root):
    _add_profile_with_sample(db, storage_root)
    response = await profile_routes.export_profile("profile-id", db)
    archive_path = Path(response.path)
    export_directory = archive_path.parent

    async def receive():
        return {"type": "http.disconnect"}

    async def broken_send(_message):
        raise ConnectionError("client disconnected")

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": "/profiles/profile-id/export",
        "raw_path": b"/profiles/profile-id/export",
        "query_string": b"",
        "headers": [],
    }
    with pytest.raises(ConnectionError, match="client disconnected"):
        await response(scope, receive, broken_send)

    assert not archive_path.exists()
    assert not export_directory.exists()
    assert export_import._active_archive_export_directories == {}
