"""Durability regressions for core generation-audio publication."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, Generation, GenerationVersion, VoiceProfile
from backend.services import deletion_journal, effects_processing, generation, versions
from backend.utils import disk_reservations
from backend.utils.audio import normalize_audio, save_audio as save_real_audio
from backend.utils.chunked_tts import _DiskBackedChunkAccumulator, release_disk_backed_audio


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="text",
            language="en",
            audio_path=None,
            status="generating",
        )
    )
    db.commit()
    return data_dir, db


def _audio():
    return np.zeros(240, dtype=np.float32)


def _run(awaitable):
    return asyncio.run(awaitable)


def _assert_no_publication_artifacts(data_dir: Path) -> None:
    assert list((data_dir / "generations").iterdir()) == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


def test_generate_discards_clean_wav_when_version_creation_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    def fail_create_version(**_kwargs):
        raise RuntimeError("version insert failed")

    monkeypatch.setattr(versions, "create_version", fail_create_version)

    with pytest.raises(RuntimeError, match="version insert failed"):
        _run(
            generation._save_generate(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                effects_chain=None,
                db=db,
            )
        )

    assert db.query(GenerationVersion).count() == 0
    assert db.query(Generation).one().audio_path is None
    _assert_no_publication_artifacts(data_dir)
    db.close()


def test_generate_uses_durable_owner_view_when_rollback_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(
        versions,
        "create_version",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("version insert failed")),
    )
    monkeypatch.setattr(
        db,
        "rollback",
        lambda: (_ for _ in ()).throw(RuntimeError("rollback unavailable")),
    )

    with pytest.raises(RuntimeError, match="version insert failed"):
        _run(
            generation._save_generate(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                effects_chain=None,
                db=db,
            )
        )

    assert db.query(GenerationVersion).count() == 0
    _assert_no_publication_artifacts(data_dir)
    db.close()


def test_generate_preserves_committed_wav_after_create_version_raises(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    real_create_version = versions.create_version

    def commit_then_raise(**kwargs):
        real_create_version(**kwargs)
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(versions, "create_version", commit_then_raise)

    with pytest.raises(RuntimeError, match="connection failed after commit"):
        _run(
            generation._save_generate(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                effects_chain=None,
                db=db,
            )
        )

    row = db.query(GenerationVersion).one()
    audio_path = config.resolve_storage_path(row.audio_path)
    assert audio_path is not None
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getnframes() == len(_audio())
    assert db.query(Generation).one().audio_path == row.audio_path
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_generate_publication_preserves_real_wav_encoding(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    expected_path = tmp_path / "expected.wav"
    save_real_audio(_audio(), str(expected_path), 24_000)

    stored_path = _run(
        generation._save_generate(
            generation_id="generation",
            audio=_audio(),
            sample_rate=24_000,
            effects_chain=None,
            db=db,
        )
    )

    audio_path = config.resolve_storage_path(stored_path)
    assert audio_path is not None
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getframerate() == 24_000
        assert wav.getnframes() == len(_audio())
    assert audio_path.read_bytes() == expected_path.read_bytes()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


@pytest.mark.parametrize(
    "process_block_frames",
    [effects_processing.EFFECTS_PROCESS_BLOCK_FRAMES, 4_096],
)
def test_foreground_file_matches_background_normalize_before_effects(
    tmp_path,
    monkeypatch,
    process_block_frames,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(
        effects_processing,
        "EFFECTS_PROCESS_BLOCK_FRAMES",
        process_block_frames,
    )
    sample_rate = 24_000
    timeline = np.arange(48_001, dtype=np.float32) / sample_rate
    source = (0.03 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    effects_chain = [
        {
            "type": "gain",
            "enabled": True,
            "params": {"gain_db": 6.0},
        }
    ]

    background_path = _run(
        generation._save_generate(
            generation_id="generation",
            audio=normalize_audio(source.copy()),
            sample_rate=sample_rate,
            effects_chain=effects_chain,
            db=db,
        )
    )
    foreground = _run(
        effects_processing.create_generated_audio_response_file(
            source.copy(),
            sample_rate,
            effects_chain,
            True,
        )
    )
    try:
        resolved_background = config.resolve_storage_path(background_path)
        assert resolved_background is not None
        assert foreground.path.read_bytes() == resolved_background.read_bytes()
    finally:
        foreground.cleanup()
        db.close()


def test_generate_publication_preserves_disk_reserve_before_encoding(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    usage = shutil.disk_usage(config.get_generations_dir())
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            usage.total,
            usage.used,
            generation.GENERATION_AUDIO_PUBLICATION_MIN_FREE_BYTES,
        ),
    )

    with pytest.raises(OSError, match="free space"):
        _run(
            generation._save_generate(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                effects_chain=None,
                db=db,
            )
        )

    _assert_no_publication_artifacts(data_dir)
    assert db.query(GenerationVersion).count() == 0
    db.close()


def test_exact_batch_uses_real_generation_audio_publication(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    db.add(
        Generation(
            id="generation-two",
            profile_id="profile",
            text="second text",
            language="en",
            audio_path=None,
            status="generating",
        )
    )
    db.commit()
    engine = db.get_bind()
    task_manager = Mock()
    backend = Mock()
    backend.is_loaded.return_value = True

    @asynccontextmanager
    async def loaded_backend(_engine, _model_size):
        yield backend

    async def run_operation(_backend, operation):
        return await operation

    monkeypatch.setattr(generation, "get_db", lambda: iter([db]))
    monkeypatch.setattr(generation, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generation, "_notify_speak_end", Mock())
    monkeypatch.setattr(generation.profiles, "_require_exact_tts_revision", Mock())
    monkeypatch.setattr(
        generation.profiles,
        "create_exact_voice_prompt_from_snapshot",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: backend)
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.loaded_tts_backend_for_request",
        loaded_backend,
    )
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.run_tts_operation_cancellation_safe",
        run_operation,
    )
    monkeypatch.setattr(
        "backend.utils.chunked_tts.generate_text_batch",
        AsyncMock(return_value=[(_audio(), 24_000), (_audio(), 24_000)]),
    )

    common = {
        "profile_id": "profile",
        "language": "en",
        "engine": "qwen",
        "model_size": "1.7B",
        "normalize": False,
        "effects_chain": None,
        "instruct": None,
        "crossfade_ms": 10,
        "expected_voice_binding_sha256": "voice-binding",
        "exact_voice_snapshot": {"voice_binding_sha256": "voice-binding"},
        "expected_tts_implementation_revision": "runtime-revision",
    }
    asyncio.run(
        generation.run_exact_generation_batch(
            [
                generation.ExactBatchGenerationSpec(
                    generation_id="generation",
                    text="first text",
                    seed=100,
                    **common,
                ),
                generation.ExactBatchGenerationSpec(
                    generation_id="generation-two",
                    text="second text",
                    seed=101,
                    **common,
                ),
            ]
        )
    )

    inspection = sessionmaker(bind=engine)()
    rows = inspection.query(Generation).order_by(Generation.id).all()
    assert [row.status for row in rows] == ["completed", "completed"]
    assert inspection.query(GenerationVersion).count() == 2
    for row in rows:
        path = config.resolve_storage_path(row.audio_path)
        assert path is not None
        with wave.open(str(path), "rb") as wav:
            assert wav.getnframes() == len(_audio())
    assert sorted(path.name for path in (data_dir / "generations").glob("*.wav")) == [
        "generation-two.wav",
        "generation.wav",
    ]
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    inspection.close()


def test_generate_keeps_owned_clean_wav_but_discards_failed_processed_wav(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    from backend.utils import effects as effects_utils

    monkeypatch.setattr(effects_utils, "validate_effects_chain", lambda _chain: None)
    monkeypatch.setattr(effects_utils, "apply_effects", lambda audio, _sample_rate, _chain: audio)
    real_create_version = versions.create_version
    calls = 0

    def fail_second_version(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("processed version insert failed")
        return real_create_version(**kwargs)

    monkeypatch.setattr(versions, "create_version", fail_second_version)

    with pytest.raises(RuntimeError, match="processed version insert failed"):
        _run(
            generation._save_generate(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                effects_chain=[{"type": "gain", "enabled": True}],
                db=db,
            )
        )

    version = db.query(GenerationVersion).one()
    clean_path = config.resolve_storage_path(version.audio_path)
    assert clean_path is not None
    with wave.open(str(clean_path), "rb") as wav:
        assert wav.getnframes() == len(_audio())
    assert not (data_dir / "generations" / "generation_processed.wav").exists()
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_generate_renders_long_memmap_effects_directly_into_publication(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    from backend.utils import effects as effects_utils

    def reject_whole_array_path(*_args, **_kwargs):
        raise AssertionError("long generation used the whole-array effects path")

    monkeypatch.setattr(effects_utils, "apply_effects", reject_whole_array_path)
    source = np.linspace(-0.2, 0.2, 1_100_003, dtype=np.float32)
    accumulator = _DiskBackedChunkAccumulator(24_000, 0)
    accumulator.append(source)
    audio = accumulator.finish()
    try:
        stored_path = _run(
            generation._save_generate(
                generation_id="generation",
                audio=audio,
                sample_rate=24_000,
                effects_chain=[
                    {
                        "type": "gain",
                        "enabled": True,
                        "params": {"gain_db": 3.0},
                    }
                ],
                db=db,
            )
        )
    finally:
        release_disk_backed_audio(audio)

    assert stored_path == "generations/generation_processed.wav"
    assert db.query(GenerationVersion).count() == 2
    for version in db.query(GenerationVersion).all():
        path = config.resolve_storage_path(version.audio_path)
        assert path is not None
        with wave.open(str(path), "rb") as wav:
            assert wav.getnframes() == len(source)
    db.close()


def test_processed_render_cancellation_reconciles_only_unpublished_output(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)

    async def cancel_render(*_args, **_kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(
        effects_processing,
        "render_generated_audio_to_descriptor",
        cancel_render,
    )
    with pytest.raises(asyncio.CancelledError):
        _run(
            generation._save_generate(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                effects_chain=[{"type": "gain", "enabled": True}],
                db=db,
            )
        )

    version = db.query(GenerationVersion).one()
    assert version.label == "original"
    assert (data_dir / "generations" / "generation.wav").is_file()
    assert not (data_dir / "generations" / "generation_processed.wav").exists()
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_regenerate_discards_wav_when_version_creation_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(
        versions,
        "create_version",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("version insert failed")),
    )

    with pytest.raises(RuntimeError, match="version insert failed"):
        _run(
            generation._save_regenerate(
                generation_id="generation",
                version_id=None,
                audio=_audio(),
                sample_rate=24_000,
                db=db,
            )
        )

    assert db.query(GenerationVersion).count() == 0
    _assert_no_publication_artifacts(data_dir)
    db.close()


def test_retry_discards_wav_when_generation_owner_commit_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    monkeypatch.setattr(
        db,
        "commit",
        lambda: (_ for _ in ()).throw(RuntimeError("owner commit failed")),
    )

    with pytest.raises(RuntimeError, match="owner commit failed"):
        _run(
            generation._save_retry(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                db=db,
            )
        )

    assert db.query(Generation).one().audio_path is None
    _assert_no_publication_artifacts(data_dir)
    db.close()


def test_retry_returns_success_after_owner_commit_then_raise(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)

    stored_path = _run(
        generation._save_retry(
            generation_id="generation",
            audio=_audio(),
            sample_rate=24_000,
            db=db,
        )
    )

    row = db.query(Generation).one()
    assert stored_path == row.audio_path
    audio_path = config.resolve_storage_path(row.audio_path)
    assert audio_path is not None
    with wave.open(str(audio_path), "rb") as wav:
        assert wav.getnframes() == len(_audio())
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_retry_commits_new_path_before_retiring_existing_owned_wav(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio_path = data_dir / "generations" / "generation.wav"
    audio_path.write_bytes(b"old retry")
    row = db.query(Generation).one()
    row.audio_path = "generations/generation.wav"
    db.commit()

    stored_path = _run(
        generation._save_retry(
            generation_id="generation",
            audio=_audio(),
            sample_rate=24_000,
            duration=0.01,
            db=db,
        )
    )

    assert stored_path.startswith("generations/generation_retry_")
    assert not audio_path.exists()
    new_audio_path = config.resolve_storage_path(stored_path)
    assert new_audio_path is not None
    with wave.open(str(new_audio_path), "rb") as wav:
        assert wav.getnframes() == len(_audio())
    row = db.query(Generation).one()
    assert row.audio_path == stored_path
    assert row.duration == 0.01
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_retry_commit_failure_restores_existing_bytes_and_metadata(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    old_audio_path = data_dir / "generations" / "generation.wav"
    old_audio_path.write_bytes(b"old retry audio")
    row = db.query(Generation).one()
    row.audio_path = "generations/generation.wav"
    row.duration = 123.0
    db.commit()
    monkeypatch.setattr(
        db,
        "commit",
        lambda: (_ for _ in ()).throw(RuntimeError("owner commit failed")),
    )

    with pytest.raises(RuntimeError, match="owner commit failed"):
        _run(
            generation._save_retry(
                generation_id="generation",
                audio=_audio(),
                sample_rate=24_000,
                duration=0.01,
                db=db,
            )
        )

    db.expire_all()
    row = db.query(Generation).one()
    assert row.audio_path == "generations/generation.wav"
    assert row.duration == 123.0
    assert old_audio_path.read_bytes() == b"old retry audio"
    assert not list((data_dir / "generations").glob("generation_retry_*.wav"))
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_retry_commit_then_raise_keeps_new_bytes_and_retires_predecessor(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    old_audio_path = data_dir / "generations" / "generation.wav"
    old_audio_path.write_bytes(b"old retry audio")
    row = db.query(Generation).one()
    row.audio_path = "generations/generation.wav"
    row.duration = 123.0
    db.commit()
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    stored_path = _run(
        generation._save_retry(
            generation_id="generation",
            audio=_audio(),
            sample_rate=24_000,
            duration=0.01,
            db=db,
        )
    )

    db.expire_all()
    row = db.query(Generation).one()
    assert stored_path == row.audio_path
    assert row.audio_path.startswith("generations/generation_retry_")
    assert row.duration == 0.01
    assert not old_audio_path.exists()
    new_audio_path = config.resolve_storage_path(row.audio_path)
    assert new_audio_path is not None
    with wave.open(str(new_audio_path), "rb") as wav:
        assert wav.getnframes() == len(_audio())
    assert not list((data_dir / "generations").glob(".voicebox-delete-*"))
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_run_retry_completes_after_owner_commit_ack_failure(
    tmp_path,
    monkeypatch,
):
    _data_dir, db = _database(tmp_path, monkeypatch)
    engine = db.get_bind()
    real_commit = db.commit
    commit_count = 0

    def fail_retry_ack_after_commit():
        nonlocal commit_count
        commit_count += 1
        real_commit()
        if commit_count == 2:
            raise RuntimeError("connection failed after retry commit")

    class FakeBackend:
        def is_loaded(self):
            return True

    fake_backend = FakeBackend()

    @asynccontextmanager
    async def loaded_backend(_engine, _model_size):
        yield fake_backend

    async def voice_prompt(*_args, **_kwargs):
        return {}

    async def generate_audio(*_args, **_kwargs):
        return _audio(), 24_000

    monkeypatch.setattr(db, "commit", fail_retry_ack_after_commit)
    monkeypatch.setattr(generation, "get_db", lambda: iter([db]))
    monkeypatch.setattr(generation, "get_task_manager", lambda: Mock(complete_generation=Mock()))
    monkeypatch.setattr(generation, "_notify_speak_end", Mock())
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", voice_prompt)
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: fake_backend)
    monkeypatch.setattr("backend.backends.engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr("backend.backends.engine_retries_runaway", lambda _engine: False)
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.loaded_tts_backend_for_request",
        loaded_backend,
    )
    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", generate_audio)

    asyncio.run(
        generation.run_generation(
            generation_id="generation",
            profile_id="profile",
            text="retry text",
            language="en",
            engine="qwen",
            model_size="1.7B",
            seed=7,
            mode="retry",
        )
    )

    with sessionmaker(bind=engine)() as durable_db:
        row = durable_db.query(Generation).one()
        assert row.status == "completed"
        assert row.error is None
        assert row.audio_path.startswith("generations/generation_retry_")
        audio_path = config.resolve_storage_path(row.audio_path)
        assert audio_path is not None
        assert audio_path.is_file()


def test_retry_keeps_predecessor_referenced_by_a_version(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    old_audio_path = data_dir / "generations" / "generation.wav"
    old_audio_path.write_bytes(b"shared version audio")
    row = db.query(Generation).one()
    row.audio_path = "generations/generation.wav"
    db.add(
        GenerationVersion(
            id="old-version",
            generation_id="generation",
            label="Old",
            audio_path="generations/generation.wav",
            is_default=True,
        )
    )
    db.commit()

    stored_path = _run(
        generation._save_retry(
            generation_id="generation",
            audio=_audio(),
            sample_rate=24_000,
            db=db,
        )
    )

    assert stored_path != "generations/generation.wav"
    assert old_audio_path.read_bytes() == b"shared version audio"
    assert db.query(GenerationVersion).one().audio_path == "generations/generation.wav"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_startup_discards_partial_wav_after_crash_during_population(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    if not hasattr(os, "fork"):
        pytest.skip("hard-crash recovery regression requires fork")
    os_temp = tmp_path / "os-temp"
    os_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(os_temp))

    def crash_during_population(descriptor, _audio, _sample_rate, _intent):
        os.ftruncate(descriptor, 0)
        os.write(descriptor, b"partial wav")
        os.fsync(descriptor)
        os._exit(23)

    monkeypatch.setattr(
        generation,
        "_populate_generation_audio_payload",
        crash_during_population,
    )

    child = os.fork()
    if child == 0:
        asyncio.run(
            generation._publish_generation_audio(
                target=data_dir / "generations" / "generation.wav",
                audio=_audio(),
                sample_rate=24_000,
                db=db,
            )
        )
        os._exit(24)
    _child, status = os.waitpid(child, 0)

    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 23

    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1
    assert len(list((data_dir / "generations").glob(".voicebox-delete-*"))) == 1
    assert list(os_temp.iterdir()) == []

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.discarded == 1
    assert report.unresolved == 0
    _assert_no_publication_artifacts(data_dir)
    db.close()
