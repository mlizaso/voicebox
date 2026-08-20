"""Resource, cancellation, and fidelity regressions for effects processing."""

from __future__ import annotations

import asyncio
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config, models
from backend.database import Base, Generation, VoiceProfile
from backend.routes import effects as effects_routes
from backend.services import effects_processing
from backend.utils import disk_reservations
from backend.utils.audio import load_audio, normalize_audio
from backend.utils.effects import MAX_EFFECTS_CHAIN_LENGTH, apply_effects, validate_effects_chain


@pytest.fixture
def storage_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", root)
    config.initialize_data_permissions()
    try:
        yield root
    finally:
        for directory in list(effects_processing._active_effects_directories):
            effects_processing._cleanup_effects_directory(directory)


@pytest.fixture
def db(storage_root):
    engine = create_engine(f"sqlite:///{storage_root / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _write_source(path: Path, *, frames: int = 48_001, sample_rate: int = 48_000) -> None:
    time = np.arange(frames, dtype=np.float32) / sample_rate
    mono = (0.15 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    sf.write(path, np.column_stack((mono, mono * 0.5)), sample_rate, subtype="PCM_16")


async def _render(source: Path, output: Path, chain: list[dict]) -> None:
    descriptor = os.open(output, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        await effects_processing.render_effects_to_descriptor(
            source,
            descriptor,
            output.parent,
            chain,
        )
    finally:
        os.close(descriptor)


def _add_generation(db, storage_root: Path, audio_path: str = "generations/source.wav") -> Generation:
    profile = VoiceProfile(id="profile", name="Narrator", voice_type="cloned")
    generation = Generation(
        id="generation",
        profile_id=profile.id,
        text="speech",
        language="en",
        audio_path=audio_path,
        duration=1.0,
        status="completed",
        created_at=datetime.utcnow(),
    )
    db.add_all((profile, generation))
    db.commit()
    return generation


def test_effect_chain_rejects_unbounded_and_nonfinite_parameters():
    too_many = [{"type": "gain", "params": {}}] * (MAX_EFFECTS_CHAIN_LENGTH + 1)
    assert validate_effects_chain(too_many) == (f"effects_chain may contain at most {MAX_EFFECTS_CHAIN_LENGTH} effects")
    assert "finite" in validate_effects_chain([{"type": "gain", "params": {"gain_db": float("nan")}}])
    assert "number" in validate_effects_chain([{"type": "gain", "params": {"gain_db": True}}])
    assert "boolean" in validate_effects_chain([{"type": "gain", "enabled": "yes"}])
    assert "Unknown effect" in validate_effects_chain([{"type": ["gain"]}])
    configs = [models.EffectConfig(type="gain")] * (MAX_EFFECTS_CHAIN_LENGTH + 1)
    with pytest.raises(ValueError, match="too_long"):
        models.GenerationRequest(profile_id="profile", text="speech", effects_chain=configs)
    with pytest.raises(ValueError, match="too_long"):
        models.ApplyEffectsRequest(effects_chain=configs)


def test_duration_contract_admits_seven_hour_audiobook_without_allocating_audio():
    frames = (7 * 60 * 60 + 43 * 60) * effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE
    assert (
        effects_processing._validate_source_shape(
            frames=frames,
            sample_rate=effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE,
            channels=1,
        )
        == frames
    )
    with pytest.raises(effects_processing.EffectsProcessingLimitError, match="24-hour"):
        effects_processing._validate_source_shape(
            frames=(effects_processing.EFFECTS_MAX_DURATION_SECONDS + 1)
            * effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE,
            sample_rate=effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE,
            channels=1,
        )


@pytest.mark.asyncio
async def test_streamed_gain_matches_legacy_quality_and_half_up_duration(storage_root, monkeypatch):
    source = storage_root / "generations" / "source.wav"
    _write_source(source)
    output = storage_root / "generations" / "processed.wav"
    chain = [{"type": "gain", "params": {"gain_db": 3.0}}]
    monkeypatch.setattr(effects_processing, "EFFECTS_PROCESS_BLOCK_FRAMES", 4096)

    expected, sample_rate = load_audio(str(source))
    expected = apply_effects(expected, sample_rate, chain)
    await _render(source, output, chain)
    actual, actual_rate = sf.read(output, dtype="float32")

    assert actual_rate == effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE
    assert len(actual) == len(expected) == 24_001
    # The only material difference is PCM16 quantization, matching the old
    # route's sf.write(..., format="WAV") output contract.
    assert np.max(np.abs(actual - expected)) <= 3.1e-5


@pytest.mark.asyncio
@pytest.mark.parametrize("frames", [2_403, 150_003])
async def test_streamed_pitch_shift_flushes_exact_duration(storage_root, monkeypatch, frames):
    source = storage_root / "generations" / "source.wav"
    _write_source(source, frames=frames, sample_rate=24_000)
    output = storage_root / "generations" / "pitched.wav"
    monkeypatch.setattr(effects_processing, "EFFECTS_PROCESS_BLOCK_FRAMES", 65_536)

    await _render(source, output, [{"type": "pitch_shift", "params": {"semitones": 2.0}}])

    audio, sample_rate = sf.read(output, dtype="float32")
    assert sample_rate == 24_000
    assert len(audio) == frames
    assert np.isfinite(audio).all()
    assert np.max(np.abs(audio)) > 0.01


@pytest.mark.asyncio
async def test_pitch_shift_preserves_tail_and_boundary_fidelity_at_production_block_size(
    storage_root,
):
    frames = effects_processing.EFFECTS_PROCESS_BLOCK_FRAMES + 151_424
    sample_rate = effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE
    timeline = np.arange(frames, dtype=np.float32) / sample_rate
    source_audio = (0.15 * np.sin(2 * np.pi * 220 * timeline) + 0.04 * np.sin(2 * np.pi * 773 * timeline)).astype(
        np.float32
    )
    source = storage_root / "generations" / "source.wav"
    output = storage_root / "generations" / "pitched.wav"
    sf.write(source, source_audio, sample_rate, subtype="PCM_16")

    await _render(source, output, [{"type": "pitch_shift", "params": {"semitones": 2.0}}])

    audio, actual_rate = sf.read(output, dtype="float32")
    boundary = effects_processing.EFFECTS_PROCESS_BLOCK_FRAMES
    boundary_rms = float(np.sqrt(np.mean(audio[boundary - 12_000 : boundary + 12_000] ** 2)))
    tail_rms = float(np.sqrt(np.mean(audio[-24_000:] ** 2)))
    assert actual_rate == sample_rate
    assert len(audio) == frames
    assert boundary_rms > 0.07
    assert tail_rms > 0.07
    assert not list((storage_root / "cache" / effects_processing.EFFECTS_EXPORT_ROOT_NAME).glob("job-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize(("phase", "subtype"), [(0.0, "FLOAT"), (0.37, "PCM_16")])
async def test_pitch_shift_crossfade_does_not_cancel_anticorrelated_windows(
    storage_root,
    phase,
    subtype,
):
    frames = effects_processing.EFFECTS_PROCESS_BLOCK_FRAMES + 151_424
    sample_rate = effects_processing.EFFECTS_OUTPUT_SAMPLE_RATE
    timeline = np.arange(frames, dtype=np.float64) / sample_rate
    source_audio = (0.1 * np.sin(2 * np.pi * 260 * timeline + phase)).astype(np.float32)
    source = storage_root / "generations" / "source.wav"
    output = storage_root / "generations" / "pitched.wav"
    sf.write(source, source_audio, sample_rate, subtype=subtype)

    await _render(source, output, [{"type": "pitch_shift", "params": {"semitones": 2.0}}])

    audio, _sample_rate = sf.read(output, dtype="float32")
    boundary = effects_processing.EFFECTS_PROCESS_BLOCK_FRAMES
    overlap = math.ceil(sample_rate * effects_processing.EFFECTS_PITCH_OVERLAP_SECONDS)
    seam = audio[boundary - overlap : boundary + overlap]
    rolling_rms = [float(np.sqrt(np.mean(seam[start : start + 256] ** 2))) for start in range(0, len(seam) - 255, 64)]
    baseline_rms = float(np.sqrt(np.mean(audio[boundary - 24_000 : boundary - overlap] ** 2)))
    assert min(rolling_rms) > baseline_rms * 0.72


@pytest.mark.asyncio
async def test_generated_audio_response_file_matches_short_background_processing(storage_root):
    sample_rate = 24_000
    timeline = np.arange(48_001, dtype=np.float32) / sample_rate
    audio = (0.15 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    chain = [{"type": "gain", "params": {"gain_db": 3.0}}]
    expected = apply_effects(normalize_audio(audio), sample_rate, chain)

    export = await effects_processing.create_generated_audio_response_file(
        audio,
        sample_rate,
        chain,
        True,
    )
    actual, actual_rate = sf.read(export.path, dtype="float32")

    assert actual_rate == sample_rate
    assert len(actual) == len(expected)
    assert np.max(np.abs(actual - expected)) <= 3.1e-5
    export.cleanup()
    assert not export.temporary_directory.exists()


@pytest.mark.asyncio
async def test_generated_audio_response_file_normalizes_long_input_before_disk_effects(
    storage_root,
    monkeypatch,
):
    monkeypatch.setattr(effects_processing, "EFFECTS_PROCESS_BLOCK_FRAMES", 4096)
    sample_rate = 24_000
    timeline = np.arange(15_003, dtype=np.float32) / sample_rate
    audio = (0.15 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
    chain = [{"type": "gain", "params": {"gain_db": -3.0}}]
    expected = apply_effects(normalize_audio(audio), sample_rate, chain)

    export = await effects_processing.create_generated_audio_response_file(
        audio,
        sample_rate,
        chain,
        True,
    )
    actual, actual_rate = sf.read(export.path, dtype="float32")

    assert actual_rate == sample_rate
    assert len(actual) == len(expected)
    assert np.max(np.abs(actual - expected)) <= 3.1e-5
    export.cleanup()
    assert not list((storage_root / "cache" / effects_processing.EFFECTS_EXPORT_ROOT_NAME).glob("job-*"))


@pytest.mark.asyncio
async def test_generated_audio_response_file_rejects_invalid_arrays_without_scratch(storage_root):
    with pytest.raises(effects_processing.EffectsProcessingError, match="float32"):
        await effects_processing.create_generated_audio_response_file(
            np.zeros(100, dtype=np.float64),
            24_000,
            [],
            False,
        )
    with pytest.raises(effects_processing.EffectsProcessingError, match="non-finite"):
        await effects_processing.create_generated_audio_response_file(
            np.array([0.0, np.nan], dtype=np.float32),
            24_000,
            [],
            False,
        )
    root = storage_root / "cache" / effects_processing.EFFECTS_EXPORT_ROOT_NAME
    assert not root.exists() or not list(root.glob("job-*"))


@pytest.mark.asyncio
async def test_cancelled_preview_signals_drains_and_removes_all_scratch(
    storage_root,
    monkeypatch,
):
    source = storage_root / "generations" / "source.wav"
    _write_source(source)
    started = threading.Event()

    def cancellable_worker(*_args, cancel_event, **_kwargs):
        started.set()
        assert cancel_event.wait(timeout=5)
        effects_processing._check_cancelled(cancel_event)

    monkeypatch.setattr(effects_processing, "_render_effects_audio_to_fd", cancellable_worker)
    task = asyncio.create_task(effects_processing.create_effects_preview(source, []))
    while not started.is_set():
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert effects_processing._active_effects_directories == set()
    root = storage_root / "cache" / effects_processing.EFFECTS_EXPORT_ROOT_NAME
    assert not list(root.glob("job-*"))
    assert not effects_processing._effects_processing_lock.locked()


@pytest.mark.asyncio
async def test_effects_worker_rejects_concurrency_without_queuing_waiters(storage_root, monkeypatch):
    source = storage_root / "generations" / "source.wav"
    _write_source(source)
    first_started = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0
    calls = 0

    def controlled_worker(_source, output_descriptor, *_args, cancel_event, **_kwargs):
        nonlocal active, maximum_active, calls
        with state_lock:
            calls += 1
            call = calls
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call == 1:
                first_started.set()
                assert release_first.wait(timeout=5)
            effects_processing._check_cancelled(cancel_event)
            os.write(output_descriptor, b"preview")
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(effects_processing, "_render_effects_audio_to_fd", controlled_worker)
    first = asyncio.create_task(effects_processing.create_effects_preview(source, []))
    while not first_started.is_set():
        await asyncio.sleep(0)
    second = asyncio.create_task(effects_processing.create_effects_preview(source, []))
    with pytest.raises(effects_processing.EffectsProcessingBusyError, match="already running"):
        await second
    assert calls == 1
    release_first.set()
    preview = await first
    preview.cleanup()

    assert calls == 1
    assert maximum_active == 1
    assert effects_processing._active_effects_directories == set()


def test_preview_scratch_leases_have_a_hard_process_bound(storage_root):
    root = config.get_cache_dir() / effects_processing.EFFECTS_EXPORT_ROOT_NAME
    root.mkdir(mode=0o700)
    directories = []
    for index in range(effects_processing.EFFECTS_MAX_ACTIVE_DIRECTORIES):
        directory = root / f"job-active-{index}"
        directory.mkdir(mode=0o700)
        directories.append(directory)
        effects_processing._active_effects_directories.add(directory)

    with pytest.raises(effects_processing.EffectsProcessingBusyError, match="Too many"):
        effects_processing._allocate_effects_directory()

    for directory in directories:
        effects_processing._cleanup_effects_directory(directory)
    assert effects_processing._active_effects_directories == set()


@pytest.mark.asyncio
async def test_preview_disconnect_removes_disk_response_and_releases_active_entry(
    db,
    storage_root,
):
    source = storage_root / "generations" / "source.wav"
    _write_source(source)
    _add_generation(db, storage_root)
    response = await effects_routes.preview_effects(
        "generation",
        models.ApplyEffectsRequest(effects_chain=[]),
        db,
    )
    preview_path = Path(response.path)
    preview_directory = preview_path.parent

    async def receive():
        return {"type": "http.disconnect"}

    async def broken_send(_message):
        raise ConnectionError("client disconnected")

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "path": "/effects/preview/generation",
        "raw_path": b"/effects/preview/generation",
        "query_string": b"",
        "headers": [],
    }
    with pytest.raises(ConnectionError, match="client disconnected"):
        await response(scope, receive, broken_send)

    assert not preview_path.exists()
    assert not preview_directory.exists()
    assert effects_processing._active_effects_directories == set()


@pytest.mark.asyncio
async def test_apply_effects_streams_directly_into_journaled_version(db, storage_root):
    source = storage_root / "generations" / "source.wav"
    _write_source(source)
    _add_generation(db, storage_root)

    version = await effects_routes.apply_effects_to_generation(
        "generation",
        models.ApplyEffectsRequest(
            effects_chain=[models.EffectConfig(type="gain", params={"gain_db": 2.0})],
            set_as_default=False,
        ),
        db,
    )

    output = config.resolve_storage_path(version.audio_path)
    assert output is not None
    info = sf.info(output)
    assert info.frames == 24_001
    assert info.samplerate == 24_000
    assert info.channels == 1
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    assert not list(config.get_generations_dir().glob(".voicebox-delete-effects-*"))
    assert effects_processing._active_effects_directories == set()


@pytest.mark.asyncio
async def test_apply_capacity_failure_reconciles_journal_and_scratch(db, storage_root, monkeypatch):
    source = storage_root / "generations" / "source.wav"
    _write_source(source)
    _add_generation(db, storage_root)
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=effects_processing.EFFECTS_MIN_FREE_BYTES),
    )

    with pytest.raises(HTTPException) as raised:
        await effects_routes.apply_effects_to_generation(
            "generation",
            models.ApplyEffectsRequest(effects_chain=[]),
            db,
        )

    assert raised.value.status_code == 507
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    assert not list(config.get_generations_dir().glob(".voicebox-delete-effects-*"))
    assert effects_processing._active_effects_directories == set()
    assert not effects_processing._effects_processing_lock.locked()


@pytest.mark.asyncio
async def test_preview_refuses_managed_symlink_source(db, storage_root):
    actual = storage_root / "actual.wav"
    _write_source(actual)
    source = storage_root / "generations" / "source.wav"
    source.symlink_to(actual)
    _add_generation(db, storage_root)

    with pytest.raises(HTTPException) as raised:
        await effects_routes.preview_effects(
            "generation",
            models.ApplyEffectsRequest(effects_chain=[]),
            db,
        )

    assert raised.value.status_code == 400
    assert "unsafe" in raised.value.detail
    assert effects_processing._active_effects_directories == set()
