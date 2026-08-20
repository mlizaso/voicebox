"""Bounded-memory regressions for long-form TTS generation."""

from __future__ import annotations

import asyncio
import gc
import shutil
import tempfile
import threading
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from backend import config
from backend.services import effects_processing, generation
from backend.utils import audio as audio_utils, chunked_tts, disk_reservations
from backend.utils.audio import normalize_audio


class _ChunkBackend:
    tts_operations_are_cancellable = True

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = iter(chunks)

    async def generate(self, *_args, **_kwargs):
        return next(self._chunks), 24_000


def _configure_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    return data_dir


@pytest.mark.asyncio
@pytest.mark.parametrize("crossfade_ms", [0, 1, 50, 500])
async def test_chunked_generation_disk_output_is_byte_identical_to_legacy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crossfade_ms: int,
):
    _configure_storage(tmp_path, monkeypatch)
    rng = np.random.default_rng(20260815)
    chunks = [rng.standard_normal(length).astype(np.float32) for length in (1700, 0, 1001, 3000, 2400, 1, 997)]
    monkeypatch.setattr(
        chunked_tts,
        "split_text_into_chunks",
        lambda _text, _max_chars: [f"chunk-{index}" for index in range(len(chunks))],
    )
    expected = chunked_tts.concatenate_audio_chunks(chunks, 24_000, crossfade_ms)

    audio, sample_rate = await chunked_tts.generate_chunked(
        _ChunkBackend(chunks.copy()),
        "long text",
        {},
        max_chunk_chars=100,
        crossfade_ms=crossfade_ms,
    )
    try:
        assert sample_rate == 24_000
        assert chunked_tts.is_disk_backed_audio(audio)
        assert audio.dtype == np.float32
        assert audio.shape == expected.shape
        assert audio.tobytes() == expected.tobytes()
    finally:
        chunked_tts.release_disk_backed_audio(audio)


@pytest.mark.asyncio
async def test_chunked_generation_does_not_retain_completed_model_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_storage(tmp_path, monkeypatch)
    references: list[weakref.ReferenceType[np.ndarray]] = []

    class Backend:
        tts_operations_are_cancellable = True

        async def generate(self, text, *_args, **_kwargs):
            audio = np.full(200_000, int(text[-1]), dtype=np.float32)
            references.append(weakref.ref(audio))
            return audio, 24_000

    monkeypatch.setattr(
        chunked_tts,
        "split_text_into_chunks",
        lambda _text, _max_chars: [f"chunk-{index}" for index in range(8)],
    )
    audio, _sample_rate = await chunked_tts.generate_chunked(
        Backend(),
        "long text",
        {},
        max_chunk_chars=100,
    )
    try:
        gc.collect()
        assert all(reference() is None for reference in references)
        assert chunked_tts.is_disk_backed_audio(audio)
        assert audio.nbytes < 8 * 200_000 * np.dtype(np.float32).itemsize
    finally:
        chunked_tts.release_disk_backed_audio(audio)


@pytest.mark.asyncio
async def test_chunked_generation_cancellation_closes_visible_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    data_dir = _configure_storage(tmp_path, monkeypatch)
    second_started = asyncio.Event()
    never = asyncio.Event()
    real_named_temporary_file = tempfile.NamedTemporaryFile

    def visible_temporary_file(**kwargs):
        return real_named_temporary_file(delete=True, **kwargs)

    monkeypatch.setattr(chunked_tts.tempfile, "TemporaryFile", visible_temporary_file)
    monkeypatch.setattr(
        chunked_tts,
        "split_text_into_chunks",
        lambda _text, _max_chars: ["first", "second"],
    )

    class Backend:
        tts_operations_are_cancellable = True

        async def generate(self, text, *_args, **_kwargs):
            if text == "first":
                return np.ones(2000, dtype=np.float32), 24_000
            second_started.set()
            await never.wait()
            return np.empty(0, dtype=np.float32), 24_000

    task = asyncio.create_task(
        chunked_tts.generate_chunked(
            Backend(),
            "long text",
            {},
            max_chunk_chars=100,
        )
    )
    await second_started.wait()
    assert list((data_dir / "cache").glob(".generation-audio-*"))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not list((data_dir / "cache").glob(".generation-audio-*"))


@pytest.mark.asyncio
async def test_chunked_generation_enforces_duration_and_disk_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_storage(tmp_path, monkeypatch)
    monkeypatch.setattr(chunked_tts, "MAX_GENERATED_AUDIO_DURATION_SECONDS", 1)
    monkeypatch.setattr(
        chunked_tts,
        "split_text_into_chunks",
        lambda _text, _max_chars: ["first", "second"],
    )
    oversized = [np.ones(20_000, dtype=np.float32), np.ones(20_000, dtype=np.float32)]
    with pytest.raises(chunked_tts.GeneratedAudioLimitError, match=r"1-hour|0-hour"):
        await chunked_tts.generate_chunked(
            _ChunkBackend(oversized),
            "long text",
            {},
            max_chunk_chars=100,
            crossfade_ms=0,
        )

    real_usage = shutil.disk_usage(config.get_cache_dir())
    monkeypatch.setattr(chunked_tts, "MAX_GENERATED_AUDIO_DURATION_SECONDS", 24 * 60 * 60)
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            real_usage.total,
            real_usage.used,
            chunked_tts.GENERATED_AUDIO_MIN_FREE_BYTES,
        ),
    )
    with pytest.raises(chunked_tts.GeneratedAudioStorageError, match="free space"):
        await chunked_tts.generate_chunked(
            _ChunkBackend([np.ones(2000, dtype=np.float32), np.ones(2000, dtype=np.float32)]),
            "long text",
            {},
            max_chunk_chars=100,
            crossfade_ms=0,
        )


def test_disk_backed_normalization_is_byte_identical_and_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_storage(tmp_path, monkeypatch)
    source = np.random.default_rng(91).standard_normal(2_200_003).astype(np.float32)
    expected = normalize_audio(source.copy())
    accumulator = chunked_tts._DiskBackedChunkAccumulator(24_000, 0)
    accumulator.append(source)
    audio = accumulator.finish()
    temporary_file = audio._voicebox_temporary_file
    try:
        normalized = normalize_audio(audio)
        assert normalized is audio
        assert normalized.tobytes() == expected.tobytes()
    finally:
        chunked_tts.release_disk_backed_audio(audio)
    assert temporary_file.closed
    assert not chunked_tts.is_disk_backed_audio(audio)


def test_disk_backed_normalization_fails_before_mutation_when_reserve_is_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_storage(tmp_path, monkeypatch)
    source = np.linspace(-0.2, 0.2, 20_001, dtype=np.float32)
    accumulator = chunked_tts._DiskBackedChunkAccumulator(24_000, 0)
    accumulator.append(source)
    audio = accumulator.finish()
    before = audio.tobytes()
    usage = shutil.disk_usage(config.get_cache_dir())
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(
            usage.total,
            usage.used,
            audio_utils._NORMALIZE_MIN_FREE_BYTES,
        ),
    )
    try:
        with pytest.raises(OSError, match="free space"):
            normalize_audio(audio)
        assert audio.tobytes() == before
    finally:
        chunked_tts.release_disk_backed_audio(audio)


def test_effects_reservation_blocks_concurrent_generation_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """One filesystem-wide lease cannot be double-spent by another writer."""
    _configure_storage(tmp_path, monkeypatch)
    cache_directory = config.get_cache_dir()
    generations_directory = config.get_generations_dir()
    assert cache_directory.stat().st_dev == generations_directory.stat().st_dev

    effect_frames = 4_096
    effect_bytes = effect_frames * 2 + 65_536
    chunk = np.ones(8_192, dtype=np.float32)
    available = chunked_tts.GENERATED_AUDIO_MIN_FREE_BYTES + effect_bytes + chunk.nbytes - 1
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: shutil._ntuple_diskusage(available * 2, available, available),
    )

    lease_held = threading.Event()
    release_lease = threading.Event()
    thread_errors: list[BaseException] = []

    def hold_effects_lease() -> None:
        try:
            with effects_processing._reserve_output_capacity(generations_directory, effect_frames):
                lease_held.set()
                if not release_lease.wait(timeout=5):
                    raise TimeoutError("test did not release the effects reservation")
        except BaseException as exc:
            thread_errors.append(exc)
            lease_held.set()

    disk_reservations._clear_reservations_for_tests()
    thread = threading.Thread(target=hold_effects_lease)
    accumulator = chunked_tts._DiskBackedChunkAccumulator(24_000, 0)
    try:
        thread.start()
        assert lease_held.wait(timeout=5)
        assert not thread_errors
        assert disk_reservations.reserved_bytes(cache_directory) == effect_bytes

        with pytest.raises(chunked_tts.GeneratedAudioStorageError, match="free space"):
            accumulator.append(chunk)
        assert accumulator.frames == 0

        release_lease.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not thread_errors
        assert disk_reservations.reserved_bytes(cache_directory) == 0

        accumulator.append(chunk)
        assert accumulator.frames == len(chunk)
    finally:
        release_lease.set()
        thread.join(timeout=5)
        accumulator.close()
        disk_reservations._clear_reservations_for_tests()


@pytest.mark.asyncio
async def test_legacy_bytes_generation_rejects_long_result_and_releases_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_storage(tmp_path, monkeypatch)
    accumulator = chunked_tts._DiskBackedChunkAccumulator(1, 0)
    accumulator.append(np.zeros(generation.SYNC_AUDIO_MAX_DURATION_SECONDS + 1, dtype=np.float32))
    audio = accumulator.finish()
    temporary_file = audio._voicebox_temporary_file

    @asynccontextmanager
    async def loaded_backend(_engine, _model_size):
        yield object()

    async def run_operation(_backend, operation):
        return await operation

    async def create_prompt(*_args, **_kwargs):
        return {}

    async def generate(*_args, **_kwargs):
        return audio, 1

    class DB:
        closed = False

        def close(self):
            self.closed = True

    db = DB()
    monkeypatch.setattr(generation, "get_db", lambda: iter([db]))
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", create_prompt)
    monkeypatch.setattr("backend.backends.engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr("backend.backends.engine_retries_runaway", lambda _engine: False)
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.loaded_tts_backend_for_request",
        loaded_backend,
    )
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.run_tts_operation_cancellation_safe",
        run_operation,
    )
    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", generate)

    with pytest.raises(ValueError, match="10 minutes"):
        await generation.generate_audio_sync(
            profile_id="profile",
            text="long",
            language="en",
            engine="qwen",
            model_size="1.7B",
        )

    assert temporary_file.closed
    assert db.closed


@pytest.mark.asyncio
async def test_background_generation_releases_mapping_after_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_storage(tmp_path, monkeypatch)
    accumulator = chunked_tts._DiskBackedChunkAccumulator(24_000, 0)
    accumulator.append(np.zeros(2000, dtype=np.float32))
    audio = accumulator.finish()
    temporary_file = audio._voicebox_temporary_file
    backend = Mock()
    backend.is_loaded.return_value = True

    @asynccontextmanager
    async def loaded_backend(_engine, _model_size):
        yield backend

    async def run_operation(_backend, operation):
        return await operation

    async def create_prompt(*_args, **_kwargs):
        return {}

    async def generate(*_args, **_kwargs):
        return audio, 24_000

    async def fail_save(**_kwargs):
        raise OSError("publication failed")

    class DB:
        closed = False

        def close(self):
            self.closed = True

    db = DB()
    task_manager = Mock()
    update_status = AsyncMock(return_value=object())
    monkeypatch.setattr(generation, "get_db", lambda: iter([db]))
    monkeypatch.setattr(generation, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", create_prompt)
    monkeypatch.setattr(generation.history, "update_generation_status", update_status)
    monkeypatch.setattr(generation, "_save_generate", fail_save)
    monkeypatch.setattr(generation, "_notify_speak_end", Mock())
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: backend)
    monkeypatch.setattr("backend.backends.engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr("backend.backends.engine_retries_runaway", lambda _engine: False)
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.loaded_tts_backend_for_request",
        loaded_backend,
    )
    monkeypatch.setattr(
        "backend.backends.mlx_tts_lifecycle.run_tts_operation_cancellation_safe",
        run_operation,
    )
    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", generate)

    await generation.run_generation(
        generation_id="generation",
        profile_id="profile",
        text="long",
        language="en",
        engine="qwen",
        model_size="1.7B",
        seed=7,
        mode="generate",
    )

    assert temporary_file.closed
    assert db.closed
    task_manager.complete_generation.assert_called_once_with("generation")
    assert any(call.kwargs.get("status") == "failed" for call in update_status.await_args_list)
