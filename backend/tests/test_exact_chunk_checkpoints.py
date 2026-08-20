"""Model-free tests for durable exact-generation chunk checkpoints."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from backend.services import exact_chunk_checkpoints as checkpoint_module
from backend.services.exact_chunk_checkpoints import (
    CheckpointCapacityError,
    ExactChunkCheckpointKey,
    ExactChunkCheckpointSession,
    ExactChunkCheckpointStore,
    InvalidCheckpointAudioError,
)
from backend.utils import chunked_tts, disk_reservations

REQUEST_SHA = "a" * 64


def _key(index: int = 0, text: str = "uno", seed: int = 41) -> ExactChunkCheckpointKey:
    return ExactChunkCheckpointKey.from_text(
        exact_request_sha256=REQUEST_SHA,
        logical_index=index,
        text=text,
        seed=seed,
    )


def test_round_trip_preserves_float32_pcm_exactly(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    audio = np.linspace(-0.75, 0.75, 1025, dtype=np.float32)

    path = store.save(_key(), audio, 24_000)
    loaded = store.load(_key())

    assert path.stat().st_mode & 0o077 == 0
    assert loaded is not None
    assert loaded.sample_rate == 24_000
    assert loaded.audio.dtype == np.float32
    assert loaded.audio.flags.writeable
    assert loaded.audio.tobytes() == audio.tobytes()


@pytest.mark.parametrize(
    ("audio", "sample_rate"),
    [
        (np.zeros((2, 2), dtype=np.float32), 24_000),
        (np.array([], dtype=np.float32), 24_000),
        (np.array([0.0, np.nan], dtype=np.float32), 24_000),
        (np.ones(2, dtype=np.float32), 0),
    ],
)
def test_save_rejects_unsafe_pcm(audio: np.ndarray, sample_rate: int, tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    with pytest.raises(InvalidCheckpointAudioError):
        store.save(_key(), audio, sample_rate)


def test_checkpoint_cache_quota_preserves_existing_resume_data(tmp_path: Path) -> None:
    initial_store = ExactChunkCheckpointStore(
        tmp_path,
        max_store_bytes=1024 * 1024,
        min_free_bytes=0,
    )
    first_path = initial_store.save(
        _key(index=0),
        np.ones(128, dtype=np.float32),
        24_000,
    )
    first_payload = first_path.read_bytes()
    constrained_store = ExactChunkCheckpointStore(
        tmp_path,
        max_store_bytes=first_path.stat().st_size + 1,
        min_free_bytes=0,
    )

    with pytest.raises(CheckpointCapacityError, match="cache limit"):
        constrained_store.save(
            _key(index=1),
            np.ones(128, dtype=np.float32),
            24_000,
        )

    assert first_path.read_bytes() == first_payload
    assert constrained_store.load(_key(index=0)) is not None


def test_checkpoint_save_preserves_free_space_reserve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExactChunkCheckpointStore(
        tmp_path,
        max_store_bytes=1024 * 1024,
        min_free_bytes=100,
    )
    monkeypatch.setattr(
        disk_reservations.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1000, used=900, free=100),
    )

    with pytest.raises(CheckpointCapacityError, match="disk reserve"):
        store.save(_key(), np.ones(8, dtype=np.float32), 24_000)

    assert not store.checkpoint_path(_key()).exists()


def test_partial_atomic_file_is_ignored_and_never_claimed_as_complete(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    key = _key()
    final_path = store.checkpoint_path(key)
    final_path.parent.mkdir(parents=True)
    partial_path = final_path.parent / f".{final_path.name}.tmp-{'d' * 32}"
    partial_path.write_bytes(b"VBX-ECP1\n")

    assert store.load(key) is None
    assert not final_path.exists()
    assert partial_path.exists()

    store.save(key, np.array([0.1, 0.2], dtype=np.float32), 24_000)
    assert store.load(key) is not None


def test_failed_atomic_replace_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExactChunkCheckpointStore(tmp_path)

    def fail_replace(_source, _destination):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated rename failure"):
        store.save(_key(), np.array([0.1], dtype=np.float32), 24_000)

    request_dir = tmp_path / REQUEST_SHA
    assert list(request_dir.iterdir()) == []


def test_corrupt_pcm_and_wrong_contract_are_rejected(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    original_key = _key(text="original")
    wrong_key = _key(text="different")
    original_path = store.save(
        original_key,
        np.array([0.1, -0.2, 0.3], dtype=np.float32),
        24_000,
    )

    # A valid file copied under another logical contract must not be reused.
    wrong_path = store.checkpoint_path(wrong_key)
    wrong_path.write_bytes(original_path.read_bytes())
    assert store.load(wrong_key) is None
    assert not wrong_path.exists()

    payload = bytearray(original_path.read_bytes())
    payload[-1] ^= 0xFF
    original_path.write_bytes(payload)
    assert store.load(original_key) is None
    assert not original_path.exists()


def test_remove_request_only_removes_managed_regular_files(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    store.save(_key(), np.ones(8, dtype=np.float32), 24_000)
    request_dir = tmp_path / REQUEST_SHA
    unexpected = request_dir / "keep-me.txt"
    unexpected.write_text("not managed", encoding="utf-8")

    assert store.remove_request(REQUEST_SHA) is False
    assert unexpected.exists()
    assert store.checkpoint_path(_key()).exists()

    unexpected.unlink()
    assert store.remove_request(REQUEST_SHA) is True
    assert not request_dir.exists()


def test_stale_cleanup_removes_only_orphan_temporary_files(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    key = _key()
    store.save(key, np.ones(8, dtype=np.float32), 24_000)
    request_dir = tmp_path / REQUEST_SHA
    old_partial = request_dir / f".{key.filename}.tmp-{'d' * 32}"
    fresh_partial = request_dir / f".{key.filename}.tmp-{'e' * 32}"
    old_partial.write_bytes(b"partial")
    fresh_partial.write_bytes(b"partial")
    os.utime(old_partial, (1_000.0, 1_000.0))
    os.utime(fresh_partial, (9_000.0, 9_000.0))

    removed = store.prune_stale(
        now=10_000.0,
        max_age_seconds=5_000,
    )

    assert removed == 1
    assert not old_partial.exists()
    assert fresh_partial.exists()
    assert store.load(key) is not None


def test_startup_cleanup_removes_all_temps_but_keeps_resume_data(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    key = _key()
    final_path = store.save(key, np.ones(8, dtype=np.float32), 24_000)
    fresh_partial = final_path.parent / f".{key.filename}.tmp-{'f' * 32}"
    fresh_partial.write_bytes(b"partial")

    assert store.prune_abandoned_temporary_files() == 1

    assert not fresh_partial.exists()
    assert final_path.exists()
    assert store.load(key) is not None


def test_valid_paused_checkpoint_never_expires_by_age(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    key = _key()
    path = store.save(key, np.ones(8, dtype=np.float32), 24_000)
    old_time = 1_000.0
    os.utime(path, (old_time, old_time))
    os.utime(path.parent, (old_time, old_time))

    assert store.prune_stale(now=10_000_000.0, max_age_seconds=1) == 0
    assert store.load(key) is not None


def test_key_rejects_non_uint32_seed_and_noncanonical_request_hash() -> None:
    with pytest.raises(ValueError, match="uint32"):
        _key(seed=1 << 32)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        ExactChunkCheckpointKey.from_text(
            exact_request_sha256="A" * 64,
            logical_index=0,
            text="text",
            seed=1,
        )


def test_session_rejects_cached_audio_with_wrong_sample_rate(tmp_path: Path) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    store.save(_key(), np.ones(8, dtype=np.float32), 48_000)
    session = ExactChunkCheckpointSession(REQUEST_SHA, store=store)

    assert session.load(logical_index=0, text="uno", seed=41) is None
    assert not store.checkpoint_path(_key()).exists()


def test_session_cleanup_is_nonfatal_and_leaves_data_for_stale_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ExactChunkCheckpointStore(tmp_path)
    session = ExactChunkCheckpointSession(REQUEST_SHA, store=store)
    session.save(
        logical_index=0,
        text="uno",
        seed=41,
        audio=np.ones(8, dtype=np.float32),
        sample_rate=24_000,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "garbage_collect_exact_chunk_checkpoints",
        Mock(side_effect=RuntimeError("simulated ownership failure")),
    )

    session.complete(object())

    assert store.checkpoint_path(_key()).exists()


def test_symlinked_root_is_never_read_followed_or_cleaned(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real"
    real_store = ExactChunkCheckpointStore(real_root)
    key = _key()
    real_store.save(key, np.ones(8, dtype=np.float32), 24_000)
    symlink_root = tmp_path / "linked"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    linked_store = ExactChunkCheckpointStore(symlink_root)

    assert linked_store.load(key) is None
    assert linked_store.remove_request(REQUEST_SHA) is False
    assert linked_store.prune_stale(now=10_000_000.0, max_age_seconds=1) == 0
    with pytest.raises(OSError, match="not a real directory"):
        linked_store.save(key, np.ones(8, dtype=np.float32), 24_000)
    assert real_store.load(key) is not None


class _ChunkBackend:
    def __init__(
        self,
        outputs: dict[str, np.ndarray],
        *,
        fail_on_call: int | None = None,
        cancel_on_call: int | None = None,
    ) -> None:
        self.outputs = outputs
        self.fail_on_call = fail_on_call
        self.cancel_on_call = cancel_on_call
        self.calls: list[tuple[str, int | None]] = []

    async def generate(self, text, _prompt, _language, seed, _instruct):
        self.calls.append((text, seed))
        call_number = len(self.calls)
        if call_number == self.fail_on_call:
            raise RuntimeError("simulated hard failure")
        if call_number == self.cancel_on_call:
            raise asyncio.CancelledError
        return self.outputs[text].copy(), 24_000


def _install_three_logical_chunks(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], str]:
    chunks = ["primero", "segundo", "tercero"]
    complete_text = " ".join(chunks)
    original_split = chunked_tts.split_text_into_chunks

    def split(text: str, max_chars: int):
        if text == complete_text:
            return chunks.copy()
        return original_split(text, max_chars)

    monkeypatch.setattr(chunked_tts, "split_text_into_chunks", split)
    return chunks, complete_text


def test_failed_generation_resumes_only_missing_chunk_with_identical_pcm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks, complete_text = _install_three_logical_chunks(monkeypatch)
    outputs = {text: np.linspace(index, index + 0.5, 80 + index, dtype=np.float32) for index, text in enumerate(chunks)}
    store = ExactChunkCheckpointStore(tmp_path)
    first_attempt = _ChunkBackend(outputs, fail_on_call=3)

    with pytest.raises(RuntimeError, match="simulated hard failure"):
        asyncio.run(
            chunked_tts.generate_chunked(
                first_attempt,
                complete_text,
                {},
                language="es",
                seed=100,
                max_chunk_chars=100,
                crossfade_ms=1,
                checkpoint_session=ExactChunkCheckpointSession(REQUEST_SHA, store=store),
            )
        )

    assert first_attempt.calls == [
        ("primero", 100),
        ("segundo", 101),
        ("tercero", 102),
    ]
    assert len(list((tmp_path / REQUEST_SHA).glob("*.vbc"))) == 2

    # A new caller/generation ID creates a new session, but the canonical exact
    # request identity is unchanged and therefore reuses the first two chunks.
    resumed_attempt = _ChunkBackend(outputs)
    resumed_audio, resumed_rate = asyncio.run(
        chunked_tts.generate_chunked(
            resumed_attempt,
            complete_text,
            {},
            language="es",
            seed=100,
            max_chunk_chars=100,
            crossfade_ms=1,
            checkpoint_session=ExactChunkCheckpointSession(REQUEST_SHA, store=store),
        )
    )
    uninterrupted_audio = chunked_tts.concatenate_audio_chunks(
        [outputs[text] for text in chunks],
        24_000,
        crossfade_ms=1,
    )

    assert resumed_attempt.calls == [("tercero", 102)]
    assert resumed_rate == 24_000
    assert resumed_audio.tobytes() == uninterrupted_audio.tobytes()


def test_cancellation_after_completed_chunk_preserves_that_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks, complete_text = _install_three_logical_chunks(monkeypatch)
    outputs = {text: np.full(64, index / 10, dtype=np.float32) for index, text in enumerate(chunks, start=1)}
    store = ExactChunkCheckpointStore(tmp_path)
    cancelled_attempt = _ChunkBackend(outputs, cancel_on_call=2)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            chunked_tts.generate_chunked(
                cancelled_attempt,
                complete_text,
                {},
                seed=200,
                max_chunk_chars=100,
                checkpoint_session=ExactChunkCheckpointSession(REQUEST_SHA, store=store),
            )
        )

    resumed_attempt = _ChunkBackend(outputs)
    asyncio.run(
        chunked_tts.generate_chunked(
            resumed_attempt,
            complete_text,
            {},
            seed=200,
            max_chunk_chars=100,
            checkpoint_session=ExactChunkCheckpointSession(REQUEST_SHA, store=store),
        )
    )

    assert resumed_attempt.calls == [("segundo", 201), ("tercero", 202)]


class _GenerationDB:
    def expire_all(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_generation_cleans_checkpoint_only_after_audio_and_completed_db_are_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from backend.services import exact_chunk_checkpoints, generation

    events: list[str] = []
    fake_db = _GenerationDB()
    fake_backend = Mock()
    fake_backend.is_loaded.return_value = True
    task_manager = Mock()

    class RecordingSession:
        def __init__(self, request_sha: str):
            assert request_sha == REQUEST_SHA
            events.append("checkpoint-session-after-snapshot")

        def complete(self, db) -> None:
            assert db is fake_db
            events.append("checkpoint-cleanup")

    async def update_status(*args, **kwargs):
        status = kwargs.get("status") or args[1]
        events.append(f"db-{status}")
        return object()

    async def exact_prompt(*_args, **_kwargs):
        events.append("immutable-voice-snapshot")
        return {}

    async def generate(*_args, **_kwargs):
        events.append("inference")
        return np.ones(32, dtype=np.float32), 24_000

    async def save_generate(**_kwargs):
        events.append("audio-and-version-save")
        return "generations/generation.wav"

    def durable_sync(*_args):
        events.append("audio-fsync")

    monkeypatch.setattr(generation, "get_db", lambda: iter([fake_db]))
    monkeypatch.setattr(generation, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        generation.profiles,
        "compute_profile_voice_binding_sha256",
        Mock(return_value="b" * 64),
    )
    monkeypatch.setattr(generation.config, "_data_dir", tmp_path)
    monkeypatch.setattr(generation.profiles, "_require_exact_tts_revision", Mock())
    monkeypatch.setattr(generation.profiles, "create_exact_voice_prompt_from_snapshot", exact_prompt)
    monkeypatch.setattr(generation.history, "update_generation_status", update_status)
    monkeypatch.setattr(generation, "_save_generate", save_generate)
    monkeypatch.setattr(generation, "_durably_sync_exact_generation_audio", durable_sync)
    monkeypatch.setattr(generation, "_notify_speak_end", Mock())
    monkeypatch.setattr(
        exact_chunk_checkpoints,
        "ExactChunkCheckpointSession",
        RecordingSession,
    )
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: fake_backend)
    monkeypatch.setattr("backend.backends.load_engine_model", AsyncMock())
    monkeypatch.setattr("backend.backends.engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr("backend.backends.engine_retries_runaway", lambda _engine: False)
    monkeypatch.setattr("backend.utils.chunked_tts.generate_chunked", generate)

    asyncio.run(
        generation.run_generation(
            generation_id="generation",
            profile_id="profile",
            text="x" * 250,
            language="es",
            engine="qwen",
            model_size="1.7B",
            seed=42,
            mode="generate",
            max_chunk_chars=100,
            expected_voice_binding_sha256="b" * 64,
            exact_voice_snapshot={"frozen": True},
            expected_tts_implementation_revision="revision",
            exact_request_sha256=REQUEST_SHA,
        )
    )

    assert events == [
        "immutable-voice-snapshot",
        "checkpoint-session-after-snapshot",
        "db-generating",
        "inference",
        "audio-and-version-save",
        "audio-fsync",
        "db-completed",
        "checkpoint-cleanup",
    ]
    task_manager.complete_generation.assert_called_once_with("generation")

    # Normal audiobook units already fit in one model chunk. They cannot reuse
    # a completed prefix after a crash, so the checkpoint cache stays untouched,
    # but the final artifact must still be durable before completed is committed.
    events.clear()
    task_manager.reset_mock()
    asyncio.run(
        generation.run_generation(
            generation_id="single-chunk",
            profile_id="profile",
            text="texto corto",
            language="es",
            engine="qwen",
            model_size="1.7B",
            seed=43,
            mode="generate",
            max_chunk_chars=100,
            expected_voice_binding_sha256="b" * 64,
            exact_voice_snapshot={"frozen": True},
            expected_tts_implementation_revision="revision",
            exact_request_sha256="c" * 64,
        )
    )

    assert events == [
        "immutable-voice-snapshot",
        "db-generating",
        "inference",
        "audio-and-version-save",
        "audio-fsync",
        "db-completed",
    ]
    assert not (tmp_path / "cache" / "exact-chunk-checkpoints-v1").exists()
    task_manager.complete_generation.assert_called_once_with("single-chunk")

    def fail_durable_sync(*_args):
        events.append("audio-fsync")
        raise OSError("simulated fsync failure")

    events.clear()
    task_manager.reset_mock()
    monkeypatch.setattr(
        generation,
        "_durably_sync_exact_generation_audio",
        fail_durable_sync,
    )
    asyncio.run(
        generation.run_generation(
            generation_id="single-chunk-fsync-failure",
            profile_id="profile",
            text="texto corto",
            language="es",
            engine="qwen",
            model_size="1.7B",
            seed=44,
            mode="generate",
            max_chunk_chars=100,
            expected_voice_binding_sha256="b" * 64,
            exact_voice_snapshot={"frozen": True},
            expected_tts_implementation_revision="revision",
            exact_request_sha256="d" * 64,
        )
    )

    assert events == [
        "immutable-voice-snapshot",
        "db-generating",
        "inference",
        "audio-and-version-save",
        "audio-fsync",
        "db-failed",
    ]
    task_manager.complete_generation.assert_called_once_with("single-chunk-fsync-failure")
