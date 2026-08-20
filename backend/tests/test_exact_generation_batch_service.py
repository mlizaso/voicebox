"""Execution semantics for one two-unit exact generation batch."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, call

import numpy as np

from backend.services import generation


class _DB:
    def expire_all(self):
        pass

    def close(self):
        pass


def _spec(generation_id, seed):
    return generation.ExactBatchGenerationSpec(
        generation_id=generation_id,
        profile_id="profile",
        text=f"text-{generation_id}",
        language="es",
        engine="qwen",
        model_size="1.7B",
        seed=seed,
        normalize=False,
        effects_chain=None,
        instruct=None,
        crossfade_ms=10,
        expected_voice_binding_sha256="voice-binding",
        exact_voice_snapshot={
            "format_version": 1,
            "snapshot_key": "raw-" + ("a" * 64),
            "voice_binding_sha256": "voice-binding",
        },
        expected_tts_implementation_revision="runtime-revision",
    )


def _install_fakes(monkeypatch, generate_batch):
    db = _DB()
    task_manager = Mock()
    backend = Mock()
    backend.is_loaded.return_value = True
    monkeypatch.setattr(generation, "get_db", lambda: iter([db]))
    monkeypatch.setattr(generation, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generation.profiles, "create_voice_prompt_for_profile", AsyncMock(return_value={}))
    monkeypatch.setattr(
        generation.profiles,
        "create_exact_voice_prompt_from_snapshot",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        generation.profiles,
        "_require_exact_tts_revision",
        Mock(),
    )
    monkeypatch.setattr(
        generation.profiles,
        "compute_profile_voice_binding_sha256",
        Mock(return_value="voice-binding"),
    )
    monkeypatch.setattr(generation.history, "update_generation_status", AsyncMock())
    monkeypatch.setattr(
        generation,
        "_save_generate",
        AsyncMock(side_effect=lambda **kw: f"{kw['generation_id']}.wav"),
    )
    monkeypatch.setattr(generation, "_durably_sync_exact_generation_audio", Mock())
    monkeypatch.setattr(generation, "_notify_speak_end", Mock())
    monkeypatch.setattr("backend.backends.get_tts_backend_for_engine", lambda _engine: backend)
    monkeypatch.setattr("backend.backends.load_engine_model", AsyncMock())
    monkeypatch.setattr("backend.backends.engine_needs_trim", lambda _engine: False)
    monkeypatch.setattr("backend.backends.engine_retries_runaway", lambda _engine: False)
    monkeypatch.setattr("backend.utils.chunked_tts.generate_text_batch", generate_batch)
    return task_manager


def test_batch_service_sends_distinct_seeds_and_completes_each_row(monkeypatch):
    generate_batch = AsyncMock(
        return_value=[
            (np.ones(10, dtype=np.float32), 24_000),
            (np.ones(12, dtype=np.float32), 24_000),
        ]
    )
    task_manager = _install_fakes(monkeypatch, generate_batch)
    specs = [_spec("first", 100), _spec("second", 101)]

    asyncio.run(generation.run_exact_generation_batch(specs))

    assert generate_batch.await_args.kwargs["seeds"] == [100, 101]
    assert generate_batch.await_args.kwargs["crossfade_ms"] == 10
    completed = [
        call.kwargs["generation_id"]
        for call in generation.history.update_generation_status.await_args_list
        if call.kwargs.get("status") == "completed"
    ]
    assert completed == ["first", "second"]
    assert generation._durably_sync_exact_generation_audio.call_args_list == [
        call("first", "first.wav"),
        call("second", "second.wav"),
    ]
    assert task_manager.complete_generation.call_args_list == [
        (("first",),),
        (("second",),),
    ]


def test_batch_never_commits_completed_before_each_output_is_durable(monkeypatch):
    events = []
    generate_batch = AsyncMock(
        return_value=[
            (np.ones(10, dtype=np.float32), 24_000),
            (np.ones(12, dtype=np.float32), 24_000),
        ]
    )
    _install_fakes(monkeypatch, generate_batch)

    async def save_generate(**kwargs):
        generation_id = kwargs["generation_id"]
        events.append((generation_id, "saved"))
        return f"{generation_id}.wav"

    def durable_sync(generation_id, _final_path):
        events.append((generation_id, "durable"))
        if generation_id == "second":
            raise OSError("simulated fsync failure")

    async def update_status(generation_id, status, db, **_kwargs):
        del db
        events.append((generation_id, status))
        return object()

    monkeypatch.setattr(generation, "_save_generate", save_generate)
    monkeypatch.setattr(
        generation,
        "_durably_sync_exact_generation_audio",
        durable_sync,
    )
    monkeypatch.setattr(generation.history, "update_generation_status", update_status)

    asyncio.run(generation.run_exact_generation_batch([_spec("first", 100), _spec("second", 101)]))

    assert events.index(("first", "saved")) < events.index(("first", "durable"))
    assert events.index(("first", "durable")) < events.index(("first", "completed"))
    assert events.index(("second", "saved")) < events.index(("second", "durable"))
    assert ("second", "completed") not in events
    assert ("second", "failed") in events


def test_batch_service_marks_both_rows_failed_when_shared_inference_is_cancelled(monkeypatch):
    generate_batch = AsyncMock(side_effect=asyncio.CancelledError)
    task_manager = _install_fakes(monkeypatch, generate_batch)
    specs = [_spec("first", 100), _spec("second", 101)]

    asyncio.run(generation.run_exact_generation_batch(specs))

    failed_ids = [
        call.args[0] for call in generation.history.update_generation_status.await_args_list if call.args[1] == "failed"
    ]
    assert failed_ids == ["first", "second"]
    assert task_manager.complete_generation.call_count == 2


def test_batch_service_fails_closed_when_runtime_changes_while_queued(monkeypatch):
    generate_batch = AsyncMock()
    task_manager = _install_fakes(monkeypatch, generate_batch)
    generation.profiles._require_exact_tts_revision.side_effect = RuntimeError(
        "TTS implementation revision changed while exact generation was queued"
    )
    specs = [_spec("first", 100), _spec("second", 101)]

    asyncio.run(generation.run_exact_generation_batch(specs))

    generate_batch.assert_not_awaited()
    failed_calls = [
        call for call in generation.history.update_generation_status.await_args_list if call.args[1] == "failed"
    ]
    assert [call.args[0] for call in failed_calls] == ["first", "second"]
    assert all("revision changed" in call.kwargs["error"].lower() for call in failed_calls)
    assert task_manager.complete_generation.call_count == 2


def test_batch_model_reads_verified_snapshot_after_live_reference_replacement(
    monkeypatch,
    tmp_path,
):
    live_reference = tmp_path / "live.wav"
    snapshot_reference = tmp_path / "snapshot.wav"
    original = b"original-reference-bytes"
    live_reference.write_bytes(original)
    snapshot_created = False

    async def create_snapshot(*_args, **_kwargs):
        nonlocal snapshot_created
        snapshot_reference.write_bytes(live_reference.read_bytes())
        snapshot_created = True
        return {"ref_audio": str(snapshot_reference), "ref_text": "reference"}

    async def generate_batch(_model, _texts, voice_prompt, **_kwargs):
        assert snapshot_created
        live_reference.write_bytes(b"replacement-reference-bytes")
        assert Path(voice_prompt["ref_audio"]).read_bytes() == original
        return [
            (np.ones(10, dtype=np.float32), 24_000),
            (np.ones(12, dtype=np.float32), 24_000),
        ]

    task_manager = _install_fakes(monkeypatch, generate_batch)
    monkeypatch.setattr(
        generation.profiles,
        "create_exact_voice_prompt_from_snapshot",
        AsyncMock(side_effect=create_snapshot),
    )

    asyncio.run(generation.run_exact_generation_batch([_spec("first", 100), _spec("second", 101)]))

    assert task_manager.complete_generation.call_count == 2
