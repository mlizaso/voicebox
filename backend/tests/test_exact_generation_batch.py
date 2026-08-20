"""Exact two-unit API contract for resumable model-level batching."""

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from starlette.testclient import TestClient

from backend import models
from backend.database import Base, Generation as DBGeneration, VoiceProfile
from backend.database.migrations import run_migrations
from backend.routes import generations


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'batch.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(
        VoiceProfile(
            id="profile-id",
            name="Narrator",
            language="es",
            voice_type="cloned",
            default_engine="qwen",
        )
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()


def _batch(*, second_text: str = "Segundo fragmento."):
    revision = "batch-runtime"
    requests = [
        models.GenerationRequest(
            profile_id="profile-id",
            text="Primer fragmento.",
            language="es",
            engine="qwen",
            model_size="1.7B",
            seed=100,
            max_chunk_chars=1200,
            crossfade_ms=10,
            tts_implementation_revision=revision,
        ),
        models.GenerationRequest(
            profile_id="profile-id",
            text=second_text,
            language="es",
            engine="qwen",
            model_size="1.7B",
            seed=101,
            max_chunk_chars=1200,
            crossfade_ms=10,
            tts_implementation_revision=revision,
        ),
    ]
    return models.ExactBatchGenerationRequest(
        items=[
            models.ExactBatchGenerationItem(
                generation_id=uuid.UUID("55e3e82d-66e2-5c84-948d-b02c496becc4"),
                request=requests[0],
            ),
            models.ExactBatchGenerationItem(
                generation_id=uuid.UUID("5b1b3a4f-625f-5067-a61f-20ae836d0d5f"),
                request=requests[1],
            ),
        ]
    )


def _install_route_fakes(monkeypatch):
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "batch-runtime",
    )
    monkeypatch.setattr(
        generations.profiles,
        "get_profile",
        AsyncMock(
            return_value=SimpleNamespace(
                id="profile-id",
                default_engine="qwen",
                preset_engine=None,
                voice_type="cloned",
            )
        ),
    )
    monkeypatch.setattr(generations.profiles, "validate_profile_engine", Mock())
    monkeypatch.setattr(
        generations.profiles,
        "freeze_exact_voice_profile",
        Mock(
            return_value={
                "format_version": 1,
                "snapshot_key": "raw-" + ("a" * 64),
                "voice_binding_sha256": "voice-binding",
            }
        ),
    )
    captured = []
    active_ids = set()

    async def fake_batch(specs):
        captured.extend(specs)

    def fake_enqueue(generation_ids, coro):
        captured.append(generation_ids)
        active_ids.update(generation_ids)
        coro.close()

    monkeypatch.setattr(generations, "run_exact_generation_batch", fake_batch)
    monkeypatch.setattr(generations, "enqueue_generation_batch", fake_enqueue)
    monkeypatch.setattr(
        generations,
        "generation_job_is_active",
        lambda generation_id: generation_id in active_ids,
    )
    return captured


def test_exact_batch_commit_then_raise_reconciles_and_enqueues_once(monkeypatch, db):
    captured = _install_route_fakes(monkeypatch)
    commit = db.commit
    raised = False

    def commit_then_raise():
        nonlocal raised
        commit()
        if not raised:
            raised = True
            raise RuntimeError("connection failed after durable commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)

    result = asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))

    db.expire_all()
    rows = db.query(DBGeneration).order_by(DBGeneration.id).all()
    assert len(rows) == 2
    assert {row.status for row in rows} == {"generating"}
    assert [row.id for row in result] == [str(item.generation_id) for item in _batch().items]
    assert tuple(str(item.generation_id) for item in _batch().items) in captured


def test_exact_batch_reattach_fails_active_rows_without_queue_owner(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    first = asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))
    assert {row.status for row in first} == {"generating"}
    monkeypatch.setattr(generations, "generation_job_is_active", lambda _generation_id: False)

    second = asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))

    assert [row.id for row in second] == [row.id for row in first]
    assert {row.status for row in second} == {"failed"}
    assert all("without a live queue owner" in (row.error or "") for row in second)


def test_exact_batch_route_is_registered():
    assert "/generate/batch/exact" in {route.path for route in generations.router.routes}


def test_generate_route_rejects_seed_outside_cross_engine_uint32_domain():
    app = FastAPI()
    app.include_router(generations.router)
    app.dependency_overrides[generations.get_db] = lambda: None

    with TestClient(app) as client:
        response = client.post(
            "/generate",
            json={"profile_id": "profile-id", "text": "hello", "seed": 1 << 32},
        )

    assert response.status_code == 422
    assert any(error["loc"][-1] == "seed" for error in response.json()["detail"])
    assert (
        models.GenerationRequest(
            profile_id="profile-id",
            text="hello",
            seed=models.MAX_GENERATION_SEED,
        ).seed
        == models.MAX_GENERATION_SEED
    )


def test_exact_batch_creates_independent_rows_and_forwards_distinct_seeds(monkeypatch, db):
    captured = _install_route_fakes(monkeypatch)

    result = asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))

    assert [row.seed for row in result] == [100, 101]
    assert all(row.exact_request_sha256 for row in result)
    assert len({row.exact_envelope_sha256 for row in result}) == 1
    assert result[0].exact_envelope_sha256
    assert {row.exact_effects_json for row in result} == {"null"}
    assert all(row.exact_voice_snapshot_json for row in result)
    assert {row.voice_binding_sha256 for row in result} == {"voice-binding"}
    assert db.query(DBGeneration).count() == 2
    assert tuple(str(item.generation_id) for item in _batch().items) in captured


def test_exact_batch_retry_is_idempotent_after_server_accepts_before_client_save(monkeypatch, db):
    captured = _install_route_fakes(monkeypatch)
    request = _batch()
    first = asyncio.run(generations.generate_speech_batch_exact(request, db=db))
    captured.clear()

    second = asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert [row.id for row in second] == [row.id for row in first]
    assert db.query(DBGeneration).count() == 2
    assert captured == []


def test_exact_numerical_hash_survives_profile_uuid_recreation():
    original = _batch().items[0].request
    recreated = original.model_copy(update={"profile_id": "recreated-profile-id"})

    original_hash = generations._exact_request_sha256(
        original,
        "same-voice-binding",
        None,
    )
    recreated_hash = generations._exact_request_sha256(
        recreated,
        "same-voice-binding",
        None,
    )

    assert recreated_hash == original_hash
    assert (
        generations._exact_request_sha256(
            recreated,
            "different-voice-binding",
            None,
        )
        != original_hash
    )


def test_exact_caller_id_ownership_still_rejects_recreated_profile_uuid(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))
    recreated = _batch()
    for item in recreated.items:
        item.request.profile_id = "recreated-profile-id"

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(recreated, db=db))

    assert raised.value.status_code == 409
    assert "belongs to another request" in raised.value.detail


def test_exact_retry_uses_frozen_profile_default_effects_after_profile_edit(monkeypatch, db):
    captured = _install_route_fakes(monkeypatch)
    profile = db.query(VoiceProfile).filter_by(id="profile-id").one()
    profile.effects_chain = json.dumps([{"effect_type": "gain", "enabled": True, "parameters": {"gain_db": 1.0}}])
    db.commit()
    request = _batch()
    first = asyncio.run(generations.generate_speech_batch_exact(request, db=db))
    frozen_effects = first[0].exact_effects_json
    captured.clear()
    profile.effects_chain = json.dumps([{"effect_type": "gain", "enabled": True, "parameters": {"gain_db": 9.0}}])
    db.commit()

    second = asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert [row.id for row in second] == [row.id for row in first]
    assert all(row.exact_effects_json == frozen_effects for row in second)
    assert captured == []


def test_exact_batch_rejects_caller_id_collision(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            generations.generate_speech_batch_exact(
                _batch(second_text="Texto diferente."),
                db=db,
            )
        )

    assert raised.value.status_code == 409
    assert "belongs to another request" in raised.value.detail


def test_idempotency_hash_rejects_non_persisted_contract_change(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    asyncio.run(generations.generate_speech_batch_exact(_batch(), db=db))
    changed = _batch()
    changed.items[1].request.normalize = False

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(changed, db=db))

    assert raised.value.status_code == 409


def test_exact_batch_rejects_multi_chunk_unit_before_history_creation(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    request = _batch(second_text=("Una frase suficientemente larga. " * 8).strip())
    request.items[1].request.max_chunk_chars = 100
    request.items[0].request.max_chunk_chars = 100

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert raised.value.status_code == 422
    assert "fit in one model chunk" in raised.value.detail
    assert db.query(DBGeneration).count() == 0


def test_exact_singleton_accepts_multi_chunk_serial_contract(monkeypatch, db):
    captured = _install_route_fakes(monkeypatch)
    request = models.ExactBatchGenerationRequest(items=[_batch().items[0]])
    request.items[0].request.text = ("Una frase suficientemente larga. " * 8).strip()
    request.items[0].request.max_chunk_chars = 100

    result = asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert len(result) == 1
    assert db.query(DBGeneration).count() == 1
    assert tuple(str(item.generation_id) for item in request.items) in captured


def test_two_item_exact_batch_rejects_instruction_instead_of_serial_fallback(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    request = _batch()
    for item in request.items:
        item.request.instruct = "Habla con calma"

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert raised.value.status_code == 422
    assert "does not support instruction" in raised.value.detail
    assert db.query(DBGeneration).count() == 0


def test_exact_batch_rejects_implicit_random_seed_before_history(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    request = _batch()
    request.items[1].request.seed = None

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert raised.value.status_code == 422
    assert "explicit seed" in raised.value.detail.lower()
    assert db.query(DBGeneration).count() == 0


def test_exact_singleton_retry_reattaches_same_caller_id(monkeypatch, db):
    captured = _install_route_fakes(monkeypatch)
    request = models.ExactBatchGenerationRequest(items=[_batch().items[0]])

    first = asyncio.run(generations.generate_speech_batch_exact(request, db=db))
    captured.clear()
    second = asyncio.run(generations.generate_speech_batch_exact(request, db=db))

    assert [row.id for row in second] == [row.id for row in first]
    assert db.query(DBGeneration).count() == 1
    assert captured == []


def test_exact_singleton_cannot_be_reattached_as_batch_member(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    pair = _batch()
    asyncio.run(
        generations.generate_speech_batch_exact(
            models.ExactBatchGenerationRequest(items=[pair.items[0]]),
            db=db,
        )
    )
    singleton_second = models.ExactBatchGenerationRequest(items=[pair.items[1]])
    asyncio.run(generations.generate_speech_batch_exact(singleton_second, db=db))

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(pair, db=db))

    assert raised.value.status_code == 409


def test_exact_batch_rejects_swapped_order(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    pair = _batch()
    asyncio.run(generations.generate_speech_batch_exact(pair, db=db))
    swapped = models.ExactBatchGenerationRequest(items=list(reversed(pair.items)))

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(swapped, db=db))

    assert raised.value.status_code == 409


def test_exact_batch_rejects_rows_created_with_different_peers(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    original = _batch()
    first_id = original.items[0].generation_id
    second_id = original.items[1].generation_id
    alternate_a = models.ExactBatchGenerationRequest(
        items=[
            original.items[0],
            models.ExactBatchGenerationItem(
                generation_id=uuid.uuid5(uuid.NAMESPACE_URL, "alternate-peer-a"),
                request=original.items[1].request.model_copy(update={"text": "Peer A"}),
            ),
        ]
    )
    alternate_b = models.ExactBatchGenerationRequest(
        items=[
            models.ExactBatchGenerationItem(
                generation_id=second_id,
                request=original.items[1].request,
            ),
            models.ExactBatchGenerationItem(
                generation_id=uuid.uuid5(uuid.NAMESPACE_URL, "alternate-peer-b"),
                request=original.items[0].request.model_copy(update={"text": "Peer B"}),
            ),
        ]
    )
    asyncio.run(generations.generate_speech_batch_exact(alternate_a, db=db))
    asyncio.run(generations.generate_speech_batch_exact(alternate_b, db=db))
    assert db.query(DBGeneration).filter_by(id=str(first_id)).count() == 1

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.generate_speech_batch_exact(original, db=db))

    assert raised.value.status_code == 409


def test_cancelling_queued_batch_member_fails_every_owned_row_immediately(monkeypatch, db):
    _install_route_fakes(monkeypatch)
    request = _batch()
    asyncio.run(generations.generate_speech_batch_exact(request, db=db))
    generation_ids = tuple(str(item.generation_id) for item in request.items)
    completed = []

    monkeypatch.setattr(generations, "generation_job_ids", lambda _id: generation_ids)
    monkeypatch.setattr(generations, "cancel_generation_job", lambda _id: "queued")
    monkeypatch.setattr(
        generations,
        "get_task_manager",
        lambda: SimpleNamespace(complete_generation=completed.append),
    )

    response = asyncio.run(generations.cancel_generation(generation_ids[0], db=db))
    db.expire_all()

    assert response == {"message": "Queued generation job cancelled"}
    assert completed == list(generation_ids)
    assert {row.id: (row.status, row.error) for row in db.query(DBGeneration).order_by(DBGeneration.id).all()} == {
        generation_id: ("failed", "Generation batch cancelled") for generation_id in generation_ids
    }


def test_generation_contract_hash_column_migrates_existing_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE generations ("
                "id VARCHAR PRIMARY KEY, profile_id VARCHAR NOT NULL, text TEXT NOT NULL, "
                "language VARCHAR, audio_path VARCHAR, duration FLOAT, seed INTEGER, "
                "instruct TEXT, created_at DATETIME)"
            )
        )

    run_migrations(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("generations")}
    assert "exact_request_sha256" in columns
    assert "exact_envelope_sha256" in columns
    assert "exact_effects_json" in columns
    assert "exact_voice_snapshot_json" in columns
    assert "voice_binding_sha256" in columns


@pytest.mark.parametrize(("operation", "status"), [("retry", "failed"), ("regenerate", "completed")])
def test_generic_retry_and_regenerate_reject_exact_rows(operation, status, db):
    row = DBGeneration(
        id=f"exact-{operation}",
        profile_id="profile-id",
        text="Frozen exact text",
        language="es",
        status=status,
        exact_request_sha256="frozen-contract",
        voice_binding_sha256="frozen-voice",
    )
    db.add(row)
    db.commit()

    endpoint = generations.retry_generation if operation == "retry" else generations.regenerate_generation
    with pytest.raises(HTTPException) as raised:
        asyncio.run(endpoint(row.id, db=db))

    assert raised.value.status_code == 409
    db.refresh(row)
    assert row.status == status


def test_generic_retry_keeps_previous_audio_owned_until_replacement(monkeypatch, db):
    row = DBGeneration(
        id="retry-with-audio",
        profile_id="profile-id",
        text="Retry text",
        language="es",
        engine="qwen",
        model_size="1.7B",
        seed=7,
        status="failed",
        error="previous failure",
        audio_path="generations/retry-with-audio.wav",
        duration=12.5,
    )
    db.add(row)
    db.commit()
    task_manager = Mock()

    def capture_enqueue(generation_id, generation_coro):
        assert generation_id == row.id
        generation_coro.close()

    monkeypatch.setattr(generations, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generations, "enqueue_generation", capture_enqueue)

    asyncio.run(generations.retry_generation(row.id, db=db))

    db.refresh(row)
    assert row.status == "generating"
    assert row.error is None
    assert row.audio_path == "generations/retry-with-audio.wav"
    assert row.duration == 12.5


@pytest.mark.parametrize(
    ("operation", "initial_status"),
    [("retry", "failed"), ("regenerate", "completed")],
)
def test_generic_restart_enqueues_after_commit_then_refresh_failure(
    monkeypatch,
    db,
    operation,
    initial_status,
):
    row = DBGeneration(
        id=f"ambiguous-{operation}",
        profile_id="profile-id",
        text="Retry text",
        language="es",
        engine="qwen",
        model_size="1.7B",
        seed=7,
        status=initial_status,
        error="previous failure",
        audio_path="generations/old.wav",
        duration=12.5,
    )
    db.add(row)
    db.commit()
    task_manager = Mock()
    enqueued = []

    def capture_enqueue(generation_id, generation_coro):
        enqueued.append(generation_id)
        generation_coro.close()

    monkeypatch.setattr(generations, "generation_job_is_active", lambda _generation_id: False)
    monkeypatch.setattr(generations, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generations, "enqueue_generation", capture_enqueue)
    monkeypatch.setattr(
        db,
        "refresh",
        lambda _row: (_ for _ in ()).throw(RuntimeError("connection failed while refreshing committed acceptance")),
    )

    endpoint = generations.retry_generation if operation == "retry" else generations.regenerate_generation
    result = asyncio.run(endpoint(row.id, db=db))

    assert result.status == "generating"
    assert result.error is None
    assert enqueued == [row.id]
    fresh = sessionmaker(bind=db.get_bind())()
    durable = fresh.query(DBGeneration).filter_by(id=row.id).one()
    assert durable.status == "generating"
    assert durable.error is None
    assert durable.audio_path == "generations/old.wav"
    fresh.close()


def test_generic_retry_refuses_cancelled_job_until_queue_releases_id(monkeypatch, db):
    row = DBGeneration(
        id="cancelled-queued-retry",
        profile_id="profile-id",
        text="Retry text",
        language="es",
        status="failed",
        error="Generation cancelled",
        audio_path="generations/old.wav",
        duration=12.5,
    )
    db.add(row)
    db.commit()
    task_manager = Mock()
    monkeypatch.setattr(generations, "generation_job_is_active", lambda _generation_id: True)
    monkeypatch.setattr(generations, "get_task_manager", lambda: task_manager)

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.retry_generation(row.id, db=db))

    assert raised.value.status_code == 409
    db.refresh(row)
    assert row.status == "failed"
    assert row.error == "Generation cancelled"
    assert row.audio_path == "generations/old.wav"
    assert row.duration == 12.5
    task_manager.start_generation.assert_not_called()


def test_retry_queue_race_closes_coroutine_and_restores_failed_row(monkeypatch, db):
    row = DBGeneration(
        id="retry-queue-race",
        profile_id="profile-id",
        text="Retry text",
        language="es",
        status="failed",
        error="previous failure",
        audio_path="generations/old.wav",
        duration=12.5,
    )
    db.add(row)
    db.commit()
    task_manager = Mock()

    class FakeCoroutine:
        closed = False

        def close(self):
            self.closed = True

    generation_coro = FakeCoroutine()
    monkeypatch.setattr(generations, "generation_job_is_active", lambda _generation_id: False)
    monkeypatch.setattr(generations, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generations, "run_generation", lambda **_kwargs: generation_coro)
    monkeypatch.setattr(
        generations,
        "enqueue_generation",
        lambda *_args: (_ for _ in ()).throw(ValueError("generation is already queued or running")),
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.retry_generation(row.id, db=db))

    assert raised.value.status_code == 409
    assert generation_coro.closed is True
    db.refresh(row)
    assert row.status == "failed"
    assert row.error == "previous failure"
    assert row.audio_path == "generations/old.wav"
    assert row.duration == 12.5
    task_manager.complete_generation.assert_called_once_with(row.id)


def test_retry_queue_capacity_rejection_restores_failed_row(monkeypatch, db):
    row = DBGeneration(
        id="retry-queue-full",
        profile_id="profile-id",
        text="Retry text",
        language="es",
        status="failed",
        error="previous failure",
        audio_path="generations/old.wav",
        duration=12.5,
    )
    db.add(row)
    db.commit()
    task_manager = Mock()

    class FakeCoroutine:
        closed = False

        def close(self):
            self.closed = True

    generation_coro = FakeCoroutine()
    monkeypatch.setattr(generations, "generation_job_is_active", lambda _generation_id: False)
    monkeypatch.setattr(generations, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(generations, "run_generation", lambda **_kwargs: generation_coro)
    monkeypatch.setattr(
        generations,
        "enqueue_generation",
        lambda *_args: (_ for _ in ()).throw(generations.GenerationQueueFullError("queue is full")),
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(generations.retry_generation(row.id, db=db))

    assert raised.value.status_code == 503
    assert generation_coro.closed is True
    db.refresh(row)
    assert row.status == "failed"
    assert row.error == "previous failure"
    assert row.audio_path == "generations/old.wav"
    task_manager.complete_generation.assert_called_once_with(row.id)


@pytest.mark.parametrize("voice_type", ["preset", "designed"])
def test_exact_qwen_rejects_non_cloned_profiles_before_history(
    monkeypatch,
    db,
    voice_type,
):
    _install_route_fakes(monkeypatch)
    monkeypatch.setattr(
        generations.profiles,
        "get_profile",
        AsyncMock(
            return_value=SimpleNamespace(
                id="profile-id",
                default_engine="qwen",
                preset_engine="qwen" if voice_type == "preset" else None,
                voice_type=voice_type,
            )
        ),
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            generations.generate_speech_batch_exact(
                models.ExactBatchGenerationRequest(items=[_batch().items[0]]),
                db=db,
            )
        )

    assert raised.value.status_code == 422
    assert "requires a cloned profile" in raised.value.detail
    assert db.query(DBGeneration).count() == 0
