"""Ownership and ordering tests for exact checkpoint garbage collection."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import app, config
from backend.database import Base, Generation, VoiceProfile
from backend.services import history, profiles
from backend.services.exact_chunk_checkpoints import (
    CheckpointGarbageCollectionError,
    ExactChunkCheckpointKey,
    ExactChunkCheckpointSession,
    ExactChunkCheckpointStore,
    garbage_collect_exact_chunk_checkpoints,
)


def _database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(
        f"sqlite:///{data_dir / 'voicebox.db'}",
        connect_args={"check_same_thread": False, "timeout": 3},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return engine, factory, ExactChunkCheckpointStore()


def _checkpoint(store: ExactChunkCheckpointStore, request_sha256: str, index: int = 0) -> Path:
    key = ExactChunkCheckpointKey.from_text(
        exact_request_sha256=request_sha256,
        logical_index=index,
        text=f"chunk {index}",
        seed=100 + index,
    )
    return store.save(key, np.ones(8, dtype=np.float32), 24_000)


def _generation(
    generation_id: str,
    request_sha256: str,
    status: str | None,
    *,
    profile_id: str = "profile",
) -> Generation:
    return Generation(
        id=generation_id,
        profile_id=profile_id,
        text="text",
        language="es",
        audio_path="",
        status=status,
        exact_request_sha256=request_sha256,
    )


def test_single_delete_preserves_shared_failed_attempt_then_reclaims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    db = factory()
    request_sha256 = "1" * 64
    path = _checkpoint(store, request_sha256)
    db.add_all(
        [
            _generation("completed-attempt", request_sha256, "completed"),
            _generation("failed-attempt", request_sha256, "failed"),
        ]
    )
    db.commit()

    assert asyncio.run(history.delete_generation("completed-attempt", db)) is True
    assert path.exists()

    assert asyncio.run(history.delete_generation("failed-attempt", db)) is True
    assert not path.parent.exists()
    db.close()
    engine.dispose()


def test_completed_session_does_not_discard_other_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    db = factory()
    request_sha256 = "2" * 64
    path = _checkpoint(store, request_sha256)
    db.add_all(
        [
            _generation("completed-attempt", request_sha256, "completed"),
            _generation("failed-attempt", request_sha256, "failed"),
        ]
    )
    db.commit()
    # Match update_generation_status(): its post-commit refresh leaves the
    # request Session in a read transaction while GC opens a fresh connection.
    db.refresh(db.get(Generation, "completed-attempt"))

    ExactChunkCheckpointSession(request_sha256, store=store).complete(db)

    assert path.exists()
    db.close()
    engine.dispose()


def test_startup_gc_scans_beyond_oldest_64_and_preserves_stale_failed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    db = factory()
    request_hashes = [f"{index:064x}" for index in range(1, 72)]
    paths = {request_sha256: _checkpoint(store, request_sha256) for request_sha256 in request_hashes}
    db.add_all(
        [
            _generation("stale-active", request_hashes[0], "generating"),
            _generation("completed", request_hashes[1], "completed"),
            _generation("legacy-null", request_hashes[2], None),
        ]
    )
    db.commit()

    app._reconcile_stale_generations(db)
    app._prune_abandoned_exact_checkpoints(db)

    db.expire_all()
    assert db.get(Generation, "stale-active").status == "failed"
    assert paths[request_hashes[0]].exists()
    assert all(not paths[value].exists() for value in request_hashes[1:])
    db.close()
    engine.dispose()


def test_gc_is_fail_closed_for_unexpected_or_unsafe_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    db = factory()
    safe_sha256 = "3" * 64
    unsafe_sha256 = "4" * 64
    safe_path = _checkpoint(store, safe_sha256)
    unsafe_path = _checkpoint(store, unsafe_sha256)
    unexpected = unsafe_path.parent / "unmanaged.txt"
    unexpected.write_text("preserve", encoding="utf-8")

    with pytest.raises(CheckpointGarbageCollectionError, match="unsafe entry"):
        garbage_collect_exact_chunk_checkpoints(db, store=store)
    assert safe_path.exists()
    assert unsafe_path.exists()

    report = garbage_collect_exact_chunk_checkpoints(
        db,
        request_hashes=(unsafe_sha256,),
        store=store,
    )
    assert report.refused == 1
    assert unsafe_path.exists()
    assert unexpected.exists()
    db.close()
    engine.dispose()


def test_concurrent_exact_row_creation_waits_for_owner_query_and_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    request_sha256 = "5" * 64
    old_path = _checkpoint(store, request_sha256)
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    writer_started = threading.Event()
    writer_committed = threading.Event()
    errors: list[BaseException] = []
    original_remove_request = store.remove_request

    def paused_remove(candidate: str) -> bool:
        cleanup_entered.set()
        if not release_cleanup.wait(timeout=3):
            raise TimeoutError("test did not release checkpoint cleanup")
        return original_remove_request(candidate)

    monkeypatch.setattr(store, "remove_request", paused_remove)

    def run_gc() -> None:
        gc_db = factory()
        try:
            garbage_collect_exact_chunk_checkpoints(
                gc_db,
                request_hashes=(request_sha256,),
                store=store,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            gc_db.close()

    def create_retry() -> None:
        writer_db = factory()
        try:
            writer_started.set()
            writer_db.add(_generation("new-attempt", request_sha256, "generating"))
            writer_db.commit()
            writer_committed.set()
        except BaseException as exc:
            errors.append(exc)
        finally:
            writer_db.close()

    gc_thread = threading.Thread(target=run_gc)
    gc_thread.start()
    assert cleanup_entered.wait(timeout=3)
    writer_thread = threading.Thread(target=create_retry)
    writer_thread.start()
    assert writer_started.wait(timeout=3)
    assert not writer_committed.wait(timeout=0.1)

    release_cleanup.set()
    gc_thread.join(timeout=3)
    writer_thread.join(timeout=3)

    assert not gc_thread.is_alive()
    assert not writer_thread.is_alive()
    assert errors == []
    assert writer_committed.is_set()
    assert not old_path.parent.exists()

    # The serialized retry commits after cleanup and can safely recreate the
    # same content-addressed request directory.
    new_path = _checkpoint(store, request_sha256)
    assert new_path.exists()
    engine.dispose()


def test_bulk_failed_delete_reclaims_hash_even_if_completed_history_remains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    db = factory()
    shared_sha256 = "6" * 64
    second_sha256 = "7" * 64
    shared_path = _checkpoint(store, shared_sha256)
    second_path = _checkpoint(store, second_sha256)
    db.add_all(
        [
            _generation("failed-one", shared_sha256, "failed"),
            _generation("failed-two", second_sha256, "failed"),
            _generation("completed", shared_sha256, "completed"),
        ]
    )
    db.commit()

    assert asyncio.run(history.delete_failed_generations(db)) == 2

    assert not shared_path.parent.exists()
    assert not second_path.parent.exists()
    assert db.get(Generation, "completed") is not None
    db.close()
    engine.dispose()


def test_profile_delete_reclaims_only_hashes_without_surviving_failed_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, factory, store = _database(tmp_path, monkeypatch)
    db = factory()
    removed_sha256 = "8" * 64
    shared_sha256 = "9" * 64
    removed_path = _checkpoint(store, removed_sha256)
    shared_path = _checkpoint(store, shared_sha256)
    db.add_all(
        [
            VoiceProfile(id="profile", name="Deleted", voice_type="cloned"),
            VoiceProfile(id="survivor", name="Survivor", voice_type="cloned"),
            _generation("deleted-only", removed_sha256, "failed"),
            _generation("deleted-shared", shared_sha256, "failed"),
            _generation(
                "surviving-shared",
                shared_sha256,
                "failed",
                profile_id="survivor",
            ),
        ]
    )
    db.commit()

    assert asyncio.run(profiles.delete_profile("profile", db)) is True

    assert not removed_path.parent.exists()
    assert shared_path.exists()
    assert db.get(Generation, "surviving-shared") is not None
    db.close()
    engine.dispose()
