"""Resource and correctness tests for disk-backed Story audio export."""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import (
    app as _app,
    config,
    models,
)
from backend.database import Base, Generation, Story, StoryItem, VoiceProfile
from backend.routes import stories as story_routes
from backend.services import stories


@pytest.fixture
def story_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "data"
    generations_dir = data_dir / "generations"
    generations_dir.mkdir(parents=True)
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'stories.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(Story(id="story", name="Bounded Story"))
    db.commit()
    try:
        yield db, generations_dir
    finally:
        db.close()
        engine.dispose()


def _add_generation(db, generations_dir: Path, generation_id: str, audio: np.ndarray, sample_rate: int = 24_000):
    audio_path = generations_dir / f"{generation_id}.wav"
    sf.write(audio_path, audio, sample_rate, subtype="FLOAT")
    db.add(
        Generation(
            id=generation_id,
            profile_id="profile",
            text=generation_id,
            language="en",
            audio_path=f"generations/{audio_path.name}",
            duration=len(audio) / sample_rate,
            status="completed",
        )
    )
    db.commit()
    return audio_path


def _add_item(
    db,
    generation_id: str,
    *,
    item_id: str | None = None,
    start_time_ms: int = 0,
    trim_start_ms: int = 0,
    trim_end_ms: int = 0,
    volume: float = 1.0,
):
    db.add(
        StoryItem(
            id=item_id or f"item-{generation_id}",
            story_id="story",
            generation_id=generation_id,
            start_time_ms=start_time_ms,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            volume=volume,
        )
    )
    db.commit()


@pytest.mark.parametrize("start_time_ms", [-1, models.STORY_MAX_TIMELINE_MS + 1])
def test_story_request_models_reject_out_of_range_timecodes(start_time_ms: int):
    with pytest.raises(ValidationError):
        models.StoryItemCreate(generation_id="generation", start_time_ms=start_time_ms)
    with pytest.raises(ValidationError):
        models.StoryItemMove(start_time_ms=start_time_ms, track=0)
    with pytest.raises(ValidationError):
        models.StoryItemBatchUpdate(
            updates=[models.StoryItemUpdateTime(item_id="item-generation", start_time_ms=start_time_ms)]
        )


def test_export_rejects_persisted_extreme_timecode_before_allocating(story_db, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.zeros(240, dtype=np.float32))
    _add_item(db, "generation", start_time_ms=10**12)

    monkeypatch.setattr(stories.np, "memmap", lambda *args, **kwargs: pytest.fail("timeline was allocated"))
    with pytest.raises(stories.StoryAudioExportLimitError, match="start time"):
        asyncio.run(stories.export_story_audio("story", db))


def test_story_services_repeat_timecode_validation_for_direct_callers(story_db):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.zeros(240, dtype=np.float32))
    invalid_create = models.StoryItemCreate.model_construct(
        generation_id="generation",
        start_time_ms=-1,
        track=0,
    )
    assert asyncio.run(stories.add_item_to_story("story", invalid_create, db)) is None
    assert db.query(StoryItem).count() == 0

    _add_item(db, "generation", start_time_ms=100)
    invalid_move = models.StoryItemMove.model_construct(
        start_time_ms=models.STORY_MAX_TIMELINE_MS + 1,
        track=0,
    )
    assert asyncio.run(stories.move_story_item("story", "item-generation", invalid_move, db)) is None
    invalid_batch = models.StoryItemBatchUpdate.model_construct(
        updates=[
            models.StoryItemUpdateTime.model_construct(
                item_id="item-generation",
                start_time_ms=-1,
            )
        ]
    )
    assert not asyncio.run(stories.update_story_item_times("story", invalid_batch, db))
    db.expire_all()
    assert db.query(StoryItem).filter_by(id="item-generation").one().start_time_ms == 100


def test_overlapping_mix_preserves_trim_volume_and_peak_normalization(story_db):
    db, generations_dir = story_db
    one_second = np.full(24_000, 0.75, dtype=np.float32)
    _add_generation(db, generations_dir, "first", one_second)
    _add_generation(db, generations_dir, "second", one_second)
    _add_item(db, "first", start_time_ms=0, trim_start_ms=100, trim_end_ms=100)
    _add_item(db, "second", start_time_ms=400)

    exported = asyncio.run(stories.export_story_audio("story", db))
    assert exported is not None
    try:
        output, sample_rate = sf.read(exported.path, dtype="float32")
        assert sample_rate == 24_000
        assert len(output) == 33_600
        assert output[1_000] == pytest.approx(0.5, abs=4e-5)
        assert output[11_000] == pytest.approx(1.0, abs=4e-5)
        assert output[30_000] == pytest.approx(0.5, abs=4e-5)
        assert float(np.max(np.abs(output))) <= 1.0
    finally:
        exported.cleanup()


def test_zero_volume_is_silent_in_export(story_db):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "silent", np.full(2_400, 0.75, dtype=np.float32))
    _add_item(db, "silent", volume=0.0)

    exported = asyncio.run(stories.export_story_audio("story", db))
    assert exported is not None
    try:
        output, _sample_rate = sf.read(exported.path, dtype="float32")
        assert not np.any(output)
    finally:
        exported.cleanup()


def test_resample_uses_libsoxr_half_up_output_length(story_db):
    db, generations_dir = story_db
    source = np.sin(np.arange(48_001, dtype=np.float32) * 0.01)
    source_path = _add_generation(db, generations_dir, "odd-48k", source, sample_rate=48_000)
    _add_item(db, "odd-48k")
    legacy_mix_input, _sample_rate = librosa.load(source_path, sr=24_000, mono=True)
    legacy_peak = float(np.max(np.abs(legacy_mix_input)))
    if legacy_peak > 1.0:
        legacy_mix_input = legacy_mix_input / legacy_peak

    exported = asyncio.run(stories.export_story_audio("story", db))
    assert exported is not None
    try:
        output, sample_rate = sf.read(exported.path, dtype="float32")
        assert sample_rate == 24_000
        assert len(output) == 24_001
        assert np.allclose(output, legacy_mix_input, atol=4e-5)
    finally:
        exported.cleanup()


def test_export_uses_disk_backed_timeline_and_bounded_decode_blocks(story_db, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "late", np.full(120_000, 0.25, dtype=np.float32))
    _add_item(db, "late", start_time_ms=60_000)

    real_memmap = stories.np.memmap
    timeline_shapes: list[tuple[int, ...]] = []
    largest_decode_block = 0
    real_mono_block = stories._mono_block

    def tracked_memmap(*args, **kwargs):
        result = real_memmap(*args, **kwargs)
        timeline_shapes.append(result.shape)
        return result

    def tracked_mono_block(block, channels, generation_id):
        nonlocal largest_decode_block
        largest_decode_block = max(largest_decode_block, len(block))
        return real_mono_block(block, channels, generation_id)

    monkeypatch.setattr(stories.np, "memmap", tracked_memmap)
    monkeypatch.setattr(stories, "_mono_block", tracked_mono_block)
    exported = asyncio.run(stories.export_story_audio("story", db))
    assert exported is not None
    try:
        assert timeline_shapes == [(65 * 24_000,)]
        assert largest_decode_block <= stories.STORY_EXPORT_BLOCK_FRAMES
        assert exported.path.stat().st_size == 44 + 65 * 24_000 * 2
    finally:
        exported.cleanup()


def test_item_count_is_rejected_before_audio_decode(story_db, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "shared", np.zeros(240, dtype=np.float32))
    db.add_all(
        StoryItem(
            id=f"item-{index}",
            story_id="story",
            generation_id="shared",
            start_time_ms=0,
        )
        for index in range(models.STORY_MAX_ITEMS + 1)
    )
    db.commit()
    monkeypatch.setattr(stories, "_probe_clip", lambda *args, **kwargs: pytest.fail("audio was probed"))

    with pytest.raises(stories.StoryAudioExportLimitError, match="item export limit"):
        asyncio.run(stories.export_story_audio("story", db))


def test_long_clip_metadata_is_supported_up_to_timeline_bound_and_then_rejected():
    stories._validate_source_shape(
        frames=60 * 60 * 24_000,
        sample_rate=24_000,
        channels=1,
        generation_id="one-hour",
    )
    with pytest.raises(stories.StoryAudioExportLimitError, match="clip limit"):
        stories._validate_source_shape(
            frames=(models.STORY_MAX_CLIP_MS // 1000 + 1) * 24_000,
            sample_rate=24_000,
            channels=1,
            generation_id="long",
        )


def test_malformed_source_duration_is_rejected():
    with pytest.raises(stories.StoryAudioExportError, match="invalid duration"):
        stories._validate_source_shape(frames=0, sample_rate=24_000, channels=1, generation_id="malformed")


def test_high_rate_multichannel_decode_bomb_is_rejected_by_sample_count():
    frames = stories.STORY_EXPORT_MAX_SOURCE_SAMPLE_VALUES // 8 + 1
    with pytest.raises(stories.StoryAudioExportLimitError, match="decoded-sample limit"):
        stories._validate_source_shape(
            frames=frames,
            sample_rate=192_000,
            channels=8,
            generation_id="compressed-bomb",
        )


def test_one_decode_failure_fails_the_whole_story_and_cleans_temp(story_db, tmp_path, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "valid", np.ones(240, dtype=np.float32))
    corrupt_path = generations_dir / "corrupt.wav"
    corrupt_path.write_bytes(b"not audio")
    db.add(
        Generation(
            id="corrupt",
            profile_id="profile",
            text="corrupt",
            audio_path="generations/corrupt.wav",
            duration=1.0,
            status="completed",
        )
    )
    db.commit()
    _add_item(db, "valid")
    _add_item(db, "corrupt")

    real_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(
        stories.tempfile,
        "mkdtemp",
        lambda **kwargs: real_mkdtemp(prefix=kwargs["prefix"], dir=tmp_path),
    )
    with pytest.raises(stories.StoryAudioExportError, match="could not be decoded"):
        asyncio.run(stories.export_story_audio("story", db))
    assert not list(tmp_path.glob("job-*"))


def test_capacity_failure_happens_before_timeline_creation(story_db, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.zeros(24_000, dtype=np.float32))
    _add_item(db, "generation")
    monkeypatch.setattr(stories.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    monkeypatch.setattr(stories.os, "ftruncate", lambda *args, **kwargs: pytest.fail("scratch file was created"))

    with pytest.raises(stories.StoryAudioExportLimitError, match="disk space"):
        asyncio.run(stories.export_story_audio("story", db))


def test_startup_reclaims_crash_abandoned_export_without_following_links(story_db, tmp_path):
    _db, _generations_dir = story_db
    root = stories._story_export_root()
    abandoned = root / "job-abandoned"
    abandoned.mkdir()
    (abandoned / "timeline.f32").write_bytes(b"abandoned mix")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    linked = root / "job-linked"
    linked.symlink_to(outside, target_is_directory=True)
    unexpected = root / "operator-note"
    unexpected.write_text("do not delete", encoding="utf-8")

    _app._prune_abandoned_story_audio_exports()

    assert not abandoned.exists()
    assert not linked.exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert unexpected.read_text(encoding="utf-8") == "do not delete"


def test_allocation_cleanup_preserves_live_response_directory(story_db):
    _db, _generations_dir = story_db
    root = stories._story_export_root()
    active = root / "job-active"
    active.mkdir()
    (active / "story.wav").write_bytes(b"live response")
    stories._active_story_export_directories.add(active)
    try:
        removed, refused, truncated = stories.cleanup_abandoned_story_audio_exports()
        assert (removed, refused, truncated) == (0, 0, False)
        assert (active / "story.wav").read_bytes() == b"live response"
    finally:
        stories._cleanup_story_export(active)


def test_stale_cleanup_releases_reservation_after_transient_remove_failure(story_db, monkeypatch):
    _db, _generations_dir = story_db
    root = stories._story_export_root()
    active = root / "job-transient"
    active.mkdir()
    (active / "story.wav").write_bytes(b"rendered response")
    released: list[bool] = []
    stories._active_story_export_directories.add(active)
    stories._story_export_reservations[active] = SimpleNamespace(release=lambda: released.append(True))
    real_remove = stories._remove_story_export_directory
    monkeypatch.setattr(
        stories,
        "_remove_story_export_directory",
        lambda _path: (_ for _ in ()).throw(OSError("transient remove failure")),
    )

    stories._cleanup_story_export(active)

    assert active.exists()
    assert released == []
    assert active in stories._story_export_reservations

    monkeypatch.setattr(stories, "_remove_story_export_directory", real_remove)
    assert stories.cleanup_abandoned_story_audio_exports() == (1, 0, False)
    assert not active.exists()
    assert released == [True]
    assert active not in stories._story_export_reservations


def test_cancellation_signals_drains_worker_and_cleans_temp(story_db, tmp_path, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.zeros(240, dtype=np.float32))
    _add_item(db, "generation")
    started = threading.Event()
    stopped = threading.Event()
    real_mkdtemp = tempfile.mkdtemp

    def cancellable_worker(_clips, *, mix_path, output_path, cancel_event):
        del mix_path, output_path
        started.set()
        cancel_event.wait(timeout=5)
        stopped.set()
        raise stories._StoryAudioExportCancelledError

    monkeypatch.setattr(stories, "_render_story_audio_to_path", cancellable_worker)
    monkeypatch.setattr(
        stories.tempfile,
        "mkdtemp",
        lambda **kwargs: real_mkdtemp(prefix=kwargs["prefix"], dir=tmp_path),
    )

    async def run_and_cancel():
        task = asyncio.create_task(stories.export_story_audio("story", db))
        await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run_and_cancel())
    assert stopped.is_set()
    assert not list(tmp_path.glob("job-*"))


def test_concurrent_story_export_is_rejected_without_waiting(story_db, tmp_path, monkeypatch):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.zeros(240, dtype=np.float32))
    _add_item(db, "generation")
    started = threading.Event()
    real_mkdtemp = tempfile.mkdtemp

    def blocked_worker(_clips, *, mix_path, output_path, cancel_event):
        del mix_path, output_path
        started.set()
        cancel_event.wait(timeout=5)
        raise stories._StoryAudioExportCancelledError

    monkeypatch.setattr(stories, "_render_story_audio_to_path", blocked_worker)
    monkeypatch.setattr(
        stories.tempfile,
        "mkdtemp",
        lambda **kwargs: real_mkdtemp(prefix=kwargs["prefix"], dir=tmp_path),
    )

    async def run_concurrent_exports():
        first = asyncio.create_task(stories.export_story_audio("story", db))
        assert await asyncio.to_thread(started.wait, 2)
        with pytest.raises(stories.StoryAudioExportBusyError, match="already running"):
            await stories.export_story_audio("story", db)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(run_concurrent_exports())
    assert not list(tmp_path.glob("job-*"))


def test_route_returns_file_response_and_background_removes_export(story_db):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.ones(240, dtype=np.float32))
    _add_item(db, "generation")

    response = asyncio.run(story_routes.export_story_audio("story", db=db))
    export_path = Path(response.path)
    assert response.media_type == "audio/wav"
    assert export_path.is_file()
    assert response.background is not None
    asyncio.run(response.background())
    assert not export_path.parent.exists()


def test_stream_disconnect_still_removes_private_export(story_db):
    db, generations_dir = story_db
    _add_generation(db, generations_dir, "generation", np.ones(240, dtype=np.float32))
    _add_item(db, "generation")

    async def disconnect_during_body():
        response = await story_routes.export_story_audio("story", db=db)
        export_path = Path(response.path)

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                raise ConnectionError("client disconnected")

        scope = {"type": "http", "method": "GET", "headers": []}
        with pytest.raises(ConnectionError, match="client disconnected"):
            await response(scope, receive, send)
        assert not export_path.parent.exists()

    asyncio.run(disconnect_during_body())


def test_route_maps_resource_limits_without_exposing_internal_errors(story_db, monkeypatch):
    db, _generations_dir = story_db

    async def limited(_story_id, _db):
        raise stories.StoryAudioExportLimitError("bounded limit")

    monkeypatch.setattr(stories, "export_story_audio", limited)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(story_routes.export_story_audio("story", db=db))
    assert raised.value.status_code == 413
    assert raised.value.detail == "bounded limit"


def test_route_maps_busy_export_with_retry_hint(story_db, monkeypatch):
    db, _generations_dir = story_db

    async def busy(_story_id, _db):
        raise stories.StoryAudioExportBusyError("already running")

    monkeypatch.setattr(stories, "export_story_audio", busy)
    with pytest.raises(HTTPException) as raised:
        asyncio.run(story_routes.export_story_audio("story", db=db))
    assert raised.value.status_code == 409
    assert raised.value.headers == {"Retry-After": "1"}
