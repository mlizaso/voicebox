"""Story-item identity and durable acknowledgement regressions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import models
from backend.database import Base, Generation, Story, StoryItem, VoiceProfile
from backend.services import stories


@pytest.fixture
def database(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'stories.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        yield db, factory
    finally:
        db.close()
        engine.dispose()


def _seed_story(db, *, duration: float = 10.0) -> None:
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(Story(id="story", name="Story"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Narration",
            language="en",
            audio_path="generations/generation.wav",
            duration=duration,
            status="completed",
        )
    )
    db.commit()


def _add_item(
    db,
    item_id: str,
    *,
    start_time_ms: int = 0,
    trim_start_ms: int = 0,
    trim_end_ms: int = 0,
) -> None:
    db.add(
        StoryItem(
            id=item_id,
            story_id="story",
            generation_id="generation",
            start_time_ms=start_time_ms,
            track=0,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            volume=1.0,
        )
    )
    db.commit()


def _fail_after_durable_commit(db, monkeypatch: pytest.MonkeyPatch) -> None:
    real_commit = db.commit

    def commit_then_raise() -> None:
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)


def _fail_refresh(db, monkeypatch: pytest.MonkeyPatch) -> None:
    def refresh_failure(_row) -> None:
        raise RuntimeError("refresh failed after commit")

    monkeypatch.setattr(db, "refresh", refresh_failure)


def test_story_mutation_models_use_story_item_identity() -> None:
    update = models.StoryItemUpdateTime(item_id="item", start_time_ms=100)
    reorder = models.StoryItemReorder(item_ids=["item"])
    assert update.item_id == "item"
    assert reorder.item_ids == ["item"]

    with pytest.raises(ValidationError):
        models.StoryItemUpdateTime(generation_id="generation", start_time_ms=100)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        models.StoryItemReorder(generation_ids=["generation"])  # type: ignore[call-arg]


def test_batch_time_update_addresses_duplicate_generation_clips_by_item_id(database) -> None:
    db, _factory = database
    _seed_story(db)
    _add_item(db, "first", start_time_ms=0)
    _add_item(db, "second", start_time_ms=1_000)

    request = models.StoryItemBatchUpdate(
        updates=[
            models.StoryItemUpdateTime(item_id="first", start_time_ms=2_000),
            models.StoryItemUpdateTime(item_id="second", start_time_ms=3_000),
        ]
    )
    assert asyncio.run(stories.update_story_item_times("story", request, db))

    rows = {row.id: row.start_time_ms for row in db.query(StoryItem).all()}
    assert rows == {"first": 2_000, "second": 3_000}


def test_batch_time_update_rejects_duplicate_item_ids_without_mutation(database) -> None:
    db, _factory = database
    _seed_story(db)
    _add_item(db, "first", start_time_ms=100)

    request = models.StoryItemBatchUpdate(
        updates=[
            models.StoryItemUpdateTime(item_id="first", start_time_ms=2_000),
            models.StoryItemUpdateTime(item_id="first", start_time_ms=3_000),
        ]
    )
    assert not asyncio.run(stories.update_story_item_times("story", request, db))
    assert db.query(StoryItem).filter_by(id="first").one().start_time_ms == 100


def test_reorder_preserves_duplicate_generation_clips_and_uses_trimmed_durations(database) -> None:
    db, _factory = database
    _seed_story(db)
    _add_item(db, "first", trim_end_ms=6_000)
    _add_item(db, "second", start_time_ms=5_000, trim_start_ms=4_000)

    result = asyncio.run(stories.reorder_story_items("story", ["second", "first"], db))

    assert result is not None
    assert [item.id for item in result] == ["second", "first"]
    assert [item.start_time_ms for item in result] == [0, 6_200]
    assert db.query(StoryItem).filter_by(id="second").one().start_time_ms == 0
    assert db.query(StoryItem).filter_by(id="first").one().start_time_ms == 6_200


@pytest.mark.parametrize("failure", [_fail_after_durable_commit, _fail_refresh])
def test_create_story_returns_exact_durable_row_after_ambiguous_ack(database, monkeypatch, failure) -> None:
    db, factory = database
    failure(db, monkeypatch)

    response = asyncio.run(stories.create_story(models.StoryCreate(name="Durable", description="Exact"), db))

    with factory() as fresh:
        durable = fresh.query(Story).filter_by(id=response.id).one()
        assert durable.name == "Durable"
        assert durable.description == "Exact"
        assert fresh.query(Story).count() == 1
    assert response.item_count == 0


@pytest.mark.parametrize("failure", [_fail_after_durable_commit, _fail_refresh])
def test_split_returns_exact_durable_items_after_ambiguous_ack(database, monkeypatch, failure) -> None:
    db, factory = database
    _seed_story(db)
    _add_item(db, "original", start_time_ms=1_000, trim_start_ms=1_000, trim_end_ms=1_000)
    failure(db, monkeypatch)

    response = asyncio.run(
        stories.split_story_item(
            "story",
            "original",
            models.StoryItemSplit(split_time_ms=3_000),
            db,
        )
    )

    assert response is not None
    assert response[0].id == "original"
    assert response[1].id != "original"
    with factory() as fresh:
        rows = fresh.query(StoryItem).order_by(StoryItem.start_time_ms).all()
        assert len(rows) == 2
        assert rows[0].id == "original"
        assert rows[0].trim_start_ms == 1_000
        assert rows[0].trim_end_ms == 6_000
        assert rows[1].id == response[1].id
        assert rows[1].start_time_ms == 4_000
        assert rows[1].trim_start_ms == 4_000
        assert rows[1].trim_end_ms == 1_000


@pytest.mark.parametrize("failure", [_fail_after_durable_commit, _fail_refresh])
def test_duplicate_returns_exact_durable_item_after_ambiguous_ack(database, monkeypatch, failure) -> None:
    db, factory = database
    _seed_story(db)
    _add_item(db, "original", start_time_ms=1_000, trim_start_ms=1_000, trim_end_ms=2_000)
    failure(db, monkeypatch)

    response = asyncio.run(stories.duplicate_story_item("story", "original", db))

    assert response is not None
    assert response.id != "original"
    with factory() as fresh:
        rows = fresh.query(StoryItem).order_by(StoryItem.start_time_ms).all()
        assert len(rows) == 2
        duplicate = next(row for row in rows if row.id == response.id)
        assert duplicate.generation_id == "generation"
        assert duplicate.start_time_ms == 8_200
        assert duplicate.trim_start_ms == 1_000
        assert duplicate.trim_end_ms == 2_000
