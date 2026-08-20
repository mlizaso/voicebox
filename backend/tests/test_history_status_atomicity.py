"""Durable generation status updates under ambiguous commit outcomes."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, Generation, VoiceProfile
from backend.services import history


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'history-status.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add(VoiceProfile(id="profile-id", name="Narrator"))
    session.add_all(
        [
            Generation(
                id=generation_id,
                profile_id="profile-id",
                text="text",
                language="en",
                engine="qwen",
                status="generating",
                audio_path="",
                duration=0,
            )
            for generation_id in ("singleton", "batch-a", "batch-b")
        ]
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.parametrize("generation_id", ["singleton", "batch-a", "batch-b"])
def test_completed_status_survives_refresh_failure_after_durable_commit(
    db,
    monkeypatch,
    generation_id,
):
    monkeypatch.setattr(
        db,
        "refresh",
        lambda _row: (_ for _ in ()).throw(RuntimeError("connection failed while refreshing committed completion")),
    )

    result = asyncio.run(
        history.update_generation_status(
            generation_id,
            "completed",
            db,
            audio_path=f"generations/{generation_id}.wav",
            duration=3600.0,
        )
    )

    assert result is not None
    assert result.status == "completed"
    assert result.audio_path == f"generations/{generation_id}.wav"
    assert result.duration == 3600.0
    db.expire_all()
    durable = db.query(Generation).filter_by(id=generation_id).one()
    assert durable.status == "completed"
    assert durable.audio_path == f"generations/{generation_id}.wav"


def test_generation_creation_survives_refresh_failure_after_durable_commit(db, monkeypatch):
    monkeypatch.setattr(
        db,
        "refresh",
        lambda _row: (_ for _ in ()).throw(RuntimeError("connection failed while refreshing committed creation")),
    )

    result = asyncio.run(
        history.create_generation(
            generation_id="created-ambiguously",
            profile_id="profile-id",
            text="created",
            language="en",
            audio_path="",
            duration=0,
            seed=9,
            status="generating",
            db=db,
        )
    )

    assert result.id == "created-ambiguously"
    assert result.status == "generating"
    db.expire_all()
    assert db.query(Generation).filter_by(id=result.id).one().status == "generating"


def test_status_update_can_durably_clear_an_error(db):
    row = db.query(Generation).filter_by(id="singleton").one()
    row.error = "previous failure"
    db.commit()

    result = asyncio.run(
        history.update_generation_status(
            "singleton",
            "generating",
            db,
            clear_error=True,
        )
    )

    assert result is not None
    assert result.error is None
    db.expire_all()
    assert db.query(Generation).filter_by(id="singleton").one().error is None
