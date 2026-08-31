"""Singleton settings initialization is safe under first-request races."""

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.database import Base, CaptureSettings, GenerationSettings
from backend.services import settings


@pytest.fixture
def database(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'settings.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    try:
        yield db, factory
    finally:
        db.close()
        engine.dispose()


def _assert_concurrent_insert_is_recovered(
    database,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model,
    getter,
) -> None:
    db, factory = database

    def commit_after_competing_insert() -> None:
        with factory() as competing_db:
            competing_db.add(model(id=settings.SINGLETON_ID))
            competing_db.commit()
        raise IntegrityError("singleton insert", {}, RuntimeError("unique constraint"))

    monkeypatch.setattr(db, "commit", commit_after_competing_insert)

    row = getter(db)

    assert row.id == settings.SINGLETON_ID
    assert db.query(model).count() == 1


def test_capture_settings_initialization_recovers_from_concurrent_insert(database, monkeypatch) -> None:
    _assert_concurrent_insert_is_recovered(
        database,
        monkeypatch,
        model=CaptureSettings,
        getter=settings.get_capture_settings,
    )


def test_generation_settings_initialization_recovers_from_concurrent_insert(database, monkeypatch) -> None:
    _assert_concurrent_insert_is_recovered(
        database,
        monkeypatch,
        model=GenerationSettings,
        getter=settings.get_generation_settings,
    )
