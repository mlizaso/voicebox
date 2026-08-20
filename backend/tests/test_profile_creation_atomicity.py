"""Atomic acknowledgement tests for random-ID profile creation."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import Base, VoiceProfile
from backend.models import VoiceProfileCreate
from backend.services import profiles


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def profile_data():
    return VoiceProfileCreate(
        name="Narrator",
        description="Stable voice",
        language="en",
    )


@pytest.mark.asyncio
async def test_create_returns_durable_profile_when_commit_reports_failure(
    db,
    tmp_path,
    profile_data,
    monkeypatch,
):
    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path / "profiles")
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)

    created = await profiles.create_profile(profile_data, db)

    durable = db.query(VoiceProfile).filter_by(id=created.id).one()
    assert durable.name == profile_data.name
    assert db.query(VoiceProfile).count() == 1
    assert (tmp_path / "profiles" / created.id).is_dir()


@pytest.mark.asyncio
async def test_create_returns_durable_profile_when_refresh_fails(
    db,
    tmp_path,
    profile_data,
    monkeypatch,
):
    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path / "profiles")
    monkeypatch.setattr(
        db,
        "refresh",
        lambda _profile: (_ for _ in ()).throw(RuntimeError("refresh failed after commit")),
    )

    created = await profiles.create_profile(profile_data, db)

    durable = db.query(VoiceProfile).filter_by(id=created.id).one()
    assert durable.name == profile_data.name
    assert db.query(VoiceProfile).count() == 1
    assert (tmp_path / "profiles" / created.id).is_dir()
