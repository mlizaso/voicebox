"""Version creation/default selection is transaction and acknowledgement atomic."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, Generation, GenerationVersion, VoiceProfile
from backend.services import versions


def _database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'versions.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        Generation(
            id="generation",
            profile_id="profile",
            text="Versioned audio",
            language="es",
            audio_path="generations/old.wav",
            status="completed",
        )
    )
    db.add(
        GenerationVersion(
            id="old",
            generation_id="generation",
            label="old",
            audio_path="generations/old.wav",
            is_default=True,
        )
    )
    db.commit()
    return db


def test_default_version_creation_updates_parent_in_one_commit(monkeypatch, tmp_path):
    db = _database(tmp_path)
    commit_calls = 0
    real_commit = db.commit

    def counted_commit():
        nonlocal commit_calls
        commit_calls += 1
        return real_commit()

    monkeypatch.setattr(db, "commit", counted_commit)

    created = versions.create_version(
        generation_id="generation",
        label="processed",
        audio_path="generations/processed.wav",
        db=db,
        is_default=True,
    )

    assert commit_calls == 1
    assert created.is_default is True
    assert db.query(Generation).filter_by(id="generation").one().audio_path == ("generations/processed.wav")
    defaults = db.query(GenerationVersion).filter_by(generation_id="generation", is_default=True).all()
    assert [version.id for version in defaults] == [created.id]
    db.close()


def test_default_version_creation_accepts_commit_then_refresh_failure(monkeypatch, tmp_path):
    db = _database(tmp_path)
    real_refresh = db.refresh

    def fail_refresh(instance, *args, **kwargs):
        if isinstance(instance, GenerationVersion) and instance.id != "old":
            raise RuntimeError("connection failed after durable commit")
        return real_refresh(instance, *args, **kwargs)

    monkeypatch.setattr(db, "refresh", fail_refresh)

    created = versions.create_version(
        generation_id="generation",
        label="processed",
        audio_path="generations/processed.wav",
        db=db,
        is_default=True,
    )

    assert created.label == "processed"
    fresh = sessionmaker(bind=db.get_bind())()
    durable = fresh.query(GenerationVersion).filter_by(id=created.id).one()
    assert durable.is_default is True
    assert fresh.query(Generation).filter_by(id="generation").one().audio_path == ("generations/processed.wav")
    assert fresh.query(GenerationVersion).filter_by(generation_id="generation", is_default=True).count() == 1
    fresh.close()
    db.close()


def test_set_default_updates_flags_and_parent_in_one_commit(monkeypatch, tmp_path):
    db = _database(tmp_path)
    db.add(
        GenerationVersion(
            id="new",
            generation_id="generation",
            label="new",
            audio_path="generations/new.wav",
            is_default=False,
        )
    )
    db.commit()

    commit_calls = 0
    real_commit = db.commit

    def counted_commit():
        nonlocal commit_calls
        commit_calls += 1
        return real_commit()

    monkeypatch.setattr(db, "commit", counted_commit)

    selected = versions.set_default_version("new", db)

    assert selected is not None
    assert commit_calls == 1
    assert selected.is_default is True
    assert db.query(GenerationVersion).filter_by(id="old").one().is_default is False
    assert db.query(Generation).filter_by(id="generation").one().audio_path == "generations/new.wav"
    db.close()


def test_set_default_accepts_commit_then_refresh_failure(monkeypatch, tmp_path):
    db = _database(tmp_path)
    db.add(
        GenerationVersion(
            id="new",
            generation_id="generation",
            label="new",
            audio_path="generations/new.wav",
            is_default=False,
        )
    )
    db.commit()
    monkeypatch.setattr(
        db,
        "refresh",
        lambda _row: (_ for _ in ()).throw(RuntimeError("connection failed while refreshing committed default")),
    )

    selected = versions.set_default_version("new", db)

    assert selected is not None
    assert selected.id == "new"
    assert selected.is_default is True
    fresh = sessionmaker(bind=db.get_bind())()
    assert fresh.query(Generation).filter_by(id="generation").one().audio_path == "generations/new.wav"
    defaults = fresh.query(GenerationVersion).filter_by(generation_id="generation", is_default=True).all()
    assert [version.id for version in defaults] == ["new"]
    fresh.close()
    db.close()
