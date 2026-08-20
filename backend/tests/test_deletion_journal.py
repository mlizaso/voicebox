"""Crash recovery and shared-ownership tests for managed deletion staging."""

import asyncio
import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend.app as app_module
from backend import config
from backend.database import (
    Base,
    Capture,
    Generation,
    ProfileSample,
    VoiceProfile,
)
from backend.services import captures, deletion_journal, history, profiles


def _database(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    (data_dir / "generations").mkdir(parents=True)
    (data_dir / "profiles").mkdir()
    (data_dir / "captures").mkdir()
    monkeypatch.setattr(config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    return data_dir, sessionmaker(bind=engine)()


def _add_generation(db, generation_id: str, audio_path: str) -> Generation:
    if db.query(VoiceProfile).filter_by(id="profile").one_or_none() is None:
        db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    generation = Generation(
        id=generation_id,
        profile_id="profile",
        text="Durable audio",
        language="es",
        audio_path=audio_path,
        status="completed",
    )
    db.add(generation)
    db.commit()
    return generation


def test_startup_recovery_restores_staged_audio_still_owned_by_database(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "owned.wav"
    audio.write_bytes(b"owned audio")
    _add_generation(db, "generation", "generations/owned.wav")

    staged = history._stage_managed_generation_audio("generations/owned.wav")
    assert staged is not None
    assert not audio.exists()

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.restored == 1
    assert report.unresolved == 0
    assert audio.read_bytes() == b"owned audio"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_startup_recovery_discards_staged_audio_after_database_commit(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "deleted.wav"
    audio.write_bytes(b"committed deletion")
    generation = _add_generation(db, "generation", "generations/deleted.wav")
    staged = history._stage_managed_generation_audio("generations/deleted.wav")
    assert staged is not None
    db.delete(generation)
    db.commit()

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.discarded == 1
    assert report.unresolved == 0
    assert not audio.exists()
    assert not (data_dir / staged.staged).exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_startup_recovery_clears_intent_when_rename_never_happened(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "untouched.wav"
    audio.write_bytes(b"rename did not run")
    _add_generation(db, "generation", "generations/untouched.wav")
    entry_stat = audio.lstat()
    deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.GENERATION_AUDIO,
        original=Path("generations/untouched.wav"),
        staged=Path("generations/.voicebox-delete-never.tmp"),
        entry_stat=entry_stat,
    )

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.cleared == 1
    assert report.unresolved == 0
    assert audio.read_bytes() == b"rename did not run"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_startup_recovery_discards_replayed_original_after_committed_delete(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "replayed.wav"
    audio.write_bytes(b"directory replay after committed delete")
    deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.GENERATION_AUDIO,
        original=Path("generations/replayed.wav"),
        staged=Path("generations/.voicebox-delete-replayed.tmp"),
        entry_stat=audio.lstat(),
    )

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.discarded == 1
    assert not audio.exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_startup_recovery_leaves_ambiguous_replacement_untouched(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "ambiguous.wav"
    audio.write_bytes(b"original")
    _add_generation(db, "generation", "generations/ambiguous.wav")
    staged = history._stage_managed_generation_audio("generations/ambiguous.wav")
    assert staged is not None
    audio.write_bytes(b"replacement")

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.unresolved == 1
    assert audio.read_bytes() == b"replacement"
    assert (data_dir / staged.staged).read_bytes() == b"original"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1
    db.close()


def test_startup_recovery_restores_profile_container_before_nested_sample(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    audio = profile_dir / "sample.wav"
    audio.write_bytes(b"nested staged sample")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text="text",
        )
    )
    db.commit()

    sample_stage = profiles._stage_profile_sample_audio(
        "profiles/profile/sample.wav",
        "profile",
    )
    profile_stage = profiles._stage_profile_storage("profile")
    assert sample_stage is not None
    assert profile_stage is not None

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.restored == 2
    assert report.unresolved == 0
    assert audio.read_bytes() == b"nested staged sample"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_delete_generation_preserves_canonically_shared_audio(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "shared.wav"
    audio.write_bytes(b"shared")
    _add_generation(db, "first", "generations/shared.wav")
    _add_generation(db, "second", "data/generations/shared.wav")

    assert asyncio.run(history.delete_generation("first", db)) is True

    assert audio.read_bytes() == b"shared"
    assert db.query(Generation).filter_by(id="second").one().audio_path == "data/generations/shared.wav"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_delete_generation_preserves_audio_owned_by_absolute_legacy_alias(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "shared-absolute.wav"
    audio.write_bytes(b"absolute legacy owner")
    _add_generation(db, "first", "generations/shared-absolute.wav")
    legacy = "/retired/voicebox/data/generations/shared-absolute.wav"
    _add_generation(db, "second", legacy)

    assert config.resolve_storage_path(legacy) == audio
    assert asyncio.run(history.delete_generation("first", db)) is True

    assert audio.read_bytes() == b"absolute legacy owner"
    assert db.query(Generation).filter_by(id="second").one().audio_path == legacy
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_delete_sample_preserves_file_owned_through_legacy_alias(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "first"
    profile_dir.mkdir()
    audio = profile_dir / "shared.wav"
    audio.write_bytes(b"shared voice")
    db.add_all(
        [
            VoiceProfile(id="first", name="First", voice_type="cloned"),
            VoiceProfile(id="second", name="Second", voice_type="cloned"),
            ProfileSample(
                id="first-sample",
                profile_id="first",
                ordinal=0,
                audio_path="profiles/first/shared.wav",
                reference_text="same",
            ),
            ProfileSample(
                id="second-sample",
                profile_id="second",
                ordinal=0,
                audio_path="data/profiles/first/shared.wav",
                reference_text="same",
            ),
        ]
    )
    db.commit()

    assert asyncio.run(profiles.delete_profile_sample("first-sample", db)) is True

    assert audio.read_bytes() == b"shared voice"
    assert db.query(ProfileSample).filter_by(id="second-sample").one_or_none() is not None
    db.close()


def test_delete_profile_retains_directory_referenced_by_surviving_sample(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "first"
    profile_dir.mkdir()
    audio = profile_dir / "shared.wav"
    audio.write_bytes(b"cross-profile legacy reference")
    own_audio = profile_dir / "own.wav"
    own_audio.write_bytes(b"private first-profile voice")
    avatar = profile_dir / "avatar.jpg"
    avatar.write_bytes(b"private first-profile avatar")
    db.add_all(
        [
            VoiceProfile(
                id="first",
                name="First",
                voice_type="cloned",
                avatar_path="profiles/first/avatar.jpg",
            ),
            VoiceProfile(id="second", name="Second", voice_type="cloned"),
            ProfileSample(
                id="first-sample",
                profile_id="first",
                ordinal=0,
                audio_path="profiles/first/own.wav",
                reference_text="private",
            ),
            ProfileSample(
                id="second-sample",
                profile_id="second",
                ordinal=0,
                audio_path="profiles/first/shared.wav",
                reference_text="same",
            ),
        ]
    )
    db.commit()

    assert asyncio.run(profiles.delete_profile("first", db)) is True

    assert audio.read_bytes() == b"cross-profile legacy reference"
    assert not own_audio.exists()
    assert not avatar.exists()
    assert db.query(ProfileSample).filter_by(id="second-sample").one_or_none() is not None
    db.close()


def test_delete_avatar_restores_managed_file_when_commit_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    avatar = profile_dir / "avatar.jpg"
    avatar.write_bytes(b"avatar survives")
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            avatar_path="profiles/profile/avatar.jpg",
        )
    )
    db.commit()

    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    with pytest.raises(RuntimeError, match="locked"):
        asyncio.run(profiles.delete_avatar("profile", db))

    assert db.query(VoiceProfile).filter_by(id="profile").one().avatar_path == ("profiles/profile/avatar.jpg")
    assert avatar.read_bytes() == b"avatar survives"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_delete_avatar_discards_file_after_outcome_ambiguous_commit(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    avatar = profile_dir / "avatar.jpg"
    avatar.write_bytes(b"deleted avatar")
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            avatar_path="profiles/profile/avatar.jpg",
        )
    )
    db.commit()
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    with pytest.raises(RuntimeError, match="connection failed after commit"):
        asyncio.run(profiles.delete_avatar("profile", db))

    assert db.query(VoiceProfile).filter_by(id="profile").one().avatar_path is None
    assert not avatar.exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_upload_avatar_keeps_old_file_when_processing_fails(
    tmp_path,
    monkeypatch,
):
    from PIL import Image

    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    old_avatar = profile_dir / "avatar.jpg"
    old_avatar.write_bytes(b"old avatar")
    uploaded = tmp_path / "new.png"
    Image.new("RGB", (2, 2), "white").save(uploaded)
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            avatar_path="profiles/profile/avatar.jpg",
        )
    )
    db.commit()

    def fail_processing(_input_path, pending_path):
        assert Path(pending_path).stat().st_size == 0
        assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1
        raise RuntimeError("process failed")

    monkeypatch.setattr(profiles, "process_avatar", fail_processing)
    with pytest.raises(RuntimeError, match="process failed"):
        asyncio.run(profiles.upload_avatar("profile", str(uploaded), db))

    assert db.query(VoiceProfile).filter_by(id="profile").one().avatar_path == ("profiles/profile/avatar.jpg")
    assert old_avatar.read_bytes() == b"old avatar"
    assert list(profile_dir.iterdir()) == [old_avatar]
    db.close()


def test_upload_avatar_publishes_new_before_retiring_old(
    tmp_path,
    monkeypatch,
):
    from PIL import Image

    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    old_avatar = profile_dir / "avatar.jpg"
    old_avatar.write_bytes(b"old avatar")
    uploaded = tmp_path / "new.png"
    Image.new("RGB", (2, 2), "white").save(uploaded)
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            avatar_path="profiles/profile/avatar.jpg",
        )
    )
    db.commit()

    response = asyncio.run(profiles.upload_avatar("profile", str(uploaded), db))

    published = config.resolve_storage_path(response.avatar_path)
    assert published is not None
    assert published.is_file()
    assert published.name.startswith("avatar-")
    assert not old_avatar.exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_upload_avatar_commit_failure_restores_pointer_and_removes_new_file(
    tmp_path,
    monkeypatch,
):
    from PIL import Image

    data_dir, db = _database(tmp_path, monkeypatch)
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    old_avatar = profile_dir / "avatar.jpg"
    old_avatar.write_bytes(b"old avatar")
    uploaded = tmp_path / "new.png"
    Image.new("RGB", (2, 2), "white").save(uploaded)
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            avatar_path="profiles/profile/avatar.jpg",
        )
    )
    db.commit()

    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    with pytest.raises(RuntimeError, match="locked"):
        asyncio.run(profiles.upload_avatar("profile", str(uploaded), db))

    assert db.query(VoiceProfile).filter_by(id="profile").one().avatar_path == ("profiles/profile/avatar.jpg")
    assert old_avatar.read_bytes() == b"old avatar"
    assert list(profile_dir.iterdir()) == [old_avatar]
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_upload_avatar_uses_durable_owner_view_when_rollback_fails(
    tmp_path,
    monkeypatch,
):
    from PIL import Image

    data_dir, db = _database(tmp_path, monkeypatch)
    engine = db.get_bind()
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir()
    old_avatar = profile_dir / "avatar.jpg"
    old_avatar.write_bytes(b"old avatar")
    uploaded = tmp_path / "new.png"
    Image.new("RGB", (2, 2), "white").save(uploaded)
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            avatar_path="profiles/profile/avatar.jpg",
        )
    )
    db.commit()

    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    monkeypatch.setattr(
        db,
        "rollback",
        lambda: (_ for _ in ()).throw(RuntimeError("rollback failed")),
    )
    with pytest.raises(RuntimeError, match="rollback failed"):
        asyncio.run(profiles.upload_avatar("profile", str(uploaded), db))

    db.close()
    durable_db = sessionmaker(bind=engine)()
    assert durable_db.query(VoiceProfile).filter_by(id="profile").one().avatar_path == ("profiles/profile/avatar.jpg")
    assert old_avatar.read_bytes() == b"old avatar"
    assert list(profile_dir.iterdir()) == [old_avatar]
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    durable_db.close()


def test_delete_capture_restores_managed_file_when_commit_fails(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "captures" / "capture.wav"
    audio.write_bytes(b"capture survives")
    db.add(
        Capture(
            id="capture",
            audio_path="captures/capture.wav",
            source="file",
            transcript_raw="text",
        )
    )
    db.commit()

    monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("locked")))
    with pytest.raises(RuntimeError, match="locked"):
        captures.delete_capture("capture", db)

    assert db.query(Capture).filter_by(id="capture").one_or_none() is not None
    assert audio.read_bytes() == b"capture survives"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_delete_capture_discards_file_after_outcome_ambiguous_commit(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "captures" / "capture.wav"
    audio.write_bytes(b"deleted capture")
    db.add(
        Capture(
            id="capture",
            audio_path="captures/capture.wav",
            source="file",
            transcript_raw="text",
        )
    )
    db.commit()
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("connection failed after commit")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    with pytest.raises(RuntimeError, match="connection failed after commit"):
        captures.delete_capture("capture", db)

    assert db.query(Capture).filter_by(id="capture").one_or_none() is None
    assert not audio.exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_path_fallback_supports_managed_delete_without_dir_fd(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "portable.wav"
    audio.write_bytes(b"portable")
    _add_generation(db, "generation", "generations/portable.wav")
    monkeypatch.setattr(deletion_journal, "secure_dir_fd_supported", lambda: False)

    assert asyncio.run(history.delete_generation("generation", db)) is True

    assert not audio.exists()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


def test_failed_initial_restore_keeps_intent_for_startup_recovery(
    tmp_path,
    monkeypatch,
):
    data_dir, db = _database(tmp_path, monkeypatch)
    audio = data_dir / "generations" / "restore-later.wav"
    audio.write_bytes(b"recover after failed fsync")
    _add_generation(db, "generation", "generations/restore-later.wav")
    real_rename = deletion_journal.rename_managed_entry
    staged_path = None
    calls = 0

    def fail_after_first_rename(source, destination):
        nonlocal calls, staged_path
        calls += 1
        if calls == 1:
            staged_path = Path(destination)
            real_rename(source, destination)
            raise OSError("simulated parent fsync failure")
        raise OSError("simulated restore failure")

    monkeypatch.setattr(deletion_journal, "rename_managed_entry", fail_after_first_rename)
    try:
        with pytest.raises(OSError, match="restore failure"):
            history._stage_managed_generation_audio("generations/restore-later.wav")
    finally:
        monkeypatch.setattr(deletion_journal, "rename_managed_entry", real_rename)

    assert staged_path is not None
    assert not audio.exists()
    assert (data_dir / staged_path).read_bytes() == b"recover after failed fsync"
    assert len(list(config.get_deletion_journal_dir().glob("*.json"))) == 1

    report = deletion_journal.recover_interrupted_deletions(db)
    assert report.restored == 1
    assert audio.read_bytes() == b"recover after failed fsync"
    assert not list(config.get_deletion_journal_dir().glob("*.json"))
    db.close()


@pytest.mark.skipif(os.name != "posix", reason="FIFO entries are POSIX-only")
def test_fifo_journal_entry_is_rejected_without_blocking(tmp_path, monkeypatch):
    _data_dir, db = _database(tmp_path, monkeypatch)
    fifo = config.get_deletion_journal_dir() / "malicious.json"
    os.mkfifo(fifo)

    report = deletion_journal.recover_interrupted_deletions(db)

    assert report.unresolved == 0
    assert report.malformed == 1
    assert fifo.exists()
    db.close()


def test_startup_refuses_valid_unresolved_recovery(monkeypatch):
    report = deletion_journal.RecoveryReport(unresolved=1)
    monkeypatch.setattr(
        deletion_journal,
        "recover_interrupted_deletions",
        lambda _db: report,
    )

    with pytest.raises(RuntimeError, match="recovery is unresolved"):
        app_module._recover_managed_storage(object())


def test_startup_keeps_malformed_journal_for_inspection_without_dos(monkeypatch):
    report = deletion_journal.RecoveryReport(malformed=1)
    monkeypatch.setattr(
        deletion_journal,
        "recover_interrupted_deletions",
        lambda _db: report,
    )

    app_module._recover_managed_storage(object())


def test_startup_refuses_recovery_exception(monkeypatch):
    def fail_recovery(_db):
        raise OSError("injected recovery failure")

    monkeypatch.setattr(
        deletion_journal,
        "recover_interrupted_deletions",
        fail_recovery,
    )

    with pytest.raises(RuntimeError, match="recovery failed"):
        app_module._recover_managed_storage(object())


def test_startup_refuses_stale_generation_reconciliation_failure():
    class FailingDatabase:
        rolled_back = False

        def execute(self, _statement):
            raise OSError("injected sqlite write failure")

        def rollback(self):
            self.rolled_back = True

    db = FailingDatabase()
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        app_module._reconcile_stale_generations(db)
    assert db.rolled_back is True
