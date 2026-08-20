"""Bounded, crash-safe lifecycle tests for immutable exact voice snapshots."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import app as backend_app
from backend.database import Base, Generation, ProfileSample, VoiceProfile
from backend.services import profiles


def _database(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(profiles.config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'voicebox.db'}")
    Base.metadata.create_all(engine)
    return data_dir, sessionmaker(bind=engine)()


def _profile_with_sample(data_dir: Path, db, *, transcript: str = "Reference text."):
    audio = data_dir / "profiles" / "profile" / "sample.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"immutable-reference-audio")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/profile/sample.wav",
            reference_text=transcript,
        )
    )
    db.commit()
    return audio


def _exact_generation(db, descriptor: dict, *, generation_id: str, status: str):
    db.add(
        Generation(
            id=generation_id,
            profile_id="profile",
            text="Exact text",
            status=status,
            exact_request_sha256="f" * 64,
            exact_voice_snapshot_json=json.dumps(descriptor, sort_keys=True),
            voice_binding_sha256=descriptor["voice_binding_sha256"],
        )
    )
    db.commit()


def _derived_snapshot(root: Path, binding: str, *, revision: str) -> Path:
    derived_hash = hashlib.sha256(f"exact-derived-voice-prompt-v1\0{binding}\0qwen\0{revision}".encode()).hexdigest()
    directory = root / f"prompt-{derived_hash}"
    directory.mkdir()
    audio = directory / "combined.wav"
    audio.write_bytes(b"combined-reference")
    profiles._write_private_json(
        directory / "snapshot.json",
        {
            "format_version": 1,
            "kind": "derived",
            "voice_binding_sha256": binding,
            "engine": "qwen",
            "tts_implementation_revision": revision,
            "combined_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
            "reference_text": "Reference text.",
        },
    )
    return directory


def test_startup_gc_removes_abandoned_pending_directories(monkeypatch, tmp_path):
    data_dir, db = _database(monkeypatch, tmp_path)
    root = data_dir / "exact_voice_snapshots"
    root.mkdir()
    for name in (".pending-dead", ".pending-prompt-dead", ".gc-dead"):
        directory = root / name
        directory.mkdir()
        (directory / "partial.wav").write_bytes(b"partial")

    report = profiles.garbage_collect_exact_voice_snapshots(db)

    assert report.pending_removed == 3
    assert report.finalized_removed == 0
    assert report.refused == 0
    assert not list(root.iterdir())
    db.close()


def test_application_startup_hook_runs_voice_snapshot_cleanup(monkeypatch, tmp_path):
    data_dir, db = _database(monkeypatch, tmp_path)
    pending = data_dir / "exact_voice_snapshots" / ".pending-killed-worker"
    pending.mkdir(parents=True)
    (pending / "sample-0000.wav").write_bytes(b"partial")

    backend_app._prune_abandoned_exact_voice_snapshots(db)

    assert not pending.exists()
    db.close()


def test_gc_preserves_failed_and_current_bindings_but_removes_completed(
    monkeypatch,
    tmp_path,
):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(data_dir, db, transcript="Failed binding.")
    failed = profiles.freeze_exact_voice_profile("profile", db)
    root = data_dir / "exact_voice_snapshots"
    failed_prompt = _derived_snapshot(
        root,
        failed["voice_binding_sha256"],
        revision="failed-runtime",
    )

    sample = db.query(ProfileSample).filter_by(id="sample").one()
    sample.reference_text = "Completed binding."
    db.commit()
    completed = profiles.freeze_exact_voice_profile("profile", db)
    completed_prompt = _derived_snapshot(
        root,
        completed["voice_binding_sha256"],
        revision="completed-runtime",
    )
    _exact_generation(db, failed, generation_id="failed", status="failed")
    _exact_generation(db, completed, generation_id="completed", status="completed")
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "failed-runtime",
    )

    # A third, not-yet-frozen live binding closes the route freeze-before-row
    # window without accidentally retaining the completed binding.
    sample.reference_text = "Current live binding."
    db.commit()
    report = profiles.garbage_collect_exact_voice_snapshots(db)

    assert report.finalized_removed == 2
    assert (root / failed["snapshot_key"]).is_dir()
    assert failed_prompt.is_dir()
    assert not (root / completed["snapshot_key"]).exists()
    assert not completed_prompt.exists()
    db.close()


def test_gc_reclaims_orphans_but_refuses_corrupt_and_unsafe_entries(
    monkeypatch,
    tmp_path,
):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(data_dir, db, transcript="Old binding.")
    orphan = profiles.freeze_exact_voice_profile("profile", db)
    sample = db.query(ProfileSample).filter_by(id="sample").one()
    sample.reference_text = "Current binding."
    db.commit()
    root = data_dir / "exact_voice_snapshots"

    corrupt = root / ("raw-" + "a" * 64)
    corrupt.mkdir()
    (corrupt / "snapshot.json").write_text("{}", encoding="utf-8")
    unsafe_target = data_dir / "outside"
    unsafe_target.mkdir()
    unsafe = root / ("raw-" + "b" * 64)
    unsafe.symlink_to(unsafe_target, target_is_directory=True)

    report = profiles.garbage_collect_exact_voice_snapshots(db)

    assert report.finalized_removed == 1
    assert report.refused == 2
    assert not (root / orphan["snapshot_key"]).exists()
    assert corrupt.is_dir()
    assert unsafe.is_symlink()
    db.close()


def test_restart_gc_prunes_obsolete_prompt_revisions_and_restores_quota(
    monkeypatch,
    tmp_path,
):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(data_dir, db)
    descriptor = profiles.freeze_exact_voice_profile("profile", db)
    root = data_dir / "exact_voice_snapshots"
    current_prompt = _derived_snapshot(
        root,
        descriptor["voice_binding_sha256"],
        revision="current-runtime",
    )
    obsolete_prompts = [
        _derived_snapshot(
            root,
            descriptor["voice_binding_sha256"],
            revision=f"obsolete-runtime-{index}",
        )
        for index in range(3)
    ]
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "current-runtime",
    )
    retained_directories = [root / descriptor["snapshot_key"], current_prompt]
    retained_bytes = sum(child.stat().st_size for directory in retained_directories for child in directory.iterdir())
    monkeypatch.setattr(
        profiles,
        "EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES",
        retained_bytes + 1,
    )
    monkeypatch.setattr(profiles, "EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES", 0)

    with (
        pytest.raises(profiles.ExactVoiceSnapshotCapacityError),
        profiles._reserve_exact_snapshot_capacity(1, required_entries=1),
    ):
        pass

    report = profiles.garbage_collect_exact_voice_snapshots(db)

    assert report.finalized_removed == 3
    assert current_prompt.is_dir()
    assert all(not prompt.exists() for prompt in obsolete_prompts)
    with profiles._reserve_exact_snapshot_capacity(1, required_entries=1):
        pass
    db.close()


def test_uncertain_database_owner_retains_all_finalized_snapshots(
    monkeypatch,
    tmp_path,
):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(data_dir, db)
    descriptor = profiles.freeze_exact_voice_profile("profile", db)
    _exact_generation(db, descriptor, generation_id="failed", status="failed")
    row = db.query(Generation).filter_by(id="failed").one()
    row.exact_voice_snapshot_json = "not-json"
    db.commit()

    with pytest.raises(
        profiles.ExactVoiceSnapshotGarbageCollectionError,
        match="invalid snapshot ownership",
    ):
        profiles.garbage_collect_exact_voice_snapshots(db)

    assert (data_dir / "exact_voice_snapshots" / descriptor["snapshot_key"]).is_dir()
    db.close()


def test_oversize_transcript_is_rejected_before_snapshot_publication(
    monkeypatch,
    tmp_path,
):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(
        data_dir,
        db,
        transcript="x" * (profiles.EXACT_VOICE_SNAPSHOT_MAX_REFERENCE_TEXT_BYTES + 1),
    )

    with pytest.raises(ValueError, match="reference text exceeds"):
        profiles.freeze_exact_voice_profile("profile", db)

    root = data_dir / "exact_voice_snapshots"
    assert root.is_dir()
    assert not list(root.iterdir())
    db.close()


def test_store_quota_bounds_repeated_profile_edits(monkeypatch, tmp_path):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(data_dir, db)
    monkeypatch.setattr(profiles, "EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES", 1800)
    monkeypatch.setattr(profiles, "EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES", 0)
    sample = db.query(ProfileSample).filter_by(id="sample").one()
    accepted = 0
    for index in range(100):
        sample.reference_text = f"Reference revision {index}."
        db.commit()
        try:
            profiles.freeze_exact_voice_profile("profile", db)
        except profiles.ExactVoiceSnapshotCapacityError:
            break
        accepted += 1

    assert 1 <= accepted < 100
    root = data_dir / "exact_voice_snapshots"
    usage = profiles._bounded_exact_snapshot_usage(root)
    assert usage.bytes <= profiles.EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES
    assert not list(root.glob(".pending-*"))
    db.close()


def test_free_space_reserve_fails_before_copy(monkeypatch, tmp_path):
    data_dir, db = _database(monkeypatch, tmp_path)
    _profile_with_sample(data_dir, db)
    free = profiles.shutil.disk_usage(data_dir).free
    monkeypatch.setattr(profiles, "EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES", free + 1)

    with pytest.raises(
        profiles.ExactVoiceSnapshotCapacityError,
        match="reserved free space",
    ):
        profiles.freeze_exact_voice_profile("profile", db)

    root = data_dir / "exact_voice_snapshots"
    assert not list(root.iterdir())
    db.close()


def test_derived_reserve_is_checked_before_combining(monkeypatch, tmp_path):
    data_dir, db = _database(monkeypatch, tmp_path)
    audio = _profile_with_sample(data_dir, db, transcript="First.")
    second_audio = audio.with_name("second.wav")
    second_audio.write_bytes(b"second-reference-audio")
    db.add(
        ProfileSample(
            id="second",
            profile_id="profile",
            ordinal=1,
            audio_path="profiles/profile/second.wav",
            reference_text="Second.",
        )
    )
    db.commit()
    descriptor = profiles.freeze_exact_voice_profile("profile", db)
    combined = False

    class Backend:
        async def combine_voice_prompts(self, _paths, _texts):
            nonlocal combined
            combined = True
            raise AssertionError("combine must not run without reserved capacity")

    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda _engine: Backend(),
    )
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "runtime-revision",
    )
    free = profiles.shutil.disk_usage(data_dir).free
    monkeypatch.setattr(profiles, "EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES", free + 1)

    with pytest.raises(profiles.ExactVoiceSnapshotCapacityError):
        asyncio.run(
            profiles.create_exact_voice_prompt_from_snapshot(
                descriptor,
                expected_voice_binding_sha256=descriptor["voice_binding_sha256"],
                expected_tts_implementation_revision="runtime-revision",
            )
        )

    assert combined is False
    assert not list((data_dir / "exact_voice_snapshots").glob(".pending-prompt-*"))
    db.close()


def test_ordinary_prompt_rejects_unbounded_sample_count_before_model_use(
    monkeypatch,
    tmp_path,
):
    data_dir, db = _database(monkeypatch, tmp_path)
    audio = _profile_with_sample(data_dir, db)
    for ordinal in range(1, profiles.EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES + 1):
        db.add(
            ProfileSample(
                id=f"sample-{ordinal}",
                profile_id="profile",
                ordinal=ordinal,
                audio_path=str(audio.relative_to(data_dir)),
                reference_text="Reference.",
            )
        )
    db.commit()

    with pytest.raises(ValueError, match="at most 64 samples"):
        asyncio.run(profiles.create_voice_prompt_for_profile("profile", db))

    db.close()
