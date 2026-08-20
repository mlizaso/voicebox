"""Exact voice binding covers stable sample order, full text, and audio bytes."""

import asyncio
import errno
import json
import stat
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, ProfileSample, VoiceProfile
from backend.services import profiles


def test_voice_binding_changes_with_transcript_audio_or_ordinal(monkeypatch, tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'voice-binding.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"processed-audio-first")
    second_audio.write_bytes(b"processed-audio-second")
    monkeypatch.setattr(
        profiles.config,
        "resolve_storage_path",
        lambda value: tmp_path / value,
    )
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            default_engine="qwen",
        )
    )
    # Insert out of order: conditioning identity must follow ordinal, never UUID/row order.
    second = ProfileSample(
        id="a-id",
        profile_id="profile",
        ordinal=1,
        audio_path=second_audio.name,
        reference_text="Full second transcript.",
    )
    first = ProfileSample(
        id="z-id",
        profile_id="profile",
        ordinal=0,
        audio_path=first_audio.name,
        reference_text="Full first transcript.",
    )
    db.add_all([second, first])
    db.commit()

    original = profiles.compute_profile_voice_binding_sha256("profile", db)
    first.reference_text = "Full first transcript changed."
    db.commit()
    transcript_changed = profiles.compute_profile_voice_binding_sha256("profile", db)
    assert transcript_changed != original

    first.reference_text = "Full first transcript."
    db.commit()
    first_audio.write_bytes(b"processed-audio-FIRST")
    audio_changed = profiles.compute_profile_voice_binding_sha256("profile", db)
    assert audio_changed != original

    first_audio.write_bytes(b"processed-audio-first")
    first.ordinal = 2
    db.commit()
    second.ordinal = 0
    db.commit()
    ordinal_changed = profiles.compute_profile_voice_binding_sha256("profile", db)
    assert ordinal_changed != original
    db.close()


def test_exact_prompt_uses_private_content_snapshot_after_live_file_replacement(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(profiles.config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'voice-binding.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    live_audio = data_dir / "profiles" / "live.wav"
    live_audio.parent.mkdir()
    original_bytes = b"exact-original-reference"
    live_audio.write_bytes(original_bytes)
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            default_engine="qwen",
        )
    )
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/live.wav",
            reference_text="Full exact transcript.",
        )
    )
    db.commit()
    expected = profiles.compute_profile_voice_binding_sha256("profile", db)

    class Backend:
        async def create_voice_prompt(self, audio_path, reference_text, use_cache=True):
            return {
                "ref_audio": audio_path,
                "ref_text": reference_text,
            }, False

    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda _engine: Backend(),
    )
    descriptor = profiles.freeze_exact_voice_profile("profile", db, engine="qwen")
    live_audio.write_bytes(b"replacement-reference")
    sample = db.query(ProfileSample).filter_by(id="sample").one()
    db.delete(sample)
    db.commit()
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "runtime-revision",
    )
    prompt = asyncio.run(
        profiles.create_exact_voice_prompt_from_snapshot(
            descriptor,
            expected_voice_binding_sha256=expected,
            expected_tts_implementation_revision="runtime-revision",
            engine="qwen",
        )
    )

    snapshot_path = Path(prompt["ref_audio"])
    assert snapshot_path != live_audio
    assert snapshot_path.read_bytes() == original_bytes
    assert stat.S_IMODE(snapshot_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(snapshot_path.parent.stat().st_mode) == 0o700
    db.close()


def test_repeated_exact_freeze_deduplicates_macos_nonempty_destination(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(profiles.config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'dedupe.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    live_audio = data_dir / "profiles" / "live.wav"
    live_audio.parent.mkdir()
    live_audio.write_bytes(b"same-exact-reference")
    db.add(
        VoiceProfile(
            id="profile",
            name="Narrator",
            voice_type="cloned",
            default_engine="qwen",
        )
    )
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/live.wav",
            reference_text="Exact transcript.",
        )
    )
    db.commit()

    original_rename = Path.rename

    def macos_rename(source, target):
        if Path(target).exists():
            raise OSError(errno.ENOTEMPTY, "Directory not empty", str(target))
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", macos_rename)
    first = profiles.freeze_exact_voice_profile("profile", db, engine="qwen")
    second = profiles.freeze_exact_voice_profile("profile", db, engine="qwen")
    assert second == first

    snapshot_root = data_dir / "exact_voice_snapshots"
    snapshot_dir = snapshot_root / first["snapshot_key"]
    assert snapshot_dir.is_dir()
    assert not list(snapshot_root.glob(".pending-*"))

    # Content addressing must never turn the dedupe race into overwrite semantics.
    snapshot_audio = snapshot_dir / "sample-0000.wav"
    snapshot_audio.write_bytes(b"corrupt-existing-snapshot")
    with pytest.raises(ValueError, match="was modified"):
        profiles.freeze_exact_voice_profile("profile", db, engine="qwen")
    assert snapshot_audio.read_bytes() == b"corrupt-existing-snapshot"
    assert not list(snapshot_root.glob(".pending-*"))
    db.close()


def test_raw_exact_snapshot_rejects_transcript_metadata_tampering(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(profiles.config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'raw-tamper.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    live_audio = data_dir / "profiles" / "live.wav"
    live_audio.parent.mkdir()
    live_audio.write_bytes(b"exact-reference")
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add(
        ProfileSample(
            id="sample",
            profile_id="profile",
            ordinal=0,
            audio_path="profiles/live.wav",
            reference_text="Original transcript.",
        )
    )
    db.commit()
    descriptor = profiles.freeze_exact_voice_profile("profile", db, engine="qwen")
    metadata_path = data_dir / "exact_voice_snapshots" / descriptor["snapshot_key"] / "snapshot.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["samples"][0]["reference_text"] = "Tampered transcript."
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="contents do not match"):
        profiles._verify_raw_exact_snapshot(
            descriptor,
            expected_binding_sha256=descriptor["voice_binding_sha256"],
        )
    db.close()


def test_derived_exact_snapshot_rejects_reference_text_tampering(
    monkeypatch,
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(profiles.config, "_data_dir", data_dir)
    engine = create_engine(f"sqlite:///{data_dir / 'derived-tamper.db'}")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    profile_dir = data_dir / "profiles" / "profile"
    profile_dir.mkdir(parents=True)
    for index, payload in enumerate((b"first-reference", b"second-reference")):
        (profile_dir / f"sample-{index}.wav").write_bytes(payload)
    db.add(VoiceProfile(id="profile", name="Narrator", voice_type="cloned"))
    db.add_all(
        [
            ProfileSample(
                id="first",
                profile_id="profile",
                ordinal=0,
                audio_path="profiles/profile/sample-0.wav",
                reference_text="First transcript.",
            ),
            ProfileSample(
                id="second",
                profile_id="profile",
                ordinal=1,
                audio_path="profiles/profile/sample-1.wav",
                reference_text="Second transcript.",
            ),
        ]
    )
    db.commit()

    class Backend:
        async def combine_voice_prompts(self, _paths, texts):
            return np.zeros(32, dtype=np.float32), " ".join(texts)

        async def create_voice_prompt(self, audio_path, reference_text, use_cache=True):
            return {"ref_audio": audio_path, "ref_text": reference_text}, False

    monkeypatch.setattr(
        "backend.backends.get_tts_backend_for_engine",
        lambda _engine: Backend(),
    )
    monkeypatch.setattr(
        "backend.backends.get_tts_implementation_revision",
        lambda: "runtime-revision",
    )
    monkeypatch.setattr(
        profiles,
        "save_audio",
        lambda _audio, path, _sample_rate: Path(path).write_bytes(b"combined"),
    )
    descriptor = profiles.freeze_exact_voice_profile("profile", db, engine="qwen")
    prompt = asyncio.run(
        profiles.create_exact_voice_prompt_from_snapshot(
            descriptor,
            expected_voice_binding_sha256=descriptor["voice_binding_sha256"],
            expected_tts_implementation_revision="runtime-revision",
            engine="qwen",
        )
    )
    derived_metadata_path = Path(prompt["ref_audio"]).parent / "snapshot.json"
    derived_metadata = json.loads(derived_metadata_path.read_text(encoding="utf-8"))
    derived_metadata["reference_text"] = "Tampered combined transcript."
    derived_metadata_path.write_text(json.dumps(derived_metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata is invalid"):
        asyncio.run(
            profiles.create_exact_voice_prompt_from_snapshot(
                descriptor,
                expected_voice_binding_sha256=descriptor["voice_binding_sha256"],
                expected_tts_implementation_revision="runtime-revision",
                engine="qwen",
            )
        )
    db.close()
