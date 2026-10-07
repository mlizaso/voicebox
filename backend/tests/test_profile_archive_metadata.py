"""Profile archives preserve supported voice configuration across export/import."""

import io
import json
import random
import wave
import zipfile

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend import config
from backend.database import Base, ProfileSample, VoiceProfile
from backend.services import export_import


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path / "data")
    monkeypatch.delenv("VOICEBOX_SHARED_GENERATIONS", raising=False)
    engine = create_engine(f"sqlite:///{tmp_path / 'metadata.db'}")
    Base.metadata.create_all(engine)
    with Session(bind=engine) as session:
        yield session
    engine.dispose()


def _wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(random.Random(42).randbytes(2 * 2 * 24_000))
    return output.getvalue()


def _profile_archive(metadata, *, sample: bytes | None = None):
    output = io.BytesIO()
    samples = {"reference.wav": "Reference transcript"} if sample is not None else {}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"version": "1.0", "profile": metadata, "sample_order": list(samples)}),
        )
        archive.writestr("samples.json", json.dumps(samples))
        if sample is not None:
            archive.writestr("samples/reference.wav", sample)
    return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice_fields",
    [
        {"voice_type": "cloned", "default_engine": "qwen"},
        {
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_heart",
            "default_engine": "kokoro",
        },
        {
            "voice_type": "designed",
            "design_prompt": "A warm, calm storyteller.",
            "default_engine": "qwen_voice_design",
        },
    ],
    ids=["cloned", "preset-without-samples", "designed-without-samples"],
)
async def test_profile_archive_roundtrip_preserves_supported_metadata(db, voice_fields):
    effects = [{"type": "gain", "enabled": False, "params": {"gain_db": -4.5}}]
    expected = {
        "name": "Narrator",
        "description": "Voice description",
        "language": "es",
        "personality": "Speak as a curious science teacher.",
        "voice_type": "cloned",
        "preset_engine": None,
        "preset_voice_id": None,
        "design_prompt": None,
        "default_engine": None,
        **voice_fields,
        "effects_chain": effects,
    }
    db.add(VoiceProfile(id="source", **{**expected, "effects_chain": json.dumps(effects)}))
    with_samples = voice_fields["voice_type"] == "cloned"
    if with_samples:
        sample_path = config.get_profiles_dir() / "source" / "reference.wav"
        sample_path.parent.mkdir(mode=0o700)
        sample_path.write_bytes(_wav_bytes())
        db.add(
            ProfileSample(
                id="reference",
                profile_id="source",
                ordinal=0,
                audio_path=config.to_storage_path(sample_path),
                reference_text="Reference transcript",
            )
        )
    db.commit()

    exported = await export_import.export_profile_to_zip("source", db)
    try:
        with zipfile.ZipFile(exported.path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["profile"] == expected
            if not with_samples:
                assert json.loads(archive.read("samples.json")) == {}
                assert manifest["sample_order"] == []
        imported = await export_import.import_profile_from_zip(exported.path, db)
    finally:
        exported.cleanup()

    assert imported.id != "source"
    assert imported.name == "Narrator (1)"
    assert imported.sample_count == int(with_samples)
    row = db.query(VoiceProfile).filter_by(id=imported.id).one()
    for field, value in expected.items():
        if field in {"name", "effects_chain"}:
            continue
        assert getattr(row, field) == value
        assert getattr(imported, field) == value
    assert json.loads(row.effects_chain) == effects
    assert [effect.model_dump() for effect in imported.effects_chain] == effects
    samples = db.query(ProfileSample).filter_by(profile_id=imported.id).all()
    if with_samples:
        assert len(samples) == 1
        assert samples[0].ordinal == 0
        assert samples[0].reference_text == "Reference transcript"
        assert config.resolve_storage_path(samples[0].audio_path).is_file()
    else:
        assert samples == []
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
@pytest.mark.parametrize("voice_fields", [{}, {"voice_type": None}], ids=["omitted", "null"])
async def test_profile_archive_legacy_metadata_defaults_and_ignores_identity_claims(db, voice_fields):
    archive = _profile_archive(
        {"name": "Legacy", "id": "untrusted-id", "voice_binding_sha256": "untrusted-binding", **voice_fields},
        sample=_wav_bytes(),
    )
    imported = await export_import.import_profile_from_zip(archive, db)
    assert imported.id != "untrusted-id"
    assert imported.name == "Legacy"
    assert imported.language == "en"
    assert imported.voice_type == "cloned"
    for field in (
        "description",
        "personality",
        "default_engine",
        "effects_chain",
        "preset_engine",
        "preset_voice_id",
        "design_prompt",
    ):
        assert getattr(imported, field) is None
    assert imported.sample_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("original_name", "existing_suffixes", "expected_name"),
    [
        ("a" * 100, [], "a" * 96 + " (1)"),
        ("語" * 100, [" (1)"], "語" * 96 + " (2)"),
        ("😀" * 100, [f" ({number})" for number in range(1, 10)], "😀" * 95 + " (10)"),
        ("Short", [" (1)"], "Short (2)"),
    ],
    ids=["ascii-limit", "unicode-limit", "two-digit-collision", "short-control"],
)
async def test_profile_archive_disambiguates_names_within_schema_limit(
    db, original_name, existing_suffixes, expected_name
):
    db.add(VoiceProfile(id="original", name=original_name))
    for index, suffix in enumerate(existing_suffixes):
        db.add(VoiceProfile(id=f"collision-{index}", name=original_name[: 100 - len(suffix)] + suffix))
    db.commit()
    archive = _profile_archive(
        {
            "name": original_name,
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_heart",
        }
    )
    imported = await export_import.import_profile_from_zip(archive, db)
    assert imported.name == expected_name
    assert len(imported.name) <= 100
    assert db.query(VoiceProfile).filter_by(id=imported.id).one().name == expected_name
    assert db.query(VoiceProfile).filter_by(id="original").one().name == original_name
    assert db.query(VoiceProfile).count() == len(existing_suffixes) + 2


@pytest.mark.asyncio
@pytest.mark.parametrize("effects_chain", [None, []], ids=["null", "empty"])
async def test_profile_archive_empty_effects_and_preset_defaults_survive_ambiguous_commit(
    db, monkeypatch, effects_chain
):
    real_commit = db.commit

    def commit_then_raise():
        real_commit()
        raise RuntimeError("post-commit acknowledgement failed")

    monkeypatch.setattr(db, "commit", commit_then_raise)
    archive = _profile_archive(
        {
            "name": "Preset",
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_heart",
            "personality": "A curious narrator",
            "effects_chain": effects_chain,
        }
    )
    imported = await export_import.import_profile_from_zip(archive, db)
    assert imported.voice_type == "preset"
    assert imported.preset_engine == imported.default_engine == "kokoro"
    assert imported.preset_voice_id == "af_heart"
    assert imported.personality == "A curious narrator"
    assert imported.effects_chain == effects_chain
    assert imported.sample_count == 0
    row = db.query(VoiceProfile).filter_by(id=imported.id).one()
    assert row.effects_chain == (None if effects_chain is None else "[]")
    assert (config.get_profiles_dir() / imported.id).is_dir()
    assert not list(config.get_deletion_journal_dir().glob("*.json"))


@pytest.mark.asyncio
async def test_profile_archives_still_require_samples_for_cloned_voices(db):
    archive = _profile_archive({"name": "Empty clone"})
    with pytest.raises(ValueError, match="contains no voice samples"):
        await export_import.import_profile_from_zip(archive, db)
    db.add(VoiceProfile(id="clone", name="Empty clone", voice_type="cloned"))
    db.commit()
    with pytest.raises(ValueError, match="has no samples"):
        await export_import.export_profile_to_zip("clone", db)
    assert not config._data_dir.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"voice_type": "unknown"},
        {"voice_type": []},
        {"personality": []},
        {"personality": "x" * 2001},
        {"default_engine": []},
        {"default_engine": "kokoro"},
        {"preset_engine": {}},
        {"preset_voice_id": []},
        {"design_prompt": []},
        {"design_prompt": "x" * 2001},
        {"design_prompt": "Cloned profiles do not use a prompt"},
        {"voice_type": "preset"},
        {"voice_type": "preset", "preset_engine": "kokoro"},
        {"voice_type": "preset", "preset_engine": "kokoro", "preset_voice_id": "unknown"},
        {
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_heart",
            "default_engine": "qwen",
        },
        {"voice_type": "designed", "design_prompt": " "},
        {"voice_type": "designed", "design_prompt": "Warm voice", "preset_engine": "kokoro"},
        {"effects_chain": {}},
        {"effects_chain": "[]"},
        {"effects_chain": [{"type": "unknown"}]},
        {"effects_chain": [{"type": "gain", "enabled": "false"}]},
        {"effects_chain": [{"type": "gain", "params": []}]},
        {"effects_chain": [{"type": "gain", "params": {"gain_db": True}}]},
        {"effects_chain": [{"type": "gain", "params": {"gain_db": float("nan")}}]},
        {"effects_chain": [{"type": "gain", "params": {"gain_db": 10**1000}}]},
        {"effects_chain": [{"type": "gain", "params": {"gain_db": -40.1}}]},
        {"effects_chain": [{"type": "gain", "params": {"unknown": 1}}]},
        {"effects_chain": [{"type": "gain"}] * 33},
    ],
)
async def test_profile_archive_rejects_malformed_metadata_before_side_effects(db, monkeypatch, metadata):
    def reject_allocation(_projected_bytes):
        raise AssertionError("Invalid metadata reached staging allocation")

    monkeypatch.setattr(export_import, "_allocate_archive_import", reject_allocation)
    archive = _profile_archive({"name": "Hostile", **metadata}, sample=b"unused")
    with pytest.raises(ValueError, match=r"Invalid manifest\.json"):
        await export_import.import_profile_from_zip(archive, db)

    assert db.query(VoiceProfile).count() == 0
    assert db.query(ProfileSample).count() == 0
    assert not config._data_dir.exists()
