"""Local checkpoint identities and profile-specific model routing."""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend import config
from backend.backends import mlx_tts_lifecycle
from backend.backends.qwen_finetuned_backend import LocalQwenCustomVoiceBackend
from backend.services import finetuned_voices, profiles


@pytest.fixture
def local_voice(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_data_dir", lambda: tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    names = ("config.json", "model.safetensors", "speech_tokenizer/model.safetensors", "tokenizer_config.json")
    for name in names:
        path = model / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"fixture model bytes")
    manifest = {
        "format": 1,
        "dtype": "bfloat16",
        "language": "es",
        "model_size": "1.7B",
        "speaker": "fabian",
        "files": {name: finetuned_voices.file_sha256(model / name) for name in names},
    }
    path = model / "voicebox_finetune.json"
    path.write_text(json.dumps(manifest))
    spec = {
        "voice_id": "finetuned:fabian",
        "name": "Fabián — fine-tuned",
        "speaker": "fabian",
        "model_path": str(model),
        "manifest_sha256": finetuned_voices.file_sha256(path),
    }
    registry = finetuned_voices.registry_directory()
    registry.mkdir()
    (registry / "fabian.json").write_text(json.dumps(spec))
    return spec


def test_local_voices_are_valid_presets_without_exposing_paths(local_voice):
    result = finetuned_voices.list_voices()
    assert result == [{"voice_id": "finetuned:fabian", "name": local_voice["name"], "gender": "male", "language": "es"}]
    assert "finetuned:fabian" in profiles._get_preset_voice_ids("qwen_custom_voice")
    assert "Ryan" in profiles._get_preset_voice_ids("qwen_custom_voice")
    finetuned_voices.read_voice("finetuned:fabian", verify_weights=True)


@pytest.mark.parametrize("identifier", ["finetuned:../fabian", "finetuned:/tmp/model", "finetuned:", "fabian"])
def test_remote_ids_cannot_select_arbitrary_paths(local_voice, identifier):
    with pytest.raises(ValueError, match="Invalid fine-tuned voice identifier"):
        finetuned_voices.read_voice(identifier)


def test_missing_local_checkpoint_never_falls_back_to_a_builtin_voice(local_voice):
    with pytest.raises(ValueError, match="not installed"):
        finetuned_voices.read_voice("finetuned:missing")


def test_model_weight_mutation_is_rejected_before_load(local_voice):
    (Path(local_voice["model_path"]) / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="artifact changed"):
        finetuned_voices.read_voice("finetuned:fabian", verify_weights=True)


@pytest.mark.asyncio
async def test_finetuned_profile_binds_its_model_before_any_builtin_download(local_voice, monkeypatch):
    from backend.backends import qwen_finetuned_backend

    captured = []

    @asynccontextmanager
    async def local_request(voice, size):
        captured.append((voice["voice_id"], size))
        yield "local backend"

    def forbidden(*_args, **_kwargs):
        pytest.fail("A local voice must never load the built-in preset checkpoint")

    monkeypatch.setattr(
        qwen_finetuned_backend, "get_local_backend", lambda: SimpleNamespace(voice_request=local_request)
    )
    monkeypatch.setattr(mlx_tts_lifecycle, "loaded_tts_backend_for_request", forbidden)
    profile = SimpleNamespace(
        voice_type="preset", preset_engine="qwen_custom_voice", preset_voice_id="finetuned:fabian"
    )
    async with finetuned_voices.loaded_backend_for_profile(
        "qwen_custom_voice", "1.7B", profile_id="p", db=None, profile=profile
    ) as backend:
        assert backend == "local backend"
    assert captured == [("finetuned:fabian", "1.7B")]


@pytest.mark.asyncio
async def test_builtin_preset_retains_its_normal_model_binding(local_voice, monkeypatch):
    calls = []

    @asynccontextmanager
    async def original(engine, size):
        calls.append((engine, size))
        yield "original backend"

    monkeypatch.setattr(mlx_tts_lifecycle, "loaded_tts_backend_for_request", original)
    profile = SimpleNamespace(voice_type="preset", preset_engine="qwen_custom_voice", preset_voice_id="Ryan")
    async with finetuned_voices.loaded_backend_for_profile(
        "qwen_custom_voice", "0.6B", profile_id="p", db=None, profile=profile
    ) as backend:
        assert backend == "original backend"
    assert calls == [("qwen_custom_voice", "0.6B")]


@pytest.mark.asyncio
async def test_local_model_rejects_unsupported_size_before_loading(local_voice, monkeypatch):
    backend = LocalQwenCustomVoiceBackend()
    load = AsyncMock()
    monkeypatch.setattr(backend, "_load", load)
    with pytest.raises(ValueError, match=r"requires Qwen 1\.7B"):
        async with backend.voice_request(local_voice, "0.6B"):
            pytest.fail("Unsupported model size was accepted")
    load.assert_not_called()


@pytest.mark.asyncio
async def test_local_backend_rejects_unbounded_direct_calls_before_inference():
    backend = LocalQwenCustomVoiceBackend()
    with pytest.raises(ValueError, match="generate_chunked"):
        await backend.generate("Una frase de prueba. " * 30, {}, language="es")


@pytest.mark.asyncio
async def test_installation_is_idempotent_and_preserves_existing_voices(local_voice, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from backend.database.models import Base, VoiceProfile
    from backend.models import VoiceProfileCreate
    from scripts.finetune_qwen.install import register_profile

    engine = create_engine(f"sqlite:///{tmp_path / 'install.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        original = await profiles.create_profile(VoiceProfileCreate(name="Existing Spanish voice", language="es"), db)
        first = await register_profile(local_voice, db)
        second = await register_profile(local_voice, db)
        assert first == second
        assert first["preset_voice_id"] == local_voice["voice_id"]
        assert first["language"] == "es"
        assert db.query(VoiceProfile).count() == 2
        assert (await profiles.get_profile(original.id, db)).model_dump() == original.model_dump()
        with pytest.raises(ValueError, match="different installation"):
            await register_profile({**local_voice, "manifest_sha256": "changed"}, db)
        assert db.query(VoiceProfile).count() == 2
    engine.dispose()


@pytest.fixture
def evaluated_voice(local_voice, tmp_path):
    from scripts.finetune_qwen import evaluate
    from scripts.finetune_qwen.evaluate import inference_source_identity, summarize

    audio = tmp_path / "evaluated.wav"
    audio.write_bytes(b"hash-checked fixture audio")
    cases = [
        {"id": f"case-{i}", "text": "La voz narra una historia.", "chapter": i % 2 if i < 4 else None, "seed": i}
        for i in range(8)
    ]
    rows = [
        {
            **case,
            "variant": variant,
            "audio": str(audio),
            "sha256": finetuned_voices.file_sha256(audio),
            "transcribed": case["text"],
            "word_errors": 0,
            "words": 5,
            "wer": 0.0,
            "seconds": 1.0,
            "duration": 5.0,
            "speaker_cosine": 0.8,
            "hit_token_limit": False,
        }
        for variant in ("candidate", "baseline")
        for case in cases
    ]
    model = Path(local_voice["model_path"])
    report = {
        **summarize(rows),
        "identity": {
            "inference_sources": inference_source_identity(),
            "evaluation_code_sha256": finetuned_voices.file_sha256(Path(evaluate.__file__)),
            "config": {"split": "test", "candidate": str(model)},
            "manifest_sha256": local_voice["manifest_sha256"],
            "candidate_sha256": finetuned_voices.file_sha256(model / "model.safetensors"),
            "cases": cases,
        },
        "results": rows,
    }
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps(report))
    return model, evaluation, report


def test_installation_rejects_changed_inference_code(evaluated_voice):
    from scripts.finetune_qwen.install import checked_registration

    model, path, report = evaluated_voice
    report["identity"]["inference_sources"]["backend/utils/chunked_tts.py"] = "changed"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="inference implementation changed"):
        checked_registration(model, path, "fabian", "Fabián")


def test_only_evaluated_unchanged_model_can_be_registered(evaluated_voice):
    from scripts.finetune_qwen.install import checked_registration

    model, path, report = evaluated_voice
    spec = checked_registration(model, path, "fabian", "Fabián")
    assert spec["evaluation_sha256"] == finetuned_voices.file_sha256(path)
    report["identity"]["config"]["split"] = "validation"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="final test evaluation"):
        checked_registration(model, path, "fabian", "Fabián")


def test_installation_recomputes_quality_gates(evaluated_voice):
    from scripts.finetune_qwen.install import checked_registration

    model, path, report = evaluated_voice
    report["results"][0]["hit_token_limit"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="every quality and speed gate"):
        checked_registration(model, path, "fabian", "Fabián")


def test_installation_requires_every_paired_sample(evaluated_voice):
    from scripts.finetune_qwen.install import checked_registration

    model, path, report = evaluated_voice
    report["results"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="missing paired samples"):
        checked_registration(model, path, "fabian", "Fabián")
