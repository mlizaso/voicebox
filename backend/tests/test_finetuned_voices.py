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


def test_exact_finetuned_snapshot_pins_model_but_not_display_name(local_voice):
    snapshot = finetuned_voices.freeze_exact_voice(local_voice["voice_id"])
    assert "model_path" not in snapshot
    assert "name" not in snapshot
    registration = finetuned_voices.registry_directory() / "fabian.json"
    registration.write_text(json.dumps({**local_voice, "name": "Renamed narrator"}))
    assert finetuned_voices.freeze_exact_voice(local_voice["voice_id"]) == snapshot
    finetuned_voices.resolve_exact_voice(snapshot, snapshot["voice_binding_sha256"])

    manifest_path = Path(local_voice["model_path"]) / "voicebox_finetune.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["training_update"] = 42
    manifest_path.write_text(json.dumps(manifest))
    registration.write_text(json.dumps({**local_voice, "manifest_sha256": finetuned_voices.file_sha256(manifest_path)}))
    with pytest.raises(ValueError, match="checkpoint changed"):
        finetuned_voices.resolve_exact_voice(snapshot, snapshot["voice_binding_sha256"])


@pytest.mark.parametrize("field", ["speaker", "manifest_sha256", "preset_voice_id", "voice_binding_sha256"])
def test_exact_finetuned_snapshot_rejects_tampering(local_voice, field):
    snapshot = finetuned_voices.freeze_exact_voice(local_voice["voice_id"])
    original = snapshot["voice_binding_sha256"]
    snapshot[field] = "changed"
    with pytest.raises(ValueError, match="snapshot"):
        finetuned_voices.resolve_exact_voice(snapshot, original)


@pytest.fixture
def audiobook_api(local_voice, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from starlette.testclient import TestClient

    from backend.database import Base, VoiceProfile, get_db
    from backend.routes import generations, profiles as profile_routes

    engine = create_engine(f"sqlite:///{tmp_path / 'audiobook.db'}")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine)
    with sessions() as db:
        db.add(
            VoiceProfile(
                id="narrator",
                name="Fabián",
                language="es",
                voice_type="preset",
                preset_engine="qwen_custom_voice",
                default_engine="qwen_custom_voice",
                preset_voice_id=local_voice["voice_id"],
            )
        )
        db.commit()

    def database():
        with sessions() as db:
            yield db

    app = FastAPI()
    app.include_router(generations.router)
    app.include_router(profile_routes.router)
    app.dependency_overrides[get_db] = database
    monkeypatch.setattr("backend.backends.get_tts_implementation_revision", lambda: "audiobook-runtime")
    with TestClient(app) as client:
        yield client, sessions
    engine.dispose()


def test_finetuned_audiobook_identity_and_exact_admission(audiobook_api, monkeypatch):
    from backend.routes import generations

    client, _sessions = audiobook_api
    identity = client.get("/profiles/narrator/audiobook")
    assert identity.status_code == 200
    snapshot = identity.json()["snapshot"]
    captured = []

    def enqueue(_id, coro):
        captured.append(coro.cr_frame.f_locals.copy())
        coro.close()

    monkeypatch.setattr(generations, "enqueue_generation", enqueue)
    request = {
        "profile_id": "narrator",
        "text": "El libro empieza aquí.",
        "language": "es",
        "engine": "qwen_custom_voice",
        "model_size": "1.7B",
        "seed": 123,
        "tts_implementation_revision": "audiobook-runtime",
        "expected_voice_binding_sha256": snapshot["voice_binding_sha256"],
    }
    result = client.post("/generate/exact", json=request)
    assert result.status_code == 200, result.text
    assert captured[0]["exact_voice_snapshot"] == snapshot
    assert captured[0]["expected_voice_binding_sha256"] == snapshot["voice_binding_sha256"]
    for route in ("/generate/exact", "/generate/stream/exact"):
        result = client.post(route, json={**request, "expected_voice_binding_sha256": "0" * 64})
        assert result.status_code == 409, result.text
    assert len(captured) == 1


def test_finetuned_snapshot_does_not_block_reference_cache_gc(audiobook_api):
    from backend.database import Generation

    client, sessions = audiobook_api
    snapshot = client.get("/profiles/narrator/audiobook").json()["snapshot"]
    with sessions() as db:
        db.add(
            Generation(
                id="incomplete",
                profile_id="narrator",
                text="Test",
                language="es",
                status="failed",
                audio_path="",
                duration=0,
                exact_voice_snapshot_json=json.dumps(snapshot),
                voice_binding_sha256=snapshot["voice_binding_sha256"],
            )
        )
        db.commit()
        profiles.garbage_collect_exact_voice_snapshots(db)
        assert finetuned_voices.read_voice("finetuned:fabian", verify_weights=True)


def test_exact_finetuned_stream_preserves_long_text_and_seeds(audiobook_api, monkeypatch):
    import numpy as np

    from backend.backends import qwen_finetuned_backend
    from backend.routes import generations

    client, _sessions = audiobook_api
    snapshot = client.get("/profiles/narrator/audiobook").json()["snapshot"]
    calls = []

    class TestBackend:
        max_input_chars = 200

        async def generate(self, text, prompt, language, seed, instruct):
            assert prompt["preset_voice_id"] == "finetuned:fabian"
            calls.append((text, seed))
            return np.full(24000, 0.1, dtype=np.float32), 24000

    @asynccontextmanager
    async def voice_request(voice, _size):
        assert voice["manifest_sha256"] == snapshot["manifest_sha256"]
        yield TestBackend()

    async def queue(_id, operation, **_kwargs):
        return await operation

    monkeypatch.setattr(generations, "run_queued_generation", queue)
    monkeypatch.setattr(
        qwen_finetuned_backend, "get_local_backend", lambda: SimpleNamespace(voice_request=voice_request)
    )
    text = " ".join(f"La palabra número {i} pertenece al libro." for i in range(20))
    request = {
        "profile_id": "narrator",
        "text": text,
        "language": "es",
        "seed": 81,
        "engine": "qwen_custom_voice",
        "model_size": "1.7B",
        "max_chunk_chars": 1200,
        "effects_chain": [],
        "tts_implementation_revision": "audiobook-runtime",
        "expected_voice_binding_sha256": snapshot["voice_binding_sha256"],
    }
    first = client.post("/generate/stream/exact", json=request)
    assert first.status_code == 200, first.text
    assert first.content[:4] == b"RIFF"
    first_calls = calls.copy()
    assert len(calls) > 1
    assert all(len(chunk) <= 200 for chunk, _seed in calls)
    assert " ".join(chunk for chunk, _seed in calls).split() == text.split()
    assert len({seed for _text, seed in calls}) == len(calls)
    second = client.post("/generate/stream/exact", json=request)
    assert second.status_code == 200, second.text
    assert calls[len(first_calls) :] == first_calls
    assert second.content == first.content


@pytest.mark.asyncio
async def test_queued_finetuned_snapshot_does_not_follow_live_profile_edits(local_voice, monkeypatch):
    from backend.backends import qwen_finetuned_backend

    snapshot = finetuned_voices.freeze_exact_voice(local_voice["voice_id"])
    captured = []

    @asynccontextmanager
    async def voice_request(voice, _size):
        captured.append(voice["voice_id"])
        yield "saved narrator"

    monkeypatch.setattr(
        qwen_finetuned_backend, "get_local_backend", lambda: SimpleNamespace(voice_request=voice_request)
    )
    edited_profile = SimpleNamespace(voice_type="preset", preset_engine="qwen_custom_voice", preset_voice_id="Ryan")
    async with finetuned_voices.loaded_backend_for_profile(
        "qwen_custom_voice",
        "1.7B",
        profile_id="narrator",
        db=None,
        profile=edited_profile,
        exact_voice_snapshot=snapshot,
    ) as backend:
        assert backend == "saved narrator"
    assert captured == ["finetuned:fabian"]


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


@pytest.mark.parametrize("tokens", [74, 75, 76])
def test_local_backend_classifies_duration_exhaustion_without_changing_budget(monkeypatch, tokens):
    import sys

    import numpy as np

    from backend.utils.chunked_tts import SynthesisDurationLimitError

    core = SimpleNamespace(random=SimpleNamespace(seed=lambda _seed: None))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    calls = []
    expected_audio = np.ones(2400, dtype=np.float32)

    def generate(**kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(audio=expected_audio, sample_rate=24000, token_count=tokens)

    backend = LocalQwenCustomVoiceBackend()
    backend.voice = {"speaker": "fabian"}
    backend.model = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda _text: [1, 2, 3]),
        _voicebox_finetuned_seed=lambda _seed: None,
        generate_custom_voice=generate,
    )
    if tokens >= 75:
        with pytest.raises(SynthesisDurationLimitError, match="duration limit before finishing"):
            backend._generate_on_stream("Texto breve.", 17)
    else:
        audio, rate = backend._generate_on_stream("Texto breve.", 17)
        np.testing.assert_array_equal(audio, expected_audio)
        assert rate == 24000
    assert calls == [
        {
            "text": "Texto breve.",
            "speaker": "fabian",
            "language": "Spanish",
            "max_tokens": 75,
            "stream": False,
            "verbose": False,
        }
    ]


@pytest.mark.parametrize("recovers", [False, True])
def test_exact_finetuned_duration_limit_recovers_or_returns_synthesis_fault(audiobook_api, monkeypatch, recovers):
    import numpy as np

    from backend.backends import qwen_finetuned_backend
    from backend.routes import generations
    from backend.utils.chunked_tts import SynthesisDurationLimitError

    client, _sessions = audiobook_api
    snapshot = client.get("/profiles/narrator/audiobook").json()["snapshot"]
    calls = []

    class LimitedBackend:
        max_input_chars = 200

        async def generate(self, text, *_args):
            calls.append(text)
            if not recovers or len(text) > 100:
                raise SynthesisDurationLimitError("Fine-tuned model reached its duration limit before finishing")
            return np.full(24000, 0.1, dtype=np.float32), 24000

    @asynccontextmanager
    async def voice_request(_voice, _size):
        yield LimitedBackend()

    async def queue(_id, operation, **_kwargs):
        return await operation

    monkeypatch.setattr(generations, "run_queued_generation", queue)
    monkeypatch.setattr(
        qwen_finetuned_backend, "get_local_backend", lambda: SimpleNamespace(voice_request=voice_request)
    )
    response = client.post(
        "/generate/stream/exact",
        json={
            "profile_id": "narrator",
            "text": "Una frase breve. " * 10,
            "language": "es",
            "seed": 81,
            "engine": "qwen_custom_voice",
            "model_size": "1.7B",
            "effects_chain": [],
            "tts_implementation_revision": "audiobook-runtime",
            "expected_voice_binding_sha256": snapshot["voice_binding_sha256"],
        },
    )
    if recovers:
        assert response.status_code == 200, response.text
        assert response.content[:4] == b"RIFF"
        assert len(calls) == 5
    else:
        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "Fine-tuned model reached its duration limit before finishing"
        assert len(calls) == 6


@pytest.mark.asyncio
async def test_installation_is_idempotent_and_preserves_existing_voices(local_voice, tmp_path):
    from scripts.finetune_qwen.install import register_profile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from backend.database.models import Base, VoiceProfile
    from backend.models import VoiceProfileCreate

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
