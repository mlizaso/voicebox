"""Regression tests for Qwen cloning correctness and the mlx-audio 0.4.1 backport."""

import asyncio
import sys
from contextlib import nullcontext
from importlib import metadata
from types import ModuleType, SimpleNamespace

import pytest

import backend.backends.mlx_backend as mlx_backend
from backend.backends.mlx_backend import (
    MLXTTSBackend,
    _apply_mlx_audio_qwen_dtype_backport,
)
from backend.backends.mlx_runtime import (
    MLX_QWEN_TTS_IMPLEMENTATION_REVISION,
    get_mlx_qwen_tts_implementation_revision,
)


class _FakeArray:
    def __init__(self, dtype):
        self.dtype = dtype
        self.astype_calls = []

    def astype(self, dtype):
        self.astype_calls.append(dtype)
        return _FakeArray(dtype)


class _FakeTalker:
    def __init__(self, dtype):
        self.codec_embedding = SimpleNamespace(weight=SimpleNamespace(dtype=dtype))
        self.get_input_embeddings_calls = 0

    def get_input_embeddings(self):
        self.get_input_embeddings_calls += 1
        return self.codec_embedding


class _FakeQwenModel:
    model_type = "qwen3_tts"

    def __init__(self, talker_dtype="bfloat16"):
        self.talker = _FakeTalker(talker_dtype)
        self.extract_calls = []
        self.last_speaker_embedding = None

    def extract_speaker_embedding(self, audio, sr=24000):
        self.extract_calls.append((audio, sr))
        self.last_speaker_embedding = _FakeArray("float32")
        return self.last_speaker_embedding


def test_runtime_revision_requires_the_guarded_mlx_audio_version(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda _distribution: "0.4.1")

    assert get_mlx_qwen_tts_implementation_revision() == MLX_QWEN_TTS_IMPLEMENTATION_REVISION

    monkeypatch.setattr(metadata, "version", lambda _distribution: "0.4.8")
    assert get_mlx_qwen_tts_implementation_revision() is None


def test_runtime_revision_is_absent_without_mlx_audio(monkeypatch):
    def missing(_distribution):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", missing)

    assert get_mlx_qwen_tts_implementation_revision() is None


def test_backport_casts_speaker_embedding_to_talker_dtype():
    model = _FakeQwenModel()

    assert _apply_mlx_audio_qwen_dtype_backport(model, "0.4.1") is True

    result = model.extract_speaker_embedding("reference audio", sr=16000)

    assert model.extract_calls == [("reference audio", 16000)]
    assert model.last_speaker_embedding.astype_calls == ["bfloat16"]
    assert result.dtype == "bfloat16"


def test_backport_does_not_touch_other_mlx_audio_versions():
    class _ModelThatMustNotBeInspected:
        @property
        def model_type(self):
            raise AssertionError("newer mlx-audio versions must not be patched")

    assert _apply_mlx_audio_qwen_dtype_backport(_ModelThatMustNotBeInspected(), "0.4.8") is False


def test_backport_skips_an_unexpected_model_interface(caplog):
    model = SimpleNamespace(model_type="qwen3_tts", talker=SimpleNamespace())

    assert _apply_mlx_audio_qwen_dtype_backport(model, "0.4.1") is False
    assert "unexpected model interface" in caplog.text


def test_backport_rejects_a_non_bf16_qwen_talker(caplog):
    model = _FakeQwenModel(talker_dtype="float32")

    assert _apply_mlx_audio_qwen_dtype_backport(model, "0.4.1") is False
    assert "expected a BF16 talker" in caplog.text


def test_backport_is_idempotent():
    model = _FakeQwenModel()

    assert _apply_mlx_audio_qwen_dtype_backport(model, "0.4.1") is True
    assert _apply_mlx_audio_qwen_dtype_backport(model, "0.4.1") is True

    result = model.extract_speaker_embedding("reference audio")

    assert model.talker.get_input_embeddings_calls == 1
    assert model.extract_calls == [("reference audio", 24000)]
    assert model.last_speaker_embedding.astype_calls == ["bfloat16"]
    assert result.dtype == "bfloat16"


def test_model_load_applies_backport_using_installed_distribution_version(monkeypatch):
    model = _FakeQwenModel()
    loaded_paths = []
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_tts = ModuleType("mlx_audio.tts")

    def fake_load(model_path):
        loaded_paths.append(model_path)
        return model

    fake_tts.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setattr(metadata, "version", lambda _distribution: "0.4.1")
    monkeypatch.setattr(mlx_backend, "model_load_progress", lambda *_args: nullcontext())

    backend = MLXTTSBackend()
    monkeypatch.setattr(backend, "_get_model_path", lambda _model_size: "fake/model")
    monkeypatch.setattr(backend, "_is_model_cached", lambda _model_size: True)

    backend._load_model_sync("1.7B")

    assert backend.model is model
    assert loaded_paths == ["fake/model"]
    assert backend._current_model_size == "1.7B"
    result = backend.model.extract_speaker_embedding("reference audio")
    assert result.dtype == "bfloat16"


def test_model_load_rejects_qwen_when_required_backport_cannot_apply(monkeypatch):
    model = SimpleNamespace(model_type="unexpected", talker=SimpleNamespace())
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_tts = ModuleType("mlx_audio.tts")
    fake_tts.load = lambda _model_path: model
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setattr(metadata, "version", lambda _distribution: "0.4.1")
    monkeypatch.setattr(mlx_backend, "model_load_progress", lambda *_args: nullcontext())

    backend = MLXTTSBackend()
    monkeypatch.setattr(
        backend,
        "_get_model_path",
        lambda _model_size: "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    )
    monkeypatch.setattr(backend, "_is_model_cached", lambda _model_size: True)

    with pytest.raises(RuntimeError, match="refusing to load an unverified TTS implementation"):
        backend._load_model_sync("1.7B")

    assert backend.model is None
    assert backend._current_model_size is None


def test_model_load_does_not_require_qwen_backport_for_other_models(monkeypatch):
    model = SimpleNamespace(model_type="whisper")
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_tts = ModuleType("mlx_audio.tts")
    fake_tts.load = lambda _model_path: model
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setattr(metadata, "version", lambda _distribution: "0.4.1")
    monkeypatch.setattr(mlx_backend, "model_load_progress", lambda *_args: nullcontext())

    backend = MLXTTSBackend()
    monkeypatch.setattr(backend, "_get_model_path", lambda _model_size: "fake/whisper")
    monkeypatch.setattr(backend, "_is_model_cached", lambda _model_size: True)

    backend._load_model_sync("1.7B")

    assert backend.model is model
    assert backend._current_model_size == "1.7B"


def test_clone_failure_never_falls_back_to_a_generic_voice(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")

    class _FailingCloneModel:
        def __init__(self):
            self.calls = []

        def generate(self, text, ref_audio=None, ref_text=None, lang_code=None):
            self.calls.append(
                {
                    "text": text,
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                    "lang_code": lang_code,
                }
            )
            raise ValueError("clone inference failed")
            yield  # pragma: no cover - makes this a generator like mlx-audio

    model = _FailingCloneModel()
    backend = MLXTTSBackend()
    backend.model = model
    backend._current_model_size = "1.7B"

    with pytest.raises(RuntimeError, match="refusing to generate with a generic voice"):
        asyncio.run(
            backend.generate(
                "Texto de prueba.",
                {"ref_audio": str(reference), "ref_text": "Referencia."},
                language="es",
            )
        )
    assert len(model.calls) == 1
    assert model.calls[0]["ref_audio"] == str(reference)


def test_clone_request_rejects_a_model_without_clone_arguments(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")

    class _GenericOnlyModel:
        def __init__(self):
            self.calls = 0

        def generate(self, text, lang_code=None):
            self.calls += 1
            raise AssertionError("generic generation must not run for a clone request")
            yield  # pragma: no cover - keeps the fake interface generator-shaped

    model = _GenericOnlyModel()
    backend = MLXTTSBackend()
    backend.model = model
    backend._current_model_size = "1.7B"

    with pytest.raises(RuntimeError, match="refusing to generate with a generic voice"):
        asyncio.run(
            backend.generate(
                "Texto de prueba.",
                {"ref_audio": str(reference), "ref_text": "Referencia."},
                language="es",
            )
        )
    assert model.calls == 0


def test_missing_clone_reference_fails_before_model_generation(tmp_path):
    class _ModelThatMustNotRun:
        def generate(self, *_args, **_kwargs):
            raise AssertionError("model must not run without the requested reference")

    backend = MLXTTSBackend()
    backend.model = _ModelThatMustNotRun()
    backend._current_model_size = "1.7B"

    with pytest.raises(FileNotFoundError, match="refusing to generate with a different voice"):
        asyncio.run(
            backend.generate(
                "Texto de prueba.",
                {"ref_audio": str(tmp_path / "missing.wav"), "ref_text": "Referencia."},
                language="es",
            )
        )
