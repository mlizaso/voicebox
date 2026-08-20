"""Regression tests for Qwen cloning correctness and the mlx-audio 0.4.1 backport."""

import asyncio
import hashlib
import logging
import sys
from contextlib import nullcontext
from importlib import metadata
from types import ModuleType, SimpleNamespace

import pytest

import backend.backends.mlx_backend as mlx_backend
import backend.backends.mlx_runtime as mlx_runtime
from backend.backends.base import is_model_cached
from backend.backends.mlx_backend import (
    MLXTTSBackend,
    _apply_mlx_audio_qwen_dtype_backport,
)
from backend.backends.mlx_runtime import (
    MLX_QWEN_TTS_IMPLEMENTATION_REVISION,
    MLX_QWEN_TTS_LOCAL_NUMERICAL_SOURCE_PATHS,
    MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS,
    MLX_QWEN_TTS_PINNED_MODELS,
    MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS,
    build_mlx_qwen_tts_implementation_revision,
    get_current_mlx_qwen_tts_source_fingerprints,
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
        self.config = SimpleNamespace(tts_model_type="base")
        self.speech_tokenizer = SimpleNamespace(has_encoder=True)
        self.speaker_encoder = SimpleNamespace()
        self.tokenizer = SimpleNamespace()
        self.extract_calls = []
        self.last_speaker_embedding = None

    def _prepare_icl_generation_inputs(self, text, ref_audio, ref_text, language="auto"):
        return text, ref_audio, ref_text, language

    def extract_speaker_embedding(self, audio, sr=24000):
        self.extract_calls.append((audio, sr))
        self.last_speaker_embedding = _FakeArray("float32")
        return self.last_speaker_embedding


def test_runtime_revision_requires_the_guarded_dependency_set(monkeypatch):
    monkeypatch.setattr(metadata, "version", MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS.__getitem__)

    assert get_mlx_qwen_tts_implementation_revision() == MLX_QWEN_TTS_IMPLEMENTATION_REVISION


@pytest.mark.parametrize("mismatched_package", MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS)
def test_runtime_revision_is_absent_when_any_dependency_version_mismatches(monkeypatch, mismatched_package):
    def installed_version(distribution):
        if distribution == mismatched_package:
            return "999.0"
        return MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS[distribution]

    monkeypatch.setattr(metadata, "version", installed_version)

    assert get_mlx_qwen_tts_implementation_revision() is None


@pytest.mark.parametrize("missing_package", MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS)
def test_runtime_revision_is_absent_when_any_dependency_is_missing(monkeypatch, missing_package):
    def installed_version(distribution):
        if distribution == missing_package:
            raise metadata.PackageNotFoundError(distribution)
        return MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS[distribution]

    monkeypatch.setattr(metadata, "version", installed_version)

    assert get_mlx_qwen_tts_implementation_revision() is None


def test_runtime_revision_canonically_covers_package_patch_and_pinned_weights():
    assert build_mlx_qwen_tts_implementation_revision() == MLX_QWEN_TTS_IMPLEMENTATION_REVISION
    assert len(MLX_QWEN_TTS_IMPLEMENTATION_REVISION) <= 128
    assert dict(MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS) == {
        "mlx-audio": "0.4.1",
        "mlx": "0.32.0",
        "mlx-lm": "0.31.1",
        "mlx-metal": "0.32.0",
        "numpy": "1.26.4",
        "transformers": "4.57.3",
        "tokenizers": "0.22.2",
        "miniaudio": "1.71",
        "librosa": "0.11.0",
        "soundfile": "0.14.0",
        "soxr": "1.1.0",
        "pedalboard": "0.9.24",
    }
    assert set(MLX_QWEN_TTS_LOCAL_NUMERICAL_SOURCE_PATHS) == {
        "backend/models.py",
        "backend/routes/generations.py",
        "backend/routes/llm.py",
        "backend/routes/models.py",
        "backend/routes/tasks.py",
        "backend/routes/transcription.py",
        "backend/backends/__init__.py",
        "backend/backends/base.py",
        "backend/backends/mlx_backend.py",
        "backend/backends/mlx_qwen_optimizations.py",
        "backend/backends/mlx_tts_lifecycle.py",
        "backend/backends/qwen_llm_backend.py",
        "backend/services/exact_chunk_checkpoints.py",
        "backend/services/effects_processing.py",
        "backend/services/generation.py",
        "backend/services/profiles.py",
        "backend/services/task_queue.py",
        "backend/services/tts.py",
        "backend/utils/audio.py",
        "backend/utils/cache.py",
        "backend/utils/chunked_tts.py",
        "backend/utils/disk_reservations.py",
        "backend/utils/effects.py",
    }
    assert get_current_mlx_qwen_tts_source_fingerprints() == MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS

    reversed_packages = dict(reversed(MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS.items()))
    reversed_specs = dict(reversed(MLX_QWEN_TTS_PINNED_MODELS.items()))
    reversed_sources = dict(reversed(MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS.items()))
    assert (
        build_mlx_qwen_tts_implementation_revision(
            runtime_packages=reversed_packages,
            model_revisions=reversed_specs,
            source_fingerprints=reversed_sources,
        )
        == MLX_QWEN_TTS_IMPLEMENTATION_REVISION
    )

    for package, version in MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS.items():
        changed_packages = dict(MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS)
        changed_packages[package] = f"{version}.changed"
        assert (
            build_mlx_qwen_tts_implementation_revision(runtime_packages=changed_packages)
            != MLX_QWEN_TTS_IMPLEMENTATION_REVISION
        )

    changed_specs = dict(MLX_QWEN_TTS_PINNED_MODELS)
    repo, _revision = changed_specs["1.7B"]
    changed_specs["1.7B"] = (repo, "0" * 40)
    assert (
        build_mlx_qwen_tts_implementation_revision(model_revisions=changed_specs)
        != MLX_QWEN_TTS_IMPLEMENTATION_REVISION
    )
    assert (
        build_mlx_qwen_tts_implementation_revision(patch_revision="bf16-speaker-v2")
        != MLX_QWEN_TTS_IMPLEMENTATION_REVISION
    )
    changed_sources = dict(MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS)
    changed_sources["voicebox-mlx"] = "0" * 64
    assert (
        build_mlx_qwen_tts_implementation_revision(source_fingerprints=changed_sources)
        != MLX_QWEN_TTS_IMPLEMENTATION_REVISION
    )


def test_runtime_revision_is_absent_when_numerical_source_is_edited(monkeypatch):
    monkeypatch.setattr(
        metadata,
        "version",
        MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS.__getitem__,
    )
    changed_sources = dict(MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS)
    changed_sources["mlx-audio-qwen3-tts"] = "0" * 64
    monkeypatch.setattr(
        mlx_runtime,
        "get_current_mlx_qwen_tts_source_fingerprints",
        lambda: changed_sources,
    )

    assert get_mlx_qwen_tts_implementation_revision() is None


def test_frozen_runtime_uses_build_verified_embedded_source_fingerprints(
    monkeypatch,
):
    get_current_mlx_qwen_tts_source_fingerprints.cache_clear()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    try:
        assert get_current_mlx_qwen_tts_source_fingerprints() == MLX_QWEN_TTS_NUMERICAL_SOURCE_FINGERPRINTS
    finally:
        get_current_mlx_qwen_tts_source_fingerprints.cache_clear()


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
    loaded_models = []
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_tts = ModuleType("mlx_audio.tts")

    def fake_load(model_path, **kwargs):
        loaded_models.append((model_path, kwargs))
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
    assert loaded_models == [
        (
            "fake/model",
            {"revision": "a6eb4f68e4b056f1215157bb696209bc82a6db48"},
        )
    ]
    assert backend._current_model_size == "1.7B"
    result = backend.model.extract_speaker_embedding("reference audio")
    assert result.dtype == "bfloat16"


@pytest.mark.parametrize(
    ("model_size", "expected_repo", "expected_revision"),
    [
        (
            "1.7B",
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "a6eb4f68e4b056f1215157bb696209bc82a6db48",
        ),
        (
            "0.6B",
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
        ),
    ],
)
def test_model_load_uses_pinned_repository_and_revision(
    monkeypatch,
    model_size,
    expected_repo,
    expected_revision,
):
    model = _FakeQwenModel()
    loaded_models = []
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_tts = ModuleType("mlx_audio.tts")

    def fake_load(model_path, **kwargs):
        loaded_models.append((model_path, kwargs))
        return model

    fake_tts.load = fake_load
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setattr(metadata, "version", lambda _distribution: "0.4.1")
    monkeypatch.setattr(mlx_backend, "model_load_progress", lambda *_args: nullcontext())

    backend = MLXTTSBackend()
    monkeypatch.setattr(backend, "_is_model_cached", lambda _model_size: True)

    backend._load_model_sync(model_size)

    assert loaded_models == [(expected_repo, {"revision": expected_revision})]


def test_model_load_rejects_an_unpinned_qwen_size():
    backend = MLXTTSBackend()

    with pytest.raises(ValueError, match="Unsupported pinned MLX Qwen TTS model size"):
        backend._load_model_sync("4B")

    assert backend.model is None


def test_model_cache_probe_is_scoped_to_the_pinned_snapshot(monkeypatch):
    calls = []

    def fake_is_model_cached(repo, **kwargs):
        calls.append((repo, kwargs))
        return True

    monkeypatch.setattr(mlx_backend, "is_model_cached", fake_is_model_cached)

    assert MLXTTSBackend()._is_model_cached("0.6B") is True
    assert calls == [
        (
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            {
                "weight_extensions": (".safetensors", ".bin", ".npz"),
                "revision": "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
            },
        )
    ]


def test_exact_model_cache_check_ignores_weights_from_another_snapshot(tmp_path, monkeypatch):
    from huggingface_hub import constants as hf_constants

    repo = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    pinned_revision = "1eccf1cb2519b5a4e8a95b5f0544f3303568164f"
    snapshots = tmp_path / "models--mlx-community--Qwen3-TTS-12Hz-0.6B-Base-bf16" / "snapshots"
    other_snapshot = snapshots / ("f" * 40)
    other_snapshot.mkdir(parents=True)
    (other_snapshot / "model.safetensors").write_bytes(b"other weights")
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))

    assert is_model_cached(repo, revision=pinned_revision) is False

    pinned_snapshot = snapshots / pinned_revision
    pinned_snapshot.mkdir()
    (pinned_snapshot / "model.safetensors").write_bytes(b"pinned weights")

    assert is_model_cached(repo, revision=pinned_revision) is True


def test_model_load_rejects_qwen_when_required_backport_cannot_apply(monkeypatch):
    model = SimpleNamespace(model_type="unexpected", talker=SimpleNamespace())
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_tts = ModuleType("mlx_audio.tts")
    fake_tts.load = lambda _model_path, **_kwargs: model
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
    fake_tts.load = lambda _model_path, **_kwargs: model
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


def test_clone_failure_never_falls_back_to_a_generic_voice(tmp_path, caplog):
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

    private_text = "Texto privado del libro: el vaquero cabalga al amanecer."
    with (
        caplog.at_level(logging.INFO, logger=mlx_backend.__name__),
        pytest.raises(
            RuntimeError,
            match="refusing to generate with a generic voice",
        ),
    ):
        asyncio.run(
            backend.generate(
                private_text,
                {"ref_audio": str(reference), "ref_text": "Referencia."},
                language="es",
            )
        )

    assert private_text not in caplog.text
    assert f"chars={len(private_text)}" in caplog.text
    assert f"text_sha256={hashlib.sha256(private_text.encode('utf-8')).hexdigest()[:12]}" in caplog.text
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
