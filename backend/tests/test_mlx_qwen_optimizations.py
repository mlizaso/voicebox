"""Focused tests for the pinned Qwen ICL cache and batch contract."""

import ast
import asyncio
import sys
import threading
from importlib import metadata
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import backend.backends.mlx_backend as mlx_backend
import backend.backends.mlx_qwen_optimizations as optimizations
from backend.backends.mlx_backend import MLXTTSBackend
from backend.backends.mlx_qwen_optimizations import (
    QwenReferenceConditioning,
    apply_qwen_icl_cache_backport,
    build_reference_cache_key,
    clear_qwen_reference_cache,
    prepare_reference_conditioning,
)


class _PatchableModel:
    model_type = "qwen3_tts"

    def __init__(self):
        self.config = SimpleNamespace(tts_model_type="base")
        self.talker = SimpleNamespace()
        self.speech_tokenizer = SimpleNamespace(has_encoder=True)
        self.speaker_encoder = SimpleNamespace()
        self.tokenizer = SimpleNamespace()
        self.original_calls = []

    def _prepare_icl_generation_inputs(self, text, ref_audio, ref_text, language="auto"):
        self.original_calls.append((text, ref_audio, ref_text, language))
        return "original", text, ref_audio, ref_text


def test_reference_cache_key_covers_audio_bytes_and_transcript(tmp_path):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice-a")
    original = build_reference_cache_key(str(audio), "transcript-a")

    assert original == build_reference_cache_key(str(audio), "transcript-a")
    assert original != build_reference_cache_key(str(audio), "transcript-b")

    audio.write_bytes(b"voice-b")
    assert original != build_reference_cache_key(str(audio), "transcript-a")


def test_backend_rehashes_persisted_prompt_when_reference_file_changes(tmp_path, monkeypatch):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice-a")
    backend = MLXTTSBackend()
    backend.model = object()
    observed_keys = []

    def fake_prepare(_model, **kwargs):
        assert "cache_key" not in kwargs
        observed_keys.append(build_reference_cache_key(kwargs["audio_path"], kwargs["reference_text"]))
        return object()

    monkeypatch.setattr(mlx_backend, "prepare_reference_conditioning", fake_prepare)
    prompt = {
        "ref_audio": str(audio),
        "ref_text": "reference",
        "mlx_conditioning_key": "stale-persisted-key",
    }

    backend._prepare_reference_sync(prompt)
    audio.write_bytes(b"voice-b")
    backend._prepare_reference_sync(prompt)

    assert observed_keys[0] != observed_keys[1]
    assert "stale-persisted-key" not in observed_keys


def test_prompt_cache_hit_rebinds_identical_content_to_callers_path(tmp_path, monkeypatch):
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    first_path.write_bytes(b"same-voice")
    second_path.write_bytes(b"same-voice")
    cached_prompt = {
        "ref_audio": str(first_path),
        "ref_text": "reference",
        "mlx_conditioning_key": "old-key",
    }
    backend = MLXTTSBackend()
    backend.model = object()
    backend._current_model_size = "1.7B"
    monkeypatch.setattr(mlx_backend, "get_cached_voice_prompt", lambda _key: cached_prompt)

    prompt, was_cached = asyncio.run(backend.create_voice_prompt(str(second_path), "reference"))
    first_path.write_bytes(b"different-voice")

    assert was_cached is True
    assert prompt["ref_audio"] == str(second_path)
    assert prompt["mlx_conditioning_key"] == build_reference_cache_key(
        str(second_path),
        "reference",
    )
    assert cached_prompt["ref_audio"] == str(first_path)


def test_cache_backport_is_exact_version_guarded_and_idempotent():
    model = _PatchableModel()

    assert apply_qwen_icl_cache_backport(model, "0.4.0") is False
    assert apply_qwen_icl_cache_backport(model, "0.4.1") is True
    patched_method = model._prepare_icl_generation_inputs
    assert apply_qwen_icl_cache_backport(model, "0.4.1") is True
    assert model._prepare_icl_generation_inputs == patched_method

    audio = object()
    assert model._prepare_icl_generation_inputs("target", audio, "reference", "spanish") == (
        "original",
        "target",
        audio,
        "reference",
    )
    assert model.original_calls == [("target", audio, "reference", "spanish")]


def test_cache_backport_routes_registered_audio_to_cached_builder(monkeypatch):
    model = _PatchableModel()
    assert apply_qwen_icl_cache_backport(model, "0.4.1") is True
    audio = object()
    conditioning = QwenReferenceConditioning(
        cache_key="key",
        ref_text="reference",
        audio=audio,
        ref_codes=object(),
        ref_text_ids=object(),
        speaker_embedding=object(),
    )
    model._voicebox_qwen_reference_conditioning["key"] = conditioning
    calls = []

    def fake_cached_builder(bound_model, text, cached_conditioning, language):
        calls.append((bound_model, text, cached_conditioning, language))
        return "cached"

    monkeypatch.setattr(optimizations, "_prepare_icl_from_conditioning", fake_cached_builder)

    assert model._prepare_icl_generation_inputs("target", audio, "reference", "spanish") == "cached"
    assert calls == [(model, "target", conditioning, "spanish")]
    assert model.original_calls == []


def _fake_mlx_modules(monkeypatch):
    fake_mlx = ModuleType("mlx")
    fake_core = ModuleType("mlx.core")
    fake_core.array = np.array
    fake_core.eval = lambda *_arrays: None
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    return fake_core


def test_conditioning_cache_runs_expensive_encoders_once_per_reference(tmp_path, monkeypatch):
    fake_core = _fake_mlx_modules(monkeypatch)
    counters = {"load": 0, "speech": 0, "text": 0, "speaker": 0}
    fake_mlx_audio = ModuleType("mlx_audio")
    fake_mlx_audio.__path__ = []
    fake_utils = ModuleType("mlx_audio.utils")

    def load_audio(_path, sample_rate):
        counters["load"] += 1
        assert sample_rate == 24000
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)

    fake_utils.load_audio = load_audio
    monkeypatch.setitem(sys.modules, "mlx_audio", fake_mlx_audio)
    monkeypatch.setitem(sys.modules, "mlx_audio.utils", fake_utils)

    class SpeechTokenizer:
        has_encoder = True

        def encode(self, audio):
            counters["speech"] += 1
            assert audio.shape == (1, 1, 3)
            return np.array([[[1, 2], [3, 4]]])

    class Tokenizer:
        def encode(self, _text):
            counters["text"] += 1
            return [1, 2, 3, 10, 11, 98, 99]

    model = _PatchableModel()
    model.sample_rate = 24000
    model.speech_tokenizer = SpeechTokenizer()
    model.tokenizer = Tokenizer()

    def extract_speaker_embedding(_audio):
        counters["speaker"] += 1
        return np.array([[0.5, 0.6]], dtype=np.float32)

    model.extract_speaker_embedding = extract_speaker_embedding
    assert apply_qwen_icl_cache_backport(model, "0.4.1") is True
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice-a")
    first = prepare_reference_conditioning(
        model,
        audio_path=str(audio_path),
        reference_text="reference",
    )
    second = prepare_reference_conditioning(
        model,
        audio_path=str(audio_path),
        reference_text="reference",
    )

    assert first is second
    assert counters == {"load": 1, "speech": 1, "text": 1, "speaker": 1}
    assert fake_core is sys.modules["mlx.core"]

    audio_path.write_bytes(b"voice-b")
    prepare_reference_conditioning(
        model,
        audio_path=str(audio_path),
        reference_text="reference",
    )
    prepare_reference_conditioning(
        model,
        audio_path=str(audio_path),
        reference_text="different",
    )
    assert counters == {"load": 3, "speech": 3, "text": 3, "speaker": 3}


def test_conditioning_hash_and_decoder_use_one_immutable_byte_snapshot(tmp_path, monkeypatch):
    _fake_mlx_modules(monkeypatch)
    audio_path = tmp_path / "voice.wav"
    audio_path.write_bytes(b"voice-a")
    decoded_snapshots = []

    def load_snapshot(audio_bytes, _suffix, _sample_rate):
        decoded_snapshots.append(audio_bytes)
        # Replace the live path during decoding.  The cache key and decoded
        # conditioning must still both describe voice-a.
        audio_path.write_bytes(b"voice-b")
        return np.array([0.1, 0.2], dtype=np.float32)

    monkeypatch.setattr(optimizations, "_load_reference_snapshot", load_snapshot)
    model = _PatchableModel()
    model.sample_rate = 24000
    model.speech_tokenizer = SimpleNamespace(
        has_encoder=True,
        encode=lambda _audio: np.array([[[1, 2], [3, 4]]]),
    )
    model.tokenizer = SimpleNamespace(encode=lambda _text: [1, 2, 3, 10, 11, 98, 99])
    model.extract_speaker_embedding = lambda _audio: np.array([[0.5]], dtype=np.float32)
    assert apply_qwen_icl_cache_backport(model, "0.4.1") is True

    conditioning = prepare_reference_conditioning(
        model,
        audio_path=str(audio_path),
        reference_text="reference",
    )

    expected_path = tmp_path / "expected.wav"
    expected_path.write_bytes(b"voice-a")
    assert decoded_snapshots == [b"voice-a"]
    assert conditioning.cache_key == build_reference_cache_key(
        str(expected_path),
        "reference",
    )


def test_reference_cache_is_bounded_and_cleared_on_model_unload():
    model = _PatchableModel()
    assert apply_qwen_icl_cache_backport(model, "0.4.1") is True
    for index in range(optimizations.MAX_QWEN_REFERENCE_CACHE_ENTRIES + 2):
        optimizations._register_reference_conditioning(
            model,
            QwenReferenceConditioning(
                cache_key=f"key-{index}",
                ref_text=f"reference-{index}",
                audio=object(),
                ref_codes=object(),
                ref_text_ids=object(),
                speaker_embedding=object(),
            ),
        )

    registry = model._voicebox_qwen_reference_conditioning
    assert list(registry) == ["key-2", "key-3", "key-4", "key-5"]

    backend = MLXTTSBackend()
    backend.model = model
    backend._current_model_size = "1.7B"
    backend.unload_model()

    assert registry == {}
    assert backend.model is None
    assert clear_qwen_reference_cache(model) == 0


class _FakeMX:
    int32 = np.int32

    @staticmethod
    def array(value):
        return np.array(value)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def broadcast_to(value, shape):
        return np.broadcast_to(value, shape)

    @staticmethod
    def eval(*_values):
        return None


def _load_pinned_original_icl_prepare():
    source_path = metadata.distribution("mlx-audio").locate_file("mlx_audio/tts/models/qwen3_tts/qwen3_tts.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "_prepare_icl_generation_inputs"
    )
    method.returns = None
    for argument in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs):
        argument.annotation = None
    namespace = {"mx": _FakeMX}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source_path), "exec"),
        namespace,
    )
    return namespace[method.name]


@pytest.mark.parametrize("language", ["auto", "spanish"])
def test_cached_preparation_matches_pinned_private_icl_algorithm(monkeypatch, language):
    fake_core = _fake_mlx_modules(monkeypatch)
    fake_core.concatenate = _FakeMX.concatenate
    fake_core.broadcast_to = _FakeMX.broadcast_to

    class Tokenizer:
        @staticmethod
        def encode(text):
            if text == "<|im_start|>assistant\nReference<|im_end|>\n":
                return [1, 2, 3, 11, 12, 98, 99]
            return [1, 2, 3, 21, 22, 23, 91, 92, 93, 94, 95]

    hidden_size = 3

    def embedding(ids):
        values = np.asarray(ids, dtype=np.float32)
        return np.repeat(values[..., None], hidden_size, axis=-1)

    code_embeddings = [lambda ids: embedding(ids) * 0.1]
    config = SimpleNamespace(
        num_code_groups=2,
        codec_bos_id=31,
        codec_pad_id=32,
        codec_nothink_id=33,
        codec_think_bos_id=34,
        codec_think_eos_id=35,
        codec_think_id=36,
        codec_language_id={"spanish": 37},
    )
    ref_codes = np.array([[[4, 5], [6, 7]]])
    speaker_embedding = np.array([[0.25, 0.5, 0.75]], dtype=np.float32)
    model = SimpleNamespace(
        config=SimpleNamespace(
            talker_config=config,
            tts_bos_token_id=41,
            tts_eos_token_id=42,
            tts_pad_token_id=43,
        ),
        tokenizer=Tokenizer(),
        speaker_encoder=object(),
        speech_tokenizer=SimpleNamespace(encode=lambda _audio: ref_codes),
        extract_speaker_embedding=lambda _audio: speaker_embedding,
        talker=SimpleNamespace(
            text_projection=lambda values: values * 0.5,
            get_text_embeddings=lambda: embedding,
            get_input_embeddings=lambda: embedding,
            code_predictor=SimpleNamespace(codec_embedding=code_embeddings),
        ),
    )
    audio = np.array([0.1, 0.2], dtype=np.float32)
    original = _load_pinned_original_icl_prepare()
    expected = original(model, "Target", audio, "Reference", language)
    conditioning = QwenReferenceConditioning(
        cache_key="key",
        ref_text="Reference",
        audio=audio,
        ref_codes=ref_codes,
        ref_text_ids=np.array([[11, 12]]),
        speaker_embedding=speaker_embedding,
    )

    actual = optimizations._prepare_icl_from_conditioning(model, "Target", conditioning, language)

    assert len(actual) == len(expected) == 4
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(actual_value, expected_value)


def test_uneven_batch_padding_and_attention_mask_preserve_each_row(monkeypatch):
    fake_core = _fake_mlx_modules(monkeypatch)
    fake_core.int32 = np.int32
    fake_core.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    fake_core.ones = lambda shape, dtype=np.float32: np.ones(shape, dtype=dtype)
    fake_core.concatenate = np.concatenate
    fake_core.broadcast_to = np.broadcast_to
    short_embeds = np.arange(6, dtype=np.float32).reshape(1, 2, 3) + 10
    long_embeds = np.arange(15, dtype=np.float32).reshape(1, 5, 3) + 100
    short_trailing = np.full((1, 1, 3), 7, dtype=np.float32)
    long_trailing = np.arange(12, dtype=np.float32).reshape(1, 4, 3) + 200
    pad_embed = np.full((1, 1, 3), -9, dtype=np.float32)
    prepared = {
        "short": (short_embeds, short_trailing, pad_embed, object()),
        "long": (long_embeds, long_trailing, pad_embed, object()),
    }
    model = SimpleNamespace(
        _prepare_icl_generation_inputs=lambda text, **_kwargs: prepared[text],
    )
    conditioning = SimpleNamespace(audio=object(), ref_text="reference")

    embeds, trailing, actual_pad, mask = optimizations._prepare_batch_inputs(
        model,
        ["short", "long"],
        conditioning,
        "spanish",
    )

    assert embeds.shape == (2, 5, 3)
    np.testing.assert_array_equal(embeds[0, :3], np.zeros((3, 3)))
    np.testing.assert_array_equal(embeds[0, 3:], short_embeds[0])
    np.testing.assert_array_equal(embeds[1], long_embeds[0])
    np.testing.assert_array_equal(mask, [[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])
    assert mask.dtype == np.int32
    np.testing.assert_array_equal(trailing[0, 0], short_trailing[0, 0])
    np.testing.assert_array_equal(trailing[0, 1:], np.broadcast_to(pad_embed, (1, 3, 3))[0])
    np.testing.assert_array_equal(trailing[1], long_trailing[0])
    np.testing.assert_array_equal(actual_pad, pad_embed)


def test_mlx_combined_reference_resamples_mixed_rates_to_qwen_24khz(monkeypatch):
    import backend.backends.base as base_backend

    calls = []

    def fake_load_audio(path, sample_rate=None):
        calls.append((path, sample_rate))
        if sample_rate == 24000:
            return np.ones(24000, dtype=np.float32), 24000
        source_rate = 16000 if path == "16k.wav" else 48000
        return np.ones(source_rate, dtype=np.float32), source_rate

    monkeypatch.setattr(base_backend, "load_audio", fake_load_audio)
    backend = MLXTTSBackend()

    combined, text = asyncio.run(
        backend.combine_voice_prompts(
            ["16k.wav", "48k.wav"],
            ["first", "second"],
        )
    )

    assert calls == [("16k.wav", 24000), ("48k.wav", 24000)]
    assert combined.shape == (48000,)
    assert text == "first second"


def test_exact_mlx_runtime_overrides_are_installed_after_base_requirements():
    backend_root = Path(__file__).resolve().parents[1]
    repository_root = backend_root.parent
    mlx_requirements = (backend_root / "requirements-mlx.txt").read_text(encoding="utf-8")
    exact_overrides = {
        "mlx": "0.32.0",
        "numpy": "1.26.4",
        "librosa": "0.11.0",
        "soundfile": "0.14.0",
        "soxr": "1.1.0",
        "pedalboard": "0.9.24",
        "transformers": "4.57.3",
        "tokenizers": "0.22.2",
        "miniaudio": "1.71",
    }
    requirement_lines = {
        line.strip() for line in mlx_requirements.splitlines() if line.strip() and not line.lstrip().startswith("#")
    }
    assert {f"{name}=={version}" for name, version in exact_overrides.items()} <= requirement_lines

    justfile = (repository_root / "justfile").read_text(encoding="utf-8")
    assert justfile.index("QwenLM/Qwen3-TTS.git") < justfile.index("requirements-mlx.txt")

    for setup_path in (
        repository_root / "justfile",
        repository_root / ".github/workflows/release.yml",
    ):
        setup = setup_path.read_text(encoding="utf-8")
        assert setup.index("requirements.txt") < setup.index("requirements-mlx.txt")
        assert setup.index("requirements-mlx.txt") < setup.index("mlx-audio==0.4.1")


def test_cache_backport_rejects_drifted_private_interface(caplog):
    model = SimpleNamespace(
        model_type="qwen3_tts",
        config=object(),
        talker=object(),
        speech_tokenizer=object(),
        tokenizer=object(),
        _prepare_icl_generation_inputs=lambda text: text,
    )

    assert apply_qwen_icl_cache_backport(model, "0.4.1") is False
    assert "unexpected preparation interface" in caplog.text


def test_per_row_key_is_stable_and_seed_specific():
    assert optimizations._row_random_key(100, 0) == optimizations._row_random_key(100, 0)
    assert optimizations._row_random_key(100, 0) != optimizations._row_random_key(101, 0)
    assert optimizations._row_random_key(100, 0) != optimizations._row_random_key(100, 1)


def test_per_row_keys_do_not_repeat_across_a_maximum_length_stream():
    keys = [optimizations._row_random_key(20260812, counter) for counter in range(65536)]

    assert len(keys) == len(set(keys))
    assert optimizations._row_random_key(1, 451) != optimizations._row_random_key(1, 23849)
    assert all(len(key) == 2 and all(0 <= word < 2**32 for word in key) for key in keys)


def test_long_generation_periodically_clears_graph_cache():
    fake_mx = SimpleNamespace(clear_cache_calls=0)

    def clear_cache():
        fake_mx.clear_cache_calls += 1

    fake_mx.clear_cache = clear_cache
    for step in range(151):
        optimizations._clear_generation_cache_if_due(fake_mx, step)

    assert fake_mx.clear_cache_calls == 3


def test_batch_sampler_uses_independent_explicit_row_keys(monkeypatch):
    fake_mlx = ModuleType("mlx")
    fake_core = ModuleType("mlx.core")
    observed_keys = []

    class FakeRandom:
        @staticmethod
        def categorical(logits, *, key):
            key_words = tuple(int(word) for word in key)
            observed_keys.append(key_words)
            return np.array([(key_words[0] ^ key_words[1]) % logits.shape[-1]], dtype=np.int32)

    fake_core.int32 = np.int32
    fake_core.uint32 = np.uint32
    fake_core.random = FakeRandom()
    fake_core.array = np.array
    fake_core.concatenate = lambda values, axis=0: np.concatenate(values, axis=axis)
    fake_core.argmax = lambda values, axis, keepdims: np.argmax(values, axis=axis, keepdims=keepdims)
    fake_mlx.core = fake_core
    fake_mlx_lm = ModuleType("mlx_lm")
    fake_sample_utils = ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.apply_min_p = lambda values, _threshold: values
    fake_sample_utils.apply_top_k = lambda values, _count: values
    fake_sample_utils.apply_top_p = lambda values, _threshold: values
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)
    logits = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)

    first_counters = [0, 0]
    first = optimizations._sample_rows(
        object(),
        logits,
        seeds=[100, 101],
        counters=first_counters,
        temperature=0.9,
        top_k=0,
        top_p=1.0,
    )
    first_keys = observed_keys[:]
    observed_keys.clear()
    second_counters = [0, 0]
    second = optimizations._sample_rows(
        object(),
        logits,
        seeds=[100, 999],
        counters=second_counters,
        temperature=0.9,
        top_k=0,
        top_p=1.0,
    )

    assert first_keys[0] == observed_keys[0] == optimizations._row_random_key(100, 0)
    assert first_keys[1] != first_keys[0]
    assert observed_keys[1] != first_keys[1]
    assert first[0].tolist() == second[0].tolist()
    assert first_counters == second_counters == [1, 1]


def test_generate_batch_preserves_distinct_seeds_and_input_order(monkeypatch):
    backend = MLXTTSBackend()
    backend.model = SimpleNamespace()
    backend._current_model_size = "1.7B"
    conditioning = object()
    calls = []
    monkeypatch.setattr(backend, "_prepare_reference_sync", lambda _prompt: conditioning)

    def fake_generate(model, texts, cached_conditioning, **kwargs):
        calls.append((model, list(texts), cached_conditioning, kwargs))
        return [
            (1, np.array([0.2], dtype=np.float32), 24000),
            (0, np.array([0.1], dtype=np.float32), 24000),
        ]

    monkeypatch.setattr(mlx_backend, "generate_qwen_icl_batch", fake_generate)

    results = asyncio.run(
        backend.generate_batch(
            ["first", "second"],
            {"ref_audio": "voice.wav", "ref_text": "reference"},
            language="es",
            seeds=[100, 101],
        )
    )

    assert [result[0].tolist() for result in results] == [[pytest.approx(0.1)], [pytest.approx(0.2)]]
    assert [result[1] for result in results] == [24000, 24000]
    assert calls[0][1:] == (
        ["first", "second"],
        conditioning,
        {"language": "spanish", "seeds": [100, 101]},
    )


def test_cancelled_mlx_inference_finishes_before_serial_queue_starts_next_job(
    monkeypatch,
):
    first_started = threading.Event()
    release_first = threading.Event()
    cancellation_sent = asyncio.Event()
    second_started = threading.Event()
    backend = MLXTTSBackend()
    backend.model = SimpleNamespace()
    backend._current_model_size = "1.7B"
    monkeypatch.setattr(backend, "_prepare_reference_sync", lambda _prompt: object())

    def fake_generate(_model, texts, _conditioning, **_kwargs):
        if texts[0].startswith("first"):
            first_started.set()
            assert release_first.wait(timeout=2)
        else:
            second_started.set()
        return [
            (0, np.array([0.1], dtype=np.float32), 24000),
            (1, np.array([0.2], dtype=np.float32), 24000),
        ]

    monkeypatch.setattr(mlx_backend, "generate_qwen_icl_batch", fake_generate)

    async def wait_for_thread_event(event: threading.Event) -> None:
        while not event.is_set():
            await asyncio.sleep(0)

    async def serial_worker() -> None:
        first_task = asyncio.create_task(
            backend.generate_batch(
                ["first-a", "first-b"],
                {"ref_audio": "voice.wav", "ref_text": "reference"},
                seeds=[100, 101],
            )
        )
        await wait_for_thread_event(first_started)
        first_task.cancel()
        cancellation_sent.set()
        with pytest.raises(asyncio.CancelledError):
            await first_task
        await backend.generate_batch(
            ["second-a", "second-b"],
            {"ref_audio": "voice.wav", "ref_text": "reference"},
            seeds=[102, 103],
        )

    async def scenario() -> None:
        worker = asyncio.create_task(serial_worker())
        await cancellation_sent.wait()
        try:
            for _ in range(10):
                await asyncio.sleep(0)
            assert not worker.done()
            assert not second_started.is_set()
        finally:
            release_first.set()
        await asyncio.wait_for(worker, timeout=2)
        assert second_started.is_set()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "indexed_results",
    [
        [(0, np.array([0.1]), 24000)],
        [(0, np.array([0.1]), 24000), (0, np.array([0.2]), 24000)],
        [(0, np.array([0.1]), 24000), (2, np.array([0.2]), 24000)],
    ],
)
def test_generate_batch_fails_closed_on_invalid_model_results(monkeypatch, indexed_results):
    backend = MLXTTSBackend()
    backend.model = SimpleNamespace()
    backend._current_model_size = "1.7B"
    monkeypatch.setattr(backend, "_prepare_reference_sync", lambda _prompt: object())
    monkeypatch.setattr(mlx_backend, "generate_qwen_icl_batch", lambda *_args, **_kwargs: indexed_results)

    with pytest.raises(RuntimeError, match="Batched voice cloning failed"):
        asyncio.run(
            backend.generate_batch(
                ["first", "second"],
                {"ref_audio": "voice.wav", "ref_text": "reference"},
                seeds=[100, 101],
            )
        )


def test_generate_batch_does_not_hide_inference_failure_as_serial_fallback(monkeypatch):
    backend = MLXTTSBackend()
    backend.model = SimpleNamespace()
    backend._current_model_size = "1.7B"
    monkeypatch.setattr(backend, "_prepare_reference_sync", lambda _prompt: object())

    def fail(*_args, **_kwargs):
        raise MemoryError("Metal allocation failed")

    monkeypatch.setattr(mlx_backend, "generate_qwen_icl_batch", fail)

    with pytest.raises(RuntimeError, match="Batched voice cloning failed") as error:
        asyncio.run(
            backend.generate_batch(
                ["first", "second"],
                {"ref_audio": "voice.wav", "ref_text": "reference"},
                seeds=[100, 101],
            )
        )
    assert isinstance(error.value.__cause__, MemoryError)


def test_generate_batch_does_not_treat_internal_not_implemented_as_capability_fallback(
    monkeypatch,
):
    backend = MLXTTSBackend()

    async def unsupported_model_load(_model_size):
        raise NotImplementedError("unexpected pinned model interface")

    # Batch generation holds the lifecycle guard across both model ensure and
    # inference, so exercise the guarded internal load boundary directly.
    monkeypatch.setattr(backend, "_ensure_model_loaded_locked", unsupported_model_load)

    with pytest.raises(RuntimeError, match="refusing serial or generic fallback") as error:
        asyncio.run(
            backend.generate_batch(
                ["first", "second"],
                {"ref_audio": "voice.wav", "ref_text": "reference"},
                seeds=[100, 101],
            )
        )
    assert isinstance(error.value.__cause__, NotImplementedError)


def test_generate_batch_rejects_unsupported_instruction_without_loading_model():
    backend = MLXTTSBackend()

    with pytest.raises(NotImplementedError, match="does not support instructions"):
        asyncio.run(
            backend.generate_batch(
                ["first", "second"],
                {"ref_audio": "voice.wav", "ref_text": "reference"},
                seeds=[100, 101],
                instruct="whisper",
            )
        )
