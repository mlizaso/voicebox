"""The deployed MLX prefix must match the PyTorch training prefix numerically."""

import numpy as np
import torch

from scripts.finetune_qwen.test_torch_model import _tiny_model
from scripts.finetune_qwen.torch_model import training_inputs


def test_mlx_non_streaming_prefix_matches_training_weights():
    import mlx.core as mx
    from mlx_audio.tts.models.qwen3_tts.config import ModelConfig
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model

    from backend.backends.qwen_finetuned_backend import apply_non_streaming_prefix

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            role = [1, 2, 3]
            words = [20, 21, 22, 23]
            if text == "<|im_start|>assistant\n":
                return role
            if text.startswith("<|im_start|>assistant\n"):
                return role + words + [4, 3, 1, 2, 3]
            return words

    torch.manual_seed(2)
    source = _tiny_model()
    target = Model(ModelConfig.from_dict(source.config.to_dict()))
    target.load_weights(
        [(name, mx.array(value.detach().numpy())) for name, value in source.state_dict().items()], strict=True
    )
    target.tokenizer = Tokenizer()
    apply_non_streaming_prefix(target)
    actual, trailing, pad = target._prepare_generation_inputs(
        text="Texto de prueba", language="Spanish", speaker="fabian"
    )
    mx.eval(actual, trailing, pad)
    speaker = source.talker.get_input_embeddings()(torch.tensor([3000]))
    expected, _labels, prefix = training_inputs(
        source, Tokenizer(), "Texto de prueba", torch.ones((3, 16), dtype=torch.long), speaker
    )
    np.testing.assert_allclose(np.asarray(actual), expected[0, :prefix].detach().numpy()[None], rtol=3e-5, atol=3e-6)
    np.testing.assert_array_equal(np.asarray(trailing), np.asarray(pad))


def test_local_model_survives_load_and_inference_on_different_workers(monkeypatch):
    """Keep a real lazy MLX buffer alive across executor/loop lifetimes."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_audio.tts import utils
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model

    from backend.backends import qwen_finetuned_backend

    class TinyLazyModel:
        _sample_token = Model._sample_token

        def __init__(self):
            self.config = SimpleNamespace(tts_model_type="custom_voice")
            self.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
            self.speech_tokenizer = object()
            self.supported_speakers = ["fabian"]
            self._prepare_generation_inputs = lambda **kwargs: None
            # Unlike parameters loaded from disk, derived model buffers may
            # still be lazy when the loading worker returns.
            self.buffer = mx.arange(16).astype(mx.float32) * 0.01

        def generate_custom_voice(self, **kwargs):
            audio = self.buffer + mx.ones(16) * 0.1
            mx.eval(audio)
            yield SimpleNamespace(audio=audio, sample_rate=24000, token_count=1)
            # Cover deferred state created during inference as well as load.
            self.buffer = mx.arange(16).astype(mx.float32) * 0.01

    voice = {"voice_id": "finetuned:test", "speaker": "fabian", "model_path": "tiny-test-model"}
    monkeypatch.setattr(qwen_finetuned_backend, "read_voice", lambda *args, **kwargs: voice)
    monkeypatch.setattr(utils, "load_model", lambda path: TinyLazyModel())
    backend = qwen_finetuned_backend.LocalQwenCustomVoiceBackend()
    # Keep the load thread alive to guarantee subsequent executor workers are
    # different threads, even on platforms that immediately recycle thread IDs.
    with ThreadPoolExecutor(max_workers=1) as loader:
        loader.submit(backend._load, voice).result()
        for _ in range(2):
            audio, rate = asyncio.run(backend.generate("Una prueba.", {"preset_voice_id": voice["voice_id"]}, seed=7))
            assert rate == 24000
            np.testing.assert_allclose(audio, np.arange(16) * 0.01 + 0.1, atol=1e-7)
    backend.unload_model()
    assert not backend.is_loaded()


def test_local_sampling_matches_seeded_upstream_across_worker_lifetimes(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_audio.tts import utils
    from mlx_audio.tts.models.qwen3_tts.qwen3_tts import Model

    from backend.backends.qwen_finetuned_backend import LocalQwenCustomVoiceBackend, load_finetuned_model

    class TinySamplerModel:
        _sample_token = Model._sample_token

        def __init__(self):
            self.config = SimpleNamespace(tts_model_type="custom_voice")
            self.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3])
            self._prepare_generation_inputs = lambda **kwargs: None

        def generate_custom_voice(self, **kwargs):
            logits = mx.arange(64).astype(mx.float32).reshape(1, 1, 64) * 0.01
            draws = [self._sample_token(logits, temperature=0.9, top_k=50) for _ in range(32)]
            audio = mx.concatenate(draws, axis=1).reshape(-1).astype(mx.float32) / 64
            mx.eval(audio)
            yield SimpleNamespace(audio=audio, sample_rate=24000, token_count=32)

    # Upstream's compiled sampler captures the importing thread's RNG list.
    # This is the valid single-thread sequence that the local worker must match.
    mx.random.seed(798)
    expected = np.asarray(next(TinySamplerModel().generate_custom_voice()).audio)
    original_sampling = Model._sample_token
    original_categorical = original_sampling.__globals__["categorical_sampling"]
    monkeypatch.setattr(utils, "load_model", lambda path: TinySamplerModel())
    backend = LocalQwenCustomVoiceBackend()
    backend.model = load_finetuned_model("tiny-sampler")
    backend.voice = {"voice_id": "finetuned:test", "speaker": "fabian"}
    for _ in range(2):
        audio, rate = asyncio.run(backend.generate("Prueba.", {"preset_voice_id": "finetuned:test"}, seed=798))
        assert rate == 24000
        np.testing.assert_array_equal(audio, expected)
    changed, _ = asyncio.run(backend.generate("Prueba.", {"preset_voice_id": "finetuned:test"}, seed=799))
    assert not np.array_equal(changed, expected)
    assert Model._sample_token is original_sampling
    assert Model._sample_token.__globals__["categorical_sampling"] is original_categorical
    backend.unload_model()
