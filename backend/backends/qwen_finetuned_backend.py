"""MLX inference for merged local CustomVoice checkpoints selected by profile."""

import gc
import secrets
from contextlib import asynccontextmanager
from types import FunctionType, MethodType

import numpy as np

from ..services.finetuned_voices import read_voice
from ..utils.chunked_tts import SynthesisDurationLimitError
from .mlx_tts_lifecycle import mlx_tts_lifecycle_guard, run_blocking_operation_cancellation_safe


def apply_non_streaming_prefix(model) -> None:
    """Match the fine-tuning prefix on this model instance only.

    mlx-audio 0.4.1 CustomVoice normally interleaves text with generated audio.
    These local checkpoints were trained on Qwen's non-streaming text prefix.
    The decoder and sampling path are unchanged.
    """
    import mlx.core as mx  # lazy: native runtime

    if getattr(model, "_voicebox_finetuned_prefix", False):
        return
    if model.config.tts_model_type != "custom_voice":
        raise ValueError("A fine-tuned voice requires a CustomVoice checkpoint")
    original = model._prepare_generation_inputs

    def prepare(text, language="auto", speaker=None, ref_audio=None, ref_text=None, instruct=None):
        if ref_audio is not None or ref_text is not None or instruct:
            raise ValueError("This fine-tuned narrator uses its fixed speaker and does not support instructions")
        prefix, _trailing, pad = original(text=text, language=language, speaker=speaker)
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        text_ids = mx.array([[*tokens, model.config.tts_eos_token_id]])
        text_hidden = model.talker.text_projection(model.talker.get_text_embeddings()(text_ids))
        codec = model.config.talker_config
        codec_pad = model.talker.get_input_embeddings()(mx.array([[codec.codec_pad_id]]))
        codec_bos = model.talker.get_input_embeddings()(mx.array([[codec.codec_bos_id]]))
        full = mx.concatenate([prefix[:, :-1], text_hidden + codec_pad, pad + codec_bos], axis=1)
        return full, pad, pad

    model._prepare_generation_inputs = prepare
    model._voicebox_finetuned_prefix = True


def load_finetuned_model(path: str):
    """Load lazy buffers on a stream shared by strictly serialized workers.

    MLX's default streams belong to the creating thread, whereas asyncio's
    executor may use a different thread for the next request. The lifecycle
    guard must protect all use of this model: this stream permits cross-thread
    evaluation, not concurrent evaluation.
    """
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    stream = mx.new_thread_unsafe_stream(mx.gpu)
    with mx.stream(stream):
        model = load_model(path)
        apply_non_streaming_prefix(model)
        apply_request_local_sampling(model)
    model._voicebox_finetuned_stream = stream
    return model


def apply_request_local_sampling(model) -> None:
    """Keep upstream filtering but pass an explicit key to compiled sampling.

    mlx-lm's compiled categorical sampler captures its importing thread's RNG
    list. Seeding another worker does not advance that captured state correctly.
    Bind a private copy of Qwen's method globals to this model only, leaving
    upstream classes and every other engine untouched.
    """
    import mlx.core as mx

    original = model._sample_token.__func__
    if "categorical_sampling" not in original.__code__.co_names:
        raise RuntimeError("Unsupported Qwen categorical sampler implementation")
    key = mx.random.key(0)

    @mx.compile
    def sample_and_advance(logits, temperature, current_key):
        next_key, draw_key = mx.random.split(current_key)
        return mx.random.categorical(logits * (1 / temperature), key=draw_key), next_key

    def categorical(logits, temperature):
        nonlocal key
        token, key = sample_and_advance(logits, temperature, key)
        return token

    def seed_request(seed):
        nonlocal key
        key = mx.random.key(seed)

    local_sample = FunctionType(
        original.__code__,
        {**original.__globals__, "categorical_sampling": categorical},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    model._sample_token = MethodType(local_sample, model)
    model._voicebox_finetuned_seed = seed_request


class LocalQwenCustomVoiceBackend:
    """Retain one warm local model under the shared accelerator lifecycle lock."""

    uses_shared_mlx_lifecycle_guard = True
    # Validation exposed premature EOS on long single sentences. Keep each
    # model call bounded; the shared chunker preserves the complete narration.
    max_input_chars = 200

    def __init__(self) -> None:
        self.model = None
        self.voice = None
        self.model_size = "1.7B"

    def is_loaded(self) -> bool:
        return self.model is not None

    @asynccontextmanager
    async def voice_request(self, voice: dict, model_size: str):
        if model_size not in ("default", "1.7B"):
            raise ValueError("This fine-tuned narrator requires Qwen 1.7B")
        async with mlx_tts_lifecycle_guard.hold("local fine-tuned voice request"):
            if self.voice != voice or self.model is None:
                await run_blocking_operation_cancellation_safe(self._load, voice)
            yield self

    def _load(self, voice: dict) -> None:
        import mlx.core as mx  # lazy: native model import

        verified = read_voice(voice["voice_id"], verify_weights=True)
        if verified != voice:
            raise ValueError("Fine-tuned model registration changed while the request was waiting")
        self.model = None
        self.voice = None
        gc.collect()
        mx.clear_cache()
        model = load_finetuned_model(voice["model_path"])
        if model.tokenizer is None or model.speech_tokenizer is None:
            raise RuntimeError("Fine-tuned checkpoint failed to load its tokenizers")
        if voice["speaker"] not in model.supported_speakers:
            raise ValueError("Fine-tuned checkpoint is missing its registered speaker")
        self.model = model
        self.voice = voice

    def unload_model(self) -> None:
        with mlx_tts_lifecycle_guard.try_hold("local fine-tuned voice unload"):
            self.model = None
            self.voice = None
            import mlx.core as mx  # lazy: native cache cleanup

            gc.collect()
            mx.clear_cache()

    async def generate(self, text: str, voice_prompt: dict, language="es", seed=None, instruct=None):
        if language not in ("es", "Spanish", "spanish"):
            raise ValueError("This fine-tuned narrator was trained and validated for Spanish")
        if instruct:
            raise ValueError("Delivery instructions are not supported by this fine-tuned narrator")
        if len(text.strip()) > self.max_input_chars:
            raise ValueError("Use generate_chunked for long fine-tuned narration")
        async with mlx_tts_lifecycle_guard.hold("local fine-tuned inference"):
            if self.voice is None or voice_prompt.get("preset_voice_id") != self.voice["voice_id"]:
                raise ValueError("Fine-tuned voice binding is missing or changed")
            return await run_blocking_operation_cancellation_safe(self._generate, text, seed)

    def _generate(self, text: str, seed: int | None) -> tuple[np.ndarray, int]:
        import mlx.core as mx  # lazy: native inference

        stream = self.model._voicebox_finetuned_stream
        with mx.stream(stream):
            try:
                return self._generate_on_stream(text, seed)
            finally:
                # Ownership must outlive submitted GPU work even on failure.
                mx.synchronize(stream)

    def _generate_on_stream(self, text: str, seed: int | None) -> tuple[np.ndarray, int]:
        import mlx.core as mx

        self.model._voicebox_finetuned_seed(seed if seed is not None else secrets.randbits(32))
        if seed is not None:
            mx.random.seed(seed)
        token_limit = min(4096, max(75, len(self.model.tokenizer.encode(text)) * 6))
        results = list(
            self.model.generate_custom_voice(
                text=text,
                speaker=self.voice["speaker"],
                language="Spanish",
                max_tokens=token_limit,
                stream=False,
                verbose=False,
            )
        )
        if not results:
            raise RuntimeError("Fine-tuned model returned no audio")
        rates = {result.sample_rate for result in results}
        if rates != {24000}:
            raise RuntimeError("Unexpected fine-tuned sample rate")
        audio = np.concatenate([np.asarray(result.audio, dtype=np.float32).reshape(-1) for result in results])
        if not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError("Fine-tuned model returned empty or nonfinite audio")
        if sum(result.token_count for result in results) >= token_limit:
            raise SynthesisDurationLimitError("Fine-tuned model reached its duration limit before finishing")
        return audio, 24000


_local_backend = LocalQwenCustomVoiceBackend()


def get_local_backend() -> LocalQwenCustomVoiceBackend:
    """Return the local preset worker; the request guard serializes all access."""
    return _local_backend
