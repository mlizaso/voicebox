"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

# ruff: noqa: E402 -- the offline patch must run before imports that touch HF.

import hashlib
import logging
import secrets
from collections.abc import Sequence
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# PATCH: Import and apply offline patch BEFORE any huggingface_hub usage
# This prevents mlx_audio from making network requests when models are cached
from ..utils.hf_offline_patch import ensure_original_qwen_config_cached, patch_huggingface_hub_offline

patch_huggingface_hub_offline()
ensure_original_qwen_config_cached()

from ..utils.cache import cache_voice_prompt, get_cache_key, get_cached_voice_prompt
from . import LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import combine_voice_prompts as _combine_voice_prompts, is_model_cached, model_load_progress
from .mlx_qwen_optimizations import (
    MAX_QWEN_CLONE_BATCH_SIZE,
    apply_qwen_icl_cache_backport,
    apply_qwen_reference_decode_backport,
    build_reference_cache_key,
    clear_qwen_reference_cache,
    clear_qwen_reference_decode_handoff,
    generate_qwen_icl_batch,
    has_qwen_icl_cache_backport,
    prepare_reference_conditioning,
)
from .mlx_runtime import (
    MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION as _MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION,
    get_installed_mlx_audio_version,
    get_mlx_qwen_tts_model_spec,
)
from .mlx_tts_lifecycle import (
    mlx_tts_lifecycle_guard,
    run_blocking_operation_cancellation_safe as _run_blocking_mlx_operation,
)

_MLX_AUDIO_QWEN_MODEL_TYPE = "qwen3_tts"
_MLX_AUDIO_QWEN_DTYPE_BACKPORT_MARKER = "_voicebox_mlx_audio_qwen_dtype_backport"


def _apply_mlx_audio_qwen_dtype_backport(model: object, mlx_audio_version: str | None) -> bool:
    """Backport the Qwen speaker-embedding dtype fix to mlx-audio 0.4.1.

    In mlx-audio 0.4.1 the speaker encoder returns float32 even when the
    talker is BF16. Concatenating those arrays promotes the talker prefill and
    KV cache to float32. Upstream fixed both cloning paths in PR #879 by
    casting the speaker embedding to the talker embedding dtype:
    https://github.com/Blaizzy/mlx-audio/pull/879

    The wrapper is intentionally restricted to the one affected dependency
    version and the expected Qwen model interface. Newer mlx-audio versions
    contain the upstream fix and must not be monkey-patched here.
    """
    if mlx_audio_version != _MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION:
        return False

    if getattr(model, _MLX_AUDIO_QWEN_DTYPE_BACKPORT_MARKER, False):
        return True

    if getattr(model, "model_type", None) != _MLX_AUDIO_QWEN_MODEL_TYPE:
        logger.warning(
            "Skipping mlx-audio %s Qwen dtype backport: unexpected model type %r",
            mlx_audio_version,
            getattr(model, "model_type", None),
        )
        return False

    extract_speaker_embedding = getattr(model, "extract_speaker_embedding", None)
    talker = getattr(model, "talker", None)
    get_input_embeddings = getattr(talker, "get_input_embeddings", None)
    if not callable(extract_speaker_embedding) or not callable(get_input_embeddings):
        logger.warning(
            "Skipping mlx-audio %s Qwen dtype backport: unexpected model interface",
            mlx_audio_version,
        )
        return False

    codec_embedding = get_input_embeddings()
    target_dtype = getattr(getattr(codec_embedding, "weight", None), "dtype", None)
    if target_dtype is None:
        logger.warning(
            "Skipping mlx-audio %s Qwen dtype backport: codec embedding dtype is unavailable",
            mlx_audio_version,
        )
        return False
    if not str(target_dtype).endswith("bfloat16"):
        logger.warning(
            "Skipping mlx-audio %s Qwen dtype backport: expected a BF16 talker, got %s",
            mlx_audio_version,
            target_dtype,
        )
        return False

    def extract_speaker_embedding_with_talker_dtype(audio, sr=24000):
        speaker_embedding = extract_speaker_embedding(audio, sr=sr)
        return speaker_embedding.astype(target_dtype)

    model.extract_speaker_embedding = extract_speaker_embedding_with_talker_dtype
    setattr(model, _MLX_AUDIO_QWEN_DTYPE_BACKPORT_MARKER, True)
    logger.info(
        "Applied mlx-audio %s Qwen speaker-embedding dtype backport (%s)",
        mlx_audio_version,
        target_dtype,
    )
    return True


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""

    uses_mlx_request_context = True
    uses_shared_mlx_lifecycle_guard = True

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX model path.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID for MLX
        """
        hf_model_id, _revision = get_mlx_qwen_tts_model_spec(model_size)
        logger.info("Will download MLX model from HuggingFace Hub: %s", hf_model_id)

        return hf_model_id

    def _get_model_revision(self, model_size: str) -> str:
        """Return the immutable Hugging Face commit for a supported model size."""
        _hf_model_id, revision = get_mlx_qwen_tts_model_spec(model_size)
        return revision

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
            revision=self._get_model_revision(model_size),
        )

    async def load_model_async(self, model_size: str | None = None):
        """
        Lazy load the MLX TTS model.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size

        async with mlx_tts_lifecycle_guard.hold("model loading"):
            await self._ensure_model_loaded_locked(model_size)

    @asynccontextmanager
    async def mlx_request_context(self, model_size: str):
        """Retain one immutable model-size binding for a complete request."""
        requested_model_size = model_size or self.model_size
        async with mlx_tts_lifecycle_guard.hold(f"{requested_model_size} model request"):
            await self._ensure_model_loaded_locked(requested_model_size)
            if self._current_model_size != requested_model_size:
                raise RuntimeError("MLX TTS loaded a different model size than the request")
            yield

    async def _ensure_model_loaded_locked(self, model_size: str) -> None:
        """Ensure one model size while the process-wide lifecycle guard is held."""

        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return

        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self._unload_model_locked()

        # Run blocking load in thread pool
        await _run_blocking_mlx_operation(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_path = self._get_model_path(model_size)
        model_revision = self._get_model_revision(model_size)
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts import load

            mlx_audio_version = get_installed_mlx_audio_version()

            logger.info("Loading MLX TTS model %s...", model_size)

            loaded_model = load(model_path, revision=model_revision)
            backport_applied = _apply_mlx_audio_qwen_dtype_backport(loaded_model, mlx_audio_version)
            optimizations_applied = apply_qwen_icl_cache_backport(loaded_model, mlx_audio_version)
            decode_applied = apply_qwen_reference_decode_backport(loaded_model, mlx_audio_version)
            is_qwen_model = (
                getattr(loaded_model, "model_type", None) == _MLX_AUDIO_QWEN_MODEL_TYPE
                or "qwen" in model_path.casefold()
            )
            if mlx_audio_version == _MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION and is_qwen_model and not backport_applied:
                raise RuntimeError(
                    "mlx-audio 0.4.1 Qwen model does not support Voicebox's required "
                    "speaker-embedding dtype guard; refusing to load an unverified TTS "
                    "implementation"
                )
            if (
                mlx_audio_version == _MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION
                and is_qwen_model
                and not optimizations_applied
            ):
                raise RuntimeError(
                    "mlx-audio 0.4.1 Qwen model does not support Voicebox's required "
                    "ICL conditioning-cache and batch interfaces; refusing to load an "
                    "unverified TTS implementation"
                )
            if mlx_audio_version == _MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION and is_qwen_model and not decode_applied:
                raise RuntimeError(
                    "mlx-audio 0.4.1 Qwen model does not expose the windowed decoder "
                    "contract Voicebox's discarded-reference decode requires; refusing "
                    "to load an unverified TTS implementation"
                )
            self.model = loaded_model

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MLX TTS model %s loaded successfully", model_size)

    def _unload_model_locked(self) -> None:
        if self.model is not None:
            clear_qwen_reference_decode_handoff(self.model)
            clear_qwen_reference_cache(self.model)
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("MLX TTS model unloaded")

    def unload_model(self):
        """Unload the model only when no MLX TTS operation is active."""
        with mlx_tts_lifecycle_guard.try_hold("model unloading"):
            self._unload_model_locked()

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        MLX backend stores voice prompt as a dict with audio path and text.
        The actual voice prompt processing happens during generation.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)

        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None and isinstance(cached_prompt, dict):
                # Cache identity is content based, so two paths can legitimately
                # share an entry.  Rebind a copy to the caller's path instead of
                # retaining the first path, which may later be overwritten.
                rebound_prompt = dict(cached_prompt)
                rebound_prompt["ref_audio"] = str(audio_path)
                rebound_prompt.pop("ref_audio_path", None)
                rebound_prompt["ref_text"] = reference_text
                rebound_prompt["mlx_conditioning_key"] = build_reference_cache_key(
                    audio_path,
                    reference_text,
                )
                return rebound_prompt, True

        # MLX voice prompt format - store audio path and text
        # The model will process this during generation
        voice_prompt_items = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
            "mlx_conditioning_key": build_reference_cache_key(audio_path, reference_text),
        }

        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)

        return voice_prompt_items, False

    def _prepare_reference_sync(self, voice_prompt: dict):
        ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
        ref_text = voice_prompt.get("ref_text", "")
        if not ref_audio or not Path(ref_audio).exists():
            raise FileNotFoundError(
                f"Voice-cloning reference audio is missing; refusing to generate with a different voice: {ref_audio}"
            )
        # Preparation reads, hashes, and decodes one immutable byte snapshot.
        # A persisted prompt may outlive an in-place edit of its WAV, and a
        # path replacement must not race cache identity construction.
        return prepare_reference_conditioning(
            self.model,
            audio_path=ref_audio,
            reference_text=ref_text,
        )

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24000)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: Language code (en or zh) - may not be fully supported by MLX
            seed: Random seed for reproducibility
            instruct: Natural language instruction (may not be supported by MLX)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        text_identifier = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        logger.info(
            "Generating audio (chars=%d, text_sha256=%s)",
            len(text),
            text_identifier,
        )

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            loaded_model = self.model
            # Never let a failed earlier ICL generation influence this request.
            clear_qwen_reference_decode_handoff(loaded_model)
            try:
                # MLX generate() returns a generator yielding GenerationResult objects
                audio_chunks = []
                sample_rate = 24000
                lang = LANGUAGE_CODE_TO_NAME.get(language, "auto")

                # Set seed if provided (MLX uses numpy random)
                if seed is not None:
                    import mlx.core as mx

                    np.random.seed(seed)
                    mx.random.seed(seed)

                # Extract voice prompt info
                ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
                ref_text = voice_prompt.get("ref_text", "")

                # Validate that the audio file exists
                if ref_audio and not Path(ref_audio).exists():
                    raise FileNotFoundError(
                        "Voice-cloning reference audio is missing; refusing to generate with a "
                        f"different voice: {ref_audio}"
                    )

                # Inference runs with the process's default HF_HUB_OFFLINE
                # state. Forcing offline here (previously used to avoid lazy
                # mlx_audio lookups hanging when the network drops mid-inference,
                # issue #462) regressed online users because libraries make
                # legitimate metadata calls during generation.
                try:
                    if ref_audio:
                        # Check if generate accepts ref_audio parameter
                        import inspect

                        sig = inspect.signature(loaded_model.generate)
                        if "ref_audio" in sig.parameters:
                            model_ref_audio = ref_audio
                            if has_qwen_icl_cache_backport(loaded_model):
                                model_ref_audio = self._prepare_reference_sync(voice_prompt).audio
                            # Generate with voice cloning
                            for result in loaded_model.generate(
                                text,
                                ref_audio=model_ref_audio,
                                ref_text=ref_text,
                                lang_code=lang,
                            ):
                                audio_chunks.append(np.array(result.audio))
                                sample_rate = result.sample_rate
                        else:
                            raise RuntimeError(
                                "Loaded MLX model does not support ref_audio; refusing to generate with a generic voice"
                            )
                    else:
                        # No voice prompt, generate normally
                        for result in loaded_model.generate(text, lang_code=lang):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                except Exception as e:
                    if ref_audio:
                        raise RuntimeError("Voice cloning failed; refusing to generate with a generic voice") from e
                    raise

                # Concatenate all chunks
                if audio_chunks:
                    audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in audio_chunks])
                else:
                    # Fallback: empty audio
                    audio = np.array([], dtype=np.float32)

                return audio, sample_rate
            finally:
                # mlx-audio can return before its decoder on immediate codec EOS.
                # In that path the one-shot handoff installed during preparation
                # would otherwise remain armed for an unrelated later decode.
                clear_qwen_reference_decode_handoff(loaded_model)

        requested_model_size = self.model_size
        async with mlx_tts_lifecycle_guard.hold("serial inference"):
            await self._ensure_model_loaded_locked(requested_model_size)
            # Cancellation drains the executor thread before this context can
            # release the shared model/global-RNG lifecycle guard.
            audio, sample_rate = await _run_blocking_mlx_operation(_generate_sync)

        return audio, sample_rate

    async def generate_batch(
        self,
        texts: Sequence[str],
        voice_prompt: dict,
        language: str = "en",
        seeds: Sequence[int | None] | None = None,
        instruct: str | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Generate up to two clone chunks in one model-level forward pass.

        The shared reference is conditioned once, while every row owns an
        independent explicit PRNG stream.  Results retain input order so the
        caller can checkpoint each logical chunk separately.
        """
        if not 1 <= len(texts) <= MAX_QWEN_CLONE_BATCH_SIZE:
            raise ValueError(f"MLX Qwen clone batch size must be between 1 and {MAX_QWEN_CLONE_BATCH_SIZE}")
        if instruct is not None:
            raise NotImplementedError("MLX Qwen ICL batching does not support instructions")
        if seeds is None:
            concrete_seeds = [secrets.randbits(32) for _ in texts]
        else:
            if len(seeds) != len(texts):
                raise ValueError("MLX Qwen clone batch requires one seed per text")
            concrete_seeds = [seed if seed is not None else secrets.randbits(32) for seed in seeds]

        try:
            # Once this pinned backend accepts a batch, every failure is an
            # inference failure.  Do not leak a third-party NotImplementedError
            # to the generic capability fallback and silently change an exact
            # audiobook request into the serial numerical algorithm.
            requested_model_size = self.model_size
            lang = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            def _generate_batch_sync() -> list[tuple[np.ndarray, int]]:
                conditioning = self._prepare_reference_sync(voice_prompt)
                indexed_results = generate_qwen_icl_batch(
                    self.model,
                    texts,
                    conditioning,
                    language=lang,
                    seeds=concrete_seeds,
                )
                by_index: dict[int, tuple[np.ndarray, int]] = {}
                for sequence_index, audio, sample_rate in indexed_results:
                    if sequence_index in by_index or not 0 <= sequence_index < len(texts):
                        raise RuntimeError("MLX Qwen clone batch returned an invalid sequence index")
                    by_index[sequence_index] = (np.asarray(audio, dtype=np.float32), sample_rate)
                if set(by_index) != set(range(len(texts))):
                    raise RuntimeError("MLX Qwen clone batch returned incomplete results")
                return [by_index[index] for index in range(len(texts))]

            async with mlx_tts_lifecycle_guard.hold("batch inference"):
                await self._ensure_model_loaded_locked(requested_model_size)
                return await _run_blocking_mlx_operation(_generate_batch_sync)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError("Batched voice cloning failed; refusing serial or generic fallback") from exc


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""

    uses_shared_mlx_lifecycle_guard = True

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo, weight_extensions=(".safetensors", ".bin", ".npz"))

    async def load_model_async(self, model_size: str | None = None):
        """
        Lazy load the MLX Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size

        async with mlx_tts_lifecycle_guard.hold("MLX STT model loading"):
            await self._ensure_model_loaded_locked(model_size)

    async def _ensure_model_loaded_locked(self, model_size: str) -> None:
        if self.model is not None and self.model_size == model_size:
            return

        if self.model is not None:
            self._unload_model_locked()

        await _run_blocking_mlx_operation(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from mlx_audio.stt import load

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading MLX Whisper model %s...", model_size)

            self.model = load(model_name)

        self.model_size = model_size
        logger.info("MLX Whisper model %s loaded successfully", model_size)

    def _unload_model_locked(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("MLX Whisper model unloaded")

    def unload_model(self):
        """Unload the model only when no shared MLX operation is active."""
        with mlx_tts_lifecycle_guard.try_hold("MLX STT model unloading"):
            self._unload_model_locked()

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        requested_model_size = model_size or self.model_size

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # MLX Whisper transcription using generate method
            # The generate method accepts audio path directly
            decode_options = {}
            if language:
                decode_options["language"] = language

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state — see the comment in MLXTTSBackend.generate for the
            # regression this revert fixes (issue #462).
            result = self.model.generate(str(audio_path), **decode_options)

            # Extract text from result
            if isinstance(result, str):
                return result.strip()
            if isinstance(result, dict):
                return result.get("text", "").strip()
            if hasattr(result, "text"):
                return result.text.strip()
            return str(result).strip()

        async with mlx_tts_lifecycle_guard.hold("MLX STT inference"):
            await self._ensure_model_loaded_locked(requested_model_size)
            return await _run_blocking_mlx_operation(_transcribe_sync)
