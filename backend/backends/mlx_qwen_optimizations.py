"""Guarded Qwen3-TTS performance backports for pinned mlx-audio 0.4.1.

The pinned release can batch preset voices, but not ICL/reference cloning, and
it recomputes reference conditioning for every generation.  These helpers
adapt the two upstream changes without replacing the package in site-packages:

* reference conditioning cache: https://github.com/Blaizzy/mlx-audio/pull/685
* shared-reference ICL batching: https://github.com/Blaizzy/mlx-audio/pull/689

Unlike the original cache patch, Voicebox also retains the decoded reference
waveform and BF16 speaker embedding.  A cache hit therefore skips file I/O,
speech-tokenizer encoding, reference tokenization, and speaker encoding.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

logger = logging.getLogger(__name__)

MLX_AUDIO_QWEN_OPTIMIZATION_VERSION = "0.4.1"
_CACHE_MARKER = "_voicebox_qwen_icl_cache_v2"
_REFERENCE_CACHE_ATTR = "_voicebox_qwen_reference_conditioning"
MAX_QWEN_CLONE_BATCH_SIZE = 2
MAX_QWEN_REFERENCE_CACHE_ENTRIES = 4


@dataclass(frozen=True)
class QwenReferenceConditioning:
    """Evaluated model inputs shared by every chunk for one cloned voice."""

    cache_key: str
    ref_text: str
    audio: Any
    ref_codes: Any
    ref_text_ids: Any
    speaker_embedding: Any


def has_qwen_icl_cache_backport(model: Any) -> bool:
    """Return whether the guarded cache-aware method is installed."""
    return bool(getattr(model, _CACHE_MARKER, False))


def build_reference_cache_key(audio_path: str, reference_text: str) -> str:
    """Return a collision-resistant identity for reference bytes and text."""
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return _build_reference_cache_key_from_bytes(audio_bytes, reference_text)


def _build_reference_cache_key_from_bytes(audio_bytes: bytes, reference_text: str) -> str:
    """Hash the same immutable byte snapshot that reference decoding consumes."""
    digest = hashlib.sha256(audio_bytes)
    digest.update(b"\0voicebox-qwen-reference-text\0")
    digest.update(reference_text.encode("utf-8"))
    return digest.hexdigest()


def _load_reference_snapshot(audio_bytes: bytes, audio_suffix: str, sample_rate: int) -> Any:
    """Decode an immutable copy so cache identity cannot race a path replacement."""
    from mlx_audio.utils import load_audio

    # mlx-audio 0.4.1's loader accepts paths or MLX arrays, but not bytes.  A
    # mode-0600 temporary preserves the source suffix (and therefore decoder
    # selection) while guaranteeing the decoder sees exactly the bytes hashed
    # for this cache entry.  Reference preparation is a cache miss-only path.
    with tempfile.NamedTemporaryFile(
        prefix="voicebox-qwen-reference-",
        suffix=audio_suffix,
    ) as snapshot:
        snapshot.write(audio_bytes)
        snapshot.flush()
        return load_audio(snapshot.name, sample_rate=sample_rate)


def _required_parameters(callable_object: Any) -> set[str]:
    try:
        return set(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError):
        return set()


def apply_qwen_icl_cache_backport(model: Any, mlx_audio_version: str | None) -> bool:
    """Install the cache-aware ICL input builder on the exact pinned model.

    The guard deliberately checks the private 0.4.1 method contract.  If a
    later package changes it, Voicebox refuses to advertise this numerical
    runtime rather than silently running an unverified monkey patch.
    """
    if mlx_audio_version != MLX_AUDIO_QWEN_OPTIMIZATION_VERSION:
        return False
    if getattr(model, _CACHE_MARKER, False):
        return True
    if getattr(model, "model_type", None) != "qwen3_tts":
        return False

    original_prepare = getattr(model, "_prepare_icl_generation_inputs", None)
    required = {"text", "ref_audio", "ref_text", "language"}
    if not callable(original_prepare) or not required.issubset(_required_parameters(original_prepare)):
        logger.warning(
            "Skipping mlx-audio %s Qwen ICL cache: unexpected preparation interface",
            mlx_audio_version,
        )
        return False
    if not all(hasattr(model, name) for name in ("config", "talker", "speech_tokenizer", "tokenizer")):
        logger.warning(
            "Skipping mlx-audio %s Qwen ICL cache: incomplete model interface",
            mlx_audio_version,
        )
        return False
    if (
        getattr(model.config, "tts_model_type", None) != "base"
        or not getattr(model.speech_tokenizer, "has_encoder", False)
        or getattr(model, "speaker_encoder", None) is None
    ):
        logger.warning(
            "Skipping mlx-audio %s Qwen ICL cache: model is not a clone-capable Base checkpoint",
            mlx_audio_version,
        )
        return False

    setattr(model, _REFERENCE_CACHE_ATTR, OrderedDict())

    def prepare_with_cache(
        bound_model: Any,
        text: str,
        ref_audio: Any,
        ref_text: str,
        language: str = "auto",
    ) -> tuple[Any, Any, Any, Any]:
        conditioning = _find_registered_conditioning(bound_model, ref_audio, ref_text)
        if conditioning is None:
            return original_prepare(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
            )
        return _prepare_icl_from_conditioning(bound_model, text, conditioning, language)

    model._prepare_icl_generation_inputs = MethodType(prepare_with_cache, model)
    setattr(model, _CACHE_MARKER, True)
    logger.info("Applied mlx-audio %s Qwen ICL conditioning cache backport", mlx_audio_version)
    return True


def prepare_reference_conditioning(
    model: Any,
    *,
    audio_path: str,
    reference_text: str,
) -> QwenReferenceConditioning:
    """Encode and register one reusable cloned-voice reference."""
    if not has_qwen_icl_cache_backport(model):
        raise RuntimeError("Qwen ICL conditioning cache backport is not installed")

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    cache_key = _build_reference_cache_key_from_bytes(audio_bytes, reference_text)
    registry = getattr(model, _REFERENCE_CACHE_ATTR)
    existing = registry.get(cache_key)
    if existing is not None:
        if existing.ref_text != reference_text:
            raise RuntimeError("Qwen reference cache key collision")
        registry.move_to_end(cache_key)
        return existing

    import mlx.core as mx

    audio = _load_reference_snapshot(
        audio_bytes,
        Path(audio_path).suffix,
        model.sample_rate,
    )
    encoder_audio = audio
    if audio.ndim == 1:
        encoder_audio = audio[None, None, :]
    elif audio.ndim == 2:
        encoder_audio = audio[None, :]

    ref_codes = model.speech_tokenizer.encode(encoder_audio)
    ref_chat = f"<|im_start|>assistant\n{reference_text}<|im_end|>\n"
    ref_ids = mx.array(model.tokenizer.encode(ref_chat))[None, :]
    ref_text_ids = ref_ids[:, 3:-2]
    speaker_embedding = model.extract_speaker_embedding(audio)
    mx.eval(audio, ref_codes, ref_text_ids, speaker_embedding)

    conditioning = QwenReferenceConditioning(
        cache_key=cache_key,
        ref_text=reference_text,
        audio=audio,
        ref_codes=ref_codes,
        ref_text_ids=ref_text_ids,
        speaker_embedding=speaker_embedding,
    )
    _register_reference_conditioning(model, conditioning)
    return conditioning


def _register_reference_conditioning(model: Any, conditioning: QwenReferenceConditioning) -> None:
    registry = getattr(model, _REFERENCE_CACHE_ATTR)
    registry[conditioning.cache_key] = conditioning
    registry.move_to_end(conditioning.cache_key)
    while len(registry) > MAX_QWEN_REFERENCE_CACHE_ENTRIES:
        registry.popitem(last=False)


def clear_qwen_reference_cache(model: Any) -> int:
    """Release all evaluated reference tensors owned by a loaded model."""
    registry = getattr(model, _REFERENCE_CACHE_ATTR, None)
    if registry is None:
        return 0
    cleared = len(registry)
    registry.clear()
    return cleared


def _find_registered_conditioning(
    model: Any,
    ref_audio: Any,
    ref_text: str,
) -> QwenReferenceConditioning | None:
    for conditioning in getattr(model, _REFERENCE_CACHE_ATTR, {}).values():
        if conditioning.audio is ref_audio and conditioning.ref_text == ref_text:
            return conditioning
    return None


def _prepare_icl_from_conditioning(
    model: Any,
    text: str,
    conditioning: QwenReferenceConditioning,
    language: str,
) -> tuple[Any, Any, Any, Any]:
    """Pinned 0.4.1 ICL preparation with reference-only work reused."""
    import mlx.core as mx

    config = model.config.talker_config
    target_chat = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    target_ids = mx.array(model.tokenizer.encode(target_chat))[None, :]
    text_ids = target_ids[:, 3:-5]

    tts_tokens = mx.array(
        [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]]
    )
    tts_embeds = model.talker.text_projection(model.talker.get_text_embeddings()(tts_tokens))
    tts_bos_embed = tts_embeds[:, 0:1, :]
    tts_eos_embed = tts_embeds[:, 1:2, :]
    tts_pad_embed = tts_embeds[:, 2:3, :]

    combined_text_ids = mx.concatenate([conditioning.ref_text_ids, text_ids], axis=1)
    text_embed = model.talker.text_projection(model.talker.get_text_embeddings()(combined_text_ids))
    text_embed = mx.concatenate([text_embed, tts_eos_embed], axis=1)
    text_len = text_embed.shape[1]

    ref_codes = conditioning.ref_codes
    ref_codec_embed = model.talker.get_input_embeddings()(ref_codes[:, 0, :])
    for index in range(config.num_code_groups - 1):
        ref_codec_embed = ref_codec_embed + model.talker.code_predictor.codec_embedding[index](
            ref_codes[:, index + 1, :]
        )

    codec_bos_embed = model.talker.get_input_embeddings()(mx.array([[config.codec_bos_id]]))
    codec_embed_icl = mx.concatenate([codec_bos_embed, ref_codec_embed], axis=1)
    codec_len = codec_embed_icl.shape[1]
    codec_pad_embed = model.talker.get_input_embeddings()(mx.array([[config.codec_pad_id]]))
    text_with_codec_pad = text_embed + mx.broadcast_to(codec_pad_embed, (1, text_len, codec_pad_embed.shape[-1]))
    codec_with_text_pad = codec_embed_icl + mx.broadcast_to(tts_pad_embed, (1, codec_len, tts_pad_embed.shape[-1]))
    icl_input_embed = mx.concatenate([text_with_codec_pad, codec_with_text_pad], axis=1)
    trailing_text_hidden = tts_pad_embed

    language_id = None
    if language.lower() != "auto" and config.codec_language_id:
        language_id = config.codec_language_id.get(language.lower())
    if language_id is None:
        codec_prefill = [config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id]
    else:
        codec_prefill = [config.codec_think_id, config.codec_think_bos_id, language_id, config.codec_think_eos_id]

    codec_prefix_embed = model.talker.get_input_embeddings()(mx.array([codec_prefill]))
    codec_prefix_suffix = model.talker.get_input_embeddings()(mx.array([[config.codec_pad_id, config.codec_bos_id]]))
    codec_prefix_embed = mx.concatenate(
        [
            codec_prefix_embed,
            conditioning.speaker_embedding.reshape(1, 1, -1),
            codec_prefix_suffix,
        ],
        axis=1,
    )
    role_embed = model.talker.text_projection(model.talker.get_text_embeddings()(target_ids[:, :3]))
    pad_count = codec_prefix_embed.shape[1] - 2
    pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
    combined_prefix = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
    combined_prefix = combined_prefix + codec_prefix_embed[:, :-1, :]
    input_embeds = mx.concatenate([role_embed, combined_prefix, icl_input_embed], axis=1)
    return input_embeds, trailing_text_hidden, tts_pad_embed, ref_codes


def _prepare_batch_inputs(
    model: Any,
    texts: Sequence[str],
    conditioning: QwenReferenceConditioning,
    language: str,
) -> tuple[Any, Any, Any, Any]:
    import mlx.core as mx

    prepared = [
        model._prepare_icl_generation_inputs(
            text=text,
            ref_audio=conditioning.audio,
            ref_text=conditioning.ref_text,
            language=language,
        )
        for text in texts
    ]
    per_sequence_embeds = [item[0] for item in prepared]
    per_sequence_trailing = [item[1] for item in prepared]
    shared_pad_embed = prepared[0][2]

    hidden_size = per_sequence_embeds[0].shape[-1]
    max_prefill = max(embeds.shape[1] for embeds in per_sequence_embeds)
    padded_embeds = []
    masks = []
    for embeds in per_sequence_embeds:
        sequence_len = embeds.shape[1]
        pad_len = max_prefill - sequence_len
        if pad_len:
            padding = mx.zeros((1, pad_len, hidden_size), dtype=embeds.dtype)
            embeds = mx.concatenate([padding, embeds], axis=1)
            mask = mx.concatenate(
                [mx.zeros((1, pad_len), dtype=mx.int32), mx.ones((1, sequence_len), dtype=mx.int32)],
                axis=1,
            )
        else:
            mask = mx.ones((1, sequence_len), dtype=mx.int32)
        padded_embeds.append(embeds)
        masks.append(mask)

    max_trailing = max(trailing.shape[1] for trailing in per_sequence_trailing)
    padded_trailing = []
    for trailing in per_sequence_trailing:
        pad_len = max_trailing - trailing.shape[1]
        if pad_len:
            fill = mx.broadcast_to(shared_pad_embed, (1, pad_len, hidden_size))
            trailing = mx.concatenate([trailing, fill], axis=1)
        padded_trailing.append(trailing)

    return (
        mx.concatenate(padded_embeds, axis=0),
        mx.concatenate(padded_trailing, axis=0),
        shared_pad_embed,
        mx.concatenate(masks, axis=0),
    )


def _row_random_key(seed: int, counter: int) -> tuple[int, int]:
    """Derive both uint32 words of an MLX key for one row-local sample."""
    payload = f"voicebox-qwen-batch-v1:{seed}:{counter}".encode("ascii")
    digest = hashlib.sha256(payload).digest()
    # MLX 0.32 validates categorical keys as uint32[2].  Supplying both words
    # directly avoids the previous four-byte truncation, which has a material
    # birthday-collision rate across the tens of thousands of samples in a long
    # row (and a known seed=1 collision at counters 451 and 23,849).
    return int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big")


def _sample_rows(
    model: Any,
    logits: Any,
    *,
    seeds: Sequence[int],
    counters: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float = 1.0,
    generated_tokens_per_sequence: Sequence[Sequence[int]] | None = None,
    suppress_tokens: Sequence[int] | None = None,
    eos_token_id: int | None = None,
    min_p: float = 0.0,
) -> Any:
    """Sample every row with an explicit independent MLX PRNG key."""
    import mlx.core as mx
    from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p

    del model  # Kept in the interface to make the pinned sampling owner explicit.
    logits = logits[:, -1, :]
    batch_size = logits.shape[0]
    if len(seeds) != batch_size or len(counters) != batch_size:
        raise ValueError("Qwen batch sampler seed state does not match batch size")

    if suppress_tokens:
        suppress_index = mx.array(suppress_tokens, dtype=mx.int32)
        indices = mx.broadcast_to(suppress_index[None, :], (batch_size, len(suppress_tokens)))
        logits = mx.put_along_axis(logits, indices, mx.array(float("-inf"), logits.dtype), axis=-1)

    if generated_tokens_per_sequence and repetition_penalty != 1.0:
        rows = []
        for row_index in range(batch_size):
            row = logits[row_index : row_index + 1]
            valid = sorted(token for token in set(generated_tokens_per_sequence[row_index]) if token < logits.shape[-1])
            if valid:
                token_ids = mx.array(valid, dtype=mx.int32)
                selected = mx.take(row, token_ids, axis=-1)
                penalized = mx.where(
                    selected < 0,
                    selected * repetition_penalty,
                    selected / repetition_penalty,
                )
                row = mx.put_along_axis(row, token_ids[None, :], penalized, axis=-1)
            rows.append(row)
        logits = mx.concatenate(rows, axis=0)

    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    eos_logits = None
    if eos_token_id is not None and eos_token_id < logits.shape[-1]:
        eos_logits = logits[:, eos_token_id : eos_token_id + 1]
    if 0 < top_k < logits.shape[-1]:
        logits = apply_top_k(logits, top_k)
    if 0.0 < top_p < 1.0:
        logits = apply_top_p(logits, top_p)
    if min_p > 0.0:
        logits = apply_min_p(logits, min_p)
    if eos_logits is not None:
        eos_indices = mx.full((batch_size, 1), eos_token_id, dtype=mx.int32)
        logits = mx.put_along_axis(logits, eos_indices, eos_logits, axis=-1)

    sampled = []
    for row_index in range(batch_size):
        key = mx.array(
            _row_random_key(seeds[row_index], counters[row_index]),
            dtype=mx.uint32,
        )
        counters[row_index] += 1
        token = mx.random.categorical(logits[row_index : row_index + 1] * (1 / temperature), key=key)
        sampled.append(token[:, None])
    return mx.concatenate(sampled, axis=0)


def _clear_generation_cache_if_due(mx: Any, step: int) -> None:
    """Mirror pinned Qwen's periodic release of evaluated graph intermediates."""
    if step > 0 and step % 50 == 0:
        mx.clear_cache()


def generate_qwen_icl_batch(
    model: Any,
    texts: Sequence[str],
    conditioning: QwenReferenceConditioning,
    *,
    language: str,
    seeds: Sequence[int],
    temperature: float = 0.9,
    max_tokens: int = 4096,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.5,
) -> list[tuple[int, Any, int]]:
    """Generate two independently checkpointable cloned-voice results.

    This is the pinned 0.4.1 batch loop adapted to the shared ICL reference
    contract merged upstream in PR #689.  Sampling uses explicit per-row keys,
    so a row's output never depends on the other row's seed or completion time.
    """
    import mlx.core as mx

    if not 1 <= len(texts) <= MAX_QWEN_CLONE_BATCH_SIZE:
        raise ValueError(f"Qwen clone batch size must be between 1 and {MAX_QWEN_CLONE_BATCH_SIZE}")
    if len(seeds) != len(texts):
        raise ValueError("Qwen clone batch requires one seed per text")
    if not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds):
        raise TypeError("Qwen clone batch seeds must be integers")
    if not has_qwen_icl_cache_backport(model):
        raise RuntimeError("Qwen ICL optimization backport is not installed")

    batch_size = len(texts)
    input_embeds, trailing_text_hidden, tts_pad_embed, attention_mask = _prepare_batch_inputs(
        model, texts, conditioning, language
    )
    mx.eval(input_embeds, trailing_text_hidden, tts_pad_embed, attention_mask)
    if batch_size == 1:
        attention_mask = None

    cache = model.talker.make_cache()
    code_cache = model.talker.code_predictor.make_cache()
    generated_codes: list[list[Any]] = [[] for _ in range(batch_size)]
    generated_token_ids: list[list[int]] = [[] for _ in range(batch_size)]
    finished = mx.zeros((batch_size,), dtype=mx.bool_)
    trailing_indices = mx.zeros((batch_size, 1), dtype=mx.int32)
    batch_indices = mx.arange(batch_size)
    max_trailing_len = trailing_text_hidden.shape[1]
    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    eos_fill = mx.full((batch_size, 1), eos_token_id, dtype=mx.int32)
    suppress_tokens = [token for token in range(config.vocab_size - 1024, config.vocab_size) if token != eos_token_id]
    per_sequence_max_tokens = [min(max_tokens, max(75, len(model.tokenizer.encode(text)) * 6)) for text in texts]
    random_counters = [0] * batch_size

    for _step in range(max(per_sequence_max_tokens)):
        logits, hidden = model.talker(input_embeds, cache=cache, attention_mask=attention_mask)
        sampled_tokens = _sample_rows(
            model,
            logits,
            seeds=seeds,
            counters=random_counters,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=max(repetition_penalty, 1.5),
            generated_tokens_per_sequence=generated_token_ids,
            suppress_tokens=suppress_tokens,
            eos_token_id=eos_token_id,
        )
        next_tokens = mx.where(finished[:, None], eos_fill, sampled_tokens)
        finished = finished | (next_tokens[:, 0] == eos_token_id)

        code_tokens = [next_tokens]
        code_hidden = hidden[:, -1:, :]
        for code_layer_cache in code_cache:
            code_layer_cache.keys = None
            code_layer_cache.values = None
            code_layer_cache.offset = 0
        for code_index in range(config.num_code_groups - 1):
            if code_index == 0:
                first_code_embed = model.talker.get_input_embeddings()(next_tokens)
                code_input = mx.concatenate([code_hidden, first_code_embed], axis=1)
            else:
                code_input = model.talker.code_predictor.codec_embedding[code_index - 1](code_tokens[-1])
            code_logits, code_cache, _ = model.talker.code_predictor(
                code_input,
                cache=code_cache,
                generation_step=code_index,
            )
            next_code = _sample_rows(
                model,
                code_logits,
                seeds=seeds,
                counters=random_counters,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            code_tokens.append(next_code)

        all_codes = mx.concatenate(code_tokens, axis=1)
        clamped_indices = mx.minimum(trailing_indices[:, 0], max_trailing_len - 1)
        text_embeds = trailing_text_hidden[batch_indices, clamped_indices, :][:, None, :]
        exhausted = clamped_indices >= max_trailing_len - 1
        text_embeds = mx.where(
            exhausted[:, None, None],
            mx.broadcast_to(tts_pad_embed, text_embeds.shape),
            text_embeds,
        )
        trailing_indices = trailing_indices + (~finished).astype(mx.int32)[:, None]
        codec_embed = model.talker.get_input_embeddings()(next_tokens)
        for index, code in enumerate(code_tokens[1:]):
            codec_embed = codec_embed + model.talker.code_predictor.codec_embedding[index](code)
        input_embeds = text_embeds + codec_embed
        mx.eval(all_codes, input_embeds, finished)

        finished_rows = finished.tolist()
        sampled_ids = next_tokens[:, 0].tolist()
        for row_index in range(batch_size):
            if not finished_rows[row_index]:
                generated_token_ids[row_index].append(sampled_ids[row_index])
                generated_codes[row_index].append(all_codes[row_index : row_index + 1])
                if len(generated_codes[row_index]) >= per_sequence_max_tokens[row_index]:
                    finished_rows[row_index] = True
        finished = mx.array(finished_rows, dtype=mx.bool_)
        if all(finished_rows):
            break
        if attention_mask is not None:
            attention_mask = mx.concatenate(
                [attention_mask, mx.ones((batch_size, 1), dtype=attention_mask.dtype)],
                axis=1,
            )
        # Retains live KV/reference tensors while releasing evaluated graph
        # intermediates, matching the pinned serial and preset-batch loops.
        _clear_generation_cache_if_due(mx, _step)

    del cache, code_cache, input_embeds, trailing_text_hidden, tts_pad_embed
    results: list[tuple[int, Any, int]] = []
    reference_codes = mx.transpose(conditioning.ref_codes, (0, 2, 1))
    reference_len = conditioning.ref_codes.shape[2]
    for row_index, row_codes in enumerate(generated_codes):
        if not row_codes:
            raise RuntimeError(f"Qwen clone batch produced no codec tokens for row {row_index}")
        generated = mx.stack(row_codes, axis=1)
        full_codes = mx.concatenate([reference_codes, generated], axis=1)
        audio, audio_lengths = model.speech_tokenizer.decode(full_codes)
        audio = audio[0]
        valid_len = int(audio_lengths[0])
        if 0 < valid_len < audio.shape[0]:
            audio = audio[:valid_len]
        cut = int(reference_len / max(full_codes.shape[1], 1) * audio.shape[0])
        if 0 < cut < audio.shape[0]:
            audio = audio[cut:]
        mx.eval(audio)
        results.append((row_index, audio, model.sample_rate))

    mx.clear_cache()
    return results
