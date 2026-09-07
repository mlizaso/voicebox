"""Regression coverage for runaway MLX Qwen TTS output."""

from unittest.mock import patch

import numpy as np
import pytest

from backend.backends import engine_needs_trim, engine_retries_runaway
from backend.utils.audio import has_tts_runaway
from backend.utils.chunked_tts import (
    SynthesisDurationLimitError,
    generate_chunked,
    is_disk_backed_audio,
    release_disk_backed_audio,
)

SAMPLE_RATE = 1000


def test_mlx_qwen_enables_runaway_retry_without_aggressive_trim():
    with patch("backend.backends.get_backend_type", return_value="mlx"):
        assert engine_needs_trim("qwen") is False
        assert engine_retries_runaway("qwen") is True


def test_pytorch_qwen_keeps_runaway_retry_disabled():
    with patch("backend.backends.get_backend_type", return_value="pytorch"):
        assert engine_needs_trim("qwen") is False
        assert engine_retries_runaway("qwen") is False


def test_detector_flags_long_internal_silence():
    speech = np.full(2 * SAMPLE_RATE, 0.2, dtype=np.float32)
    runaway_gap = np.zeros(2500, dtype=np.float32)
    hallucinated_noise = np.full(2 * SAMPLE_RATE, 0.8, dtype=np.float32)
    audio = np.concatenate([speech, runaway_gap, hallucinated_noise])

    assert has_tts_runaway(audio, SAMPLE_RATE) is True


def test_detector_ignores_normal_internal_pause():
    speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    normal_pause = np.zeros(1200, dtype=np.float32)
    audio = np.concatenate([speech, normal_pause, speech])

    assert has_tts_runaway(audio, SAMPLE_RATE) is False


def test_trailing_silence_is_not_a_runaway():
    speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
    trailing_silence = np.zeros(2 * SAMPLE_RATE, dtype=np.float32)

    assert (
        has_tts_runaway(
            np.concatenate([speech, trailing_silence]),
            SAMPLE_RATE,
        )
        is False
    )


@pytest.mark.asyncio
async def test_runaway_chunk_is_retried_as_smaller_chunks():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            if len(text) > 200:
                speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
                silence = np.zeros(2500, dtype=np.float32)
                noise = np.full(SAMPLE_RATE, 0.8, dtype=np.float32)
                return np.concatenate([speech, silence, noise]), SAMPLE_RATE
            return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE

    backend = FakeBackend()
    text = f"{'A' * 119}. {'B' * 119}."

    audio, sample_rate = await generate_chunked(
        backend,
        text,
        {},
        max_chunk_chars=800,
        crossfade_ms=50,
        runaway_detector=has_tts_runaway,
    )

    try:
        assert sample_rate == SAMPLE_RATE
        assert backend.calls == [text, f"{'A' * 119}.", f"{'B' * 119}."]
        assert len(audio) == 1950
        assert is_disk_backed_audio(audio)
    finally:
        release_disk_backed_audio(audio)


@pytest.mark.asyncio
async def test_persistent_runaway_fails_instead_of_returning_corrupt_audio():
    class AlwaysRunawayBackend:
        def __init__(self):
            self.calls = []

        async def generate(self, text, *_args):
            self.calls.append(text)
            speech = np.full(SAMPLE_RATE, 0.2, dtype=np.float32)
            silence = np.zeros(2500, dtype=np.float32)
            noise = np.full(SAMPLE_RATE, 0.8, dtype=np.float32)
            return np.concatenate([speech, silence, noise]), SAMPLE_RATE

    backend = AlwaysRunawayBackend()
    text = f"{'A' * 119}. {'B' * 119}."

    with pytest.raises(
        RuntimeError,
        match="remained unstable after retrying smaller text chunks",
    ):
        await generate_chunked(
            backend,
            text,
            {},
            max_chunk_chars=800,
            runaway_detector=has_tts_runaway,
        )

    assert [len(call) for call in backend.calls] == [241, 120, 100]


@pytest.mark.asyncio
async def test_duration_limit_recovers_without_detector_and_preserves_text_and_seeds():
    class LimitedBackend:
        max_input_chars = 200

        def __init__(self):
            self.calls = []

        async def generate(self, text, _prompt, _language, seed, _instruct):
            self.calls.append((text, seed))
            if len(text) > 100:
                raise SynthesisDurationLimitError("duration limit")
            return np.full(SAMPLE_RATE, 0.2, dtype=np.float32), SAMPLE_RATE

    backend = LimitedBackend()
    first, second = "Una frase breve. " * 5, "Otra frase clara. " * 5
    text = first + second
    for _ in range(2):
        audio, rate = await generate_chunked(backend, text, {}, seed=42, crossfade_ms=0)
        try:
            assert rate == SAMPLE_RATE
            assert len(audio) == 2 * SAMPLE_RATE
            np.testing.assert_array_equal(audio, np.full(2 * SAMPLE_RATE, 0.2, dtype=np.float32))
        finally:
            release_disk_backed_audio(audio)
    expected_calls = [(text, 42), (text, 10042), (text, 20042), (first.strip(), 1042), (second.strip(), 1043)]
    assert backend.calls == expected_calls * 2
    assert " ".join(chunk for chunk, _seed in expected_calls[3:]).split() == text.split()


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [80, 160, 700])
async def test_duration_limit_recovery_is_bounded_and_retains_real_error(length):
    calls = []

    class AlwaysLimitedBackend:
        async def generate(self, text, *_args):
            calls.append(text)
            raise SynthesisDurationLimitError("Fine-tuned model reached its duration limit before finishing")

    with pytest.raises(SynthesisDurationLimitError, match="duration limit before finishing"):
        await generate_chunked(AlwaysLimitedBackend(), "a " * (length // 2), {}, seed=7)
    assert len(calls) <= 9
    assert all(calls[index : index + 3] == [calls[index]] * 3 for index in range(0, len(calls), 3))
    assert all(len(child) < len(parent) for parent, child in zip(calls[::3], calls[3::3], strict=False))


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [20261305, 2**32 - 1])
@pytest.mark.parametrize("success_offset", [0, 10000, 20000])
async def test_short_duration_limit_uses_fixed_seeds_and_leaves_successful_audio_unchanged(seed, success_offset):
    calls = []
    expected = np.arange(1000, dtype=np.float32) / 1000

    class SeedLimitedBackend:
        async def generate(self, text, _prompt, _language, attempt_seed, _instruct):
            calls.append((text, attempt_seed))
            if attempt_seed != (seed + success_offset) % 2**32:
                raise SynthesisDurationLimitError("duration limit")
            return expected, SAMPLE_RATE

    text = "Porque entonces, los oyes."
    for _ in range(2):
        audio, rate = await generate_chunked(SeedLimitedBackend(), text, {}, seed=seed)
        assert rate == SAMPLE_RATE
        assert audio is expected
    offsets = [offset for offset in (0, 10000, 20000) if offset <= success_offset]
    assert calls == [(text, (seed + offset) % 2**32) for offset in offsets] * 2


@pytest.mark.asyncio
async def test_infrastructure_error_is_not_retried_as_duration_limit():
    calls = []

    class BrokenBackend:
        async def generate(self, text, *_args):
            calls.append(text)
            raise RuntimeError("model load failed")

    text = "Una frase sin problema. " * 8
    with pytest.raises(RuntimeError, match="model load failed"):
        await generate_chunked(BrokenBackend(), text, {}, seed=7)
    assert calls == [text]
