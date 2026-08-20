"""Model-level batching contract for long-form TTS chunks."""

import asyncio

import numpy as np
import pytest

from backend.utils import chunked_tts


class _BatchBackend:
    tts_operations_are_cancellable = True

    def __init__(self):
        self.batch_calls = []
        self.single_calls = []

    async def generate_batch(
        self,
        texts,
        _voice_prompt,
        language="en",
        seeds=None,
        instruct=None,
    ):
        self.batch_calls.append((texts, language, seeds, instruct))
        return [(np.full(2, index + 1, dtype=np.float32), 24_000) for index, _text in enumerate(texts)]

    async def generate(
        self,
        text,
        _voice_prompt,
        language="en",
        seed=None,
        instruct=None,
    ):
        self.single_calls.append((text, language, seed, instruct))
        return np.full(2, 9, dtype=np.float32), 24_000


@pytest.mark.asyncio
async def test_generate_text_batch_preserves_order_and_per_unit_seeds():
    backend = _BatchBackend()

    results = await chunked_tts.generate_text_batch(
        backend,
        ["first", "second"],
        {"ref_audio": "voice.wav"},
        language="es",
        seeds=[100, 101],
        instruct="steady",
    )

    assert [audio.tolist() for audio, _sample_rate in results] == [[1, 1], [2, 2]]
    assert [sample_rate for _audio, sample_rate in results] == [24_000, 24_000]
    assert backend.batch_calls == [(["first", "second"], "es", [100, 101], "steady")]
    assert backend.single_calls == []


@pytest.mark.asyncio
async def test_not_implemented_batch_falls_back_to_serial_generation():
    class Backend(_BatchBackend):
        async def generate_batch(self, *_args, **_kwargs):
            raise NotImplementedError

    backend = Backend()
    results = await chunked_tts.generate_text_batch(
        backend,
        ["first", "second"],
        {},
        seeds=[20, 21],
    )

    assert len(results) == 2
    assert [call[2] for call in backend.single_calls] == [20, 21]


@pytest.mark.asyncio
async def test_batch_inference_error_is_not_hidden_by_serial_fallback():
    class Backend(_BatchBackend):
        async def generate_batch(self, *_args, **_kwargs):
            raise RuntimeError("Metal allocation failed")

    backend = Backend()
    with pytest.raises(RuntimeError, match="Metal allocation failed"):
        await chunked_tts.generate_text_batch(
            backend,
            ["first", "second"],
            {},
            seeds=[1, 2],
        )

    assert backend.single_calls == []


@pytest.mark.asyncio
async def test_batch_result_must_be_complete_and_in_input_order():
    class Backend(_BatchBackend):
        async def generate_batch(self, *_args, **_kwargs):
            return [(np.ones(2, dtype=np.float32), 24_000)]

    with pytest.raises(RuntimeError, match="returned 1 result for 2 texts"):
        await chunked_tts.generate_text_batch(
            Backend(),
            ["first", "second"],
            {},
            seeds=[1, 2],
        )


@pytest.mark.asyncio
async def test_batch_cancellation_propagates_without_serial_fallback():
    started = asyncio.Event()

    class Backend(_BatchBackend):
        async def generate_batch(self, *_args, **_kwargs):
            started.set()
            await asyncio.Event().wait()

    backend = Backend()
    task = asyncio.create_task(
        chunked_tts.generate_text_batch(
            backend,
            ["first", "second"],
            {},
            seeds=[1, 2],
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.single_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [[], ["one"], ["one", "two", "three"]])
async def test_batch_requires_exactly_two_units(texts):
    with pytest.raises(ValueError, match="exactly two texts"):
        await chunked_tts.generate_text_batch(
            _BatchBackend(),
            texts,
            {},
            seeds=list(range(len(texts))),
        )


@pytest.mark.asyncio
async def test_batch_requires_one_seed_per_text():
    with pytest.raises(ValueError, match="one seed per text"):
        await chunked_tts.generate_text_batch(
            _BatchBackend(),
            ["one", "two"],
            {},
            seeds=[1],
        )


@pytest.mark.asyncio
async def test_chunk_seed_derivation_stays_in_uint32_domain():
    backend = _BatchBackend()
    max_seed = (1 << 32) - 1

    await chunked_tts.generate_chunked(
        backend,
        ("a" * 100) + ("b" * 100),
        {},
        seed=max_seed,
        max_chunk_chars=100,
        crossfade_ms=0,
    )

    assert [call[2] for call in backend.single_calls] == [max_seed, 0]


@pytest.mark.asyncio
async def test_runaway_retry_keeps_crossfade_and_does_not_trim_twice(monkeypatch):
    class Backend(_BatchBackend):
        async def generate_batch(self, *_args, **_kwargs):
            runaway = np.array([99], dtype=np.float32)
            stable = np.array([2], dtype=np.float32)
            return [(runaway, 24_000), (stable, 24_000)]

    retry_calls = []
    trim_calls = []

    async def fake_generate_chunked(*_args, **kwargs):
        retry_calls.append(kwargs)
        return np.array([7], dtype=np.float32), 24_000

    def trim(audio, _sample_rate):
        trim_calls.append(audio.tolist())
        return audio

    monkeypatch.setattr(chunked_tts, "generate_chunked", fake_generate_chunked)

    results = await chunked_tts.generate_text_batch(
        Backend(),
        ["first", "second"],
        {},
        seeds=[1, 2],
        crossfade_ms=10,
        trim_fn=trim,
        runaway_detector=lambda audio, _sample_rate: bool(audio[0] == 99),
    )

    assert retry_calls[0]["crossfade_ms"] == 10
    assert retry_calls[0]["trim_fn"] is trim
    assert trim_calls == [[2.0]]
    assert [audio.tolist() for audio, _sample_rate in results] == [[7.0], [2.0]]
