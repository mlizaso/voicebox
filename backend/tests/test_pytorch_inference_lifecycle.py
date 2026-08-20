"""Process-wide lifecycle regressions for PyTorch STT and LLM work."""

from __future__ import annotations

import asyncio
import threading

import numpy as np
import pytest

from backend.backends import pytorch_backend
from backend.backends.mlx_tts_lifecycle import (
    MLXTTSLifecycleBusyError,
    mlx_tts_lifecycle_guard,
)
from backend.backends.pytorch_backend import PyTorchSTTBackend
from backend.backends.qwen_llm_backend import PyTorchQwenLLMBackend


async def _wait_for_thread_event(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0)


class _BlockingLLM(PyTorchQwenLLMBackend):
    def __init__(self) -> None:
        self.model = object()
        self.tokenizer = object()
        self.model_size = "0.6B"
        self._current_model_size = "0.6B"
        self.device = "cpu"
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()

    def _generate_sync(self, *_args, **_kwargs) -> str:
        self.entered.set()
        try:
            assert self.release.wait(timeout=3)
            return "done"
        finally:
            self.exited.set()


class _BlockingWhisperModel:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, *_args, **_kwargs):
        self.entered.set()
        assert self.release.wait(timeout=3)
        return [[1]]


class _WhisperInputs(dict):
    def to(self, _device: str):
        return self


class _WhisperProcessor:
    def __call__(self, *_args, **_kwargs):
        return _WhisperInputs(input_features=np.zeros((1, 1), dtype=np.float32))

    def batch_decode(self, *_args, **_kwargs):
        return ["transcribed"]


def _blocking_stt() -> tuple[PyTorchSTTBackend, _BlockingWhisperModel]:
    backend = PyTorchSTTBackend.__new__(PyTorchSTTBackend)
    model = _BlockingWhisperModel()
    backend.model = model
    backend.processor = _WhisperProcessor()
    backend.model_size = "base"
    backend.device = "cpu"
    return backend, model


@pytest.mark.asyncio
async def test_pytorch_stt_and_llm_inference_never_overlap(monkeypatch):
    monkeypatch.setattr(
        pytorch_backend,
        "load_audio",
        lambda *_args, **_kwargs: (np.zeros(16, dtype=np.float32), 16_000),
    )
    stt, whisper_model = _blocking_stt()
    llm = _BlockingLLM()

    transcription = asyncio.create_task(stt.transcribe("audio.wav"))
    await _wait_for_thread_event(whisper_model.entered)
    completion = asyncio.create_task(llm.generate("hello"))
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not llm.entered.is_set()
    finally:
        whisper_model.release.set()

    assert await asyncio.wait_for(transcription, timeout=1) == "transcribed"
    await _wait_for_thread_event(llm.entered)
    llm.release.set()
    assert await asyncio.wait_for(completion, timeout=1) == "done"


@pytest.mark.asyncio
async def test_cancelled_pytorch_inference_drains_before_releasing_guard():
    llm = _BlockingLLM()
    operation = asyncio.create_task(llm.generate("hello"))
    await _wait_for_thread_event(llm.entered)

    operation.cancel()
    for _ in range(20):
        await asyncio.sleep(0)

    assert not operation.done()
    assert mlx_tts_lifecycle_guard.is_active()
    with pytest.raises(MLXTTSLifecycleBusyError):
        with mlx_tts_lifecycle_guard.try_hold("competing unload"):
            pass

    llm.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(operation, timeout=1)
    assert llm.exited.is_set()
    assert not mlx_tts_lifecycle_guard.is_active()
