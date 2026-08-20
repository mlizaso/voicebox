"""Concurrency regressions for the process-wide MLX TTS lifecycle guard."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi import HTTPException

import backend.backends as backends
import backend.backends.mlx_backend as mlx_backend
from backend.backends.mlx_backend import MLXSTTBackend, MLXTTSBackend
from backend.backends.mlx_tts_lifecycle import (
    MLXTTSLifecycleGuard,
    loaded_tts_backend_for_request,
    mlx_tts_lifecycle_guard,
    run_tts_operation_cancellation_safe,
)
from backend.backends.qwen_llm_backend import MLXQwenLLMBackend
from backend.routes import (
    llm as llm_routes,
    models as model_routes,
    tasks as task_routes,
    transcription as transcription_routes,
)
from backend.services import llm, task_queue, transcribe, tts
from backend.utils.progress import ProgressManager
from backend.utils.tasks import TaskManager


class _BlockingSerialModel:
    def __init__(self, blocked_text: str = "first") -> None:
        self.blocked_text = blocked_text
        self.entered = {
            "first": threading.Event(),
            "second": threading.Event(),
            "stream": threading.Event(),
            "queued": threading.Event(),
        }
        self.release = threading.Event()
        self._state_lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def generate(self, text: str, *, lang_code: str):
        del lang_code
        with self._state_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        self.entered.setdefault(text, threading.Event()).set()
        try:
            if text == self.blocked_text:
                assert self.release.wait(timeout=3)
            yield SimpleNamespace(
                audio=np.array([0.1, 0.2], dtype=np.float32),
                sample_rate=24_000,
            )
        finally:
            with self._state_lock:
                self._active -= 1


class _SizedSerialModel:
    def __init__(self, model_size: str, calls: list[tuple[str, str]]) -> None:
        self.model_size = model_size
        self.calls = calls

    def generate(self, text: str, *, lang_code: str):
        del lang_code
        self.calls.append((text, self.model_size))
        yield SimpleNamespace(
            audio=np.array([0.1, 0.2], dtype=np.float32),
            sample_rate=24_000,
        )


class _BlockingAsyncTTSBackend:
    def __init__(self) -> None:
        self.model = object()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.unload_calls = 0

    def is_loaded(self) -> bool:
        return self.model is not None

    async def load_model(self) -> None:
        if self.model is None:
            self.model = object()

    async def generate(self, *_args, **_kwargs):
        def generate_sync():
            self.entered.set()
            assert self.release.wait(timeout=3)
            if self.model is None:
                raise RuntimeError("model was unloaded during inference")
            return np.array([0.1], dtype=np.float32), 24_000

        return await asyncio.to_thread(generate_sync)

    def unload_model(self) -> None:
        self.unload_calls += 1
        self.model = None


class _BlockingModelDownloadBackend:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.exited = threading.Event()

    async def load_model(self) -> None:
        def load_sync() -> None:
            self.entered.set()
            try:
                assert self.release.wait(timeout=3)
            finally:
                self.exited.set()

        await asyncio.to_thread(load_sync)


class _BlockingEndpointLoaderBackend(_BlockingModelDownloadBackend):
    def __init__(self, model_size: str) -> None:
        super().__init__()
        self.model_size = model_size

    def is_loaded(self) -> bool:
        return False

    def _is_model_cached(self, _model_size: str) -> bool:
        return False

    async def load_model(self, _model_size: str | None = None) -> None:
        del _model_size
        await super().load_model()

    async def load_model_async(self, _model_size: str) -> None:
        await self.load_model()


def _loaded_backend(model: object) -> MLXTTSBackend:
    backend = MLXTTSBackend()
    backend.model = model
    backend._current_model_size = "1.7B"
    backend.model_size = "1.7B"
    return backend


async def _wait_for_thread_event(event: threading.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0)


def _capture_background_tasks(monkeypatch, route_module) -> list[asyncio.Task]:
    tasks: list[asyncio.Task] = []

    def create_task(coroutine):
        task = asyncio.create_task(coroutine)
        tasks.append(task)
        return task

    monkeypatch.setattr(route_module, "create_background_task", create_task)
    return tasks


def test_direct_streams_on_different_event_loop_threads_never_overlap():
    model = _BlockingSerialModel()
    backend = _loaded_backend(model)
    second_scheduled = threading.Event()
    errors: list[BaseException] = []

    def run(text: str) -> None:
        async def invoke() -> None:
            if text == "second":
                second_scheduled.set()
            await backend.generate(text, {})

        try:
            asyncio.run(invoke())
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=run, args=("first",), daemon=True)
    second = threading.Thread(target=run, args=("second",), daemon=True)
    first.start()
    assert model.entered["first"].wait(timeout=1)
    second.start()
    assert second_scheduled.wait(timeout=1)
    try:
        assert not model.entered["second"].wait(timeout=0.1)
        assert model.max_active == 1
    finally:
        model.release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors
    assert model.entered["second"].is_set()
    assert model.max_active == 1


@pytest.mark.asyncio
async def test_direct_stream_and_queued_batch_never_overlap(monkeypatch):
    model = _BlockingSerialModel(blocked_text="stream")
    backend = _loaded_backend(model)
    batch_attempted = asyncio.Event()
    batch_entered = threading.Event()
    batch_finished = asyncio.Event()
    monkeypatch.setattr(backend, "_prepare_reference_sync", lambda _prompt: object())

    def generate_batch(_model, _texts, _conditioning, **_kwargs):
        batch_entered.set()
        return [
            (0, np.array([0.3], dtype=np.float32), 24_000),
            (1, np.array([0.4], dtype=np.float32), 24_000),
        ]

    monkeypatch.setattr(mlx_backend, "generate_qwen_icl_batch", generate_batch)
    task_queue.init_queue(force=True)

    direct = asyncio.create_task(backend.generate("stream", {}))
    await _wait_for_thread_event(model.entered["stream"])

    async def queued_batch() -> None:
        batch_attempted.set()
        await backend.generate_batch(
            ["queued-a", "queued-b"],
            {"ref_audio": "unused.wav", "ref_text": "reference"},
            seeds=[100, 101],
        )
        batch_finished.set()

    task_queue.enqueue_generation_batch(("lifecycle-a", "lifecycle-b"), queued_batch())
    await asyncio.wait_for(batch_attempted.wait(), timeout=1)
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not batch_entered.is_set()
    finally:
        model.release.set()

    await asyncio.wait_for(direct, timeout=2)
    await asyncio.wait_for(batch_finished.wait(), timeout=2)
    assert batch_entered.is_set()


@pytest.mark.asyncio
async def test_cancelled_inference_drains_thread_before_guard_allows_next_stream():
    model = _BlockingSerialModel()
    backend = _loaded_backend(model)

    first = asyncio.create_task(backend.generate("first", {}))
    await _wait_for_thread_event(model.entered["first"])
    second = asyncio.create_task(backend.generate("second", {}))
    first.cancel()
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not model.entered["second"].is_set()
        assert mlx_tts_lifecycle_guard.is_active()
    finally:
        model.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(first, timeout=2)
    await asyncio.wait_for(second, timeout=2)
    assert model.entered["second"].is_set()
    assert model.max_active == 1


@pytest.mark.asyncio
async def test_cancelled_guard_waiter_never_orphans_a_later_acquisition():
    guard = MLXTTSLifecycleGuard()
    waiter_started = asyncio.Event()

    async def waiter() -> None:
        waiter_started.set()
        async with guard.hold("waiter"):
            raise AssertionError("cancelled waiter must not acquire")

    async with guard.hold("holder"):
        task = asyncio.create_task(waiter())
        await waiter_started.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    with guard.try_hold("follow-up"):
        assert guard.is_active()
    assert not guard.is_active()


@pytest.mark.asyncio
async def test_complete_request_keeps_model_size_across_interleaved_chunks(
    monkeypatch,
):
    backend = MLXTTSBackend()
    load_calls: list[str] = []
    inference_calls: list[tuple[str, str]] = []
    first_chunk_done = asyncio.Event()
    second_request_attempted = asyncio.Event()
    continue_first_request = asyncio.Event()

    def load_model(model_size: str) -> None:
        load_calls.append(model_size)
        backend.model = _SizedSerialModel(model_size, inference_calls)
        backend._current_model_size = model_size
        backend.model_size = model_size

    monkeypatch.setattr(backend, "_load_model_sync", load_model)
    monkeypatch.setitem(backends._tts_backends, "qwen", backend)

    async def request_1_7b() -> None:
        async with loaded_tts_backend_for_request(
            "qwen",
            "1.7B",
        ) as request_backend:
            await request_backend.generate("first-chunk", {})
            first_chunk_done.set()
            await second_request_attempted.wait()
            await continue_first_request.wait()
            await request_backend.generate("second-chunk", {})

    async def request_0_6b() -> None:
        await first_chunk_done.wait()
        second_request_attempted.set()
        async with loaded_tts_backend_for_request(
            "qwen",
            "0.6B",
        ) as request_backend:
            await request_backend.generate("other-request", {})

    first = asyncio.create_task(request_1_7b())
    second = asyncio.create_task(request_0_6b())
    await asyncio.wait_for(second_request_attempted.wait(), timeout=1)
    for _ in range(20):
        await asyncio.sleep(0)

    assert load_calls == ["1.7B"]
    assert backend._current_model_size == "1.7B"
    continue_first_request.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=2)

    assert load_calls == ["1.7B", "0.6B"]
    assert inference_calls == [
        ("first-chunk", "1.7B"),
        ("second-chunk", "1.7B"),
        ("other-request", "0.6B"),
    ]


@pytest.mark.asyncio
async def test_non_mlx_request_refuses_unload_until_inference_finishes(
    monkeypatch,
):
    backend = _BlockingAsyncTTSBackend()
    monkeypatch.setitem(backends._tts_backends, "luxtts", backend)

    async def request() -> None:
        async with loaded_tts_backend_for_request(
            "luxtts",
            "default",
        ) as request_backend:
            await run_tts_operation_cancellation_safe(
                request_backend,
                request_backend.generate("text", {}),
            )

    active = asyncio.create_task(request())
    await _wait_for_thread_event(backend.entered)
    try:
        with pytest.raises(HTTPException) as raised:
            await model_routes.unload_model_by_name("luxtts")
        assert raised.value.status_code == 409
        assert backend.unload_calls == 0
        assert backend.model is not None
    finally:
        backend.release.set()
    await asyncio.wait_for(active, timeout=2)


@pytest.mark.asyncio
async def test_cancelled_non_mlx_request_drains_executor_before_guard_release(
    monkeypatch,
):
    backend = _BlockingAsyncTTSBackend()
    monkeypatch.setitem(backends._tts_backends, "luxtts", backend)

    async def request() -> None:
        async with loaded_tts_backend_for_request(
            "luxtts",
            "default",
        ) as request_backend:
            await run_tts_operation_cancellation_safe(
                request_backend,
                request_backend.generate("text", {}),
            )

    active = asyncio.create_task(request())
    await _wait_for_thread_event(backend.entered)
    active.cancel()
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert mlx_tts_lifecycle_guard.is_active()
        with pytest.raises(HTTPException) as raised:
            await model_routes.unload_model_by_name("luxtts")
        assert raised.value.status_code == 409
        assert backend.unload_calls == 0
    finally:
        backend.release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(active, timeout=2)


@pytest.mark.asyncio
async def test_lux_request_and_mlx_request_never_overlap(monkeypatch):
    lux_backend = _BlockingAsyncTTSBackend()
    mlx_backend_instance = MLXTTSBackend()
    mlx_loads: list[str] = []

    def load_mlx(model_size: str) -> None:
        mlx_loads.append(model_size)
        mlx_backend_instance.model = _SizedSerialModel(model_size, [])
        mlx_backend_instance._current_model_size = model_size
        mlx_backend_instance.model_size = model_size

    monkeypatch.setattr(mlx_backend_instance, "_load_model_sync", load_mlx)
    monkeypatch.setitem(backends._tts_backends, "luxtts", lux_backend)
    monkeypatch.setitem(backends._tts_backends, "qwen", mlx_backend_instance)

    async def lux_request() -> None:
        async with loaded_tts_backend_for_request("luxtts") as request_backend:
            await run_tts_operation_cancellation_safe(
                request_backend,
                request_backend.generate("lux", {}),
            )

    async def mlx_request() -> None:
        async with loaded_tts_backend_for_request(
            "qwen",
            "1.7B",
        ) as request_backend:
            await request_backend.generate("mlx", {})

    lux = asyncio.create_task(lux_request())
    await _wait_for_thread_event(lux_backend.entered)
    mlx = asyncio.create_task(mlx_request())
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert mlx_loads == []
    finally:
        lux_backend.release.set()
    await asyncio.wait_for(asyncio.gather(lux, mlx), timeout=2)
    assert mlx_loads == ["1.7B"]


@pytest.mark.asyncio
async def test_seeded_serial_tts_and_mlx_llm_never_overlap(monkeypatch):
    fake_mlx = ModuleType("mlx")
    fake_core = ModuleType("mlx.core")
    fake_core.random = SimpleNamespace(seed=lambda _seed: None)
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    tts_model = _BlockingSerialModel()
    tts_backend = _loaded_backend(tts_model)
    llm_backend = MLXQwenLLMBackend("0.6B")
    llm_backend.model = object()
    llm_backend.tokenizer = object()
    llm_backend._current_model_size = "0.6B"
    llm_entered = threading.Event()

    def generate_llm(*_args, **_kwargs) -> str:
        llm_entered.set()
        return "answer"

    llm_backend._generate_sync = generate_llm
    tts = asyncio.create_task(tts_backend.generate("first", {}, seed=42))
    await _wait_for_thread_event(tts_model.entered["first"])
    llm = asyncio.create_task(llm_backend.generate("prompt", temperature=0.2))
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not llm_entered.is_set()
    finally:
        tts_model.release.set()
    await asyncio.wait_for(asyncio.gather(tts, llm), timeout=2)
    assert llm_entered.is_set()


@pytest.mark.asyncio
async def test_seeded_serial_tts_and_mlx_stt_fallback_never_overlap(monkeypatch):
    fake_mlx = ModuleType("mlx")
    fake_core = ModuleType("mlx.core")
    fake_core.random = SimpleNamespace(seed=lambda _seed: None)
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    tts_model = _BlockingSerialModel()
    tts_backend = _loaded_backend(tts_model)
    stt_backend = MLXSTTBackend("base")
    stt_entered = threading.Event()

    class STTModel:
        def generate(self, _audio_path: str, **_options):
            stt_entered.set()
            return "transcript"

    stt_backend.model = STTModel()
    tts = asyncio.create_task(tts_backend.generate("first", {}, seed=42))
    await _wait_for_thread_event(tts_model.entered["first"])
    stt = asyncio.create_task(stt_backend.transcribe("audio.wav"))
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not stt_entered.is_set()
    finally:
        tts_model.release.set()
    await asyncio.wait_for(asyncio.gather(tts, stt), timeout=2)
    assert stt_entered.is_set()


@pytest.mark.asyncio
async def test_llm_endpoint_background_loader_drains_on_cancellation(monkeypatch):
    backend = _BlockingEndpointLoaderBackend("0.6B")
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    tasks = _capture_background_tasks(monkeypatch, llm_routes)
    monkeypatch.setattr(llm, "get_llm_model", lambda: backend)
    monkeypatch.setattr(
        llm_routes,
        "get_llm_model_configs",
        lambda: [SimpleNamespace(model_size="0.6B")],
    )
    monkeypatch.setattr(llm_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        llm_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        model_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "_model_download_tasks", {})

    response = await llm_routes.llm_generate(
        llm_routes.models.LLMGenerateRequest(
            prompt="hello",
            model_size="0.6B",
        )
    )
    assert response.status_code == 202
    assert len(tasks) == 1
    await _wait_for_thread_event(backend.entered)

    duplicate = await llm_routes.llm_generate(
        llm_routes.models.LLMGenerateRequest(
            prompt="hello again",
            model_size="0.6B",
        )
    )
    assert duplicate.status_code == 202
    assert len(tasks) == 1
    with pytest.raises(HTTPException) as duplicate_route:
        await model_routes.trigger_model_download(model_routes.models.ModelDownloadRequest(model_name="qwen3-0.6b"))
    assert duplicate_route.value.status_code == 409

    cancellation = asyncio.create_task(
        model_routes.cancel_model_download(model_routes.models.ModelDownloadRequest(model_name="qwen3-0.6b"))
    )
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not cancellation.done()
        assert not backend.exited.is_set()
        assert task_manager.is_download_active("qwen3-0.6b")
    finally:
        backend.release.set()

    response = await asyncio.wait_for(cancellation, timeout=2)
    assert response == {"message": "Download task for qwen3-0.6b cancelled"}
    assert backend.exited.is_set()
    assert not task_manager.is_download_active("qwen3-0.6b")


@pytest.mark.asyncio
async def test_transcription_endpoint_background_loader_drains_on_cancellation(
    monkeypatch,
):
    backend = _BlockingEndpointLoaderBackend("base")
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    tasks = _capture_background_tasks(monkeypatch, transcription_routes)
    monkeypatch.setattr(transcribe, "get_whisper_model", lambda: backend)
    monkeypatch.setattr(
        transcription_routes,
        "get_task_manager",
        lambda: task_manager,
    )
    monkeypatch.setattr(
        transcription_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        model_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "_model_download_tasks", {})
    monkeypatch.setattr(
        transcription_routes.librosa,
        "load",
        lambda _path, **_kwargs: (np.zeros(24, dtype=np.float32), 24_000),
    )

    class Upload:
        filename = "sample.wav"

        def __init__(self) -> None:
            self._read = False

        async def read(self, _size: int) -> bytes:
            if self._read:
                return b""
            self._read = True
            return b"not-a-real-wav"

    with pytest.raises(HTTPException) as raised:
        await transcription_routes.transcribe_audio(
            file=Upload(),
            language=None,
            model="base",
        )
    assert raised.value.status_code == 202
    assert len(tasks) == 1
    await _wait_for_thread_event(backend.entered)

    with pytest.raises(HTTPException) as duplicate:
        await transcription_routes.transcribe_audio(
            file=Upload(),
            language=None,
            model="base",
        )
    assert duplicate.value.status_code == 202
    assert len(tasks) == 1
    with pytest.raises(HTTPException) as duplicate_route:
        await model_routes.trigger_model_download(model_routes.models.ModelDownloadRequest(model_name="whisper-base"))
    assert duplicate_route.value.status_code == 409

    cancellation = asyncio.create_task(
        model_routes.cancel_model_download(model_routes.models.ModelDownloadRequest(model_name="whisper-base"))
    )
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not cancellation.done()
        assert not backend.exited.is_set()
        assert task_manager.is_download_active("whisper-base")
    finally:
        backend.release.set()

    response = await asyncio.wait_for(cancellation, timeout=2)
    assert response == {"message": "Download task for whisper-base cancelled"}
    assert backend.exited.is_set()
    assert not task_manager.is_download_active("whisper-base")


async def _run_model_download_background(monkeypatch, config, backend, load_func):
    tasks: list[asyncio.Task] = []
    task_manager = Mock()
    progress_manager = Mock()

    def create_task(coroutine):
        task = asyncio.create_task(coroutine)
        tasks.append(task)
        return task

    monkeypatch.setattr(backends, "get_model_config", lambda _name: config)
    monkeypatch.setattr(backends, "get_model_load_func", lambda _config: load_func)
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(model_routes, "get_progress_manager", lambda: progress_manager)
    monkeypatch.setattr(model_routes, "create_background_task", create_task)
    if config.engine == "whisper":
        monkeypatch.setattr(transcribe, "get_whisper_model", lambda: backend)
    else:
        monkeypatch.setattr(llm, "get_llm_model", lambda: backend)

    await model_routes.trigger_model_download(model_routes.models.ModelDownloadRequest(model_name=config.model_name))
    await asyncio.wait_for(tasks[0], timeout=2)
    task_manager.complete_download.assert_called_once_with(config.model_name)
    assert not mlx_tts_lifecycle_guard.is_active()


@pytest.mark.asyncio
async def test_model_download_cancel_drains_owned_task_and_rejects_duplicate(
    monkeypatch,
):
    backend = _BlockingModelDownloadBackend()
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    config = SimpleNamespace(
        model_name="luxtts",
        engine="luxtts",
        model_size="default",
    )
    monkeypatch.setattr(backends, "get_model_config", lambda _name: config)
    monkeypatch.setattr(
        backends,
        "get_model_load_func",
        lambda _config: backend.load_model,
    )
    monkeypatch.setattr(
        backends,
        "get_tts_backend_for_engine",
        lambda _engine: backend,
    )
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        model_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "_model_download_tasks", {})
    request = model_routes.models.ModelDownloadRequest(model_name=config.model_name)

    await model_routes.trigger_model_download(request)
    await _wait_for_thread_event(backend.entered)
    cancellation = asyncio.create_task(model_routes.cancel_model_download(request))
    try:
        for _ in range(20):
            await asyncio.sleep(0)

        assert not cancellation.done()
        assert model_routes._owned_model_download_task(config.model_name) is not None
        assert task_manager.is_download_active(config.model_name)
        assert progress_manager.get_progress(config.model_name) is not None
        assert mlx_tts_lifecycle_guard.is_active()

        with pytest.raises(HTTPException) as raised:
            await model_routes.trigger_model_download(request)
        assert raised.value.status_code == 409
        assert "already running or cancelling" in raised.value.detail
    finally:
        backend.release.set()

    response = await asyncio.wait_for(cancellation, timeout=2)
    assert response == {"message": "Download task for luxtts cancelled"}
    assert backend.exited.is_set()
    assert model_routes._owned_model_download_task(config.model_name) is None
    assert not task_manager.is_download_active(config.model_name)
    assert progress_manager.get_progress(config.model_name) is None
    assert not mlx_tts_lifecycle_guard.is_active()


@pytest.mark.asyncio
async def test_model_download_cancel_before_task_start_cleans_ownership(
    monkeypatch,
):
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    config = SimpleNamespace(
        model_name="luxtts",
        engine="luxtts",
        model_size="default",
    )
    load_called = False

    async def load_model() -> None:
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(backends, "get_model_config", lambda _name: config)
    monkeypatch.setattr(
        backends,
        "get_model_load_func",
        lambda _config: load_model,
    )
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        model_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "_model_download_tasks", {})
    request = model_routes.models.ModelDownloadRequest(model_name=config.model_name)

    await model_routes.trigger_model_download(request)
    response = await model_routes.cancel_model_download(request)

    assert response == {"message": "Download task for luxtts cancelled"}
    assert not load_called
    assert model_routes._owned_model_download_task(config.model_name) is None
    assert not task_manager.is_download_active(config.model_name)
    assert progress_manager.get_progress(config.model_name) is None
    assert not mlx_tts_lifecycle_guard.is_active()


@pytest.mark.asyncio
async def test_stale_cancelled_download_callback_cannot_clear_replacement(
    monkeypatch,
):
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    model_name = "luxtts"

    stale = asyncio.create_task(asyncio.sleep(1))
    stale.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stale
    replacement = asyncio.create_task(asyncio.sleep(1))
    monkeypatch.setattr(
        model_routes,
        "_model_download_tasks",
        {model_name: replacement},
    )
    task_manager.start_download(model_name)
    progress_manager.update_progress(model_name, 0, 0, status="downloading")

    model_routes._finish_model_download_task(
        model_name,
        stale,
        task_manager,
        progress_manager,
    )

    assert model_routes._owned_model_download_task(model_name) is replacement
    assert task_manager.is_download_active(model_name)
    assert progress_manager.get_progress(model_name) is not None
    replacement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await replacement


@pytest.mark.asyncio
async def test_clear_tasks_refuses_running_download_and_preserves_error_metadata(
    monkeypatch,
):
    backend = _BlockingModelDownloadBackend()
    task_manager = TaskManager()
    progress_manager = ProgressManager()
    config = SimpleNamespace(
        model_name="luxtts",
        engine="luxtts",
        model_size="default",
    )
    monkeypatch.setattr(backends, "get_model_config", lambda _name: config)
    monkeypatch.setattr(
        backends,
        "get_model_load_func",
        lambda _config: backend.load_model,
    )
    monkeypatch.setattr(
        backends,
        "get_tts_backend_for_engine",
        lambda _engine: backend,
    )
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        model_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(task_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        task_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "_model_download_tasks", {})
    task_manager.start_download("failed-model")
    task_manager.error_download("failed-model", "network failed")
    task_manager.start_generation("active-generation", "profile", "text")
    progress_manager.update_progress("failed-model", 0, 1, status="error")
    request = model_routes.models.ModelDownloadRequest(model_name=config.model_name)

    await model_routes.trigger_model_download(request)
    await _wait_for_thread_event(backend.entered)
    try:
        with pytest.raises(HTTPException) as raised:
            await task_routes.clear_all_tasks()
        assert raised.value.status_code == 409
        assert config.model_name in raised.value.detail
        assert task_manager.is_download_active(config.model_name)
        assert task_manager.is_download_active("failed-model")
        assert progress_manager.get_progress(config.model_name) is not None
        assert progress_manager.get_progress("failed-model") is not None
        assert task_manager.is_generation_active("active-generation")
    finally:
        backend.release.set()

    owned_task = model_routes._owned_model_download_task(config.model_name)
    assert owned_task is not None
    await asyncio.wait_for(owned_task, timeout=2)
    assert await task_routes.clear_all_tasks() == {"message": "All download task state cleared"}
    assert task_manager.get_active_downloads() == []
    assert task_manager.is_generation_active("active-generation")
    assert progress_manager.get_progress(config.model_name) is None
    assert progress_manager.get_progress("failed-model") is None


@pytest.mark.asyncio
async def test_model_migration_cancel_drains_move_and_rejects_duplicate(
    monkeypatch,
    tmp_path,
):
    from huggingface_hub import constants as hf_constants

    source = tmp_path / "cache"
    (source / "models--voice").mkdir(parents=True)
    destination = tmp_path / "destination"
    progress_manager = ProgressManager()
    task_manager = TaskManager()
    move_entered = threading.Event()
    move_release = threading.Event()
    move_exited = threading.Event()

    def blocking_move(_source, _destination):
        move_entered.set()
        try:
            assert move_release.wait(timeout=3)
        finally:
            move_exited.set()

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(source))
    monkeypatch.setattr(model_routes, "get_progress_manager", lambda: progress_manager)
    monkeypatch.setattr(task_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(
        task_routes,
        "get_progress_manager",
        lambda: progress_manager,
    )
    monkeypatch.setattr(model_routes, "_model_migration_task", None)
    monkeypatch.setattr(
        model_routes,
        "_move_model_cache_directory",
        blocking_move,
    )
    request = model_routes.models.ModelMigrateRequest(destination=str(destination))

    await model_routes.migrate_models(request)
    migration = model_routes._owned_model_migration_task()
    assert migration is not None
    await _wait_for_thread_event(move_entered)
    migration.cancel()
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not migration.done()
        assert not move_exited.is_set()
        assert mlx_tts_lifecycle_guard.is_active()

        with pytest.raises(HTTPException) as clear_error:
            await task_routes.clear_all_tasks()
        assert clear_error.value.status_code == 409
        assert "model-cache-migration" in clear_error.value.detail
        assert progress_manager.get_progress("migration") is not None

        with pytest.raises(HTTPException) as raised:
            await model_routes.migrate_models(request)
        assert raised.value.status_code == 409
        assert "already running or cancelling" in raised.value.detail
    finally:
        move_release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(migration, timeout=2)
    await asyncio.sleep(0)
    assert move_exited.is_set()
    assert model_routes._owned_model_migration_task() is None
    assert not mlx_tts_lifecycle_guard.is_active()
    assert progress_manager.get_progress("migration")["status"] == "error"


@pytest.mark.asyncio
async def test_model_migration_and_model_download_never_overlap(
    monkeypatch,
    tmp_path,
):
    from huggingface_hub import constants as hf_constants

    source = tmp_path / "cache"
    (source / "models--voice").mkdir(parents=True)
    destination = tmp_path / "destination"
    progress_manager = ProgressManager()
    task_manager = TaskManager()
    migration_entered = threading.Event()
    migration_release = threading.Event()
    migration_exited = threading.Event()
    download_entered = asyncio.Event()

    def blocking_move(_source, _destination):
        migration_entered.set()
        assert migration_release.wait(timeout=3)
        migration_exited.set()

    class DownloadBackend:
        async def load_model(self) -> None:
            assert migration_exited.is_set()
            download_entered.set()

    download_backend = DownloadBackend()
    download_config = SimpleNamespace(
        model_name="luxtts",
        engine="luxtts",
        model_size="default",
    )
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(source))
    monkeypatch.setattr(model_routes, "get_progress_manager", lambda: progress_manager)
    monkeypatch.setattr(model_routes, "get_task_manager", lambda: task_manager)
    monkeypatch.setattr(model_routes, "_model_migration_task", None)
    monkeypatch.setattr(model_routes, "_model_download_tasks", {})
    monkeypatch.setattr(
        model_routes,
        "_move_model_cache_directory",
        blocking_move,
    )
    monkeypatch.setattr(backends, "get_model_config", lambda _name: download_config)
    monkeypatch.setattr(
        backends,
        "get_model_load_func",
        lambda _config: download_backend.load_model,
    )
    monkeypatch.setattr(
        backends,
        "get_tts_backend_for_engine",
        lambda _engine: download_backend,
    )

    await model_routes.migrate_models(model_routes.models.ModelMigrateRequest(destination=str(destination)))
    migration = model_routes._owned_model_migration_task()
    assert migration is not None
    await _wait_for_thread_event(migration_entered)

    await model_routes.trigger_model_download(model_routes.models.ModelDownloadRequest(model_name="luxtts"))
    download = model_routes._owned_model_download_task("luxtts")
    assert download is not None
    try:
        for _ in range(20):
            await asyncio.sleep(0)
        assert not download_entered.is_set()
    finally:
        migration_release.set()

    await asyncio.wait_for(migration, timeout=2)
    await asyncio.wait_for(download, timeout=2)
    assert download_entered.is_set()
    assert not mlx_tts_lifecycle_guard.is_active()


@pytest.mark.asyncio
async def test_whisper_model_download_reenters_shared_guard_without_deadlock(
    monkeypatch,
):
    backend = MLXSTTBackend("base")

    def load_model(model_size: str) -> None:
        backend.model = object()
        backend.model_size = model_size

    monkeypatch.setattr(backend, "_load_model_sync", load_model)
    config = SimpleNamespace(
        model_name="whisper-base",
        engine="whisper",
        model_size="base",
    )
    await _run_model_download_background(
        monkeypatch,
        config,
        backend,
        lambda: backend.load_model_async("base"),
    )
    assert backend.is_loaded()


@pytest.mark.asyncio
async def test_mlx_llm_model_download_reenters_shared_guard_without_deadlock(
    monkeypatch,
):
    backend = MLXQwenLLMBackend("0.6B")

    def load_model(model_size: str) -> None:
        backend.model = object()
        backend.tokenizer = object()
        backend._current_model_size = model_size
        backend.model_size = model_size

    monkeypatch.setattr(backend, "_load_model_sync", load_model)
    config = SimpleNamespace(
        model_name="qwen3-0.6b",
        engine="qwen_llm",
        model_size="0.6B",
    )
    await _run_model_download_background(
        monkeypatch,
        config,
        backend,
        lambda: backend.load_model("0.6B"),
    )
    assert backend.is_loaded()


@pytest.mark.asyncio
async def test_unload_route_returns_409_without_mutating_active_model(monkeypatch):
    model = _BlockingSerialModel()
    backend = _loaded_backend(model)
    unload_called = False

    def unload() -> None:
        nonlocal unload_called
        unload_called = True
        backend.unload_model()

    monkeypatch.setattr(tts, "unload_tts_model", unload)
    active = asyncio.create_task(backend.generate("first", {}))
    await _wait_for_thread_event(model.entered["first"])
    try:
        with pytest.raises(HTTPException) as raised:
            await model_routes.unload_model()
        assert raised.value.status_code == 409
        assert "busy with serial inference" in raised.value.detail
        assert not unload_called
        assert backend.model is model
    finally:
        model.release.set()
    await asyncio.wait_for(active, timeout=2)


@pytest.mark.asyncio
async def test_load_route_returns_409_without_touching_backend(monkeypatch):
    model = _BlockingSerialModel()
    backend = _loaded_backend(model)
    get_model_calls = 0

    def get_model():
        nonlocal get_model_calls
        get_model_calls += 1
        return backend

    monkeypatch.setattr(tts, "get_tts_model", get_model)
    active = asyncio.create_task(backend.generate("first", {}))
    await _wait_for_thread_event(model.entered["first"])
    try:
        with pytest.raises(HTTPException) as raised:
            await model_routes.load_model("0.6B")
        assert raised.value.status_code == 409
        assert "busy with serial inference" in raised.value.detail
        assert get_model_calls == 0
        assert backend.model is model
        assert backend.model_size == "1.7B"
    finally:
        model.release.set()
    await asyncio.wait_for(active, timeout=2)


@pytest.mark.asyncio
async def test_delete_route_returns_409_without_unload_or_cache_mutation(
    monkeypatch,
    tmp_path,
):
    from huggingface_hub import constants as hf_constants

    model = _BlockingSerialModel()
    backend = _loaded_backend(model)
    repo_id = "example/qwen"
    repo_cache = tmp_path / "models--example--qwen"
    repo_cache.mkdir()
    unload_calls: list[object] = []
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(
        backends,
        "get_model_config",
        lambda _name: SimpleNamespace(
            engine="qwen",
            model_size="1.7B",
            hf_repo_id=repo_id,
        ),
    )
    monkeypatch.setattr(
        backends,
        "unload_model_by_config",
        lambda config: unload_calls.append(config),
    )

    active = asyncio.create_task(backend.generate("first", {}))
    await _wait_for_thread_event(model.entered["first"])
    try:
        with pytest.raises(HTTPException) as raised:
            await model_routes.delete_model("qwen-tts-1.7B")
        assert raised.value.status_code == 409
        assert "busy with serial inference" in raised.value.detail
        assert unload_calls == []
        assert repo_cache.is_dir()
        assert backend.model is model
    finally:
        model.release.set()
    await asyncio.wait_for(active, timeout=2)
