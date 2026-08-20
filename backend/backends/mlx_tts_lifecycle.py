"""Process-wide serialization for local model and accelerator lifecycle."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress

_ACQUIRE_POLL_SECONDS = 0.01


class MLXTTSLifecycleBusyError(RuntimeError):
    """Raised when a non-waiting model mutation cannot acquire the accelerator."""


class MLXTTSLifecycleGuard:
    """Serialize local accelerator work across loops and process threads.

    ``asyncio.Lock`` is bound to one event loop and cannot protect direct API
    calls running on another loop/thread. A process-wide ``threading.Lock`` is
    the actual exclusion primitive here. Async waiters poll with non-blocking
    acquisition, so cancelling a waiter can never leave an executor thread
    behind that later acquires the lock without an owner.

    The guard is reentrant only for the same asyncio task (or the same plain
    thread). This permits a lifecycle route to hold the guard while calling a
    backend method that also enforces it, without letting inherited context in
    a child task bypass serialization.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._owner: tuple[str, int] | None = None
        self._operation: str | None = None

    @staticmethod
    def _execution_owner() -> tuple[str, int]:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            return ("task", id(task))
        return ("thread", threading.get_ident())

    def _try_claim(self, owner: tuple[str, int], operation: str) -> bool | None:
        """Return True for a new claim, False for reentry, None if busy."""
        with self._state_lock:
            if self._owner == owner:
                return False
        if not self._lock.acquire(blocking=False):
            return None
        with self._state_lock:
            self._owner = owner
            self._operation = operation
        return True

    def _release(self, owner: tuple[str, int]) -> None:
        with self._state_lock:
            if self._owner != owner:
                raise RuntimeError("MLX TTS lifecycle guard released by a non-owner")
            self._owner = None
            self._operation = None
        self._lock.release()

    def _busy_error(self) -> MLXTTSLifecycleBusyError:
        with self._state_lock:
            operation = self._operation or "another model operation"
        return MLXTTSLifecycleBusyError(f"Local inference is busy with {operation}; retry after it finishes")

    @asynccontextmanager
    async def hold(self, operation: str) -> AsyncIterator[None]:
        """Wait cancellation-safely for exclusive lifecycle ownership."""
        owner = self._execution_owner()
        releases_lock = False
        while True:
            claim = self._try_claim(owner, operation)
            if claim is not None:
                releases_lock = claim
                break
            await asyncio.sleep(_ACQUIRE_POLL_SECONDS)
        try:
            yield
        finally:
            if releases_lock:
                self._release(owner)

    @contextmanager
    def try_hold(self, operation: str) -> Iterator[None]:
        """Acquire atomically without waiting, or raise a UI-safe busy error."""
        owner = self._execution_owner()
        claim = self._try_claim(owner, operation)
        if claim is None:
            raise self._busy_error()
        try:
            yield
        finally:
            if claim:
                self._release(owner)

    def is_active(self) -> bool:
        """Return a diagnostic snapshot; never use this for check-then-mutate."""
        return self._lock.locked()


mlx_tts_lifecycle_guard = MLXTTSLifecycleGuard()


async def run_blocking_operation_cancellation_safe(function, /, *args, **kwargs):
    """Run executor work and drain its real thread before propagating cancel."""
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError as cancellation:
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                continue
        if not operation.cancelled():
            with suppress(BaseException):
                operation.result()
        raise cancellation


async def run_tts_operation_cancellation_safe(backend, operation):
    """Await one backend operation without orphaning non-cancellable work.

    Most non-MLX engines use ``asyncio.to_thread`` internally. Cancelling the
    awaiting coroutine cannot stop that executor thread, so the global request
    guard must keep ownership until the real load, prompt, or inference call
    exits. MLX operations stay in the owner task because their nested lifecycle
    guard is deliberately task-local and their blocking work already drains.
    """
    if getattr(backend, "uses_shared_mlx_lifecycle_guard", False) is True:
        return await operation
    if getattr(backend, "tts_operations_are_cancellable", False) is True:
        return await operation

    task = asyncio.create_task(operation)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancellation:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        if not task.cancelled():
            with suppress(BaseException):
                task.result()
        raise cancellation


@asynccontextmanager
async def loaded_tts_backend_for_request(
    engine: str,
    model_size: str = "default",
):
    """Load and retain one engine/model binding for a complete TTS request.

    MLX Qwen owns one mutable process-wide model. Its request context keeps the
    lifecycle guard from model selection through prompt preparation and every
    logical inference chunk. Other engines preserve their existing load-only
    behavior and do not contend on the MLX guard.
    """
    from . import get_tts_backend_for_engine, load_engine_model

    backend = get_tts_backend_for_engine(engine)
    async with mlx_tts_lifecycle_guard.hold(f"{engine} {model_size} TTS request"):
        if getattr(backend, "uses_mlx_request_context", False) is True:
            async with backend.mlx_request_context(model_size):
                yield backend
            return

        await run_tts_operation_cancellation_safe(
            backend,
            load_engine_model(engine, model_size),
        )
        # Callers wrap every backend-owned prompt/inference await with the
        # cancellation-safe helper, so ownership outlives executor work.
        yield backend
