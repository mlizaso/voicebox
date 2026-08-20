"""
Serial generation queue — ensures only one TTS inference runs at a time
to avoid GPU contention.
"""

import asyncio
import os
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Literal

# Keep references to fire-and-forget background tasks to prevent GC
_background_tasks: set = set()
_accepting_background_tasks = True


@dataclass
class GenerationJob:
    """Queued inference work and every generation row it owns."""

    generation_ids: tuple[str, ...]
    coro: Coroutine


# Generation queue — serializes TTS inference to avoid GPU contention
_generation_queue: asyncio.Queue = None  # type: ignore  # initialized at startup
_generation_worker_task: asyncio.Task | None = None
_queued_generation_ids: set[str] = set()
_running_generation_tasks: dict[str, asyncio.Task] = {}
_cancelled_generation_ids: set[str] = set()
_generation_job_ids: dict[str, tuple[str, ...]] = {}

DEFAULT_MAX_QUEUED_GENERATION_JOBS = 32
MAX_QUEUED_GENERATION_JOBS_ENV = "VOICEBOX_MAX_QUEUED_GENERATIONS"


class GenerationQueueFullError(RuntimeError):
    """Raised before accepting work beyond the bounded inference backlog."""


def _max_queued_generation_jobs() -> int:
    raw = os.getenv(
        MAX_QUEUED_GENERATION_JOBS_ENV,
        str(DEFAULT_MAX_QUEUED_GENERATION_JOBS),
    )
    try:
        limit = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{MAX_QUEUED_GENERATION_JOBS_ENV} must be an integer") from exc
    if not 1 <= limit <= 1024:
        raise RuntimeError(f"{MAX_QUEUED_GENERATION_JOBS_ENV} must be between 1 and 1024")
    return limit


def create_background_task(coro) -> asyncio.Task:
    """Create a background task and prevent it from being garbage collected."""
    if not _accepting_background_tasks:
        coro.close()
        raise RuntimeError("Voicebox is shutting down and cannot accept background work")
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _generation_worker(
    queue: asyncio.Queue,
    queued_generation_ids: set[str],
    running_generation_tasks: dict[str, asyncio.Task],
    cancelled_generation_ids: set[str],
    generation_job_ids_by_id: dict[str, tuple[str, ...]],
):
    """Worker that processes generation tasks one at a time."""
    while True:
        job = await queue.get()
        try:
            if any(generation_id in cancelled_generation_ids for generation_id in job.generation_ids):
                cancelled_generation_ids.difference_update(job.generation_ids)
                job.coro.close()
                for generation_id in job.generation_ids:
                    await _force_fail_if_active(
                        generation_id,
                        "Generation batch cancelled before inference",
                    )
                continue

            task = asyncio.create_task(job.coro)
            for generation_id in job.generation_ids:
                running_generation_tasks[generation_id] = task
                queued_generation_ids.discard(generation_id)
            try:
                await task
                # A generation coroutine may deliberately drain blocking work
                # and suppress CancelledError after writing its terminal state.
                # If the worker itself was cancelled, that normal-looking
                # return must still terminate the worker instead of looping
                # back to queue.get() and hanging application shutdown.
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
            except asyncio.CancelledError:
                # Cancellation of the worker itself propagates to the awaited
                # child. It must terminate this worker, rather than being
                # mistaken for an ordinary per-generation cancellation.
                if asyncio.current_task().cancelling() or not task.cancelled():
                    raise
        except Exception:
            traceback.print_exc()
            for generation_id in job.generation_ids:
                await _force_fail_if_active(
                    generation_id,
                    "Worker exited without writing terminal status",
                )
        finally:
            # A running child can be cancelled before its coroutine executes a
            # single instruction.  In that case none of the generation
            # service's status/task cleanup handlers run. Reconcile every row
            # while this queue still owns the IDs, then release ownership.
            from ..utils.tasks import get_task_manager

            task_manager = get_task_manager()
            for generation_id in job.generation_ids:
                await _force_fail_if_active(
                    generation_id,
                    "Generation cancelled before terminal status was written",
                )
                task_manager.complete_generation(generation_id)
            for generation_id in job.generation_ids:
                running_generation_tasks.pop(generation_id, None)
                queued_generation_ids.discard(generation_id)
                generation_job_ids_by_id.pop(generation_id, None)
            cancelled_generation_ids.difference_update(job.generation_ids)
            queue.task_done()


async def _force_fail_if_active(generation_id: str, error: str) -> None:
    """Best-effort recovery — flip an active row to failed if the worker
    bailed before writing a terminal status. Catches the case where the gen
    coroutine's own status-write raised (e.g. SQLite lock contention)."""
    try:
        from ..database import Generation as DBGeneration, get_db
        from . import history

        db = next(get_db())
        try:
            gen = db.query(DBGeneration).filter_by(id=generation_id).first()
            if gen is None:
                return
            if (gen.status or "completed") not in ("loading_model", "generating"):
                return
            await history.update_generation_status(
                generation_id=generation_id,
                status="failed",
                db=db,
                error=error,
            )
        finally:
            db.close()
    except Exception:
        traceback.print_exc()


def enqueue_generation(generation_id: str, coro):
    """Add a generation coroutine to the serial queue."""
    enqueue_generation_batch((generation_id,), coro)


async def run_queued_generation(
    generation_id: str,
    coro,
    *,
    discard_result: Callable[[object], None] | None = None,
):
    """Run request-owned work through the bounded serial generation queue.

    Unlike the durable generation endpoints, foreground callers need the
    result rather than a database status update. The queue still owns and
    drains the coroutine after a client cancellation, so blocking inference
    cannot escape the one-job accelerator boundary.
    """
    loop = asyncio.get_running_loop()
    result = loop.create_future()

    def discard(value: object) -> None:
        if discard_result is None:
            return
        try:
            discard_result(value)
        except Exception:
            traceback.print_exc()

    async def queued_work():
        try:
            value = await coro
        except BaseException as exc:
            if not result.done():
                result.set_exception(exc)
            if not isinstance(exc, Exception):
                raise
        else:
            if not result.done():
                result.set_result(value)
            else:
                discard(value)

    wrapper = queued_work()
    try:
        enqueue_generation(generation_id, wrapper)
    except BaseException:
        wrapper.close()
        coro.close()
        raise

    try:
        return await asyncio.shield(result)
    except asyncio.CancelledError:
        if result.done() and not result.cancelled():
            try:
                completed_value = result.result()
            except BaseException:
                pass
            else:
                discard(completed_value)
        else:
            result.cancel()
        cancel_generation(generation_id)
        raise


def enqueue_generation_batch(generation_ids: tuple[str, ...], coro) -> None:
    """Queue one inference coroutine that atomically owns multiple rows."""
    if not _accepting_background_tasks:
        coro.close()
        raise RuntimeError("Voicebox is shutting down and cannot accept generation work")
    if _generation_queue is None:
        raise RuntimeError("Generation queue has not been initialized")
    if not generation_ids or len(set(generation_ids)) != len(generation_ids):
        raise ValueError("generation batch IDs must be non-empty and unique")
    if any(
        generation_id in _queued_generation_ids or generation_id in _running_generation_tasks
        for generation_id in generation_ids
    ):
        raise ValueError("generation is already queued or running")

    try:
        _generation_queue.put_nowait(GenerationJob(generation_ids=generation_ids, coro=coro))
    except asyncio.QueueFull as exc:
        raise GenerationQueueFullError(
            "Generation queue is full; wait for queued work to finish before submitting more"
        ) from exc

    _queued_generation_ids.update(generation_ids)
    for generation_id in generation_ids:
        _generation_job_ids[generation_id] = generation_ids


def generation_job_ids(generation_id: str) -> tuple[str, ...]:
    """Return every row owned by the same queued/running inference job."""
    return _generation_job_ids.get(generation_id, (generation_id,))


def generation_job_is_active(generation_id: str) -> bool:
    """Whether a caller ID remains owned by queued or executing work."""
    return generation_id in _generation_job_ids


def cancel_generation(generation_id: str) -> Literal["queued", "running"] | None:
    """Cancel a queued or running generation if it is still active."""
    if generation_id in _cancelled_generation_ids:
        return None
    running_task = _running_generation_tasks.get(generation_id)
    if running_task is not None:
        _cancelled_generation_ids.update(generation_job_ids(generation_id))
        running_task.cancel()
        return "running"

    if generation_id in _queued_generation_ids:
        # Keep every ID reserved until the worker consumes and closes the shared coroutine.
        # Otherwise a cancelled caller ID could be enqueued again behind the still-live job.
        _cancelled_generation_ids.update(generation_job_ids(generation_id))
        return "queued"

    return None


def init_queue(force: bool = False):
    """Initialize the generation queue and start the worker.

    Must be called once during application startup (inside a running event loop).
    """
    global _generation_queue, _generation_worker_task
    global _queued_generation_ids, _running_generation_tasks, _cancelled_generation_ids
    global _generation_job_ids, _accepting_background_tasks

    if _generation_worker_task is not None and not _generation_worker_task.done():
        if not force:
            return
        _generation_worker_task.cancel()
        for task in list(_running_generation_tasks.values()):
            task.cancel()

    _generation_queue = asyncio.Queue(maxsize=_max_queued_generation_jobs())
    _queued_generation_ids = set()
    _running_generation_tasks = {}
    _cancelled_generation_ids = set()
    _generation_job_ids = {}
    _accepting_background_tasks = True
    _generation_worker_task = create_background_task(
        _generation_worker(
            _generation_queue,
            _queued_generation_ids,
            _running_generation_tasks,
            _cancelled_generation_ids,
            _generation_job_ids,
        )
    )


async def shutdown_background_tasks() -> None:
    """Drain every owned task and queued coroutine before releasing data storage."""
    global _generation_worker_task, _accepting_background_tasks

    _accepting_background_tasks = False
    tasks = list(_background_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    task_manager = None
    if _generation_queue is not None:
        from ..utils.tasks import get_task_manager

        task_manager = get_task_manager()
        while True:
            try:
                job = _generation_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            job.coro.close()
            for generation_id in job.generation_ids:
                await _force_fail_if_active(
                    generation_id,
                    "Voicebox shut down before queued generation started",
                )
                task_manager.complete_generation(generation_id)
                _queued_generation_ids.discard(generation_id)
                _running_generation_tasks.pop(generation_id, None)
                _generation_job_ids.pop(generation_id, None)
            _cancelled_generation_ids.difference_update(job.generation_ids)
            _generation_queue.task_done()

    _generation_worker_task = None
