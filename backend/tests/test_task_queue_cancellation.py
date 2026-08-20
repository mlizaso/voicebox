import asyncio
import threading

import pytest

from backend.backends import mlx_backend, mlx_tts_lifecycle
from backend.mcp_server import context as mcp_context
from backend.services import task_queue
from backend.utils.tasks import get_task_manager


@pytest.mark.asyncio
async def test_generation_queue_rejects_work_beyond_bounded_backlog(monkeypatch):
    monkeypatch.setenv(task_queue.MAX_QUEUED_GENERATION_JOBS_ENV, "1")
    task_queue.init_queue(force=True)
    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()

    async def blocker():
        blocker_started.set()
        await release_blocker.wait()

    async def queued():
        return None

    task_queue.enqueue_generation("running", blocker())
    await asyncio.wait_for(blocker_started.wait(), timeout=1)
    task_queue.enqueue_generation("queued", queued())
    rejected = queued()
    with pytest.raises(task_queue.GenerationQueueFullError, match="queue is full"):
        task_queue.enqueue_generation("rejected", rejected)
    rejected.close()

    assert task_queue.generation_job_is_active("queued") is True
    assert task_queue.generation_job_is_active("rejected") is False
    release_blocker.set()
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_foreground_work_uses_serial_queue_and_returns_result():
    task_queue.init_queue(force=True)
    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()
    foreground_started = asyncio.Event()

    async def blocker():
        blocker_started.set()
        await release_blocker.wait()

    async def foreground():
        foreground_started.set()
        return "ready"

    task_queue.enqueue_generation("blocker", blocker())
    await asyncio.wait_for(blocker_started.wait(), timeout=1)
    result_task = asyncio.create_task(task_queue.run_queued_generation("stream-request", foreground()))
    await asyncio.sleep(0)

    assert not foreground_started.is_set()
    assert task_queue.generation_job_is_active("stream-request")

    release_blocker.set()
    assert await asyncio.wait_for(result_task, timeout=1) == "ready"
    for _ in range(20):
        if not task_queue.generation_job_is_active("stream-request"):
            break
        await asyncio.sleep(0)
    assert not task_queue.generation_job_is_active("stream-request")


@pytest.mark.asyncio
async def test_cancelling_foreground_waiter_cancels_queue_owned_work():
    task_queue.init_queue(force=True)
    foreground_started = asyncio.Event()
    foreground_cancelled = asyncio.Event()

    async def foreground():
        foreground_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            foreground_cancelled.set()
            raise

    result_task = asyncio.create_task(task_queue.run_queued_generation("stream-request", foreground()))
    await asyncio.wait_for(foreground_started.wait(), timeout=1)
    result_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await result_task
    await asyncio.wait_for(foreground_cancelled.wait(), timeout=1)
    for _ in range(20):
        if not task_queue.generation_job_is_active("stream-request"):
            break
        await asyncio.sleep(0)
    assert not task_queue.generation_job_is_active("stream-request")


@pytest.mark.asyncio
async def test_cancelled_foreground_waiter_discards_late_file_result():
    task_queue.init_queue(force=True)
    foreground_started = asyncio.Event()
    release_foreground = asyncio.Event()
    discarded = []

    async def cancellation_draining_foreground():
        foreground_started.set()
        try:
            await release_foreground.wait()
        except asyncio.CancelledError:
            await release_foreground.wait()
        return "private-response-file"

    result_task = asyncio.create_task(
        task_queue.run_queued_generation(
            "stream-request",
            cancellation_draining_foreground(),
            discard_result=discarded.append,
        )
    )
    await asyncio.wait_for(foreground_started.wait(), timeout=1)
    result_task.cancel()
    await asyncio.sleep(0)
    release_foreground.set()

    with pytest.raises(asyncio.CancelledError):
        await result_task
    for _ in range(20):
        if discarded:
            break
        await asyncio.sleep(0)
    assert discarded == ["private-response-file"]


@pytest.mark.asyncio
async def test_cancel_queued_generation_skips_execution():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    release_running = asyncio.Event()
    queued_ran = asyncio.Event()

    async def running_job():
        running_started.set()
        await release_running.wait()

    async def queued_job():
        queued_ran.set()

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    task_queue.enqueue_generation("gen-queued", queued_job())
    assert task_queue.cancel_generation("gen-queued") == "queued"

    release_running.set()
    await asyncio.sleep(0.1)

    assert not queued_ran.is_set()


@pytest.mark.asyncio
async def test_cancel_running_generation_cancels_task():
    task_queue.init_queue(force=True)

    running_started = asyncio.Event()
    running_cancelled = asyncio.Event()

    async def running_job():
        running_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            running_cancelled.set()
            raise

    task_queue.enqueue_generation("gen-running", running_job())
    await asyncio.wait_for(running_started.wait(), timeout=1)

    assert task_queue.cancel_generation("gen-running") == "running"
    await asyncio.wait_for(running_cancelled.wait(), timeout=1)


@pytest.mark.asyncio
async def test_cancel_one_queued_batch_member_skips_the_shared_batch():
    task_queue.init_queue(force=True)

    blocker_started = asyncio.Event()
    release_blocker = asyncio.Event()
    batch_ran = asyncio.Event()

    async def blocker():
        blocker_started.set()
        await release_blocker.wait()

    async def batch_job():
        batch_ran.set()

    task_queue.enqueue_generation("blocker", blocker())
    await asyncio.wait_for(blocker_started.wait(), timeout=1)
    task_queue.enqueue_generation_batch(("batch-a", "batch-b"), batch_job())

    assert task_queue.generation_job_ids("batch-b") == ("batch-a", "batch-b")
    assert task_queue.cancel_generation("batch-b") == "queued"
    assert task_queue.cancel_generation("batch-a") is None
    duplicate = batch_job()
    with pytest.raises(ValueError, match="already queued"):
        task_queue.enqueue_generation("batch-a", duplicate)
    duplicate.close()
    release_blocker.set()
    await asyncio.sleep(0.1)

    assert not batch_ran.is_set()


@pytest.mark.asyncio
async def test_cancel_one_running_batch_member_cancels_shared_task():
    task_queue.init_queue(force=True)

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def batch_job():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task_queue.enqueue_generation_batch(("batch-a", "batch-b"), batch_job())
    await asyncio.wait_for(started.wait(), timeout=1)

    assert task_queue.cancel_generation("batch-a") == "running"
    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert task_queue.cancel_generation("batch-b") is None


@pytest.mark.asyncio
async def test_cancelled_batch_keeps_worker_slot_and_ids_until_coro_really_exits():
    task_queue.init_queue(force=True)

    thread_started = threading.Event()
    release_thread = threading.Event()
    next_job_started = asyncio.Event()

    def blocking_inference():
        thread_started.set()
        assert release_thread.wait(timeout=2)

    async def cancellation_safe_batch_job():
        await mlx_backend._run_blocking_mlx_operation(blocking_inference)

    async def next_job():
        next_job_started.set()

    task_queue.enqueue_generation_batch(
        ("batch-a", "batch-b"),
        cancellation_safe_batch_job(),
    )
    task_queue.enqueue_generation("next", next_job())
    while not thread_started.is_set():
        await asyncio.sleep(0)
    assert task_queue.cancel_generation("batch-a") == "running"
    for _ in range(10):
        await asyncio.sleep(0)

    assert task_queue.generation_job_ids("batch-b") == ("batch-a", "batch-b")
    assert not next_job_started.is_set()
    duplicate = asyncio.sleep(0)
    with pytest.raises(ValueError, match="already queued"):
        task_queue.enqueue_generation("batch-b", duplicate)
    duplicate.close()

    release_thread.set()
    await asyncio.wait_for(next_job_started.wait(), timeout=1)
    assert task_queue.generation_job_ids("batch-b") == ("batch-b",)


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_ids", [("single",), ("batch-a", "batch-b")])
async def test_cancel_before_child_first_step_reconciles_rows_and_task_metadata(
    monkeypatch,
    generation_ids,
):
    forced_failed = []
    child_entered = False

    async def force_fail(generation_id, error):
        forced_failed.append((generation_id, error))

    async def never_started():
        nonlocal child_entered
        child_entered = True

    monkeypatch.setattr(task_queue, "_force_fail_if_active", force_fail)
    task_queue.init_queue(force=True)
    task_manager = get_task_manager()
    task_manager.clear_all()
    for generation_id in generation_ids:
        task_manager.start_generation(generation_id, "profile", "text")

    task_queue.enqueue_generation_batch(generation_ids, never_started())

    async def cancel_before_child_runs():
        assert task_queue.cancel_generation(generation_ids[0]) == "running"

    # The queue worker is already ahead of this callback in the ready queue. It
    # creates and publishes the child, then awaits it; this callback cancels it
    # before the newly scheduled child receives its first instruction.
    cancellation = asyncio.create_task(cancel_before_child_runs())
    await asyncio.wait_for(cancellation, timeout=1)
    for _ in range(100):
        if not task_queue.generation_job_is_active(generation_ids[0]):
            break
        await asyncio.sleep(0)

    assert child_entered is False
    assert [item[0] for item in forced_failed] == list(generation_ids)
    assert all(not task_manager.is_generation_active(item) for item in generation_ids)
    assert all(not task_queue.generation_job_is_active(item) for item in generation_ids)


@pytest.mark.asyncio
async def test_worker_cancellation_terminates_and_does_not_touch_replacement_queue(
    monkeypatch,
):
    child_started = asyncio.Event()

    async def force_fail(_generation_id, _error):
        return None

    async def child():
        child_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(task_queue, "_force_fail_if_active", force_fail)
    task_queue.init_queue(force=True)
    task_queue.enqueue_generation("old", child())
    await asyncio.wait_for(child_started.wait(), timeout=1)
    old_worker = task_queue._generation_worker_task
    old_queue = task_queue._generation_queue

    task_queue.init_queue(force=True)
    replacement_queue = task_queue._generation_queue
    replacement_jobs = task_queue._generation_job_ids

    with pytest.raises(asyncio.CancelledError):
        await old_worker

    assert old_worker.done()
    assert old_queue is not replacement_queue
    assert replacement_queue.empty()
    assert replacement_jobs == {}


@pytest.mark.asyncio
async def test_shutdown_terminates_worker_when_generation_suppresses_cancellation(
    monkeypatch,
):
    child_started = asyncio.Event()
    child_suppressed_cancellation = asyncio.Event()

    async def force_fail(_generation_id, _error):
        return None

    async def cancellation_draining_child():
        child_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_suppressed_cancellation.set()
            # Generation services suppress cancellation after draining their
            # executor work and recording a terminal status.
            return

    monkeypatch.setattr(task_queue, "_force_fail_if_active", force_fail)
    task_queue.init_queue(force=True)
    task_queue.enqueue_generation("generation", cancellation_draining_child())
    await asyncio.wait_for(child_started.wait(), timeout=1)

    worker = task_queue._generation_worker_task
    shutdown = asyncio.create_task(task_queue.shutdown_background_tasks())
    await asyncio.wait_for(child_suppressed_cancellation.wait(), timeout=1)
    await asyncio.wait_for(shutdown, timeout=1)

    assert worker.done()
    assert worker.cancelled()
    assert not task_queue.generation_job_is_active("generation")


@pytest.mark.asyncio
async def test_shutdown_waits_for_blocking_work_before_background_ownership_ends(
    monkeypatch,
):
    thread_started = threading.Event()
    release_thread = threading.Event()

    async def force_fail(_generation_id, _error):
        return None

    def blocking_operation():
        thread_started.set()
        assert release_thread.wait(timeout=2)

    monkeypatch.setattr(task_queue, "_force_fail_if_active", force_fail)
    task_queue.init_queue(force=True)
    operation = task_queue.create_background_task(
        mlx_tts_lifecycle.run_blocking_operation_cancellation_safe(blocking_operation)
    )
    while not thread_started.is_set():
        await asyncio.sleep(0)

    shutdown = asyncio.create_task(task_queue.shutdown_background_tasks())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not shutdown.done()
    assert not operation.done()

    release_thread.set()
    await asyncio.wait_for(shutdown, timeout=1)

    assert operation.done()
    assert not task_queue._background_tasks
    rejected = asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="shutting down"):
        task_queue.create_background_task(rejected)


@pytest.mark.asyncio
async def test_shutdown_drains_detached_mcp_last_seen_database_thread(
    monkeypatch,
):
    thread_started = threading.Event()
    release_thread = threading.Event()
    thread_finished = threading.Event()

    async def force_fail(_generation_id, _error):
        return None

    def blocking_stamp(_client_id):
        thread_started.set()
        assert release_thread.wait(timeout=2)
        thread_finished.set()

    monkeypatch.setattr(task_queue, "_force_fail_if_active", force_fail)
    monkeypatch.setattr(mcp_context, "_stamp_last_seen", blocking_stamp)
    monkeypatch.setattr(mcp_context, "_pending_stamps", set())
    task_queue.init_queue(force=True)

    mcp_context._enqueue_stamp("client")
    while not thread_started.is_set():
        await asyncio.sleep(0)

    shutdown = asyncio.create_task(task_queue.shutdown_background_tasks())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not shutdown.done()
    assert mcp_context._pending_stamps

    release_thread.set()
    await asyncio.wait_for(shutdown, timeout=1)

    assert thread_finished.is_set()
    assert not mcp_context._pending_stamps
