"""Concurrency and cancellation contracts for downloadable GPU backends."""

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from fastapi import HTTPException

from backend.routes import cuda as cuda_routes, rocm as rocm_routes
from backend.services import cuda, rocm


@pytest.fixture(params=[cuda, rocm], ids=["cuda", "rocm"])
def backend_service(request, tmp_path: Path, monkeypatch) -> ModuleType:
    service = request.param
    monkeypatch.setattr(service, "get_backends_dir", lambda: tmp_path / service.__name__.rsplit(".", 1)[-1])
    if service is cuda:
        monkeypatch.setattr(cuda.sys, "platform", "win32")
    assert service._active_operation_name() is None
    yield service
    assert service._active_operation_name() is None


def _schedule_download(service: ModuleType) -> asyncio.Task:
    if service is cuda:
        return service.schedule_cuda_binary_download()
    return service.schedule_rocm_binary_download()


async def _delete_backend(service: ModuleType) -> bool:
    if service is cuda:
        return await service.delete_cuda_binary()
    return await service.delete_rocm_binary()


def _backend_directory(service: ModuleType) -> Path:
    return service.get_backends_dir() / ("cuda" if service is cuda else "rocm")


@pytest.mark.asyncio
async def test_scheduled_download_reserves_before_background_task_starts(
    backend_service: ModuleType,
    monkeypatch,
):
    async def never_started(_version=None):
        raise AssertionError("the cancelled background operation must not start")

    locked_name = "_download_cuda_binary_locked" if backend_service is cuda else "_download_rocm_binary_locked"
    monkeypatch.setattr(backend_service, locked_name, never_started)

    task = _schedule_download(backend_service)
    with pytest.raises(backend_service.BackendOperationBusyError):
        await _delete_backend(backend_service)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend_service._active_operation_name() is None


@pytest.mark.asyncio
async def test_cancelled_delete_drains_rmtree_before_releasing_reservation(
    backend_service: ModuleType,
    monkeypatch,
):
    backend_dir = _backend_directory(backend_service)
    backend_dir.mkdir(parents=True)
    (backend_dir / "backend.bin").write_bytes(b"installed")

    real_delete = backend_service.delete_backend_install
    worker_started = threading.Event()
    allow_worker_to_finish = threading.Event()

    def blocking_delete(root: Path, backend_name: str, executable_name: str) -> bool:
        worker_started.set()
        if not allow_worker_to_finish.wait(timeout=5):
            raise TimeoutError("test did not release the filesystem worker")
        return real_delete(root, backend_name, executable_name)

    monkeypatch.setattr(backend_service, "delete_backend_install", blocking_delete)

    deletion = asyncio.create_task(_delete_backend(backend_service))
    assert await asyncio.to_thread(worker_started.wait, 2)
    deletion.cancel()
    await asyncio.sleep(0)
    assert not deletion.done()
    assert backend_service._active_operation_name() == "delete"
    with pytest.raises(backend_service.BackendOperationBusyError):
        _schedule_download(backend_service)

    allow_worker_to_finish.set()
    with pytest.raises(asyncio.CancelledError):
        await deletion
    assert backend_service._active_operation_name() is None
    assert not backend_dir.exists()


@pytest.mark.asyncio
async def test_auto_update_reserves_checks_and_download_as_one_operation(
    backend_service: ModuleType,
    monkeypatch,
    tmp_path: Path,
):
    installed = tmp_path / "installed.exe"
    installed.write_bytes(b"old")
    update_started = asyncio.Event()
    finish_update = asyncio.Event()

    monkeypatch.setattr(
        backend_service,
        "get_cuda_binary_path" if backend_service is cuda else "get_rocm_binary_path",
        lambda: installed,
    )
    monkeypatch.setattr(backend_service, "_needs_server_download", lambda _version=None: True)
    libs_check = "_needs_cuda_libs_download" if backend_service is cuda else "_needs_rocm_libs_download"
    monkeypatch.setattr(backend_service, libs_check, lambda: False)
    version_reader = "get_cuda_binary_version" if backend_service is cuda else "get_rocm_binary_version"
    monkeypatch.setattr(backend_service, version_reader, lambda: "old")

    async def blocked_update(_version=None):
        update_started.set()
        await finish_update.wait()

    locked_name = "_download_cuda_binary_locked" if backend_service is cuda else "_download_rocm_binary_locked"
    update_name = "check_and_update_cuda_binary" if backend_service is cuda else "check_and_update_rocm_binary"
    monkeypatch.setattr(backend_service, locked_name, blocked_update)

    update = asyncio.create_task(getattr(backend_service, update_name)())
    await update_started.wait()
    assert backend_service._active_operation_name() == "update"
    with pytest.raises(backend_service.BackendOperationBusyError):
        await _delete_backend(backend_service)

    finish_update.set()
    await update
    assert backend_service._active_operation_name() is None


@pytest.mark.parametrize(
    ("service", "download_route", "delete_route"),
    [
        (cuda, cuda_routes.download_cuda_backend, cuda_routes.delete_cuda_backend),
        (rocm, rocm_routes.download_rocm_backend, rocm_routes.delete_rocm_backend),
    ],
    ids=["cuda", "rocm"],
)
@pytest.mark.asyncio
async def test_routes_map_operation_conflicts_to_409(
    service: ModuleType,
    download_route: Callable,
    delete_route: Callable,
    monkeypatch,
):
    if service is cuda:
        monkeypatch.setattr(service, "get_cuda_download_unsupported_reason", lambda: None)
        schedule_name = "schedule_cuda_binary_download"
        active_name = "is_cuda_active"
        delete_name = "delete_cuda_binary"
    else:
        schedule_name = "schedule_rocm_binary_download"
        active_name = "is_rocm_active"
        delete_name = "delete_rocm_binary"

    def busy_schedule():
        raise service.BackendOperationBusyError("operation busy")

    async def busy_delete():
        raise service.BackendOperationBusyError("operation busy")

    monkeypatch.setattr(service, schedule_name, busy_schedule)
    monkeypatch.setattr(service, active_name, lambda: False)
    monkeypatch.setattr(service, delete_name, busy_delete)

    with pytest.raises(HTTPException) as download_error:
        await download_route()
    assert download_error.value.status_code == 409

    with pytest.raises(HTTPException) as delete_error:
        await delete_route()
    assert delete_error.value.status_code == 409
