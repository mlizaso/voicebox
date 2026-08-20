"""Model management endpoints."""

import asyncio
import shutil
from contextlib import suppress
from pathlib import Path
from threading import Lock

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import models
from ..backends.mlx_tts_lifecycle import (
    MLXTTSLifecycleBusyError,
    mlx_tts_lifecycle_guard,
    run_blocking_operation_cancellation_safe,
    run_tts_operation_cancellation_safe,
)
from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager

router = APIRouter()

# TaskManager exposes UI metadata, but cancellation must target the actual
# coroutine that owns the model download/load and lifecycle guard. Keep that
# ownership separate so metadata can never claim a cancelled task is gone while
# its executor thread is still draining.
_model_download_tasks: dict[str, asyncio.Task] = {}
_model_download_tasks_lock = Lock()
_model_migration_task: asyncio.Task | None = None


class ModelDownloadAlreadyActiveError(RuntimeError):
    """Raised when a model already has a live owned download task."""


def _remove_model_download_progress(progress_manager, model_name: str) -> bool:
    with progress_manager._lock:
        removed = progress_manager._progress.pop(model_name, None) is not None
        progress_manager._last_notify_time.pop(model_name, None)
        progress_manager._last_notify_progress.pop(model_name, None)
        return removed


def _finish_model_download_task(
    model_name: str,
    task: asyncio.Task,
    task_manager,
    progress_manager,
) -> None:
    """Release ownership only after the real background task terminates."""
    released_ownership = False
    with _model_download_tasks_lock:
        if _model_download_tasks.get(model_name) is task:
            _model_download_tasks.pop(model_name, None)
            released_ownership = True
    if released_ownership and task.cancelled():
        task_manager.cancel_download(model_name)
        _remove_model_download_progress(progress_manager, model_name)


def _owned_model_download_task(model_name: str) -> asyncio.Task | None:
    with _model_download_tasks_lock:
        return _model_download_tasks.get(model_name)


def start_owned_model_download_task(
    model_name: str,
    coroutine,
    *,
    task_manager,
    progress_manager,
    task_factory,
) -> asyncio.Task:
    """Create and register the one cancellable loader for ``model_name``."""
    with _model_download_tasks_lock:
        existing_task = _model_download_tasks.get(model_name)
        if existing_task is not None and not existing_task.done():
            coroutine.close()
            raise ModelDownloadAlreadyActiveError(model_name)
        if existing_task is not None:
            _model_download_tasks.pop(model_name, None)

        try:
            task = task_factory(coroutine)
        except BaseException:
            coroutine.close()
            raise
        _model_download_tasks[model_name] = task
        task.add_done_callback(
            lambda completed_task: _finish_model_download_task(
                model_name,
                completed_task,
                task_manager,
                progress_manager,
            )
        )

    task_manager.start_download(model_name)
    return task


def _finish_model_migration_task(task: asyncio.Task) -> None:
    global _model_migration_task
    with _model_download_tasks_lock:
        if _model_migration_task is task:
            _model_migration_task = None


def _owned_model_migration_task() -> asyncio.Task | None:
    with _model_download_tasks_lock:
        return _model_migration_task


def _move_model_cache_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.move(str(source), str(destination))


def clear_task_metadata_if_no_owned_model_tasks(
    task_manager,
    progress_manager,
) -> tuple[str, ...]:
    """Atomically clear UI metadata only when no real model task is live."""
    with _model_download_tasks_lock:
        active_operations = sorted(model_name for model_name, task in _model_download_tasks.items() if not task.done())
        if _model_migration_task is not None and not _model_migration_task.done():
            active_operations.append("model-cache-migration")
        if active_operations:
            return tuple(active_operations)

        # Remove terminal handles whose scheduled done callbacks have not run
        # yet. Ownership-aware callbacks cannot disturb a later replacement.
        _model_download_tasks.clear()
        for download in task_manager.get_active_downloads():
            task_manager.cancel_download(download.model_name)
        with progress_manager._lock:
            progress_manager._progress.clear()
            progress_manager._last_notify_time.clear()
            progress_manager._last_notify_progress.clear()
    return ()


def _get_dir_size(path: Path) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _copy_with_progress(src: Path, dst: Path, progress_manager, copied_so_far: int, total_bytes: int) -> int:
    """Copy a directory tree with byte-level progress tracking."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dest_item = dst / item.name
        if item.is_dir():
            copied_so_far = _copy_with_progress(item, dest_item, progress_manager, copied_so_far, total_bytes)
        else:
            size = item.stat().st_size
            shutil.copy2(str(item), str(dest_item))
            copied_so_far += size
            progress_manager.update_progress(
                "migration",
                copied_so_far,
                total_bytes,
                filename=item.name,
                status="downloading",
            )
    return copied_so_far


@router.post("/models/load")
async def load_model(model_size: str = "1.7B"):
    """Manually load TTS model."""
    from ..services import tts

    try:
        with mlx_tts_lifecycle_guard.try_hold("model loading"):
            tts_model = tts.get_tts_model()
            await run_tts_operation_cancellation_safe(
                tts_model,
                tts_model.load_model_async(model_size),
            )
        return {"message": f"Model {model_size} loaded successfully"}
    except MLXTTSLifecycleBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/unload")
async def unload_model():
    """Unload the default Qwen TTS model to free memory."""
    from ..services import tts

    try:
        with mlx_tts_lifecycle_guard.try_hold("model unloading"):
            tts.unload_tts_model()
        return {"message": "Model unloaded successfully"}
    except MLXTTSLifecycleBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/models/{model_name}/unload")
async def unload_model_by_name(model_name: str):
    """Unload a specific model from memory without deleting it from disk."""
    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        with mlx_tts_lifecycle_guard.try_hold("model unloading"):
            was_loaded = unload_model_by_config(config)
        if not was_loaded:
            return {"message": f"Model {model_name} is not loaded"}
        return {"message": f"Model {model_name} unloaded successfully"}
    except MLXTTSLifecycleBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/models/progress/{model_name}")
async def get_model_progress(model_name: str):
    """Get model download progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe(model_name):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/cache-dir")
async def get_models_cache_dir():
    """Get the path to the HuggingFace model cache directory."""
    from huggingface_hub import constants as hf_constants

    return {"path": str(Path(hf_constants.HF_HUB_CACHE))}


@router.post("/models/migrate")
async def migrate_models(request: models.ModelMigrateRequest):
    """Move all downloaded models to a new directory with byte-level progress via SSE."""
    from huggingface_hub import constants as hf_constants

    global _model_migration_task

    source = Path(hf_constants.HF_HUB_CACHE)
    destination = Path(request.destination)

    if not source.exists():
        raise HTTPException(status_code=404, detail="Current model cache directory not found")

    if source.resolve() == destination.resolve():
        raise HTTPException(status_code=400, detail="Source and destination are the same directory")

    if destination.resolve().is_relative_to(source.resolve()):
        raise HTTPException(status_code=400, detail="Destination cannot be inside the current cache directory")

    progress_manager = get_progress_manager()
    scan_complete = asyncio.get_running_loop().create_future()

    async def migrate_background():
        moved = 0
        errors = []
        try:
            async with mlx_tts_lifecycle_guard.hold("model cache migration"):
                if not source.exists():
                    raise FileNotFoundError("Current model cache directory not found")
                model_dirs = [item for item in source.iterdir() if item.name.startswith("models--") and item.is_dir()]
                if not scan_complete.done():
                    scan_complete.set_result(len(model_dirs))
                if not model_dirs:
                    progress_manager.update_progress("migration", 1, 1, status="complete")
                    progress_manager.mark_complete("migration")
                    return

                destination.mkdir(parents=True, exist_ok=True)
                same_fs = False
                with suppress(OSError):
                    same_fs = source.stat().st_dev == destination.stat().st_dev

                if same_fs:
                    total = len(model_dirs)
                    for i, item in enumerate(model_dirs):
                        dest_item = destination / item.name
                        try:
                            await run_blocking_operation_cancellation_safe(
                                _move_model_cache_directory,
                                item,
                                dest_item,
                            )
                            moved += 1
                            progress_manager.update_progress(
                                "migration",
                                i + 1,
                                total,
                                filename=item.name,
                                status="downloading",
                            )
                        except Exception as e:
                            errors.append(f"{item.name}: {e!s}")
                else:
                    total_bytes = await run_blocking_operation_cancellation_safe(
                        lambda: sum(_get_dir_size(item) for item in model_dirs)
                    )
                    progress_manager.update_progress(
                        "migration",
                        0,
                        total_bytes,
                        filename="Calculating...",
                        status="downloading",
                    )

                    copied = 0
                    for item in model_dirs:
                        dest_item = destination / item.name
                        try:
                            if dest_item.exists():
                                await run_blocking_operation_cancellation_safe(
                                    shutil.rmtree,
                                    dest_item,
                                )
                            copied = await run_blocking_operation_cancellation_safe(
                                _copy_with_progress,
                                item,
                                dest_item,
                                progress_manager,
                                copied,
                                total_bytes,
                            )
                            await run_blocking_operation_cancellation_safe(
                                shutil.rmtree,
                                item,
                            )
                            moved += 1
                        except Exception as e:
                            errors.append(f"{item.name}: {e!s}")

                if errors:
                    raise RuntimeError("; ".join(errors))
                progress_manager.update_progress(
                    "migration",
                    1,
                    1,
                    status="complete",
                )
                progress_manager.mark_complete("migration")
        except asyncio.CancelledError:
            if not scan_complete.done():
                scan_complete.cancel()
            progress_manager.update_progress("migration", 0, 0, status="error")
            progress_manager.mark_error("migration", "Model migration cancelled")
            raise
        except Exception as e:
            if not scan_complete.done():
                scan_complete.set_exception(e)
            progress_manager.update_progress("migration", 0, 0, status="error")
            progress_manager.mark_error("migration", str(e))

    with _model_download_tasks_lock:
        if _model_migration_task is not None and not _model_migration_task.done():
            raise HTTPException(
                status_code=409,
                detail="A model cache migration is already running or cancelling",
            )
        progress_manager.update_progress(
            "migration",
            0,
            0,
            filename="Preparing model cache migration...",
            status="downloading",
        )
        _model_migration_task = create_background_task(migrate_background())
        migration_task = _model_migration_task
        migration_task.add_done_callback(_finish_model_migration_task)

    try:
        model_count = await asyncio.shield(scan_complete)
    except asyncio.CancelledError:
        migration_task.cancel()
        while not migration_task.done():
            try:
                await asyncio.shield(migration_task)
            except asyncio.CancelledError:
                continue
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if model_count == 0:
        return {
            "moved": 0,
            "errors": [],
            "source": str(source),
            "destination": str(destination),
        }

    return {"source": str(source), "destination": str(destination)}


@router.get("/models/migrate/progress")
async def get_migration_progress():
    """Get model migration progress via Server-Sent Events."""
    progress_manager = get_progress_manager()

    async def event_generator():
        async for event in progress_manager.subscribe("migration"):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/models/status", response_model=models.ModelStatusListResponse)
async def get_model_status():
    """Get status of all available models."""
    from huggingface_hub import constants as hf_constants

    task_manager = get_task_manager()

    # Pending only — an errored task stays in the active list for the
    # error/retry UI, but reporting it as "downloading" here would mask
    # the model's real cache state until the app restarts (issue #925).
    active_download_names = {task.model_name for task in task_manager.get_pending_downloads()}

    try:
        from huggingface_hub import scan_cache_dir

        use_scan_cache = True
    except ImportError:
        use_scan_cache = False

    from ..backends import check_model_loaded, get_all_model_configs

    registry_configs = get_all_model_configs()
    model_configs = [
        {
            "model_name": cfg.model_name,
            "display_name": cfg.display_name,
            "hf_repo_id": cfg.hf_repo_id,
            "model_size": cfg.model_size,
            "check_loaded": lambda c=cfg: check_model_loaded(c),
        }
        for cfg in registry_configs
    ]

    model_to_repo = {cfg["model_name"]: cfg["hf_repo_id"] for cfg in model_configs}
    active_download_repos = {model_to_repo.get(name) for name in active_download_names if name in model_to_repo}

    cache_info = None
    if use_scan_cache:
        with suppress(Exception):
            cache_info = scan_cache_dir()

    statuses = []

    for config in model_configs:
        try:
            downloaded = False
            size_mb = None
            loaded = False

            if cache_info:
                repo_id = config["hf_repo_id"]
                for repo in cache_info.repos:
                    if repo.repo_id == repo_id:
                        has_model_weights = False
                        for rev in repo.revisions:
                            for f in rev.files:
                                fname = f.file_name.lower()
                                if fname.endswith((".safetensors", ".bin", ".pt", ".pth", ".npz")):
                                    has_model_weights = True
                                    break
                            if has_model_weights:
                                break

                        has_incomplete = False
                        try:
                            cache_dir = hf_constants.HF_HUB_CACHE
                            blobs_dir = Path(cache_dir) / ("models--" + repo_id.replace("/", "--")) / "blobs"
                            if blobs_dir.exists():
                                has_incomplete = any(blobs_dir.glob("*.incomplete"))
                        except Exception:
                            pass

                        if has_model_weights and not has_incomplete:
                            downloaded = True
                            try:
                                total_size = sum(revision.size_on_disk for revision in repo.revisions)
                                size_mb = total_size / (1024 * 1024)
                            except Exception:
                                pass
                        break

            if not downloaded:
                try:
                    cache_dir = hf_constants.HF_HUB_CACHE
                    repo_cache = Path(cache_dir) / ("models--" + config["hf_repo_id"].replace("/", "--"))

                    if repo_cache.exists():
                        blobs_dir = repo_cache / "blobs"
                        has_incomplete = blobs_dir.exists() and any(blobs_dir.glob("*.incomplete"))

                        if not has_incomplete:
                            snapshots_dir = repo_cache / "snapshots"
                            has_model_files = False
                            if snapshots_dir.exists():
                                has_model_files = (
                                    any(snapshots_dir.rglob("*.bin"))
                                    or any(snapshots_dir.rglob("*.safetensors"))
                                    or any(snapshots_dir.rglob("*.pt"))
                                    or any(snapshots_dir.rglob("*.pth"))
                                    or any(snapshots_dir.rglob("*.npz"))
                                )

                            if has_model_files:
                                downloaded = True
                                try:
                                    total_size = sum(
                                        f.stat().st_size
                                        for f in repo_cache.rglob("*")
                                        if f.is_file() and not f.name.endswith(".incomplete")
                                    )
                                    size_mb = total_size / (1024 * 1024)
                                except Exception:
                                    pass
                except Exception:
                    pass

            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False

            is_downloading = config["hf_repo_id"] in active_download_repos

            if is_downloading:
                downloaded = False
                size_mb = None

            statuses.append(
                models.ModelStatus(
                    model_name=config["model_name"],
                    display_name=config["display_name"],
                    hf_repo_id=config["hf_repo_id"],
                    downloaded=downloaded,
                    downloading=is_downloading,
                    size_mb=size_mb,
                    loaded=loaded,
                )
            )
        except Exception:
            try:
                loaded = config["check_loaded"]()
            except Exception:
                loaded = False

            is_downloading = config["hf_repo_id"] in active_download_repos

            statuses.append(
                models.ModelStatus(
                    model_name=config["model_name"],
                    display_name=config["display_name"],
                    hf_repo_id=config["hf_repo_id"],
                    downloaded=False,
                    downloading=is_downloading,
                    size_mb=None,
                    loaded=loaded,
                )
            )

    return models.ModelStatusListResponse(models=statuses)


@router.post("/models/download")
async def trigger_model_download(request: models.ModelDownloadRequest):
    """Trigger download of a specific model."""
    from ..backends import (
        get_model_config,
        get_model_load_func,
        get_tts_backend_for_engine,
    )

    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    config = get_model_config(request.model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {request.model_name}")

    load_func = get_model_load_func(config)

    async def download_in_background():
        try:
            async with mlx_tts_lifecycle_guard.hold("model loading"):
                result = load_func()
                if asyncio.iscoroutine(result):
                    if config.engine == "whisper":
                        from ..services import transcribe

                        operation_backend = transcribe.get_whisper_model()
                    elif config.engine == "qwen_llm":
                        from ..services import llm

                        operation_backend = llm.get_llm_model()
                    else:
                        operation_backend = get_tts_backend_for_engine(config.engine)
                    await run_tts_operation_cancellation_safe(
                        operation_backend,
                        result,
                    )
            task_manager.complete_download(request.model_name)
        except asyncio.CancelledError:
            # The cancellation-safe backend wrapper has already drained any
            # executor thread before this task reaches its terminal state.
            raise
        except Exception as e:
            task_manager.error_download(request.model_name, str(e))

    try:
        start_owned_model_download_task(
            request.model_name,
            download_in_background(),
            task_manager=task_manager,
            progress_manager=progress_manager,
            task_factory=create_background_task,
        )
    except ModelDownloadAlreadyActiveError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Model {request.model_name} download is already running or cancelling",
        ) from exc

    progress_manager.update_progress(
        model_name=request.model_name,
        current=0,
        total=0,
        filename="Connecting to HuggingFace...",
        status="downloading",
    )

    return {"message": f"Model {request.model_name} download started"}


@router.post("/models/download/cancel")
async def cancel_model_download(request: models.ModelDownloadRequest):
    """Cancel or dismiss an errored/stale download task."""
    task_manager = get_task_manager()
    progress_manager = get_progress_manager()

    task = _owned_model_download_task(request.model_name)
    if task is not None and not task.done():
        task.cancel()
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                # The request itself was cancelled. The owned background task
                # keeps draining and remains tracked until its done callback.
                raise
        _finish_model_download_task(
            request.model_name,
            task,
            task_manager,
            progress_manager,
        )
        return {"message": f"Download task for {request.model_name} cancelled"}

    if task is not None:
        _finish_model_download_task(
            request.model_name,
            task,
            task_manager,
            progress_manager,
        )

    removed = task_manager.cancel_download(request.model_name)

    progress_removed = _remove_model_download_progress(
        progress_manager,
        request.model_name,
    )

    if removed or progress_removed:
        return {"message": f"Download task for {request.model_name} cancelled"}
    return {"message": f"No active task found for {request.model_name}"}


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a downloaded model from the HuggingFace cache."""
    from huggingface_hub import constants as hf_constants

    from ..backends import get_model_config, unload_model_by_config

    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    hf_repo_id = config.hf_repo_id

    try:
        # Acquisition and mutation are one critical section: inference cannot
        # start after an unloaded check but before the cache directory removal.
        with mlx_tts_lifecycle_guard.try_hold("model cache deletion"):
            unload_model_by_config(config)

            cache_dir = hf_constants.HF_HUB_CACHE
            repo_cache_dir = Path(cache_dir) / ("models--" + hf_repo_id.replace("/", "--"))

            if not repo_cache_dir.exists():
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found in cache")

            try:
                shutil.rmtree(repo_cache_dir)
            except OSError as e:
                raise HTTPException(status_code=500, detail=f"Failed to delete model cache directory: {e!s}") from e

        return {"message": f"Model {model_name} deleted successfully"}

    except MLXTTSLifecycleBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {e!s}") from e
