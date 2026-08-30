"""
CUDA backend download, assembly, and verification.

Downloads two archives from GitHub Releases:
  1. Server core (voicebox-server-cuda.tar.gz) — the exe + non-NVIDIA deps,
     versioned with the app.
  2. CUDA libs (cuda-libs-{version}.tar.gz) — NVIDIA runtime libraries,
     versioned independently (only redownloaded on CUDA toolkit bump).

Both archives are extracted into {data_dir}/backends/cuda/ which forms the
complete PyInstaller --onedir directory structure that torch expects.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path

from .. import __version__
from ..config import get_data_dir
from ..utils.backend_archive import (
    BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES,
    BACKEND_ARCHIVE_MIN_FREE_BYTES,
    BackendArchiveError,
    backend_directory_allocation_bytes,
    commit_backend_install,
    copy_backend_directory,
    delete_backend_install,
    download_and_extract_backend_archive,
    recover_backend_install,
    remove_backend_directory,
    run_blocking_cancellation_safe,
)
from ..utils.disk_reservations import DiskSpaceReservation, DiskSpaceReservationError, reserve_disk_space
from ..utils.progress import get_progress_manager

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = "https://github.com/jamiepine/voicebox/releases/download"

PROGRESS_KEY = "cuda-backend"

CUDA_DOWNLOAD_UNSUPPORTED_REASON = "Downloadable CUDA backend releases are currently only published for Windows."

# The current expected CUDA libs version.  Bump this when we change the
# CUDA toolkit version or torch's CUDA dependency changes (e.g. cu126 -> cu128).
CUDA_LIBS_VERSION = "cu128-v1"

_operation_state_lock = threading.Lock()
_active_operation: tuple[object, str] | None = None


class BackendOperationBusyError(RuntimeError):
    """Raised when a conflicting CUDA install operation already owns storage."""


class BackendAlreadyInstalledError(RuntimeError):
    """Raised when a manual install targets an already installed backend."""


def _reserve_operation(operation: str) -> object:
    global _active_operation

    token = object()
    with _operation_state_lock:
        if _active_operation is not None:
            raise BackendOperationBusyError(f"CUDA backend {_active_operation[1]} operation already in progress")
        _active_operation = (token, operation)
    return token


def _release_operation(token: object) -> None:
    global _active_operation

    with _operation_state_lock:
        if _active_operation is not None and _active_operation[0] is token:
            _active_operation = None


def _active_operation_name() -> str | None:
    with _operation_state_lock:
        return _active_operation[1] if _active_operation is not None else None


def get_backends_dir() -> Path:
    """Directory where downloaded backend binaries are stored."""
    d = get_data_dir() / "backends"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cuda_dir() -> Path:
    """Directory where the CUDA backend (onedir) is extracted."""
    d = get_backends_dir() / "cuda"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cuda_exe_name() -> str:
    """Platform-specific CUDA executable filename."""
    if sys.platform == "win32":
        return "voicebox-server-cuda.exe"
    return "voicebox-server-cuda"


def is_cuda_download_supported() -> bool:
    """Return whether this platform has a matching CUDA release asset."""
    return sys.platform == "win32"


def get_cuda_download_unsupported_reason() -> str | None:
    """Explain why this platform cannot use the release-download flow."""
    if is_cuda_download_supported():
        return None
    return CUDA_DOWNLOAD_UNSUPPORTED_REASON


def ensure_cuda_download_supported() -> None:
    """Raise if downloading would fetch an asset built for another platform."""
    reason = get_cuda_download_unsupported_reason()
    if reason:
        raise RuntimeError(reason)


def get_cuda_binary_path() -> Path | None:
    """Return path to the CUDA executable if it exists inside the onedir."""
    p = get_cuda_dir() / get_cuda_exe_name()
    if p.exists():
        return p
    return None


def get_cuda_libs_manifest_path() -> Path:
    """Path to the cuda-libs.json manifest inside the CUDA dir."""
    return get_cuda_dir() / "cuda-libs.json"


def get_installed_cuda_libs_version() -> str | None:
    """Read the installed CUDA libs version from cuda-libs.json, or None."""
    manifest_path = get_cuda_libs_manifest_path()
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        return data.get("version")
    except Exception as e:
        logger.warning(f"Could not read cuda-libs.json: {e}")
        return None


def is_cuda_active() -> bool:
    """Check if the current process is the CUDA binary.

    The CUDA binary sets this env var on startup (see server.py).
    """
    return os.environ.get("VOICEBOX_BACKEND_VARIANT") == "cuda"


def get_cuda_status() -> dict:
    """Get current CUDA backend status for the API."""
    progress_manager = get_progress_manager()
    cuda_path = get_cuda_binary_path()
    progress = progress_manager.get_progress(PROGRESS_KEY)
    cuda_libs_version = get_installed_cuda_libs_version()
    unsupported_reason = get_cuda_download_unsupported_reason()
    active_operation = _active_operation_name()

    return {
        "available": cuda_path is not None,
        "active": is_cuda_active(),
        "binary_path": str(cuda_path) if cuda_path else None,
        "cuda_libs_version": cuda_libs_version,
        "download_supported": unsupported_reason is None,
        "unsupported_reason": unsupported_reason,
        "downloading": active_operation in {"download", "update"},
        "operation": active_operation,
        "download_progress": progress,
    }


def _needs_server_download(version: str | None = None) -> bool:
    """Check if the server core archive needs to be (re)downloaded."""
    cuda_path = get_cuda_binary_path()
    if not cuda_path:
        return True
    # Check if the binary version matches the expected app version
    installed = get_cuda_binary_version()
    expected = version or __version__
    if expected.startswith("v"):
        expected = expected[1:]
    return installed != expected


def _needs_cuda_libs_download() -> bool:
    """Check if the CUDA libs archive needs to be (re)downloaded."""
    installed = get_installed_cuda_libs_version()
    if installed is None:
        return True
    return installed != CUDA_LIBS_VERSION


async def _download_and_extract_archive(
    client,
    url: str,
    sha256_url: str | None,
    dest_dir: Path,
    label: str,
    progress_offset: int,
    total_size: int,
    storage_reservation: DiskSpaceReservation | None = None,
):
    """Bind CUDA progress and policy to the shared archive pipeline."""
    progress = get_progress_manager()

    def update_progress(*, current: int, total: int, filename: str, status: str) -> None:
        progress.update_progress(
            PROGRESS_KEY,
            current=current,
            total=total,
            filename=filename,
            status=status,
        )

    return await download_and_extract_backend_archive(
        client,
        url=url,
        sha256_url=sha256_url,
        dest_dir=dest_dir,
        label=label,
        progress_offset=progress_offset,
        total_size=total_size,
        update_progress=update_progress,
        log_info=logger.info,
        storage_reservation=storage_reservation,
        max_compressed_bytes=BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES,
        min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
    )


async def download_cuda_binary(version: str | None = None):
    """Download the CUDA backend (server core + CUDA libs if needed).

    Downloads both archives from GitHub Releases, extracts them into
    {data_dir}/backends/cuda/, and writes the cuda-libs.json manifest.

    Only downloads what's needed:
    - Server core: always redownloaded (versioned with app)
    - CUDA libs: only if missing or version mismatch

    Args:
        version: Version tag (e.g. "v0.3.0"). Defaults to current app version.
    """
    token = _reserve_operation("download")
    try:
        await _download_cuda_binary_locked(version)
    finally:
        _release_operation(token)


def schedule_cuda_binary_download(version: str | None = None) -> asyncio.Task:
    """Reserve storage now and schedule a manually requested download."""
    from .task_queue import create_background_task

    ensure_cuda_download_supported()
    token = _reserve_operation("download")
    operation = None
    try:
        if get_cuda_binary_path() is not None:
            raise BackendAlreadyInstalledError("CUDA backend already downloaded")
        operation = _download_cuda_binary_locked(version)
        task = create_background_task(operation)
    except BaseException:
        if operation is not None:
            operation.close()
        _release_operation(token)
        raise

    def _operation_finished(completed: asyncio.Task) -> None:
        _release_operation(token)
        if completed.cancelled():
            return
        try:
            completed.result()
        except Exception:
            logger.exception("CUDA download failed")

    task.add_done_callback(_operation_finished)
    return task


async def _download_cuda_binary_locked(version: str | None = None):
    """Inner implementation called only while an operation is reserved."""
    ensure_cuda_download_supported()

    import httpx

    if version is None:
        version = f"v{__version__}"

    progress = get_progress_manager()
    backends_dir = get_backends_dir()
    await run_blocking_cancellation_safe(
        recover_backend_install,
        backends_dir,
        "cuda",
        get_cuda_exe_name(),
    )
    cuda_dir = get_cuda_dir()

    need_server = await run_blocking_cancellation_safe(_needs_server_download, version)
    need_libs = _needs_cuda_libs_download()

    if not need_server and not need_libs:
        logger.info("CUDA backend is up to date, nothing to download")
        return

    logger.info(
        f"Starting CUDA backend download for {version} "
        f"(server={'yes' if need_server else 'cached'}, "
        f"libs={'yes' if need_libs else 'cached'})"
    )
    progress.update_progress(
        PROGRESS_KEY,
        current=0,
        total=0,
        filename="Preparing download...",
        status="downloading",
    )

    base_url = f"{GITHUB_RELEASES_URL}/{version}"
    server_archive = "voicebox-server-cuda.tar.gz"
    libs_archive = f"cuda-libs-{CUDA_LIBS_VERSION}.tar.gz"
    staging_dir = backends_dir / "cuda-staging"
    storage_reservation = None

    try:
        await run_blocking_cancellation_safe(remove_backend_directory, staging_dir)
        staging_bytes = await run_blocking_cancellation_safe(backend_directory_allocation_bytes, cuda_dir)
        try:
            storage_reservation = reserve_disk_space(
                backends_dir,
                staging_bytes,
                min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise BackendArchiveError("Insufficient shared capacity to stage the CUDA backend") from exc
        if cuda_dir.exists():
            await run_blocking_cancellation_safe(copy_backend_directory, cuda_dir, staging_dir)
        else:
            staging_dir.mkdir(parents=True, mode=0o700)
        storage_reservation.resize(
            0,
            directory=backends_dir,
            min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
        )

        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            # Estimate total download size
            total_size = 0
            if need_server:
                try:
                    head = await client.head(f"{base_url}/{server_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass
            if need_libs:
                try:
                    head = await client.head(f"{base_url}/{libs_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass

            logger.info(f"Total download size: {total_size / 1024 / 1024:.1f} MB")

            offset = 0

            # Download server core
            if need_server:
                server_downloaded = await _download_and_extract_archive(
                    client,
                    url=f"{base_url}/{server_archive}",
                    sha256_url=f"{base_url}/{server_archive}.sha256",
                    dest_dir=staging_dir,
                    label="CUDA server",
                    progress_offset=offset,
                    total_size=total_size,
                    storage_reservation=storage_reservation,
                )
                offset += server_downloaded

                # Make executable on Unix
                exe_path = staging_dir / get_cuda_exe_name()
                if sys.platform != "win32" and exe_path.exists():
                    exe_path.chmod(0o755)

            # Download CUDA libs
            if need_libs:
                await _download_and_extract_archive(
                    client,
                    url=f"{base_url}/{libs_archive}",
                    sha256_url=f"{base_url}/{libs_archive}.sha256",
                    dest_dir=staging_dir,
                    label="CUDA libraries",
                    progress_offset=offset,
                    total_size=total_size,
                    storage_reservation=storage_reservation,
                )

                # Write local cuda-libs.json manifest
                manifest = {"version": CUDA_LIBS_VERSION}
                (staging_dir / "cuda-libs.json").write_text(json.dumps(manifest, indent=2) + "\n")

        await run_blocking_cancellation_safe(
            commit_backend_install,
            backends_dir,
            "cuda",
            get_cuda_exe_name(),
        )

        logger.info(f"CUDA backend ready at {cuda_dir}")
        progress.mark_complete(PROGRESS_KEY)

    except BaseException as error:
        await run_blocking_cancellation_safe(remove_backend_directory, staging_dir)
        if isinstance(error, Exception):
            logger.error(f"CUDA backend download failed: {error}")
            progress.mark_error(PROGRESS_KEY, str(error))
        raise
    finally:
        if storage_reservation is not None:
            storage_reservation.release()


def get_cuda_binary_version() -> str | None:
    """Get the version of the installed CUDA binary, or None if not installed."""
    import subprocess

    cuda_path = get_cuda_binary_path()
    if not cuda_path:
        return None
    try:
        result = subprocess.run(
            [str(cuda_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(cuda_path.parent),  # Run from the onedir directory
        )
        # Output format: "voicebox-server 0.3.0"
        for line in result.stdout.strip().splitlines():
            if "voicebox-server" in line:
                return line.split()[-1]
    except Exception as e:
        logger.warning(f"Could not get CUDA binary version: {e}")
    return None


async def check_and_update_cuda_binary():
    """Check if the CUDA binary is outdated and auto-download if so.

    Called on server startup. Checks both server version and CUDA libs
    version. Downloads only what's needed.
    """
    unsupported_reason = get_cuda_download_unsupported_reason()
    if unsupported_reason:
        logger.info("Skipping CUDA backend auto-update: %s", unsupported_reason)
        return

    try:
        token = _reserve_operation("update")
    except BackendOperationBusyError:
        logger.info("Skipping CUDA backend auto-update while another operation is active")
        return

    try:
        await run_blocking_cancellation_safe(
            recover_backend_install,
            get_backends_dir(),
            "cuda",
            get_cuda_exe_name(),
        )
        cuda_path = get_cuda_binary_path()
        if not cuda_path:
            return  # No CUDA binary installed, nothing to update

        need_server = await run_blocking_cancellation_safe(_needs_server_download)
        need_libs = _needs_cuda_libs_download()

        if not need_server and not need_libs:
            logger.info(f"CUDA binary is up to date (server=v{__version__}, libs={get_installed_cuda_libs_version()})")
            return

        reasons = []
        if need_server:
            cuda_version = await run_blocking_cancellation_safe(get_cuda_binary_version)
            reasons.append(f"server v{cuda_version} != v{__version__}")
        if need_libs:
            installed_libs = get_installed_cuda_libs_version()
            reasons.append(f"libs {installed_libs} != {CUDA_LIBS_VERSION}")

        logger.info(f"CUDA backend needs update ({', '.join(reasons)}). Auto-downloading...")

        try:
            await _download_cuda_binary_locked()
        except Exception as error:
            logger.error(f"Auto-update of CUDA binary failed: {error}")
    finally:
        _release_operation(token)


async def recover_cuda_backend_install() -> None:
    """Reconcile an interrupted CUDA directory swap before serving requests."""
    token = _reserve_operation("recovery")
    try:
        await run_blocking_cancellation_safe(
            recover_backend_install,
            get_backends_dir(),
            "cuda",
            get_cuda_exe_name(),
        )
    finally:
        _release_operation(token)


async def delete_cuda_binary() -> bool:
    """Delete the downloaded CUDA backend directory. Returns True if deleted."""
    token = _reserve_operation("delete")
    try:
        deleted = await run_blocking_cancellation_safe(
            delete_backend_install,
            get_backends_dir(),
            "cuda",
            get_cuda_exe_name(),
        )
        if deleted:
            logger.info("Deleted CUDA backend installation")
        return deleted
    finally:
        _release_operation(token)
