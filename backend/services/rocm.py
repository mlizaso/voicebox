"""
ROCm backend download, assembly, and verification.

Downloads two archives from GitHub Releases:
  1. Server core (voicebox-server-rocm.tar.gz) — the exe + non-AMD deps,
     versioned with the app.
  2. ROCm libs (rocm-libs-{version}.tar.gz) — AMD runtime libraries,
     versioned independently (only redownloaded on ROCm toolkit bump).

Both archives are extracted into {data_dir}/backends/rocm/ which forms the
complete PyInstaller --onedir directory structure that torch expects.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import suppress
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
    extract_backend_tar_archive,
    inspect_backend_tar_archive,
    recover_backend_install,
    remove_backend_directory,
    run_blocking_cancellation_safe,
    sha256_backend_file,
)
from ..utils.disk_reservations import DiskSpaceReservation, DiskSpaceReservationError, reserve_disk_space
from ..utils.progress import get_progress_manager

logger = logging.getLogger(__name__)

GITHUB_RELEASES_URL = "https://github.com/jamiepine/voicebox/releases/download"

PROGRESS_KEY = "rocm-backend"

# The current expected ROCm libs version.  Bump this when we change the
# ROCm toolkit version or torch's ROCm dependency changes (e.g. rocm7.2 -> rocm7.4).
ROCM_LIBS_VERSION = "rocm7.2-v1"

_operation_state_lock = threading.Lock()
_active_operation: tuple[object, str] | None = None


class BackendOperationBusyError(RuntimeError):
    """Raised when a conflicting ROCm install operation already owns storage."""


def _reserve_operation(operation: str) -> object:
    global _active_operation

    token = object()
    with _operation_state_lock:
        if _active_operation is not None:
            raise BackendOperationBusyError(f"ROCm backend {_active_operation[1]} operation already in progress")
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


def get_rocm_dir() -> Path:
    """Directory where the ROCm backend (onedir) is extracted."""
    d = get_backends_dir() / "rocm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_rocm_exe_name() -> str:
    """Platform-specific ROCm executable filename."""
    if sys.platform == "win32":
        return "voicebox-server-rocm.exe"
    return "voicebox-server-rocm"


def get_rocm_binary_path() -> Path | None:
    """Return path to the ROCm executable if it exists inside the onedir."""
    p = get_rocm_dir() / get_rocm_exe_name()
    if p.exists():
        return p
    return None


def get_rocm_libs_manifest_path() -> Path:
    """Path to the rocm-libs.json manifest inside the ROCm dir."""
    return get_rocm_dir() / "rocm-libs.json"


def get_installed_rocm_libs_version() -> str | None:
    """Read the installed ROCm libs version from rocm-libs.json, or None."""
    manifest_path = get_rocm_libs_manifest_path()
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text())
        return data.get("version")
    except Exception as e:
        logger.warning(f"Could not read rocm-libs.json: {e}")
        return None


def is_rocm_active() -> bool:
    """Check if the current process is the ROCm binary.

    The ROCm binary sets this env var on startup (see server.py).
    """
    return os.environ.get("VOICEBOX_BACKEND_VARIANT") == "rocm"


def get_rocm_status() -> dict:
    """Get current ROCm backend status for the API."""
    progress_manager = get_progress_manager()
    rocm_path = get_rocm_binary_path()
    progress = progress_manager.get_progress(PROGRESS_KEY)
    rocm_libs_version = get_installed_rocm_libs_version()
    active_operation = _active_operation_name()

    return {
        "available": rocm_path is not None,
        "active": is_rocm_active(),
        "binary_path": str(rocm_path) if rocm_path else None,
        "rocm_libs_version": rocm_libs_version,
        "downloading": active_operation in {"download", "update"},
        "operation": active_operation,
        "download_progress": progress,
    }


def _needs_server_download(version: str | None = None) -> bool:
    """Check if the server core archive needs to be (re)downloaded."""
    rocm_path = get_rocm_binary_path()
    if not rocm_path:
        return True
    # Check if the binary version matches the expected app version
    installed = get_rocm_binary_version()
    expected = version or __version__
    if expected.startswith("v"):
        expected = expected[1:]
    return installed != expected


def _needs_rocm_libs_download() -> bool:
    """Check if the ROCm libs archive needs to be (re)downloaded."""
    installed = get_installed_rocm_libs_version()
    if installed is None:
        return True
    return installed != ROCM_LIBS_VERSION


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
    """Download a .tar.gz archive and extract it into dest_dir.

    Args:
        client: httpx.AsyncClient
        url: URL of the .tar.gz archive
        sha256_url: URL of the .sha256 checksum file (optional)
        dest_dir: Directory to extract into
        label: Human-readable label for progress updates
        progress_offset: Byte offset for progress reporting (when downloading
            multiple archives sequentially)
        total_size: Total bytes across all downloads (for progress bar)
    """
    progress = get_progress_manager()
    temp_path = dest_dir / f".download-{label.replace(' ', '-')}.tmp"
    owns_reservation = storage_reservation is None
    try:
        if storage_reservation is None:
            storage_reservation = reserve_disk_space(
                dest_dir,
                0,
                min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
            )
    except DiskSpaceReservationError as exc:
        raise BackendArchiveError("Insufficient shared capacity for backend download") from exc

    try:
        # Clean up leftover partial download.
        with suppress(FileNotFoundError):
            temp_path.unlink()

        # Fetch the expected checksum before reserving the archive payload.
        expected_sha = None
        if sha256_url:
            try:
                sha_resp = await client.get(sha256_url)
                sha_resp.raise_for_status()
                expected_sha = sha_resp.text.strip().split()[0].casefold()
                if len(expected_sha) != 64 or any(character not in "0123456789abcdef" for character in expected_sha):
                    raise ValueError("checksum response does not contain a SHA-256 digest")
                logger.info(f"{label}: expected SHA-256: {expected_sha[:16]}...")
            except Exception as error:
                raise RuntimeError(f"{label}: failed to fetch checksum from {sha256_url}") from error
    except BaseException:
        if owns_reservation and storage_reservation is not None:
            storage_reservation.release()
        raise

    # Stream download, verify, and extract — always clean up temp file
    downloaded = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            advertised_size = None
            if content_length is not None:
                try:
                    advertised_size = int(content_length)
                except ValueError as exc:
                    raise BackendArchiveError(f"{label} returned an invalid Content-Length") from exc
                if advertised_size < 0 or advertised_size > BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES:
                    raise BackendArchiveError(
                        f"{label} exceeds the compressed archive size limit "
                        f"({BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES} bytes)"
                    )
            projected_download_bytes = (
                advertised_size if advertised_size is not None else BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES
            )
            try:
                storage_reservation.resize(
                    projected_download_bytes,
                    directory=dest_dir,
                    min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
                )
            except DiskSpaceReservationError as exc:
                raise BackendArchiveError("Insufficient shared capacity for backend download") from exc

            open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            open_flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temp_path, open_flags, 0o600)
            with os.fdopen(descriptor, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    next_downloaded = downloaded + len(chunk)
                    if next_downloaded > BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES:
                        raise BackendArchiveError(
                            f"{label} exceeds the compressed archive size limit "
                            f"({BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES} bytes)"
                        )
                    if advertised_size is not None and next_downloaded > advertised_size:
                        raise BackendArchiveError(f"{label} exceeded its advertised download size")
                    f.write(chunk)
                    downloaded = next_downloaded
                    progress.update_progress(
                        PROGRESS_KEY,
                        current=progress_offset + downloaded,
                        total=total_size,
                        filename=f"Downloading {label}",
                        status="downloading",
                    )
            if advertised_size is not None and downloaded != advertised_size:
                raise BackendArchiveError(f"{label} did not match its advertised download size")

        # Verify integrity
        if expected_sha:
            progress.update_progress(
                PROGRESS_KEY,
                current=progress_offset + downloaded,
                total=total_size,
                filename=f"Verifying {label}...",
                status="downloading",
            )
            actual = await run_blocking_cancellation_safe(
                sha256_backend_file,
                temp_path,
                cooperative=True,
            )
            if actual != expected_sha:
                raise ValueError(
                    f"{label} integrity check failed: expected {expected_sha[:16]}..., got {actual[:16]}..."
                )
            logger.info(f"{label}: integrity verified")

        # Preflight every member before streaming regular files to disk. This
        # contract is independent of Python's version-specific tar filters.
        progress.update_progress(
            PROGRESS_KEY,
            current=progress_offset + downloaded,
            total=total_size,
            filename=f"Extracting {label}...",
            status="downloading",
        )
        required_extraction_bytes = await run_blocking_cancellation_safe(
            inspect_backend_tar_archive,
            temp_path,
            dest_dir,
            cooperative=True,
        )
        try:
            storage_reservation.resize(
                required_extraction_bytes,
                directory=dest_dir,
                min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise BackendArchiveError("Insufficient shared capacity for backend extraction") from exc
        await run_blocking_cancellation_safe(
            extract_backend_tar_archive,
            temp_path,
            dest_dir,
            cooperative=True,
        )
        storage_reservation.resize(
            0,
            directory=dest_dir,
            min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
        )

        logger.info(f"{label}: extracted to {dest_dir}")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            if storage_reservation is not None:
                try:
                    storage_reservation.resize(
                        0,
                        directory=dest_dir,
                        min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
                    )
                finally:
                    if owns_reservation:
                        storage_reservation.release()
    return downloaded


async def download_rocm_binary(version: str | None = None):
    """Download the ROCm backend (server core + ROCm libs if needed).

    Downloads both archives from GitHub Releases, extracts them into
    {data_dir}/backends/rocm/, and writes the rocm-libs.json manifest.

    Only downloads what's needed:
    - Server core: always redownloaded (versioned with app)
    - ROCm libs: only if missing or version mismatch

    Args:
        version: Version tag (e.g. "v0.3.0"). Defaults to current app version.
    """
    token = _reserve_operation("download")
    try:
        await _download_rocm_binary_locked(version)
    finally:
        _release_operation(token)


def schedule_rocm_binary_download(version: str | None = None) -> asyncio.Task:
    """Reserve storage now and schedule a manually requested download."""
    from .task_queue import create_background_task

    token = _reserve_operation("download")
    operation = None
    try:
        operation = _download_rocm_binary_locked(version)
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
            logger.exception("ROCm download failed")

    task.add_done_callback(_operation_finished)
    return task


async def _download_rocm_binary_locked(version: str | None = None):
    """Inner implementation called only while an operation is reserved."""
    import httpx

    if version is None:
        version = f"v{__version__}"

    progress = get_progress_manager()
    backends_dir = get_backends_dir()
    await run_blocking_cancellation_safe(
        recover_backend_install,
        backends_dir,
        "rocm",
        get_rocm_exe_name(),
    )
    rocm_dir = get_rocm_dir()

    need_server = await run_blocking_cancellation_safe(_needs_server_download, version)
    need_libs = _needs_rocm_libs_download()

    if not need_server and not need_libs:
        logger.info("ROCm backend is up to date, nothing to download")
        return

    logger.info(
        f"Starting ROCm backend download for {version} "
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

    # Server core and libs archive are both published under the app-version
    # release tag; the libs content version is encoded in the filename only.
    server_base_url = f"{GITHUB_RELEASES_URL}/{version}"
    libs_base_url = server_base_url
    server_archive = "voicebox-server-rocm.tar.gz"
    libs_archive = f"rocm-libs-{ROCM_LIBS_VERSION}.tar.gz"

    # Always stage when any download is needed, then atomically rename over
    # rocm_dir on success. This prevents a failed mid-extraction from leaving
    # rocm_dir in a partially-installed state that still passes the
    # get_rocm_binary_path() existence check. Existing files are pre-copied
    # into staging so partial updates (e.g. libs-only or server-only) preserve
    # whatever isn't being re-downloaded.
    staging_dir = backends_dir / "rocm-staging"
    storage_reservation = None

    try:
        await run_blocking_cancellation_safe(remove_backend_directory, staging_dir)
        staging_bytes = await run_blocking_cancellation_safe(backend_directory_allocation_bytes, rocm_dir)
        try:
            storage_reservation = reserve_disk_space(
                backends_dir,
                staging_bytes,
                min_free_bytes=BACKEND_ARCHIVE_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise BackendArchiveError("Insufficient shared capacity to stage the ROCm backend") from exc
        if rocm_dir.exists():
            await run_blocking_cancellation_safe(copy_backend_directory, rocm_dir, staging_dir)
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
                    head = await client.head(f"{server_base_url}/{server_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass
            if need_libs:
                try:
                    head = await client.head(f"{libs_base_url}/{libs_archive}")
                    total_size += int(head.headers.get("content-length", 0))
                except Exception:
                    pass

            logger.info(f"Total download size: {total_size / 1024 / 1024:.1f} MB")

            offset = 0

            # Download server core
            if need_server:
                server_downloaded = await _download_and_extract_archive(
                    client,
                    url=f"{server_base_url}/{server_archive}",
                    sha256_url=f"{server_base_url}/{server_archive}.sha256",
                    dest_dir=staging_dir,
                    label="ROCm server",
                    progress_offset=offset,
                    total_size=total_size,
                    storage_reservation=storage_reservation,
                )
                offset += server_downloaded

                # Make executable on Unix
                exe_path = staging_dir / get_rocm_exe_name()
                if sys.platform != "win32" and exe_path.exists():
                    exe_path.chmod(0o755)

            # Download ROCm libs
            if need_libs:
                await _download_and_extract_archive(
                    client,
                    url=f"{libs_base_url}/{libs_archive}",
                    sha256_url=f"{libs_base_url}/{libs_archive}.sha256",
                    dest_dir=staging_dir,
                    label="ROCm libraries",
                    progress_offset=offset,
                    total_size=total_size,
                    storage_reservation=storage_reservation,
                )

                # Write local rocm-libs.json manifest
                manifest = {"version": ROCM_LIBS_VERSION}
                (staging_dir / "rocm-libs.json").write_text(json.dumps(manifest, indent=2) + "\n")

        await run_blocking_cancellation_safe(
            commit_backend_install,
            backends_dir,
            "rocm",
            get_rocm_exe_name(),
        )

        logger.info(f"ROCm backend ready at {rocm_dir}")
        progress.mark_complete(PROGRESS_KEY)

    except BaseException as error:
        await run_blocking_cancellation_safe(remove_backend_directory, staging_dir)
        if isinstance(error, Exception):
            logger.error(f"ROCm backend download failed: {error}")
            progress.mark_error(PROGRESS_KEY, str(error))
        raise
    finally:
        if storage_reservation is not None:
            storage_reservation.release()


def get_rocm_binary_version() -> str | None:
    """Get the version of the installed ROCm binary, or None if not installed."""
    import subprocess

    rocm_path = get_rocm_binary_path()
    if not rocm_path:
        return None
    try:
        result = subprocess.run(
            [str(rocm_path), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(rocm_path.parent),  # Run from the onedir directory
        )
        # Output format: "voicebox-server 0.3.0"
        for line in result.stdout.strip().splitlines():
            if "voicebox-server" in line:
                return line.split()[-1]
    except Exception as e:
        logger.warning(f"Could not get ROCm binary version: {e}")
    return None


async def check_and_update_rocm_binary():
    """Check if the ROCm binary is outdated and auto-download if so.

    Called on server startup. Checks both server version and ROCm libs
    version. Downloads only what's needed.
    """
    if is_rocm_active():
        logger.info("ROCm backend is active; skipping auto-update to avoid replacing the running backend")
        return

    try:
        token = _reserve_operation("update")
    except BackendOperationBusyError:
        logger.info("Skipping ROCm backend auto-update while another operation is active")
        return

    try:
        await run_blocking_cancellation_safe(
            recover_backend_install,
            get_backends_dir(),
            "rocm",
            get_rocm_exe_name(),
        )
        rocm_path = get_rocm_binary_path()
        if not rocm_path:
            return  # No ROCm binary installed, nothing to update

        need_server = await run_blocking_cancellation_safe(_needs_server_download)
        need_libs = _needs_rocm_libs_download()

        if not need_server and not need_libs:
            logger.info(f"ROCm binary is up to date (server=v{__version__}, libs={get_installed_rocm_libs_version()})")
            return

        reasons = []
        if need_server:
            rocm_version = await run_blocking_cancellation_safe(get_rocm_binary_version)
            reasons.append(f"server v{rocm_version} != v{__version__}")
        if need_libs:
            installed_libs = get_installed_rocm_libs_version()
            reasons.append(f"libs {installed_libs} != {ROCM_LIBS_VERSION}")

        logger.info(f"ROCm backend needs update ({', '.join(reasons)}). Auto-downloading...")

        try:
            await _download_rocm_binary_locked()
        except Exception as error:
            logger.error(f"Auto-update of ROCm binary failed: {error}")
    finally:
        _release_operation(token)


async def recover_rocm_backend_install() -> None:
    """Reconcile an interrupted ROCm directory swap before serving requests."""
    token = _reserve_operation("recovery")
    try:
        await run_blocking_cancellation_safe(
            recover_backend_install,
            get_backends_dir(),
            "rocm",
            get_rocm_exe_name(),
        )
    finally:
        _release_operation(token)


async def delete_rocm_binary() -> bool:
    """Delete the downloaded ROCm backend directory. Returns True if deleted."""
    token = _reserve_operation("delete")
    try:
        deleted = await run_blocking_cancellation_safe(
            delete_backend_install,
            get_backends_dir(),
            "rocm",
            get_rocm_exe_name(),
        )
        if deleted:
            logger.info("Deleted ROCm backend installation")
        return deleted
    finally:
        _release_operation(token)
