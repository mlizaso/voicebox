"""
Voice profile export/import module.

Handles exporting profiles to ZIP archives and importing them back.
Also handles exporting individual generations.
"""

import io
import json
import logging
import math
import os
import shutil
import stat
import tempfile
import threading
import uuid
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from sqlalchemy.orm import Session

from .. import config
from ..backends.mlx_tts_lifecycle import run_blocking_operation_cancellation_safe
from ..database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)
from ..models import ProfileSampleCreate, VoiceProfileCreate, VoiceProfileResponse
from ..utils.audio import save_audio, validate_and_load_reference_audio
from ..utils.audio_metadata import (
    PORTABLE_AUDIO_MAX_CHANNELS,
    PORTABLE_AUDIO_MAX_DURATION_SECONDS,
    PORTABLE_AUDIO_MAX_SAMPLE_RATE,
    probe_audio_metadata,
)
from ..utils.disk_reservations import (
    DiskSpaceReservation,
    DiskSpaceReservationError,
    reserve_disk_space,
)
from ..utils.images import MAX_FILE_SIZE as AVATAR_MAX_FILE_SIZE, process_avatar, validate_image
from . import deletion_journal
from .profiles import (
    EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES,
    EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES,
    EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES,
    EXACT_VOICE_SNAPSHOT_MAX_TOTAL_REFERENCE_TEXT_BYTES,
    _profile_to_response,
)

logger = logging.getLogger(__name__)

ArchiveSource = bytes | bytearray | memoryview | str | os.PathLike[str]

PROFILE_ARCHIVE_MAX_MEMBERS = 128
PROFILE_ARCHIVE_MAX_TOTAL_BYTES = 512 * 1024 * 1024
PROFILE_ARCHIVE_MAX_ENTRY_BYTES = 64 * 1024 * 1024
PROFILE_ARCHIVE_MAX_SAMPLES = EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES
GENERATION_ARCHIVE_MAX_MEMBERS = 256
GENERATION_ARCHIVE_MAX_TOTAL_BYTES = 512 * 1024 * 1024
GENERATION_ARCHIVE_MAX_ENTRY_BYTES = 512 * 1024 * 1024
GENERATION_AUDIO_MAX_DURATION_SECONDS = PORTABLE_AUDIO_MAX_DURATION_SECONDS
GENERATION_AUDIO_MAX_CHANNELS = PORTABLE_AUDIO_MAX_CHANNELS
GENERATION_AUDIO_MAX_SAMPLE_RATE = PORTABLE_AUDIO_MAX_SAMPLE_RATE
GENERATION_ARCHIVE_AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"})
ARCHIVE_JSON_MAX_BYTES = 1024 * 1024
ARCHIVE_MAX_COMPRESSION_RATIO = 250.0
ARCHIVE_RATIO_MIN_FILE_BYTES = 64 * 1024
ARCHIVE_IO_CHUNK_BYTES = 1024 * 1024
ARCHIVE_EXPORT_MIN_FREE_BYTES = 1024**3
ARCHIVE_EXPORT_OVERHEAD_BYTES = 4 * 1024 * 1024
ARCHIVE_EXPORT_ROOT_NAME = "archive-exports-v1"
ARCHIVE_EXPORT_MAX_ACTIVE = 8
ARCHIVE_IMPORT_MIN_FREE_BYTES = ARCHIVE_EXPORT_MIN_FREE_BYTES
ARCHIVE_IMPORT_OVERHEAD_BYTES = 4 * 1024 * 1024
ARCHIVE_IMPORT_ROOT_NAME = "archive-imports-v1"
ARCHIVE_IMPORT_MAX_CONCURRENT = 2


class ArchiveExportLimitError(ValueError):
    """An export cannot fit within the corresponding import contract."""


class ArchiveExportStorageError(RuntimeError):
    """An export cannot preserve the application's free-space reserve."""


class ArchiveExportBusyError(RuntimeError):
    """The bounded archive build/response capacity is already in use."""


class ArchiveImportStorageError(RuntimeError):
    """An import cannot preserve the application's free-space reserve."""


class ArchiveImportBusyError(RuntimeError):
    """The bounded concurrent archive-import capacity is already in use."""


@dataclass(frozen=True)
class ArchiveExport:
    """A private archive file whose response owner must clean it."""

    path: Path
    temporary_directory: Path

    def cleanup(self) -> None:
        _cleanup_archive_export(self.temporary_directory)


@dataclass(frozen=True)
class _ArchiveExportMember:
    """One preflighted managed file and its immutable filesystem identity."""

    path: Path
    archive_name: str
    size: int
    identity: tuple[int, int, int, int, int]


ArchiveFingerprint = tuple[tuple[str, int, int, int, int, int, int], ...]


@dataclass(frozen=True)
class _ProfileImportSample:
    """One validated profile sample selected from an archive."""

    member_name: str
    filename: str
    reference_text: str
    size: int


@dataclass(frozen=True)
class _ProfileImportPlan:
    """Immutable, side-effect-free profile archive inspection result."""

    fingerprint: ArchiveFingerprint
    profile_data: dict
    samples: tuple[_ProfileImportSample, ...]
    avatar_member_name: str | None
    projected_bytes: int


@dataclass(frozen=True)
class _GenerationImportPlan:
    """Immutable, side-effect-free generation archive inspection result."""

    fingerprint: ArchiveFingerprint
    generation_data: dict
    profile_data: dict
    audio_member_name: str
    audio_suffix: str
    audio_size: int
    projected_bytes: int


@dataclass(frozen=True)
class _ArchiveImportWorkspace:
    """A private import workspace whose coroutine owner must clean it."""

    directory: Path

    def cleanup(self) -> None:
        _cleanup_archive_import(self.directory)


_archive_storage_lock = threading.RLock()
# Keep the historical name for focused tests and internal compatibility.
_archive_export_lock = _archive_storage_lock
_active_archive_export_directories: dict[Path, int] = {}
_active_archive_import_directories: dict[Path, int] = {}
_archive_disk_reservations: dict[Path, DiskSpaceReservation] = {}


@contextmanager
def _open_zip_archive(source: ArchiveSource) -> Iterator[zipfile.ZipFile]:
    """Open an archive without following a caller-supplied final symlink."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        with zipfile.ZipFile(io.BytesIO(bytes(source)), "r") as archive:
            yield archive
        return

    if not isinstance(source, (str, os.PathLike)):
        raise ValueError("Archive source must be bytes or a filesystem path")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(os.fspath(source), flags)
    except OSError as exc:
        raise ValueError("Archive path is not a readable regular file") from exc

    try:
        entry_stat = os.fstat(descriptor)
        if not stat.S_ISREG(entry_stat.st_mode):
            raise ValueError("Archive path is not a regular file")
        with os.fdopen(descriptor, "rb") as archive_file:
            descriptor = -1
            with zipfile.ZipFile(archive_file, "r") as archive:
                yield archive
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _canonical_zip_member_name(name: str) -> str:
    """Validate and canonicalize one POSIX ZIP member name."""
    if not name or "\x00" in name or "\\" in name or len(name) > 512:
        raise ValueError("ZIP archive contains an unsafe member name")
    if name.startswith("/"):
        raise ValueError("ZIP archive contains an absolute member path")

    is_directory = name.endswith("/")
    parts = name.split("/")
    if is_directory:
        parts = parts[:-1]
    if not parts or len(parts) > 8 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("ZIP archive contains an unsafe member path")
    canonical = "/".join(parts)
    return f"{canonical}/" if is_directory else canonical


def _validate_zip_archive(
    archive: zipfile.ZipFile,
    *,
    max_members: int,
    max_total_bytes: int,
    max_entry_bytes: int,
) -> dict[str, zipfile.ZipInfo]:
    """Preflight every member before the importer reads or mutates anything."""
    members = archive.infolist()
    if len(members) > max_members:
        raise ValueError(f"ZIP archive contains too many members (max {max_members})")

    by_name: dict[str, zipfile.ZipInfo] = {}
    total_bytes = 0
    for info in members:
        canonical_name = _canonical_zip_member_name(info.filename)
        if canonical_name in by_name:
            raise ValueError(f"ZIP archive contains duplicate member: {canonical_name}")
        by_name[canonical_name] = info

        unix_mode = info.external_attr >> 16
        file_type = stat.S_IFMT(unix_mode)
        if info.create_system == 3 and file_type not in {
            0,
            stat.S_IFREG,
            stat.S_IFDIR,
        }:
            raise ValueError("ZIP archive contains a non-regular member")
        if info.flag_bits & 0x1:
            raise ValueError("Encrypted ZIP members are not supported")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise ValueError("ZIP archive uses an unsupported compression method")
        if info.file_size < 0 or info.compress_size < 0:
            raise ValueError("ZIP archive contains invalid member sizes")
        if info.file_size > max_entry_bytes:
            raise ValueError(f"ZIP member exceeds the uncompressed size limit ({max_entry_bytes} bytes)")

        total_bytes += info.file_size
        if total_bytes > max_total_bytes:
            raise ValueError(f"ZIP archive exceeds the total uncompressed size limit ({max_total_bytes} bytes)")
        if (
            not info.is_dir()
            and info.file_size >= ARCHIVE_RATIO_MIN_FILE_BYTES
            and (info.compress_size == 0 or info.file_size / info.compress_size > ARCHIVE_MAX_COMPRESSION_RATIO)
        ):
            raise ValueError("ZIP member exceeds the compression-ratio limit")

    return by_name


def _copy_zip_member_bounded(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
    *,
    max_bytes: int,
) -> int:
    """Stream one regular member to a new private file with a hard byte cap."""
    if info.is_dir() or info.file_size > max_bytes:
        raise ValueError(f"ZIP member exceeds the allowed size ({max_bytes} bytes)")

    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    total_bytes = 0
    try:
        with archive.open(info, "r") as source_file, os.fdopen(descriptor, "wb") as destination_file:
            descriptor = -1
            while chunk := source_file.read(ARCHIVE_IO_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(f"ZIP member exceeds the allowed size ({max_bytes} bytes)")
                destination_file.write(chunk)
            destination_file.flush()
            os.fsync(destination_file.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if total_bytes != info.file_size:
        destination.unlink(missing_ok=True)
        raise ValueError("ZIP member size does not match its directory entry")
    return total_bytes


def _copy_private_file(source: Path, destination: Path) -> None:
    """Copy to a new private regular file and sync its contents."""
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with source.open("rb") as source_file, os.fdopen(descriptor, "wb") as destination_file:
            descriptor = -1
            shutil.copyfileobj(
                source_file,
                destination_file,
                length=ARCHIVE_IO_CHUNK_BYTES,
            )
            destination_file.flush()
            os.fsync(destination_file.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _create_empty_private_file(destination: Path) -> None:
    """Create and sync one private inode before recording its publish intent."""
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _populate_private_file(
    source: Path,
    destination: Path,
    *,
    expected_stat: os.stat_result,
) -> None:
    """Populate an already-journaled private inode without replacing it."""
    flags = os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags)
    try:
        entry_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry_stat.st_mode)
            or entry_stat.st_dev != expected_stat.st_dev
            or entry_stat.st_ino != expected_stat.st_ino
        ):
            raise RuntimeError("Journaled import staging file identity changed")
        with source.open("rb") as source_file, os.fdopen(descriptor, "wb") as destination_file:
            descriptor = -1
            shutil.copyfileobj(
                source_file,
                destination_file,
                length=ARCHIVE_IO_CHUNK_BYTES,
            )
            destination_file.flush()
            os.fsync(destination_file.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    """Persist directory entries where directory fsync is supported."""
    if os.name != "posix":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _archive_export_root() -> Path:
    """Return a private real directory for response-owned export files."""
    root = config.get_cache_dir() / ARCHIVE_EXPORT_ROOT_NAME
    with suppress(FileExistsError):
        root.mkdir(mode=0o700)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ArchiveExportStorageError("Archive export storage is unavailable") from exc
    is_junction = getattr(root, "is_junction", None)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or (is_junction is not None and is_junction())
    ):
        raise ArchiveExportStorageError("Archive export storage is not a real directory")
    if os.name == "posix":
        os.chmod(root, 0o700, follow_symlinks=False)
    return root


def _archive_import_root() -> Path:
    """Return a private real directory for bounded import workspaces."""
    root = config.get_cache_dir() / ARCHIVE_IMPORT_ROOT_NAME
    with suppress(FileExistsError):
        root.mkdir(mode=0o700)
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise ArchiveImportStorageError("Archive import storage is unavailable") from exc
    is_junction = getattr(root, "is_junction", None)
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or (is_junction is not None and is_junction())
    ):
        raise ArchiveImportStorageError("Archive import storage is not a real directory")
    if os.name == "posix":
        os.chmod(root, 0o700, follow_symlinks=False)
    return root


def _remove_archive_export_directory(directory: Path) -> bool:
    """Remove one generated export directory without following a replacement."""
    try:
        entry_stat = directory.lstat()
    except FileNotFoundError:
        return True
    if stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
        shutil.rmtree(directory)
        return True
    if stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
        directory.unlink(missing_ok=True)
        return True
    return False


def _archive_entry_missing(directory: Path) -> bool:
    try:
        directory.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _release_archive_reservation_locked(directory: Path) -> None:
    reservation = _archive_disk_reservations.pop(directory, None)
    if reservation is not None:
        reservation.release()


def _cleanup_stale_archive_exports_locked(root: Path) -> None:
    """Reclaim exports from a prior process while preserving live responses."""
    for entry in root.iterdir():
        if not entry.name.startswith("job-") or entry in _active_archive_export_directories:
            continue
        try:
            if _remove_archive_export_directory(entry):
                _release_archive_reservation_locked(entry)
        except OSError:
            if _archive_entry_missing(entry):
                _release_archive_reservation_locked(entry)
            logger.warning("Could not remove a stale private archive export")


def _cleanup_stale_archive_imports() -> None:
    """Reclaim prior-process import workspaces away from the event loop."""
    with _archive_storage_lock:
        root = _archive_import_root()
        for entry in root.iterdir():
            if not entry.name.startswith("job-") or entry in _active_archive_import_directories:
                continue
            try:
                if _remove_archive_export_directory(entry):
                    _release_archive_reservation_locked(entry)
            except OSError:
                if _archive_entry_missing(entry):
                    _release_archive_reservation_locked(entry)
                logger.warning("Could not remove a stale private archive import")


def _active_archive_reserved_bytes() -> int:
    """Return process-local reservations shared by imports and exports."""
    return sum(_active_archive_export_directories.values()) + sum(_active_archive_import_directories.values())


def _allocate_archive_export(projected_bytes: int) -> ArchiveExport:
    """Reserve bounded disk capacity and allocate a response-owned directory."""
    if projected_bytes <= 0:
        raise ArchiveExportLimitError("Archive export size is invalid")
    with _archive_export_lock:
        if len(_active_archive_export_directories) >= ARCHIVE_EXPORT_MAX_ACTIVE:
            raise ArchiveExportBusyError("Too many archive exports are already active; retry later")
        root = _archive_export_root()
        _cleanup_stale_archive_exports_locked(root)
        try:
            reservation = reserve_disk_space(
                root,
                projected_bytes,
                min_free_bytes=ARCHIVE_EXPORT_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise ArchiveExportStorageError("Insufficient free space to export this archive safely") from exc
        try:
            temporary_directory = Path(tempfile.mkdtemp(prefix="job-", dir=root))
        except BaseException:
            reservation.release()
            raise
        _active_archive_export_directories[temporary_directory] = projected_bytes
        _archive_disk_reservations[temporary_directory] = reservation
        return ArchiveExport(
            path=temporary_directory / "archive.voicebox.zip",
            temporary_directory=temporary_directory,
        )


def _cleanup_archive_export(directory: Path) -> None:
    """Release one export reservation and remove its private files."""
    with _archive_export_lock:
        _active_archive_export_directories.pop(directory, None)
        try:
            if _remove_archive_export_directory(directory):
                _release_archive_reservation_locked(directory)
        except OSError:
            if _archive_entry_missing(directory):
                _release_archive_reservation_locked(directory)
            logger.warning("Could not remove a private archive export")


def _allocate_archive_import(projected_bytes: int) -> _ArchiveImportWorkspace:
    """Atomically admit one bounded import and allocate its private workspace."""
    if projected_bytes <= 0:
        raise ValueError("Archive import size projection is invalid")
    with _archive_storage_lock:
        if len(_active_archive_import_directories) >= ARCHIVE_IMPORT_MAX_CONCURRENT:
            raise ArchiveImportBusyError("Too many archive imports are already in progress")
        root = _archive_import_root()
        try:
            reservation = reserve_disk_space(
                root,
                projected_bytes,
                min_free_bytes=ARCHIVE_IMPORT_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise ArchiveImportStorageError("Insufficient free space to import this archive safely") from exc
        try:
            directory = Path(tempfile.mkdtemp(prefix="job-", dir=root))
        except BaseException:
            reservation.release()
            raise
        _active_archive_import_directories[directory] = projected_bytes
        _archive_disk_reservations[directory] = reservation
        return _ArchiveImportWorkspace(directory=directory)


def _cleanup_archive_import(directory: Path) -> None:
    """Remove one import workspace and release its shared disk reservation."""
    with _archive_storage_lock:
        removed = False
        try:
            removed = _remove_archive_export_directory(directory)
        except OSError:
            if _archive_entry_missing(directory):
                _release_archive_reservation_locked(directory)
            logger.warning("Could not remove a private archive import")
        finally:
            _active_archive_import_directories.pop(directory, None)
        if removed:
            _release_archive_reservation_locked(directory)


def _export_source_identity(entry_stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        entry_stat.st_dev,
        entry_stat.st_ino,
        entry_stat.st_size,
        entry_stat.st_mtime_ns,
        entry_stat.st_ctime_ns,
    )


def _preflight_managed_export_file(
    stored_path: str | Path | None,
    *,
    expected_directory: str,
    archive_name: str,
    max_bytes: int,
) -> _ArchiveExportMember:
    """Open and inventory one managed regular file without following it."""
    relative = config.managed_storage_relative_path(stored_path)
    if relative is None or not relative.parts or relative.parts[0] != expected_directory:
        raise ValueError("Archive source file is outside managed storage")
    canonical_name = _canonical_zip_member_name(archive_name)
    if canonical_name.endswith("/"):
        raise ValueError("Archive source member name is invalid")

    path = config.get_data_dir() / relative
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Archive source file is unavailable or unsafe") from exc
    try:
        entry_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_size <= 0:
        raise ValueError("Archive source is not a non-empty regular file")
    if entry_stat.st_size > max_bytes:
        raise ArchiveExportLimitError(f"Archive member exceeds the size limit ({max_bytes} bytes)")
    return _ArchiveExportMember(
        path=path,
        archive_name=canonical_name,
        size=entry_stat.st_size,
        identity=_export_source_identity(entry_stat),
    )


@contextmanager
def _open_preflighted_export_member(member: _ArchiveExportMember) -> Iterator[object]:
    """Reopen a source and prove it did not change since preflight."""
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(member.path, flags)
    except OSError as exc:
        raise ValueError("Archive source changed after preflight") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _export_source_identity(before) != member.identity:
            raise ValueError("Archive source changed after preflight")
        with os.fdopen(descriptor, "rb", buffering=0) as source_file:
            descriptor = -1
            yield source_file
            after = os.fstat(source_file.fileno())
            if _export_source_identity(after) != member.identity:
                raise ValueError("Archive source changed while it was being exported")
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _zip_info(name: str, *, size: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    # PCM WAV already compresses poorly, and storing members guarantees that
    # archives produced here cannot trip the importer's ZIP-bomb ratio guard.
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.file_size = size
    return info


def _check_archive_output_size(output_file, max_output_bytes: int) -> None:
    if output_file.tell() > max_output_bytes:
        raise ArchiveExportLimitError("Compressed archive exceeds the export size limit")


def _write_archive_export(
    destination: Path,
    *,
    json_members: tuple[tuple[str, bytes], ...],
    file_members: tuple[_ArchiveExportMember, ...],
    max_output_bytes: int,
) -> None:
    """Build a bounded ZIP directly on disk from verified managed files."""
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w+b", buffering=0) as output_file:
            descriptor = -1
            with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
                for name, payload in json_members:
                    archive.writestr(_zip_info(name, size=len(payload)), payload)
                    _check_archive_output_size(output_file, max_output_bytes)

                for member in file_members:
                    copied_bytes = 0
                    with (
                        _open_preflighted_export_member(member) as source_file,
                        archive.open(
                            _zip_info(member.archive_name, size=member.size),
                            "w",
                            force_zip64=False,
                        ) as archive_member,
                    ):
                        while chunk := source_file.read(ARCHIVE_IO_CHUNK_BYTES):
                            copied_bytes += len(chunk)
                            if copied_bytes > member.size:
                                raise ValueError("Archive source grew while it was being exported")
                            archive_member.write(chunk)
                            _check_archive_output_size(output_file, max_output_bytes)
                    if copied_bytes != member.size:
                        raise ValueError("Archive source size changed while it was being exported")
            _check_archive_output_size(output_file, max_output_bytes)
            output_file.flush()
            os.fsync(output_file.fileno())
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


async def _build_archive_export(
    *,
    json_members: tuple[tuple[str, bytes], ...],
    file_members: tuple[_ArchiveExportMember, ...],
    max_total_bytes: int,
    max_members: int,
) -> ArchiveExport:
    """Validate aggregate limits, reserve disk, and build off the event loop."""
    member_count = len(json_members) + len(file_members)
    if member_count > max_members:
        raise ArchiveExportLimitError(f"Archive contains too many members (max {max_members})")
    total_bytes = sum(len(payload) for _name, payload in json_members) + sum(member.size for member in file_members)
    if total_bytes > max_total_bytes:
        raise ArchiveExportLimitError(f"Archive exceeds the total uncompressed size limit ({max_total_bytes} bytes)")
    max_output_bytes = total_bytes + ARCHIVE_EXPORT_OVERHEAD_BYTES
    archive_export = _allocate_archive_export(max_output_bytes)
    try:
        await run_blocking_operation_cancellation_safe(
            _write_archive_export,
            archive_export.path,
            json_members=json_members,
            file_members=file_members,
            max_output_bytes=max_output_bytes,
        )
    except BaseException:
        archive_export.cleanup()
        raise
    return archive_export


def _encoded_archive_json(label: str, value: object) -> bytes:
    payload = json.dumps(value, indent=2).encode("utf-8")
    if len(payload) > ARCHIVE_JSON_MAX_BYTES:
        raise ArchiveExportLimitError(f"{label} exceeds the archive metadata size limit")
    return payload


def _safe_audio_archive_basename(
    name: str,
    *,
    fallback: str,
    allowed_extensions: frozenset[str] = frozenset({".wav"}),
) -> str:
    """Keep compatible basenames where safe, otherwise use a stable fallback."""
    try:
        canonical = _canonical_zip_member_name(name)
    except ValueError:
        canonical = fallback
    if "/" in canonical or canonical.endswith("/") or PurePosixPath(canonical).suffix.lower() not in allowed_extensions:
        canonical = fallback
    return canonical


def _published_import_matches_intent(
    intent: deletion_journal.DeletionIntent,
) -> bool:
    """Prove that the public path still names the inode we populated."""
    entry_stat = deletion_journal.managed_entry_stat(intent.original)
    if entry_stat is None or not deletion_journal.entry_matches_intent(intent, entry_stat):
        return False
    if stat.S_ISREG(entry_stat.st_mode):
        return entry_stat.st_nlink == 1
    return stat.S_ISDIR(entry_stat.st_mode)


def _private_file_identity(relative: Path) -> tuple[int, int, int, int]:
    """Capture one immutable managed-file identity for later acknowledgement."""
    entry_stat = deletion_journal.managed_entry_stat(relative)
    if entry_stat is None or not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
        raise RuntimeError("Imported profile payload is not a private regular file")
    return (
        entry_stat.st_dev,
        entry_stat.st_ino,
        stat.S_IFMT(entry_stat.st_mode),
        entry_stat.st_size,
    )


def _durable_imported_profile_response(
    db: Session,
    *,
    profile_id: str,
    expected_profile_fields: dict[str, object],
    expected_sample_fields: tuple[tuple[object, ...], ...],
    expected_file_identities: dict[Path, tuple[int, int, int, int]],
    publish_intent: deletion_journal.DeletionIntent,
) -> VoiceProfileResponse | None:
    """Return an independently proven profile after an ambiguous commit."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        profile = durable_db.query(DBVoiceProfile).filter_by(id=profile_id).one_or_none()
        if profile is None:
            return None
        if any(getattr(profile, field) != value for field, value in expected_profile_fields.items()):
            raise RuntimeError("Imported profile ID is owned by different durable data")

        samples = (
            durable_db.query(DBProfileSample)
            .filter_by(profile_id=profile_id)
            .order_by(DBProfileSample.ordinal, DBProfileSample.id)
            .all()
        )
        durable_sample_fields = tuple(
            (
                sample.id,
                sample.profile_id,
                sample.ordinal,
                sample.audio_path,
                sample.reference_text,
            )
            for sample in samples
        )
        if durable_sample_fields != expected_sample_fields:
            raise RuntimeError("Imported profile samples differ from the committed payload")
        if not _published_import_matches_intent(publish_intent):
            raise RuntimeError("Imported profile storage identity changed after commit")
        for relative, expected_identity in expected_file_identities.items():
            if _private_file_identity(relative) != expected_identity:
                raise RuntimeError("Imported profile file identity changed after commit")
        return _profile_to_response(profile, sample_count=len(samples))


def _durable_imported_generation_result(
    db: Session,
    *,
    generation_id: str,
    expected_fields: dict[str, object],
    profile_name: str,
    publish_intent: deletion_journal.DeletionIntent,
) -> dict | None:
    """Return an independently proven generation after an ambiguous commit."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        generation = durable_db.query(DBGeneration).filter_by(id=generation_id).one_or_none()
        if generation is None:
            return None
        if any(getattr(generation, field) != value for field, value in expected_fields.items()):
            raise RuntimeError("Imported generation ID is owned by different durable data")
        if not _published_import_matches_intent(publish_intent):
            raise RuntimeError("Imported generation audio identity changed after commit")
        return {
            "id": generation.id,
            "profile_id": generation.profile_id,
            "profile_name": profile_name,
            "text": generation.text,
            "message": f"Generation imported successfully (assigned to profile: {profile_name})",
        }


def _finish_committed_import_intent(
    intent: deletion_journal.DeletionIntent,
    *,
    label: str,
) -> None:
    """Best-effort journal retirement after the DB and payload are durable."""
    try:
        deletion_journal.finish_deletion_intent(intent)
    except Exception:
        logger.warning("Deferred cleanup of a committed %s intent", label, exc_info=True)


def _reconcile_failed_publication(
    intent: deletion_journal.DeletionIntent,
    db: Session,
    *,
    commit_state: bool | None,
    label: str,
) -> None:
    """Resolve a failed publish only when durable DB ownership is knowable."""
    if commit_state is None:
        logger.warning("Retaining an indeterminate %s publish intent for startup recovery", label)
        return
    try:
        with deletion_journal.durable_reconciliation_session(db) as durable_db:
            deletion_journal.reconcile_deletion_intent(intent, durable_db)
    except BaseException:
        # The intent is already durable.  Leaving both it and the private
        # artifact intact is the only safe outcome when reconciliation fails.
        logger.warning("Retaining a failed %s publish intent for startup recovery", label)


def _read_zip_member_bounded(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    max_bytes: int,
) -> bytes:
    if info.is_dir() or info.file_size > max_bytes:
        raise ValueError(f"ZIP member exceeds the allowed size ({max_bytes} bytes)")
    output = io.BytesIO()
    total_bytes = 0
    with archive.open(info, "r") as source_file:
        while chunk := source_file.read(min(ARCHIVE_IO_CHUNK_BYTES, max_bytes + 1)):
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(f"ZIP member exceeds the allowed size ({max_bytes} bytes)")
            output.write(chunk)
    if total_bytes != info.file_size:
        raise ValueError("ZIP member size does not match its directory entry")
    return output.getvalue()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON document contains duplicate key: {key}")
        result[key] = value
    return result


def _read_json_object(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    label: str,
) -> dict:
    payload = _read_zip_member_bounded(
        archive,
        info,
        max_bytes=ARCHIVE_JSON_MAX_BYTES,
    )
    try:
        value = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {label}: must be a JSON object")
    return value


def _get_unique_profile_name(name: str, db: Session) -> str:
    """
    Get a unique profile name by appending a number if needed.

    Args:
        name: Original profile name
        db: Database session

    Returns:
        Unique profile name
    """
    base_name = name
    counter = 1

    while True:
        existing = db.query(DBVoiceProfile).filter_by(name=name).first()
        if not existing:
            return name

        name = f"{base_name} ({counter})"
        counter += 1


def _ordered_import_samples(
    manifest_data: dict,
    samples_data: dict[str, str],
) -> list[tuple[str, str]]:
    """Resolve an archive's explicit sample order, with v1 legacy fallback."""
    sample_order = manifest_data.get("sample_order")
    if sample_order is None:
        # Legacy archives predate explicit ordinals. Python's JSON decoder
        # retains their serialized member order, which is the best historical
        # ordering information those archives contain.
        return list(samples_data.items())

    if (
        not isinstance(sample_order, list)
        or any(not isinstance(filename, str) for filename in sample_order)
        or len(sample_order) != len(set(sample_order))
        or set(sample_order) != set(samples_data)
    ):
        raise ValueError("Invalid manifest.json: sample_order must list every sample exactly once")

    return [(filename, samples_data[filename]) for filename in sample_order]


def _selected_generation_audio_member(
    manifest_data: dict,
    members: dict[str, zipfile.ZipInfo],
) -> zipfile.ZipInfo:
    """Select the exported default version instead of filename ordering."""
    audio_members = sorted(
        (
            info
            for name, info in members.items()
            if PurePosixPath(name).parent == PurePosixPath("audio")
            and PurePosixPath(name).suffix.lower() in GENERATION_ARCHIVE_AUDIO_EXTENSIONS
            and not info.is_dir()
        ),
        key=lambda info: info.filename,
    )
    if not audio_members:
        raise ValueError("No audio file found in ZIP archive")

    versions_data = manifest_data.get("versions")
    if versions_data in (None, []):
        return audio_members[0]
    if not isinstance(versions_data, list) or len(versions_data) > GENERATION_ARCHIVE_MAX_MEMBERS - 1:
        raise ValueError("Invalid manifest.json: versions is invalid")

    selected: zipfile.ZipInfo | None = None
    default_count = 0
    first: zipfile.ZipInfo | None = None
    for version in versions_data:
        if not isinstance(version, dict):
            raise ValueError("Invalid manifest.json: version entry must be an object")
        filename = version.get("filename")
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename).name != filename
            or PurePosixPath(filename).suffix.lower() not in GENERATION_ARCHIVE_AUDIO_EXTENSIONS
        ):
            raise ValueError("Invalid manifest.json: version filename is invalid")
        info = members.get(f"audio/{filename}")
        if info is None or info.is_dir():
            raise ValueError("Version audio file is missing from ZIP archive")
        if first is None:
            first = info
        is_default = version.get("is_default", False)
        if not isinstance(is_default, bool):
            raise ValueError("Invalid manifest.json: version is_default is invalid")
        if is_default:
            default_count += 1
            selected = info
    if default_count > 1:
        raise ValueError("Invalid manifest.json: multiple default versions")
    assert first is not None
    return selected or first


def _archive_fingerprint(members: dict[str, zipfile.ZipInfo]) -> ArchiveFingerprint:
    """Capture the complete validated central-directory contract."""
    return tuple(
        sorted(
            (
                name,
                info.file_size,
                info.compress_size,
                info.CRC,
                info.compress_type,
                info.flag_bits,
                info.external_attr,
            )
            for name, info in members.items()
        )
    )


def _inspect_profile_import(archive_source: ArchiveSource) -> _ProfileImportPlan:
    """Inspect and size a profile archive without creating persistent files."""
    with _open_zip_archive(archive_source) as archive:
        members = _validate_zip_archive(
            archive,
            max_members=PROFILE_ARCHIVE_MAX_MEMBERS,
            max_total_bytes=PROFILE_ARCHIVE_MAX_TOTAL_BYTES,
            max_entry_bytes=PROFILE_ARCHIVE_MAX_ENTRY_BYTES,
        )
        manifest_info = members.get("manifest.json")
        samples_info = members.get("samples.json")
        if manifest_info is None or manifest_info.is_dir():
            raise ValueError("ZIP archive missing manifest.json")
        if samples_info is None or samples_info.is_dir():
            raise ValueError("ZIP archive missing samples.json")

        manifest_data = _read_json_object(
            archive,
            manifest_info,
            label="manifest.json",
        )
        if "version" not in manifest_data:
            raise ValueError("Invalid manifest.json: missing version")
        raw_profile_data = manifest_data.get("profile")
        if not isinstance(raw_profile_data, dict):
            raise ValueError("Invalid manifest.json: missing profile")
        original_name = raw_profile_data.get("name", "Imported Profile")
        if not isinstance(original_name, str):
            raise ValueError("Invalid manifest.json: profile.name must be a string")
        validated_profile = VoiceProfileCreate(
            name=original_name,
            description=raw_profile_data.get("description"),
            language=raw_profile_data.get("language", "en"),
        )
        profile_data = {
            "name": validated_profile.name,
            "description": validated_profile.description,
            "language": validated_profile.language,
        }

        samples_data = _read_json_object(
            archive,
            samples_info,
            label="samples.json",
        )
        ordered_samples = _ordered_import_samples(manifest_data, samples_data)
        if not ordered_samples:
            raise ValueError("Profile archive contains no voice samples")
        if len(ordered_samples) > PROFILE_ARCHIVE_MAX_SAMPLES:
            raise ValueError(f"Profile archive contains too many voice samples (max {PROFILE_ARCHIVE_MAX_SAMPLES})")

        planned_samples: list[_ProfileImportSample] = []
        aggregate_audio_bytes = 0
        aggregate_text_bytes = 0
        for filename, reference_text in ordered_samples:
            if (
                not isinstance(filename, str)
                or PurePosixPath(filename).name != filename
                or not filename.lower().endswith(".wav")
            ):
                raise ValueError(f"Invalid sample filename: {filename} (must be a .wav basename)")
            validated_reference = ProfileSampleCreate(reference_text=reference_text).reference_text
            aggregate_text_bytes += len(validated_reference.encode("utf-8"))
            if aggregate_text_bytes > EXACT_VOICE_SNAPSHOT_MAX_TOTAL_REFERENCE_TEXT_BYTES:
                raise ValueError("Profile sample transcripts exceed the safe aggregate size limit")

            member_name = f"samples/{filename}"
            sample_info = members.get(member_name)
            if sample_info is None or sample_info.is_dir():
                raise ValueError(f"Sample file not found in ZIP: {member_name}")
            if sample_info.file_size <= 0 or sample_info.file_size > EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES:
                raise ValueError("Reference sample audio exceeds the safe size limit")
            aggregate_audio_bytes += sample_info.file_size
            if aggregate_audio_bytes > EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES:
                raise ValueError("Reference sample audio exceeds the safe aggregate size limit")
            planned_samples.append(
                _ProfileImportSample(
                    member_name=member_name,
                    filename=filename,
                    reference_text=validated_reference,
                    size=sample_info.file_size,
                )
            )

        avatar_members = [
            (name, info)
            for name, info in members.items()
            if PurePosixPath(name).parent == PurePosixPath(".")
            and PurePosixPath(name).name.startswith("avatar.")
            and not info.is_dir()
        ]
        if len(avatar_members) > 1:
            raise ValueError("Profile archive contains multiple avatar files")
        avatar_member_name = avatar_members[0][0] if avatar_members else None
        avatar_bytes = avatar_members[0][1].file_size if avatar_members else 0
        if avatar_member_name is not None and (avatar_bytes <= 0 or avatar_bytes > AVATAR_MAX_FILE_SIZE):
            raise ValueError("Profile avatar exceeds the safe size limit")

        # Peak ownership includes extracted originals, canonical staging files,
        # and the simultaneously populated managed copies. Canonical sample
        # output is independently checked against the same 128 MiB contract.
        projected_bytes = (
            aggregate_audio_bytes
            + avatar_bytes
            + (2 * EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES)
            + (2 * AVATAR_MAX_FILE_SIZE)
            + ARCHIVE_IMPORT_OVERHEAD_BYTES
        )
        return _ProfileImportPlan(
            fingerprint=_archive_fingerprint(members),
            profile_data=profile_data,
            samples=tuple(planned_samples),
            avatar_member_name=avatar_member_name,
            projected_bytes=projected_bytes,
        )


def _inspect_generation_import(archive_source: ArchiveSource) -> _GenerationImportPlan:
    """Inspect and size a generation archive without creating persistent files."""
    with _open_zip_archive(archive_source) as archive:
        members = _validate_zip_archive(
            archive,
            max_members=GENERATION_ARCHIVE_MAX_MEMBERS,
            max_total_bytes=GENERATION_ARCHIVE_MAX_TOTAL_BYTES,
            max_entry_bytes=GENERATION_ARCHIVE_MAX_ENTRY_BYTES,
        )
        manifest_info = members.get("manifest.json")
        if manifest_info is None or manifest_info.is_dir():
            raise ValueError("ZIP archive missing manifest.json")
        manifest_data = _read_json_object(
            archive,
            manifest_info,
            label="manifest.json",
        )
        if "version" not in manifest_data:
            raise ValueError("Invalid manifest.json: missing version")
        generation_data = manifest_data.get("generation")
        if not isinstance(generation_data, dict):
            raise ValueError("Invalid manifest.json: missing generation data")
        profile_data = manifest_data.get("profile", {})
        if not isinstance(profile_data, dict):
            raise ValueError("Invalid manifest.json: profile must be an object")

        required_fields = ("text", "language", "duration")
        for field in required_fields:
            if field not in generation_data:
                raise ValueError(f"Invalid manifest.json: missing generation.{field}")
        text = generation_data["text"]
        language = generation_data["language"]
        duration = generation_data["duration"]
        if not isinstance(text, str) or not text or len(text) > 50_000:
            raise ValueError("Invalid manifest.json: generation.text is invalid")
        if not isinstance(language, str) or not language:
            raise ValueError("Invalid manifest.json: generation.language is invalid")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or duration < 0
            or duration > GENERATION_AUDIO_MAX_DURATION_SECONDS
        ):
            raise ValueError("Invalid manifest.json: generation.duration is invalid")
        seed = generation_data.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("Invalid manifest.json: generation.seed is invalid")
        instruct = generation_data.get("instruct")
        if instruct is not None and (not isinstance(instruct, str) or len(instruct) > 500):
            raise ValueError("Invalid manifest.json: generation.instruct is invalid")

        selected_audio = _selected_generation_audio_member(manifest_data, members)
        if selected_audio.file_size <= 0:
            raise ValueError("Generation archive contains empty audio")
        return _GenerationImportPlan(
            fingerprint=_archive_fingerprint(members),
            generation_data={
                "text": text,
                "language": language,
                "duration": float(duration),
                "seed": seed,
                "instruct": instruct,
            },
            profile_data=profile_data,
            audio_member_name=_canonical_zip_member_name(selected_audio.filename),
            audio_suffix=PurePosixPath(selected_audio.filename).suffix.lower(),
            audio_size=selected_audio.file_size,
            projected_bytes=(2 * selected_audio.file_size) + ARCHIVE_IMPORT_OVERHEAD_BYTES,
        )


def _validated_reopened_members(
    archive: zipfile.ZipFile,
    *,
    expected_fingerprint: ArchiveFingerprint,
    max_members: int,
    max_total_bytes: int,
    max_entry_bytes: int,
) -> dict[str, zipfile.ZipInfo]:
    """Revalidate the source and reject replacements after inspection."""
    members = _validate_zip_archive(
        archive,
        max_members=max_members,
        max_total_bytes=max_total_bytes,
        max_entry_bytes=max_entry_bytes,
    )
    if _archive_fingerprint(members) != expected_fingerprint:
        raise ValueError("ZIP archive changed after import preflight")
    return members


def _extract_profile_import(
    archive_source: ArchiveSource,
    plan: _ProfileImportPlan,
    workspace: Path,
) -> None:
    """Extract all selected profile payloads in one bounded worker."""
    with _open_zip_archive(archive_source) as archive:
        members = _validated_reopened_members(
            archive,
            expected_fingerprint=plan.fingerprint,
            max_members=PROFILE_ARCHIVE_MAX_MEMBERS,
            max_total_bytes=PROFILE_ARCHIVE_MAX_TOTAL_BYTES,
            max_entry_bytes=PROFILE_ARCHIVE_MAX_ENTRY_BYTES,
        )
        for ordinal, sample in enumerate(plan.samples):
            _copy_zip_member_bounded(
                archive,
                members[sample.member_name],
                workspace / f"sample-{ordinal}.uploaded.wav",
                max_bytes=EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES,
            )
        if plan.avatar_member_name is not None:
            _copy_zip_member_bounded(
                archive,
                members[plan.avatar_member_name],
                workspace / "avatar.uploaded",
                max_bytes=AVATAR_MAX_FILE_SIZE,
            )
        _fsync_directory(workspace)


def _extract_generation_import(
    archive_source: ArchiveSource,
    plan: _GenerationImportPlan,
    destination: Path,
) -> None:
    """Extract the selected generation audio in one bounded worker."""
    with _open_zip_archive(archive_source) as archive:
        members = _validated_reopened_members(
            archive,
            expected_fingerprint=plan.fingerprint,
            max_members=GENERATION_ARCHIVE_MAX_MEMBERS,
            max_total_bytes=GENERATION_ARCHIVE_MAX_TOTAL_BYTES,
            max_entry_bytes=GENERATION_ARCHIVE_MAX_ENTRY_BYTES,
        )
        _copy_zip_member_bounded(
            archive,
            members[plan.audio_member_name],
            destination,
            max_bytes=GENERATION_ARCHIVE_MAX_ENTRY_BYTES,
        )
        _fsync_directory(destination.parent)


def _canonicalize_profile_sample(source: Path, destination: Path) -> str | None:
    """Validate and canonicalize one reference sample without retaining PCM."""
    is_valid, error_message, audio, sample_rate = validate_and_load_reference_audio(str(source))
    if not is_valid or audio is None or sample_rate is None:
        return error_message or "unknown validation error"
    save_audio(audio, str(destination), sample_rate)
    return None


def _canonicalize_profile_avatar(source: Path, destination: Path) -> str | None:
    """Validate and canonicalize one avatar in the same worker."""
    image_valid, image_error = validate_image(str(source))
    if not image_valid:
        return image_error or "unknown validation error"
    process_avatar(str(source), str(destination))
    return None


def _bounded_private_file_size(path: Path, *, max_bytes: int, label: str) -> int:
    """Measure a private regular file without following a replacement link."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable or unsafe") from exc
    try:
        entry_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_size <= 0 or entry_stat.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the safe size limit")
    return entry_stat.st_size


def _populate_profile_import(
    copies: tuple[tuple[Path, Path], ...],
    pending_profile_dir: Path,
) -> None:
    """Populate an already-journaled profile directory and sync its entries."""
    for source, destination in copies:
        _copy_private_file(source, destination)
    _fsync_directory(pending_profile_dir)


async def export_profile_to_zip(profile_id: str, db: Session) -> ArchiveExport:
    """Export a profile into a bounded private ZIP owned by its response."""
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).order_by(DBProfileSample.ordinal.asc()).all()
    if not samples:
        raise ValueError(f"Profile {profile_id} has no samples")
    if len(samples) > PROFILE_ARCHIVE_MAX_SAMPLES:
        raise ArchiveExportLimitError(
            f"Profile contains too many voice samples to export (max {PROFILE_ARCHIVE_MAX_SAMPLES})"
        )

    sample_entries: list[tuple[DBProfileSample, str]] = []
    sample_members: list[_ArchiveExportMember] = []
    used_names: set[str] = set()
    for ordinal, sample in enumerate(samples):
        relative = config.managed_storage_relative_path(sample.audio_path)
        candidate = relative.name if relative is not None else ""
        filename = _safe_audio_archive_basename(candidate, fallback=f"sample-{ordinal}.wav")
        suffix = 1
        while filename in used_names:
            filename = f"sample-{ordinal}-{suffix}.wav"
            suffix += 1
        used_names.add(filename)
        sample_members.append(
            _preflight_managed_export_file(
                sample.audio_path,
                expected_directory="profiles",
                archive_name=f"samples/{filename}",
                max_bytes=PROFILE_ARCHIVE_MAX_ENTRY_BYTES,
            )
        )
        sample_entries.append((sample, filename))

    file_members: list[_ArchiveExportMember] = []
    has_avatar = profile.avatar_path is not None
    if profile.avatar_path:
        avatar_relative = config.managed_storage_relative_path(profile.avatar_path)
        avatar_suffix = avatar_relative.suffix.lower() if avatar_relative is not None else ".png"
        if len(avatar_suffix) < 2 or len(avatar_suffix) > 10 or not avatar_suffix[1:].isalnum():
            avatar_suffix = ".png"
        file_members.append(
            _preflight_managed_export_file(
                profile.avatar_path,
                expected_directory="profiles",
                archive_name=f"avatar{avatar_suffix}",
                max_bytes=AVATAR_MAX_FILE_SIZE,
            )
        )
    file_members.extend(sample_members)

    manifest = {
        "version": "1.0",
        "profile": {
            "name": profile.name,
            "description": profile.description,
            "language": profile.language,
        },
        "has_avatar": has_avatar,
        "sample_order": [filename for _sample, filename in sample_entries],
    }
    samples_data = {filename: sample.reference_text for sample, filename in sample_entries}
    # Archive construction and the response lease can be long-lived. All DB
    # state has been copied into immutable manifest/member descriptors, so do
    # not retain a pooled connection while the worker compresses or the client
    # downloads.
    db.rollback()
    return await _build_archive_export(
        json_members=(
            ("manifest.json", _encoded_archive_json("manifest.json", manifest)),
            ("samples.json", _encoded_archive_json("samples.json", samples_data)),
        ),
        file_members=tuple(file_members),
        max_total_bytes=PROFILE_ARCHIVE_MAX_TOTAL_BYTES,
        max_members=PROFILE_ARCHIVE_MAX_MEMBERS,
    )


async def import_profile_from_zip(
    archive_source: ArchiveSource,
    db: Session,
) -> VoiceProfileResponse:
    """Validate a profile archive fully, then publish it in one DB commit."""
    try:
        plan = await run_blocking_operation_cancellation_safe(
            _inspect_profile_import,
            archive_source,
        )
        await run_blocking_operation_cancellation_safe(_cleanup_stale_archive_imports)
        workspace = _allocate_archive_import(plan.projected_bytes)
        try:
            staging_dir = workspace.directory
            prepared_samples: list[tuple[Path, str]] = []
            prepared_avatar: Path | None = None
            await run_blocking_operation_cancellation_safe(
                _extract_profile_import,
                archive_source,
                plan,
                staging_dir,
            )

            canonical_audio_bytes = 0
            for ordinal, sample in enumerate(plan.samples):
                raw_sample = staging_dir / f"sample-{ordinal}.uploaded.wav"
                canonical_sample = staging_dir / f"sample-{ordinal}.wav"
                error_message = await run_blocking_operation_cancellation_safe(
                    _canonicalize_profile_sample,
                    raw_sample,
                    canonical_sample,
                )
                if error_message is not None:
                    raise ValueError(f"Invalid reference audio for {sample.filename}: {error_message}")
                canonical_audio_bytes += await run_blocking_operation_cancellation_safe(
                    _bounded_private_file_size,
                    canonical_sample,
                    max_bytes=EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES,
                    label="Canonical reference sample audio",
                )
                if canonical_audio_bytes > EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES:
                    raise ValueError("Reference sample audio exceeds the safe aggregate size limit")
                prepared_samples.append((canonical_sample, sample.reference_text))

            if plan.avatar_member_name is not None:
                raw_avatar = staging_dir / "avatar.uploaded"
                prepared_avatar = staging_dir / "avatar.png"
                image_error = await run_blocking_operation_cancellation_safe(
                    _canonicalize_profile_avatar,
                    raw_avatar,
                    prepared_avatar,
                )
                if image_error is not None:
                    raise ValueError(f"Invalid profile avatar: {image_error}")
                await run_blocking_operation_cancellation_safe(
                    _bounded_private_file_size,
                    prepared_avatar,
                    max_bytes=AVATAR_MAX_FILE_SIZE,
                    label="Canonical profile avatar",
                )

            original_name = plan.profile_data["name"]
            unique_name = _get_unique_profile_name(original_name, db)
            profile_create = VoiceProfileCreate(
                name=unique_name,
                description=plan.profile_data["description"],
                language=plan.profile_data["language"],
            )

            profile_id = str(uuid.uuid4())
            profiles_dir = config.get_profiles_dir()
            profile_relative = Path("profiles") / profile_id
            pending_relative = Path("profiles") / f".voicebox-delete-profile-import-{profile_id}"
            profile_dir = profiles_dir / profile_id
            pending_profile_dir = config.get_data_dir() / pending_relative
            now = datetime.utcnow()
            db_profile = DBVoiceProfile(
                id=profile_id,
                name=profile_create.name,
                description=profile_create.description,
                language=profile_create.language,
                voice_type="cloned",
                created_at=now,
                updated_at=now,
            )

            commit_started = False
            publish_intent: deletion_journal.DeletionIntent | None = None
            expected_profile_fields: dict[str, object] = {}
            expected_sample_fields: tuple[tuple[object, ...], ...] = ()
            expected_file_identities: dict[Path, tuple[int, int, int, int]] = {}
            try:
                pending_profile_dir.mkdir(mode=0o700, exist_ok=False)
                pending_stat = deletion_journal.managed_entry_stat(pending_relative)
                if pending_stat is None or not stat.S_ISDIR(pending_stat.st_mode):
                    raise RuntimeError("Imported profile staging directory is not a regular managed directory")
                publish_intent = deletion_journal.prepare_deletion_intent(
                    kind=deletion_journal.PROFILE_STORAGE,
                    original=profile_relative,
                    staged=pending_relative,
                    entry_stat=pending_stat,
                    owner_id=profile_id,
                )

                imported_samples = [
                    (str(uuid.uuid4()), reference_text) for _staged_sample, reference_text in prepared_samples
                ]
                copies = tuple(
                    (staged_sample, pending_profile_dir / f"{sample_id}.wav")
                    for (staged_sample, _reference_text), (sample_id, _validated_text) in zip(
                        prepared_samples,
                        imported_samples,
                        strict=True,
                    )
                )
                if prepared_avatar is not None:
                    copies += ((prepared_avatar, pending_profile_dir / "avatar.png"),)
                await run_blocking_operation_cancellation_safe(
                    _populate_profile_import,
                    copies,
                    pending_profile_dir,
                )
                deletion_journal.rename_managed_entry(pending_relative, profile_relative)

                db.add(db_profile)
                for ordinal, (sample_id, reference_text) in enumerate(imported_samples):
                    destination = profile_dir / f"{sample_id}.wav"
                    db.add(
                        DBProfileSample(
                            id=sample_id,
                            profile_id=profile_id,
                            ordinal=ordinal,
                            audio_path=config.to_storage_path(destination),
                            reference_text=reference_text,
                        )
                    )
                if prepared_avatar is not None:
                    db_profile.avatar_path = config.to_storage_path(profile_dir / "avatar.png")

                expected_profile_fields = {
                    field: getattr(db_profile, field)
                    for field in (
                        "id",
                        "name",
                        "description",
                        "language",
                        "avatar_path",
                        "effects_chain",
                        "voice_type",
                        "preset_engine",
                        "preset_voice_id",
                        "design_prompt",
                        "default_engine",
                        "personality",
                        "created_at",
                        "updated_at",
                    )
                }
                expected_sample_fields = tuple(
                    (
                        sample_id,
                        profile_id,
                        ordinal,
                        config.to_storage_path(profile_dir / f"{sample_id}.wav"),
                        reference_text,
                    )
                    for ordinal, (sample_id, reference_text) in enumerate(imported_samples)
                )
                expected_file_identities = {
                    Path("profiles") / profile_id / f"{sample_id}.wav": _private_file_identity(
                        Path("profiles") / profile_id / f"{sample_id}.wav"
                    )
                    for sample_id, _reference_text in imported_samples
                }
                if prepared_avatar is not None:
                    avatar_relative = Path("profiles") / profile_id / "avatar.png"
                    expected_file_identities[avatar_relative] = _private_file_identity(avatar_relative)

                response = _profile_to_response(
                    db_profile,
                    sample_count=len(prepared_samples),
                )
                commit_started = True
                db.commit()
            except BaseException as operation_error:
                rollback_error: BaseException | None = None
                try:
                    db.rollback()
                except BaseException as exc:
                    rollback_error = exc
                    logger.error("Profile import rollback failed", exc_info=True)

                durable_response: VoiceProfileResponse | None = None
                commit_state: bool | None = False
                if commit_started and publish_intent is not None:
                    try:
                        durable_response = _durable_imported_profile_response(
                            db,
                            profile_id=profile_id,
                            expected_profile_fields=expected_profile_fields,
                            expected_sample_fields=expected_sample_fields,
                            expected_file_identities=expected_file_identities,
                            publish_intent=publish_intent,
                        )
                        commit_state = durable_response is not None
                    except Exception:
                        commit_state = None
                        logger.warning(
                            "Could not prove an ambiguous profile import commit",
                            exc_info=True,
                        )
                if durable_response is not None:
                    _finish_committed_import_intent(
                        publish_intent,
                        label="profile import",
                    )
                    return durable_response

                if publish_intent is None:
                    try:
                        deletion_journal.discard_managed_entry(pending_relative)
                    except (OSError, ValueError):
                        logger.warning("Could not clean an unpublished profile import staging directory")
                else:
                    _reconcile_failed_publication(
                        publish_intent,
                        db,
                        commit_state=commit_state,
                        label="profile import",
                    )
                if rollback_error is not None:
                    raise rollback_error from operation_error
                raise

            assert publish_intent is not None
            _finish_committed_import_intent(
                publish_intent,
                label="profile import",
            )
            return response
        finally:
            await run_blocking_operation_cancellation_safe(workspace.cleanup)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError) as exc:
        raise ValueError("Invalid ZIP file") from exc


async def export_generation_to_zip(generation_id: str, db: Session) -> ArchiveExport:
    """Export a generation into a bounded private ZIP owned by its response."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        raise ValueError(f"Generation {generation_id} not found")

    profile = db.query(DBVoiceProfile).filter_by(id=generation.profile_id).first()
    if not profile:
        raise ValueError(f"Profile {generation.profile_id} not found")

    versions = (
        db.query(DBGenerationVersion)
        .filter_by(generation_id=generation_id)
        .order_by(DBGenerationVersion.created_at)
        .all()
    )
    if len(versions) > GENERATION_ARCHIVE_MAX_MEMBERS - 1:
        raise ArchiveExportLimitError(
            f"Generation contains too many versions to export (max {GENERATION_ARCHIVE_MAX_MEMBERS - 1})"
        )

    file_members: list[_ArchiveExportMember] = []
    version_entries: list[dict] = []
    identity_names: dict[tuple[int, int, int, int, int], str] = {}
    name_identities: dict[str, tuple[int, int, int, int, int]] = {}

    def register_audio(stored_path: str | Path | None, *, index: int) -> str:
        relative = config.managed_storage_relative_path(stored_path)
        candidate = relative.name if relative is not None else ""
        candidate = _safe_audio_archive_basename(
            candidate,
            fallback=f"version-{index}.wav",
            allowed_extensions=GENERATION_ARCHIVE_AUDIO_EXTENSIONS,
        )
        provisional = _preflight_managed_export_file(
            stored_path,
            expected_directory="generations",
            archive_name=f"audio/{candidate}",
            max_bytes=GENERATION_ARCHIVE_MAX_ENTRY_BYTES,
        )
        existing_name = identity_names.get(provisional.identity)
        if existing_name is not None:
            return existing_name

        filename = candidate
        filename_suffix = PurePosixPath(candidate).suffix
        suffix = 1
        while filename in name_identities:
            filename = f"version-{index}-{suffix}{filename_suffix}"
            suffix += 1
        member = _ArchiveExportMember(
            path=provisional.path,
            archive_name=f"audio/{filename}",
            size=provisional.size,
            identity=provisional.identity,
        )
        file_members.append(member)
        identity_names[member.identity] = filename
        name_identities[filename] = member.identity
        return filename

    for index, version in enumerate(versions):
        filename = register_audio(version.audio_path, index=index)
        effects_chain = None
        if version.effects_chain:
            if len(version.effects_chain.encode("utf-8")) > ARCHIVE_JSON_MAX_BYTES:
                raise ArchiveExportLimitError("Version effects metadata is too large to export")
            effects_chain = json.loads(version.effects_chain)
        version_entries.append(
            {
                "id": version.id,
                "label": version.label,
                "is_default": version.is_default,
                "effects_chain": effects_chain,
                "filename": filename,
            }
        )

    if not versions:
        register_audio(generation.audio_path, index=0)

    manifest = {
        "version": "1.0",
        "generation": {
            "id": generation.id,
            "text": generation.text,
            "language": generation.language,
            "duration": generation.duration,
            "seed": generation.seed,
            "instruct": generation.instruct,
            "created_at": generation.created_at.isoformat(),
        },
        "profile": {
            "id": profile.id,
            "name": profile.name,
            "description": profile.description,
            "language": profile.language,
        },
        "versions": version_entries,
    }
    db.rollback()
    return await _build_archive_export(
        json_members=(("manifest.json", _encoded_archive_json("manifest.json", manifest)),),
        file_members=tuple(file_members),
        max_total_bytes=GENERATION_ARCHIVE_MAX_TOTAL_BYTES,
        max_members=GENERATION_ARCHIVE_MAX_MEMBERS,
    )


async def import_generation_from_zip(
    archive_source: ArchiveSource,
    db: Session,
) -> dict:
    """Validate a generation archive before publishing audio and one DB row."""
    try:
        plan = await run_blocking_operation_cancellation_safe(
            _inspect_generation_import,
            archive_source,
        )
        await run_blocking_operation_cancellation_safe(_cleanup_stale_archive_imports)
        workspace = _allocate_archive_import(plan.projected_bytes)
        try:
            staged_audio = workspace.directory / f"generation{plan.audio_suffix}"
            await run_blocking_operation_cancellation_safe(
                _extract_generation_import,
                archive_source,
                plan,
                staged_audio,
            )
            try:
                audio_duration, audio_channels, audio_sample_rate = await run_blocking_operation_cancellation_safe(
                    probe_audio_metadata,
                    staged_audio,
                )
            except Exception as exc:
                label = "invalid WAV audio" if plan.audio_suffix == ".wav" else "invalid audio"
                raise ValueError(f"Generation archive contains {label}") from exc
            if (
                audio_sample_rate <= 0
                or audio_sample_rate > GENERATION_AUDIO_MAX_SAMPLE_RATE
                or audio_channels <= 0
                or audio_channels > GENERATION_AUDIO_MAX_CHANNELS
            ):
                label = "invalid WAV audio" if plan.audio_suffix == ".wav" else "invalid audio"
                raise ValueError(f"Generation archive contains {label}")
            if (
                not math.isfinite(audio_duration)
                or audio_duration <= 0
                or audio_duration > GENERATION_AUDIO_MAX_DURATION_SECONDS
            ):
                raise ValueError("Generation audio duration is outside the allowed range")
            duration_tolerance = max(1.0, audio_duration * 0.05)
            if abs(plan.generation_data["duration"] - audio_duration) > duration_tolerance:
                raise ValueError("Manifest duration does not match the archived audio")

            profile_id = None
            profile_name = plan.profile_data.get("name", "Unknown Profile")
            if not isinstance(profile_name, str):
                raise ValueError("Invalid manifest.json: profile.name must be a string")
            if profile_name and profile_name != "Unknown Profile":
                existing_profile = db.query(DBVoiceProfile).filter_by(name=profile_name).first()
                if existing_profile:
                    profile_id = existing_profile.id
            if not profile_id:
                any_profile = db.query(DBVoiceProfile).first()
                if any_profile is None:
                    raise ValueError("No voice profiles found. Please create a profile before importing generations.")
                profile_id = any_profile.id
                profile_name = any_profile.name

            generations_dir = config.get_generations_dir()
            new_generation_id = str(uuid.uuid4())
            audio_destination = generations_dir / f"{new_generation_id}{plan.audio_suffix}"
            audio_relative = Path("generations") / audio_destination.name
            staging_relative = Path("generations") / (
                f".voicebox-delete-generation-import-{new_generation_id}{plan.audio_suffix}"
            )
            staging_destination = config.get_data_dir() / staging_relative
            stored_audio_path: str | None = None
            commit_started = False
            publish_intent: deletion_journal.DeletionIntent | None = None
            expected_generation_fields: dict[str, object] = {}
            try:
                _create_empty_private_file(staging_destination)
                pending_stat = deletion_journal.managed_entry_stat(staging_relative)
                if pending_stat is None or not stat.S_ISREG(pending_stat.st_mode):
                    raise RuntimeError("Imported generation staging audio is not a regular managed file")
                publish_intent = deletion_journal.prepare_deletion_intent(
                    kind=deletion_journal.GENERATION_AUDIO,
                    original=audio_relative,
                    staged=staging_relative,
                    entry_stat=pending_stat,
                    owner_id=new_generation_id,
                )
                await run_blocking_operation_cancellation_safe(
                    _populate_private_file,
                    staged_audio,
                    staging_destination,
                    expected_stat=pending_stat,
                )
                deletion_journal.rename_managed_entry(staging_relative, audio_relative)
                stored_audio_path = config.to_storage_path(audio_destination)

                created_at = datetime.utcnow()
                db_generation = DBGeneration(
                    id=new_generation_id,
                    profile_id=profile_id,
                    text=plan.generation_data["text"],
                    language=plan.generation_data["language"],
                    audio_path=stored_audio_path,
                    duration=audio_duration,
                    seed=plan.generation_data["seed"],
                    instruct=plan.generation_data["instruct"],
                    engine="qwen",
                    model_size=None,
                    status="completed",
                    error=None,
                    is_favorited=False,
                    source="manual",
                    exact_request_sha256=None,
                    exact_envelope_sha256=None,
                    exact_effects_json=None,
                    exact_voice_snapshot_json=None,
                    voice_binding_sha256=None,
                    created_at=created_at,
                )
                expected_generation_fields = {
                    field: getattr(db_generation, field)
                    for field in (
                        "id",
                        "profile_id",
                        "text",
                        "language",
                        "audio_path",
                        "duration",
                        "seed",
                        "instruct",
                        "engine",
                        "model_size",
                        "status",
                        "error",
                        "is_favorited",
                        "source",
                        "exact_request_sha256",
                        "exact_envelope_sha256",
                        "exact_effects_json",
                        "exact_voice_snapshot_json",
                        "voice_binding_sha256",
                        "created_at",
                    )
                }
                result = {
                    "id": db_generation.id,
                    "profile_id": profile_id,
                    "profile_name": profile_name,
                    "text": db_generation.text,
                    "message": (f"Generation imported successfully (assigned to profile: {profile_name})"),
                }
                db.add(db_generation)
                commit_started = True
                db.commit()
            except BaseException as operation_error:
                rollback_error: BaseException | None = None
                try:
                    db.rollback()
                except BaseException as exc:
                    rollback_error = exc
                    logger.error("Generation import rollback failed", exc_info=True)

                durable_result: dict | None = None
                commit_state: bool | None = False
                if commit_started and stored_audio_path is not None and publish_intent is not None:
                    try:
                        durable_result = _durable_imported_generation_result(
                            db,
                            generation_id=new_generation_id,
                            expected_fields=expected_generation_fields,
                            profile_name=profile_name,
                            publish_intent=publish_intent,
                        )
                        commit_state = durable_result is not None
                    except Exception:
                        commit_state = None
                        logger.warning(
                            "Could not prove an ambiguous generation import commit",
                            exc_info=True,
                        )
                if durable_result is not None:
                    _finish_committed_import_intent(
                        publish_intent,
                        label="generation import",
                    )
                    return durable_result

                if publish_intent is None:
                    try:
                        deletion_journal.discard_managed_entry(staging_relative)
                    except (OSError, ValueError):
                        logger.warning("Could not clean unpublished generation import staging audio")
                else:
                    _reconcile_failed_publication(
                        publish_intent,
                        db,
                        commit_state=commit_state,
                        label="generation import",
                    )
                if rollback_error is not None:
                    raise rollback_error from operation_error
                raise

            assert publish_intent is not None
            _finish_committed_import_intent(
                publish_intent,
                label="generation import",
            )
            return result
        finally:
            await run_blocking_operation_cancellation_safe(workspace.cleanup)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError) as exc:
        raise ValueError("Invalid ZIP file") from exc
