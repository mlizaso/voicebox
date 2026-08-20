"""Bounded, portable extraction for downloadable backend release archives."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES = 8 * 1024**3
BACKEND_ARCHIVE_MAX_MEMBERS = 100_000
BACKEND_ARCHIVE_MAX_TOTAL_BYTES = 16 * 1024**3
BACKEND_ARCHIVE_MAX_MEMBER_BYTES = 8 * 1024**3
BACKEND_ARCHIVE_MAX_NAME_LENGTH = 4096
BACKEND_ARCHIVE_COPY_CHUNK_BYTES = 1024 * 1024
BACKEND_ARCHIVE_MIN_FREE_BYTES = 1024**3
BACKEND_ARCHIVE_RATIO_MIN_TOTAL_BYTES = 64 * 1024**2
BACKEND_ARCHIVE_MAX_COMPRESSION_RATIO = 200

_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class BackendArchiveError(ValueError):
    """Raised when a release archive cannot be extracted safely."""


class BackendInstallRecoveryError(RuntimeError):
    """Raised when an interrupted backend install cannot be reconciled safely."""


class _BackendArchiveCancelledError(RuntimeError):
    """Internal cooperative-cancellation signal for the extraction worker."""


async def run_blocking_cancellation_safe(function, /, *args, cooperative: bool = False, **kwargs):
    """Run, and on cancellation drain, one filesystem worker operation."""
    cancel_event = threading.Event()
    if cooperative:
        kwargs["cancel_event"] = cancel_event
    operation = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError as cancellation:
        cancel_event.set()
        while not operation.done():
            try:
                await asyncio.shield(operation)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        if not operation.cancelled():
            with suppress(BaseException):
                operation.result()
        raise cancellation


@dataclass(frozen=True)
class _ValidatedTarMember:
    info: tarfile.TarInfo
    parts: tuple[str, ...]
    is_directory: bool


def _canonical_member_parts(name: str) -> tuple[str, ...]:
    """Return one portable relative path, rejecting filesystem aliases."""
    try:
        encoded_name = name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise BackendArchiveError("Backend archive contains an invalid member name") from exc
    if not name or len(encoded_name) > BACKEND_ARCHIVE_MAX_NAME_LENGTH:
        raise BackendArchiveError("Backend archive contains an invalid member name")
    if "\x00" in name or "\\" in name or name.startswith("/"):
        raise BackendArchiveError("Backend archive contains an unsafe member path")

    canonical = name[:-1] if name.endswith("/") else name
    raw_parts = canonical.split("/")
    if not canonical or any(part in {"", ".", ".."} for part in raw_parts):
        raise BackendArchiveError("Backend archive contains an unsafe member path")
    if len(raw_parts) > 128:
        raise BackendArchiveError("Backend archive member path is too deeply nested")

    for part in raw_parts:
        if len(part.encode("utf-8")) > 255 or ":" in part or part.endswith((" ", ".")):
            raise BackendArchiveError("Backend archive contains a non-portable member path")
        if not part.isprintable() or part.startswith((".backend-extract-", ".download-")):
            raise BackendArchiveError("Backend archive contains a reserved member path")
        device_name = part.split(".", 1)[0].casefold()
        if device_name in _WINDOWS_RESERVED_NAMES:
            raise BackendArchiveError("Backend archive contains a reserved member path")

    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or path.parts != tuple(raw_parts):
        raise BackendArchiveError("Backend archive contains an unsafe member path")
    return tuple(raw_parts)


def _validate_members(
    archive: tarfile.TarFile,
    *,
    max_members: int,
    max_total_bytes: int,
    max_member_bytes: int,
    cancel_event: threading.Event | None = None,
) -> list[_ValidatedTarMember]:
    members: list[_ValidatedTarMember] = []
    paths: dict[tuple[str, ...], bool] = {}
    total_bytes = 0

    for info in archive:
        if cancel_event is not None and cancel_event.is_set():
            raise _BackendArchiveCancelledError
        if len(members) >= max_members:
            raise BackendArchiveError(f"Backend archive contains too many members (max {max_members})")
        parts = _canonical_member_parts(info.name)
        collision_key = tuple(part.casefold() for part in parts)
        if collision_key in paths:
            raise BackendArchiveError("Backend archive contains duplicate member paths")

        is_directory = info.isdir()
        if not is_directory and not info.isreg():
            raise BackendArchiveError("Backend archive contains a link or special-file member")
        if info.size < 0 or (is_directory and info.size != 0):
            raise BackendArchiveError("Backend archive contains an invalid member size")
        if info.size > max_member_bytes:
            raise BackendArchiveError(f"Backend archive member exceeds the size limit ({max_member_bytes} bytes)")

        total_bytes += info.size
        if total_bytes > max_total_bytes:
            raise BackendArchiveError(
                f"Backend archive exceeds the total extracted-size limit ({max_total_bytes} bytes)"
            )

        paths[collision_key] = is_directory
        members.append(_ValidatedTarMember(info=info, parts=parts, is_directory=is_directory))

    # Reject file/directory conflicts before creating a single archive path.
    if not members:
        raise BackendArchiveError("Backend archive contains no members")
    for path in paths:
        for depth in range(1, len(path)):
            if paths.get(path[:depth]) is False:
                raise BackendArchiveError("Backend archive contains conflicting member paths")

    return members


def _validate_compression_ratio(archive_path: Path, members: list[_ValidatedTarMember]) -> None:
    total_member_bytes = sum(member.info.size for member in members)
    compressed_bytes = archive_path.stat().st_size
    if total_member_bytes >= BACKEND_ARCHIVE_RATIO_MIN_TOTAL_BYTES and (
        compressed_bytes == 0 or total_member_bytes / compressed_bytes > BACKEND_ARCHIVE_MAX_COMPRESSION_RATIO
    ):
        raise BackendArchiveError("Backend archive exceeds the compression-ratio limit")


def _require_real_directory(path: Path) -> None:
    path_stat = path.lstat()
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise BackendArchiveError("Backend archive destination contains an unsafe directory")


def _path_present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_regular_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise BackendInstallRecoveryError("Backend install contains a non-regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_backend_tree(path: Path) -> None:
    _require_real_directory(path)
    directories = [path]
    member_count = 0
    total_bytes = 0
    for root, child_directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        directories.extend(root_path / name for name in child_directories)
        member_count += len(child_directories) + len(files)
        if member_count > BACKEND_ARCHIVE_MAX_MEMBERS:
            raise BackendInstallRecoveryError("Backend install contains too many filesystem entries")
        for name in child_directories:
            child = root_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                continue
            if not stat.S_ISDIR(child_stat.st_mode):
                raise BackendInstallRecoveryError("Backend install contains an unsafe directory entry")
        for name in files:
            child = root_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                continue
            if not stat.S_ISREG(child_stat.st_mode):
                raise BackendInstallRecoveryError("Backend install contains an unsafe file entry")
            total_bytes += child_stat.st_size
            if total_bytes > BACKEND_ARCHIVE_MAX_TOTAL_BYTES:
                raise BackendInstallRecoveryError("Backend install exceeds the durable staging size limit")
            _fsync_regular_file(child)
    for directory in reversed(directories):
        if not directory.is_symlink():
            _fsync_directory(directory)


@dataclass(frozen=True)
class _BackendInstallLayout:
    root: Path
    backend_name: str
    executable_name: str

    @property
    def live(self) -> Path:
        return self.root / self.backend_name

    @property
    def staging(self) -> Path:
        return self.root / f"{self.backend_name}-staging"

    @property
    def backup(self) -> Path:
        return self.root / f"{self.backend_name}-backup"

    @property
    def journal(self) -> Path:
        return self.root / f".{self.backend_name}-install-swap-v1.json"

    @property
    def journal_temp_prefix(self) -> str:
        return f".{self.backend_name}-install-swap-"


def _backend_install_layout(root: Path, backend_name: str, executable_name: str) -> _BackendInstallLayout:
    if not backend_name or not backend_name.replace("-", "").isalnum() or Path(executable_name).name != executable_name:
        raise BackendInstallRecoveryError("Backend install layout is invalid")
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    _require_real_directory(root)
    return _BackendInstallLayout(root=root, backend_name=backend_name, executable_name=executable_name)


def _backend_directory_is_valid(path: Path, executable_name: str) -> bool:
    try:
        directory_stat = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
        raise BackendInstallRecoveryError("Backend install path is not a real directory")
    executable = path / executable_name
    try:
        executable_stat = executable.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(executable_stat.st_mode) or stat.S_ISLNK(executable_stat.st_mode):
        raise BackendInstallRecoveryError("Backend executable is not a regular file")
    return True


def _write_swap_journal(layout: _BackendInstallLayout) -> None:
    payload = json.dumps(
        {
            "backend": layout.backend_name,
            "backup": layout.backup.name,
            "live": layout.live.name,
            "staging": layout.staging.name,
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=layout.journal_temp_prefix,
        suffix=".tmp",
        dir=layout.root,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as journal_file:
            descriptor = -1
            journal_file.write(payload)
            journal_file.flush()
            os.fsync(journal_file.fileno())
        os.replace(temporary_path, layout.journal)
        _fsync_directory(layout.root)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


def _read_swap_journal(layout: _BackendInstallLayout) -> bool:
    if not _path_present(layout.journal):
        return False
    descriptor = os.open(layout.journal, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        journal_stat = os.fstat(descriptor)
        if not stat.S_ISREG(journal_stat.st_mode) or journal_stat.st_size > 4096:
            raise BackendInstallRecoveryError("Backend install swap journal is unsafe")
        payload = os.read(descriptor, 4097)
    finally:
        os.close(descriptor)
    expected = {
        "backend": layout.backend_name,
        "backup": layout.backup.name,
        "live": layout.live.name,
        "staging": layout.staging.name,
        "version": 1,
    }
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendInstallRecoveryError("Backend install swap journal is corrupt") from exc
    if parsed != expected:
        raise BackendInstallRecoveryError("Backend install swap journal does not match its managed paths")
    return True


def _unlink_managed_file(path: Path) -> None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(path_stat.st_mode) and not stat.S_ISLNK(path_stat.st_mode):
        raise BackendInstallRecoveryError("Managed backend metadata path is an unexpected directory")
    path.unlink()


def _cleanup_journal_temps(layout: _BackendInstallLayout) -> None:
    for entry in layout.root.iterdir():
        if entry.name.startswith(layout.journal_temp_prefix) and entry.name.endswith(".tmp"):
            _unlink_managed_file(entry)


def _remove_invalid_or_missing_directory(path: Path) -> None:
    if _path_present(path):
        remove_backend_directory(path)


def recover_backend_install(root: Path, backend_name: str, executable_name: str) -> None:
    """Reconcile every durable directory-swap state after interruption."""
    layout = _backend_install_layout(root, backend_name, executable_name)
    has_journal = _read_swap_journal(layout)
    live_valid = _backend_directory_is_valid(layout.live, executable_name)
    staging_valid = _backend_directory_is_valid(layout.staging, executable_name)
    backup_valid = _backend_directory_is_valid(layout.backup, executable_name)

    if live_valid:
        remove_backend_directory(layout.staging)
        remove_backend_directory(layout.backup)
    elif has_journal and staging_valid:
        _remove_invalid_or_missing_directory(layout.live)
        layout.staging.rename(layout.live)
        _fsync_directory(layout.root)
        remove_backend_directory(layout.backup)
    elif backup_valid:
        _remove_invalid_or_missing_directory(layout.live)
        remove_backend_directory(layout.staging)
        layout.backup.rename(layout.live)
        _fsync_directory(layout.root)
    elif has_journal:
        raise BackendInstallRecoveryError("Interrupted backend install has no valid recovery candidate")
    else:
        # Staging is only known complete after the durable swap journal lands.
        # A process can die midway through extraction after the executable was
        # written, so executable presence alone must never publish an
        # unjournaled fresh install.
        remove_backend_directory(layout.live)
        remove_backend_directory(layout.staging)
        remove_backend_directory(layout.backup)

    _unlink_managed_file(layout.journal)
    _cleanup_journal_temps(layout)
    _fsync_directory(layout.root)


def commit_backend_install(
    root: Path,
    backend_name: str,
    executable_name: str,
    *,
    transition_hook: Callable[[str], None] | None = None,
) -> None:
    """Durably replace a live backend and leave all crash states recoverable."""
    layout = _backend_install_layout(root, backend_name, executable_name)
    if not _backend_directory_is_valid(layout.staging, executable_name):
        raise BackendInstallRecoveryError("Staged backend is missing its executable")
    if _path_present(layout.backup) or _path_present(layout.journal):
        raise BackendInstallRecoveryError("Backend install was not reconciled before commit")

    try:
        _fsync_backend_tree(layout.staging)
        _write_swap_journal(layout)
        if transition_hook is not None:
            transition_hook("journal")

        if _path_present(layout.live):
            layout.live.rename(layout.backup)
            _fsync_directory(layout.root)
        if transition_hook is not None:
            transition_hook("backup")

        layout.staging.rename(layout.live)
        _fsync_directory(layout.root)
        if transition_hook is not None:
            transition_hook("live")

        remove_backend_directory(layout.backup)
        _fsync_directory(layout.root)
        if transition_hook is not None:
            transition_hook("cleanup")

        _unlink_managed_file(layout.journal)
        _fsync_directory(layout.root)
        _cleanup_journal_temps(layout)
    except BaseException:
        recover_backend_install(root, backend_name, executable_name)
        raise


def delete_backend_install(root: Path, backend_name: str, executable_name: str) -> bool:
    """Delete live and interrupted-install state as one drained worker action."""
    layout = _backend_install_layout(root, backend_name, executable_name)
    deleted = False
    for directory in (layout.staging, layout.backup, layout.live):
        if _path_present(directory):
            if _backend_directory_is_valid(directory, executable_name) or backend_directory_has_entries(directory):
                deleted = True
            remove_backend_directory(directory)
    if _path_present(layout.journal):
        deleted = True
    _unlink_managed_file(layout.journal)
    _cleanup_journal_temps(layout)
    _fsync_directory(layout.root)
    return deleted


def remove_backend_directory(path: Path) -> None:
    """Remove one real managed directory without following a replacement link."""
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise BackendArchiveError("Refusing to remove an unsafe backend directory")
    shutil.rmtree(path)


def backend_directory_has_entries(path: Path) -> bool:
    """Inspect one real managed directory without following a root symlink."""
    try:
        _require_real_directory(path)
    except FileNotFoundError:
        return False
    with os.scandir(path) as entries:
        return next(entries, None) is not None


def require_backend_free_space(path: Path, required_bytes: int) -> None:
    """Preserve the backend installer disk reserve before one bounded write."""
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as exc:
        raise BackendArchiveError("Could not verify free space for backend storage") from exc
    if required_bytes < 0 or free_bytes - required_bytes < BACKEND_ARCHIVE_MIN_FREE_BYTES:
        raise BackendArchiveError("Insufficient free space for backend storage")


def backend_directory_allocation_bytes(source: Path) -> int:
    """Validate and size the future allocation needed to stage an install."""
    _require_real_directory(source)
    member_count = 0
    total_bytes = 0
    for root, directories, files in os.walk(source, followlinks=False):
        member_count += len(directories) + len(files)
        if member_count > BACKEND_ARCHIVE_MAX_MEMBERS:
            raise BackendArchiveError("Installed backend contains too many filesystem entries")
        for directory_name in directories:
            directory_stat = (Path(root) / directory_name).lstat()
            if not (stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode)):
                raise BackendArchiveError("Installed backend contains an unsafe filesystem entry")
        for filename in files:
            file_stat = (Path(root) / filename).lstat()
            if stat.S_ISREG(file_stat.st_mode):
                total_bytes += file_stat.st_size
                if total_bytes > BACKEND_ARCHIVE_MAX_TOTAL_BYTES:
                    raise BackendArchiveError("Installed backend exceeds the staging size limit")
            elif not stat.S_ISLNK(file_stat.st_mode):
                raise BackendArchiveError("Installed backend contains an unsafe filesystem entry")
    return total_bytes


def copy_backend_directory(source: Path, destination: Path) -> None:
    """Copy an installed tree without following symlinks stored inside it."""
    total_bytes = backend_directory_allocation_bytes(source)
    require_backend_free_space(destination.parent, total_bytes)
    shutil.copytree(source, destination, symlinks=True)


def sha256_backend_file(path: Path, *, cancel_event: threading.Event | None = None) -> str:
    """Hash one bounded download cooperatively outside the event loop."""
    digest = hashlib.sha256()
    total_bytes = 0
    with path.open("rb") as archive_file:
        while chunk := archive_file.read(BACKEND_ARCHIVE_COPY_CHUNK_BYTES):
            if cancel_event is not None and cancel_event.is_set():
                raise _BackendArchiveCancelledError
            total_bytes += len(chunk)
            if total_bytes > BACKEND_ARCHIVE_MAX_COMPRESSED_BYTES:
                raise BackendArchiveError("Backend archive exceeds the compressed archive size limit")
            digest.update(chunk)
    return digest.hexdigest()


def _preflight_destination(destination: Path, members: list[_ValidatedTarMember]) -> None:
    """Reject pre-existing links and type conflicts before extraction."""
    _require_real_directory(destination)
    for member in members:
        current = destination
        for part in member.parts[:-1]:
            current /= part
            try:
                _require_real_directory(current)
            except FileNotFoundError:
                break

        target = destination.joinpath(*member.parts)
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(target_stat.st_mode):
            raise BackendArchiveError("Backend archive destination contains an unsafe link")
        if member.is_directory:
            if not stat.S_ISDIR(target_stat.st_mode):
                raise BackendArchiveError("Backend archive destination contains a path-type conflict")
        elif not stat.S_ISREG(target_stat.st_mode):
            raise BackendArchiveError("Backend archive destination contains a path-type conflict")


def _required_extraction_space(destination: Path, members: list[_ValidatedTarMember]) -> int:
    """Estimate the peak additional bytes used by ordered atomic replacements."""
    committed_delta = 0
    peak_bytes = 0
    for member in members:
        if member.is_directory:
            continue
        target = destination.joinpath(*member.parts)
        try:
            replaced_size = target.lstat().st_size
        except FileNotFoundError:
            replaced_size = 0
        peak_bytes = max(peak_bytes, committed_delta + member.info.size)
        committed_delta += member.info.size - replaced_size
    return max(0, peak_bytes)


def _ensure_parent_directories(destination: Path, parts: tuple[str, ...]) -> Path:
    current = destination
    for part in parts:
        current /= part
        with suppress(FileExistsError):
            current.mkdir(mode=0o700)
        _require_real_directory(current)
    return current


def _extract_regular_member(
    archive: tarfile.TarFile,
    member: _ValidatedTarMember,
    destination: Path,
    *,
    cancel_event: threading.Event | None,
) -> int:
    parent = _ensure_parent_directories(destination, member.parts[:-1])
    target = parent / member.parts[-1]
    source = archive.extractfile(member.info)
    if source is None:
        raise BackendArchiveError("Backend archive regular member has no readable payload")

    descriptor, temporary_name = tempfile.mkstemp(prefix=".backend-extract-", dir=parent)
    temporary_path = Path(temporary_name)
    written = 0
    try:
        with source, os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            while chunk := source.read(BACKEND_ARCHIVE_COPY_CHUNK_BYTES):
                if cancel_event is not None and cancel_event.is_set():
                    raise _BackendArchiveCancelledError
                written += len(chunk)
                if written > member.info.size:
                    raise BackendArchiveError("Backend archive member exceeded its declared size")
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written != member.info.size:
            raise BackendArchiveError("Backend archive member did not match its declared size")
        os.chmod(temporary_path, 0o700 if member.info.mode & 0o111 else 0o600)
        os.replace(temporary_path, target)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
    return written


def inspect_backend_tar_archive(
    archive_path: Path,
    destination: Path,
    *,
    max_members: int = BACKEND_ARCHIVE_MAX_MEMBERS,
    max_total_bytes: int = BACKEND_ARCHIVE_MAX_TOTAL_BYTES,
    max_member_bytes: int = BACKEND_ARCHIVE_MAX_MEMBER_BYTES,
    cancel_event: threading.Event | None = None,
) -> int:
    """Validate an archive and return its peak future extraction allocation."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _validate_members(
            archive,
            max_members=max_members,
            max_total_bytes=max_total_bytes,
            max_member_bytes=max_member_bytes,
            cancel_event=cancel_event,
        )
    _validate_compression_ratio(archive_path, members)
    if cancel_event is not None and cancel_event.is_set():
        raise _BackendArchiveCancelledError
    _preflight_destination(destination, members)
    return _required_extraction_space(destination, members)


def extract_backend_tar_archive(
    archive_path: Path,
    destination: Path,
    *,
    max_members: int = BACKEND_ARCHIVE_MAX_MEMBERS,
    max_total_bytes: int = BACKEND_ARCHIVE_MAX_TOTAL_BYTES,
    max_member_bytes: int = BACKEND_ARCHIVE_MAX_MEMBER_BYTES,
    cancel_event: threading.Event | None = None,
) -> None:
    """Preflight and stream-extract only bounded regular files/directories."""
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _validate_members(
            archive,
            max_members=max_members,
            max_total_bytes=max_total_bytes,
            max_member_bytes=max_member_bytes,
            cancel_event=cancel_event,
        )
        _validate_compression_ratio(archive_path, members)
        if cancel_event is not None and cancel_event.is_set():
            raise _BackendArchiveCancelledError

        destination.mkdir(parents=True, mode=0o700, exist_ok=True)
        _preflight_destination(destination, members)
        required_bytes = _required_extraction_space(destination, members)
        require_backend_free_space(destination, required_bytes)

        total_written = 0
        for member in members:
            if cancel_event is not None and cancel_event.is_set():
                raise _BackendArchiveCancelledError
            if member.is_directory:
                _ensure_parent_directories(destination, member.parts)
                continue
            total_written += _extract_regular_member(
                archive,
                member,
                destination,
                cancel_event=cancel_event,
            )
            if total_written > max_total_bytes:
                raise BackendArchiveError("Backend archive exceeded the total extracted-size limit")
