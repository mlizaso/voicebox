"""POSIX permission hardening for Voicebox-managed local data.

All filesystem traversal in this module is descriptor-relative and refuses
symbolic links.  This keeps permission repair inside the configured data root
even when an untrusted local process can replace directory entries while the
server is starting.
"""

from __future__ import annotations

import os
import stat
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
SHARED_ROOT_MODE = 0o711
SHARED_DIR_MODE = 0o755
SHARED_FILE_MODE = 0o644

_PRIVATE_RECURSIVE_DIRS = (
    "cache",
    "captures",
    "deletion_journal",
    "exact_voice_snapshots",
    "logs",
    "profiles",
)
_PRIVATE_TOP_LEVEL_DIRS = ("backends", "models")
_DATABASE_FILES = (
    ".voicebox.lock",
    "voicebox.db",
    "voicebox.db-journal",
    "voicebox.db-shm",
    "voicebox.db-wal",
)


class UnsafeDataPathError(RuntimeError):
    """Raised when a managed data path is not a real directory."""


@dataclass
class PermissionRepairReport:
    """Aggregate result that is safe to log without revealing local paths."""

    repaired: int = 0
    skipped: int = 0


def _supports_posix_permissions() -> bool:
    """Return whether secure descriptor-relative POSIX repair is available."""
    return (
        os.name == "posix"
        and hasattr(os, "fchmod")
        and hasattr(os, "geteuid")
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
    )


def set_private_umask() -> bool:
    """Make future Voicebox-created files private on supported POSIX hosts."""
    if not _supports_posix_permissions():
        return False
    os.umask(0o077)
    return True


def data_root_mode(shared_generations: bool) -> int:
    """Return the root mode needed by the selected export policy."""
    return SHARED_ROOT_MODE if shared_generations else PRIVATE_DIR_MODE


def generation_dir_mode(shared_generations: bool) -> int:
    """Return the generated-audio directory mode for the export policy."""
    return SHARED_DIR_MODE if shared_generations else PRIVATE_DIR_MODE


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _file_open_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _owned_by_process(entry_stat: os.stat_result) -> bool:
    return entry_stat.st_uid == os.geteuid()


def _is_link_or_reparse_point(path: Path, entry_stat: os.stat_result) -> bool:
    """Recognize POSIX links plus Windows junction/reparse-point aliases."""
    if stat.S_ISLNK(entry_stat.st_mode):
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None:
        try:
            if is_junction():
                return True
        except OSError:
            return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(entry_stat, "st_file_attributes", 0) & reparse_flag)


def _ensure_directory_tree_fallback(path: Path) -> None:
    """Best available no-link directory creation where dir-fd APIs are absent."""
    current = Path(path.anchor)
    for component in path.parts[1:]:
        candidate = current / component
        try:
            entry_stat = candidate.lstat()
        except FileNotFoundError:
            try:
                candidate.mkdir()
            except FileExistsError:
                pass
            except OSError as exc:
                raise UnsafeDataPathError("Could not create a safe Voicebox data directory") from exc
            try:
                entry_stat = candidate.lstat()
            except OSError as exc:
                raise UnsafeDataPathError("Could not validate a Voicebox data directory") from exc
        except OSError as exc:
            raise UnsafeDataPathError("Could not validate a Voicebox data path") from exc
        if _is_link_or_reparse_point(candidate, entry_stat) or not stat.S_ISDIR(entry_stat.st_mode):
            raise UnsafeDataPathError("Configured Voicebox data path contains an unsafe component")
        current = candidate


def _repair_open_fd(fd: int, expected_mode: int, report: PermissionRepairReport) -> None:
    entry_stat = os.fstat(fd)
    if not _owned_by_process(entry_stat):
        report.skipped += 1
        return
    if stat.S_IMODE(entry_stat.st_mode) != expected_mode:
        os.fchmod(fd, expected_mode)
        report.repaired += 1


def _open_data_root(root: Path) -> int:
    if _supports_posix_permissions():
        return _open_directory_tree(root, create=False)
    try:
        fd = os.open(root, _directory_open_flags())
    except OSError as exc:
        raise UnsafeDataPathError("Configured Voicebox data root is not a real directory") from exc
    if not stat.S_ISDIR(os.fstat(fd).st_mode):
        os.close(fd)
        raise UnsafeDataPathError("Configured Voicebox data root is not a directory")
    return fd


def _open_directory_tree(
    path: Path,
    *,
    create: bool,
    final_mode: int = PRIVATE_DIR_MODE,
) -> int:
    """Open an absolute directory path without following any linked component."""
    if not path.is_absolute() or not path.anchor:
        raise UnsafeDataPathError("Configured Voicebox data root must be absolute")
    if path == Path(path.anchor):
        raise UnsafeDataPathError("Filesystem roots cannot be used as the Voicebox data directory")
    components = path.parts[1:]
    current_fd = os.open(path.anchor, _directory_open_flags())
    try:
        for index, component in enumerate(components):
            is_final = index == len(components) - 1
            try:
                next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(
                    component,
                    mode=final_mode if is_final else 0o777,
                    dir_fd=current_fd,
                )
                next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            except OSError as exc:
                raise UnsafeDataPathError("Configured Voicebox data path contains an unsafe component") from exc
            next_stat = os.fstat(next_fd)
            if not stat.S_ISDIR(next_stat.st_mode):
                os.close(next_fd)
                raise UnsafeDataPathError("Configured Voicebox data path contains a non-directory component")
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def ensure_data_root(root: Path, *, shared_generations: bool) -> PermissionRepairReport:
    """Create/open the data root without following any ancestor symbolic link."""
    if not root.is_absolute() or not root.anchor or root == Path(root.anchor):
        raise UnsafeDataPathError("Voicebox data must be an absolute directory below a filesystem root")
    if not _supports_posix_permissions():
        _ensure_directory_tree_fallback(root)
        return harden_data_root(root, shared_generations=shared_generations)

    report = PermissionRepairReport()
    root_fd = _open_directory_tree(
        root,
        create=True,
        final_mode=data_root_mode(shared_generations),
    )
    try:
        _repair_open_fd(root_fd, data_root_mode(shared_generations), report)
    finally:
        os.close(root_fd)
    return report


def harden_data_root(root: Path, *, shared_generations: bool) -> PermissionRepairReport:
    """Apply the root policy without resolving or following the root path."""
    report = PermissionRepairReport()
    if not _supports_posix_permissions():
        return report

    root_fd = _open_data_root(root)
    try:
        _repair_open_fd(root_fd, data_root_mode(shared_generations), report)
    finally:
        os.close(root_fd)
    return report


def ensure_managed_directory(root: Path, name: str, *, mode: int) -> Path:
    """Create or validate one direct child directory without following links."""
    path = root / name
    if not _supports_posix_permissions():
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError("Managed directory name must be one path component")
        _ensure_directory_tree_fallback(path)
        return path
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Managed directory name must be one path component")

    root_fd = _open_data_root(root)
    child_fd = None
    try:
        with suppress(FileExistsError):
            os.mkdir(name, mode=mode, dir_fd=root_fd)
        try:
            child_fd = os.open(name, _directory_open_flags(), dir_fd=root_fd)
        except OSError as exc:
            raise UnsafeDataPathError("Managed Voicebox data entry is not a real directory") from exc
        if not stat.S_ISDIR(os.fstat(child_fd).st_mode):
            raise UnsafeDataPathError("Managed Voicebox data entry is not a directory")
        _repair_open_fd(child_fd, mode, PermissionRepairReport())
    finally:
        if child_fd is not None:
            os.close(child_fd)
        os.close(root_fd)
    return path


def _repair_tree_fd(
    directory_fd: int,
    *,
    directory_mode: int,
    file_mode: int,
    report: PermissionRepairReport,
) -> None:
    try:
        with os.scandir(directory_fd) as iterator:
            entries = list(iterator)
    except OSError:
        report.skipped += 1
        return

    for entry in entries:
        try:
            if entry.is_symlink():
                report.skipped += 1
                continue
            if entry.is_dir(follow_symlinks=False):
                child_fd = os.open(entry.name, _directory_open_flags(), dir_fd=directory_fd)
                try:
                    child_stat = os.fstat(child_fd)
                    if not stat.S_ISDIR(child_stat.st_mode):
                        report.skipped += 1
                        continue
                    _repair_open_fd(child_fd, directory_mode, report)
                    if _owned_by_process(child_stat):
                        _repair_tree_fd(
                            child_fd,
                            directory_mode=directory_mode,
                            file_mode=file_mode,
                            report=report,
                        )
                finally:
                    os.close(child_fd)
                continue
            if not entry.is_file(follow_symlinks=False):
                report.skipped += 1
                continue

            child_fd = os.open(entry.name, _file_open_flags(), dir_fd=directory_fd)
            try:
                child_stat = os.fstat(child_fd)
                if not stat.S_ISREG(child_stat.st_mode) or child_stat.st_nlink != 1:
                    report.skipped += 1
                    continue
                _repair_open_fd(child_fd, file_mode, report)
            finally:
                os.close(child_fd)
        except OSError:
            report.skipped += 1


def _repair_named_tree(
    root_fd: int,
    name: str,
    *,
    directory_mode: int,
    file_mode: int,
    recurse: bool,
    report: PermissionRepairReport,
) -> None:
    try:
        directory_fd = os.open(name, _directory_open_flags(), dir_fd=root_fd)
    except FileNotFoundError:
        return
    except OSError:
        report.skipped += 1
        return

    try:
        directory_stat = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_stat.st_mode):
            report.skipped += 1
            return
        _repair_open_fd(directory_fd, directory_mode, report)
        if recurse and _owned_by_process(directory_stat):
            _repair_tree_fd(
                directory_fd,
                directory_mode=directory_mode,
                file_mode=file_mode,
                report=report,
            )
    finally:
        os.close(directory_fd)


def _repair_root_file(root_fd: int, name: str, report: PermissionRepairReport) -> None:
    try:
        file_fd = os.open(name, _file_open_flags(), dir_fd=root_fd)
    except FileNotFoundError:
        return
    except OSError:
        report.skipped += 1
        return
    try:
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            report.skipped += 1
            return
        _repair_open_fd(file_fd, PRIVATE_FILE_MODE, report)
    finally:
        os.close(file_fd)


def repair_data_permissions(
    root: Path,
    *,
    shared_generations: bool,
) -> PermissionRepairReport:
    """Repair existing Voicebox-owned sensitive paths beneath ``root``."""
    report = PermissionRepairReport()
    if not _supports_posix_permissions():
        return report

    root_fd = _open_data_root(root)
    try:
        _repair_open_fd(root_fd, data_root_mode(shared_generations), report)
        for name in _DATABASE_FILES:
            _repair_root_file(root_fd, name, report)
        for name in _PRIVATE_RECURSIVE_DIRS:
            _repair_named_tree(
                root_fd,
                name,
                directory_mode=PRIVATE_DIR_MODE,
                file_mode=PRIVATE_FILE_MODE,
                recurse=True,
                report=report,
            )
        for name in _PRIVATE_TOP_LEVEL_DIRS:
            _repair_named_tree(
                root_fd,
                name,
                directory_mode=PRIVATE_DIR_MODE,
                file_mode=PRIVATE_FILE_MODE,
                recurse=False,
                report=report,
            )
        _repair_named_tree(
            root_fd,
            "generations",
            directory_mode=generation_dir_mode(shared_generations),
            file_mode=SHARED_FILE_MODE if shared_generations else PRIVATE_FILE_MODE,
            recurse=True,
            report=report,
        )
    finally:
        os.close(root_fd)
    return report


def apply_managed_file_permissions(
    root: Path,
    path: Path,
    *,
    shared_generations: bool,
) -> bool:
    """Harden a newly written managed file and its parents, link-safely."""
    if not _supports_posix_permissions():
        return False

    absolute_root = Path(os.path.abspath(root))
    absolute_path = Path(os.path.abspath(path))
    try:
        relative = absolute_path.relative_to(absolute_root)
    except ValueError:
        return False
    if not relative.parts:
        return False

    shared = shared_generations and relative.parts[0] == "generations"
    directory_mode = SHARED_DIR_MODE if shared else PRIVATE_DIR_MODE
    file_mode = SHARED_FILE_MODE if shared else PRIVATE_FILE_MODE

    current_fd = _open_data_root(absolute_root)
    try:
        for component in relative.parts[:-1]:
            try:
                next_fd = os.open(component, _directory_open_flags(), dir_fd=current_fd)
            except OSError:
                return False
            os.close(current_fd)
            current_fd = next_fd
            current_stat = os.fstat(current_fd)
            if not stat.S_ISDIR(current_stat.st_mode) or not _owned_by_process(current_stat):
                return False
            os.fchmod(current_fd, directory_mode)

        try:
            file_fd = os.open(relative.parts[-1], _file_open_flags(), dir_fd=current_fd)
        except OSError:
            return False
        try:
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1 or not _owned_by_process(file_stat):
                return False
            os.fchmod(file_fd, file_mode)
        finally:
            os.close(file_fd)
    finally:
        os.close(current_fd)
    return True
