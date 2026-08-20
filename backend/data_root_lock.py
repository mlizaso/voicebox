"""Cross-process lifetime lock for one writable Voicebox data root."""

from __future__ import annotations

import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from . import config


class DataRootInUseError(RuntimeError):
    """Raised when another Voicebox process already owns the data directory."""


@dataclass
class DataRootLock:
    """An acquired lock held until application shutdown."""

    fd: int
    windows: bool
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        try:
            if self.windows:
                import msvcrt

                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.released = True


def _lock_path() -> Path:
    return config.get_data_dir() / ".voicebox.lock"


def acquire_data_root_lock() -> DataRootLock:
    """Acquire the configured root exclusively, without deleting the lock inode."""
    path = _lock_path()
    if path.parent != config.get_data_dir():
        raise RuntimeError("Voicebox data lock escaped its managed root")
    try:
        entry_stat = path.lstat()
    except FileNotFoundError:
        entry_stat = None
    if entry_stat is not None and (not stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode)):
        raise RuntimeError("Voicebox data lock is not a regular file")

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    windows = sys.platform == "win32"
    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
            raise RuntimeError("Voicebox data lock is not a private regular file")
        if hasattr(os, "fchmod") and hasattr(os, "geteuid") and opened_stat.st_uid == os.geteuid():
            os.fchmod(fd, 0o600)
        if opened_stat.st_size == 0:
            os.write(fd, b"\0")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            if windows:
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DataRootInUseError("Another Voicebox backend already owns this data directory") from exc

        payload = f"pid={os.getpid()}\n".encode()
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, payload)
        os.fsync(fd)
        return DataRootLock(fd=fd, windows=windows)
    except BaseException:
        os.close(fd)
        raise
