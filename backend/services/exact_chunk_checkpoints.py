"""Durable PCM checkpoints for exact, singleton Qwen generations.

The exact-generation route binds every numerical input (including the runtime
revision and immutable voice reference) into ``exact_request_sha256``.  This
module uses that digest together with the logical chunk identity to persist the
post-trim float32 PCM.  It deliberately does not know about generation IDs, so
a caller can safely retry an interrupted exact request under a new ID.

Checkpoint files use a small, non-executable binary format instead of pickle or
``numpy.load``.  A canonical JSON header is followed by little-endian float32
PCM.  Both the contract fields and PCM digest are verified before reuse.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import logging
import os
import re
import stat
import struct
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .. import config
from ..utils.disk_reservations import DiskSpaceReservationError, reserve_disk_space

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_MAGIC = b"VBX-ECP1\n"
_SCHEMA_VERSION = 1
_HEADER_LENGTH = struct.Struct(">I")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CHECKPOINT_NAME_RE = re.compile(r"(?P<index>[0-9]{6})-(?P<text>[0-9a-f]{64})-(?P<seed>[0-9a-f]{8})\.vbc\Z")
_TEMP_CHECKPOINT_NAME_RE = re.compile(r"\.[0-9]{6}-[0-9a-f]{64}-[0-9a-f]{8}\.vbc\.tmp-[0-9a-f]{32}\Z")
_MAX_HEADER_BYTES = 4096
_MAX_PCM_BYTES = 256 * 1024 * 1024
_MAX_LOGICAL_INDEX = 999_999
_MIN_SAMPLE_RATE = 8_000
_MAX_SAMPLE_RATE = 192_000
_DEFAULT_STALE_AGE_SECONDS = 30 * 24 * 60 * 60
_DEFAULT_MAX_STORE_BYTES = 8 * 1024**3
_DEFAULT_MIN_FREE_BYTES = 1024**3
_CHECKPOINT_FILESYSTEM_OVERHEAD_BYTES = 64 * 1024
_MAX_ACCOUNTING_ENTRIES = 65_536
_MAX_REQUEST_FILES = 4096
_GC_TEMP_TABLE = "voicebox_exact_checkpoint_gc_candidates"
_CHECKPOINT_FILESYSTEM_LOCK = threading.RLock()


class InvalidCheckpointAudioError(ValueError):
    """Raised when generated PCM is unsafe or cannot be represented exactly."""


class CheckpointCapacityError(OSError):
    """Raised before a checkpoint would exhaust its bounded cache volume."""


class CheckpointGarbageCollectionError(RuntimeError):
    """Raised when checkpoint ownership cannot be established safely."""


@dataclass(frozen=True, slots=True)
class CheckpointGarbageCollectionReport:
    """Bounded result of one DB-aware checkpoint collection pass."""

    candidates: int
    removed: int
    preserved: int
    refused: int


@dataclass(frozen=True, slots=True)
class ExactChunkCheckpointKey:
    """Content identity of one logical chunk in one exact request."""

    exact_request_sha256: str
    logical_index: int
    text_sha256: str
    seed: int

    @classmethod
    def from_text(
        cls,
        *,
        exact_request_sha256: str,
        logical_index: int,
        text: str,
        seed: int,
    ) -> ExactChunkCheckpointKey:
        if not isinstance(text, str):
            raise TypeError("checkpoint text must be a string")
        return cls(
            exact_request_sha256=exact_request_sha256,
            logical_index=logical_index,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            seed=seed,
        )

    def __post_init__(self) -> None:
        if not _SHA256_RE.fullmatch(self.exact_request_sha256):
            raise ValueError("exact_request_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.logical_index, bool) or not isinstance(self.logical_index, int):
            raise TypeError("logical_index must be an integer")
        if not 0 <= self.logical_index <= _MAX_LOGICAL_INDEX:
            raise ValueError("logical_index is outside the supported range")
        if not _SHA256_RE.fullmatch(self.text_sha256):
            raise ValueError("text_sha256 must be a lowercase SHA-256 digest")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("checkpoint seed must be an integer")
        if not 0 <= self.seed <= (1 << 32) - 1:
            raise ValueError("checkpoint seed must fit in uint32")

    @property
    def filename(self) -> str:
        return f"{self.logical_index:06d}-{self.text_sha256}-{self.seed:08x}.vbc"


@dataclass(frozen=True, slots=True)
class ExactChunkCheckpoint:
    """Validated post-trim PCM loaded from a durable checkpoint."""

    audio: np.ndarray
    sample_rate: int


class ExactChunkCheckpointStore:
    """Atomic, fail-closed storage for exact logical-chunk PCM."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        max_store_bytes: int = _DEFAULT_MAX_STORE_BYTES,
        min_free_bytes: int = _DEFAULT_MIN_FREE_BYTES,
    ) -> None:
        self.root = Path(root) if root is not None else config.get_cache_dir() / "exact-chunk-checkpoints-v1"
        if isinstance(max_store_bytes, bool) or not isinstance(max_store_bytes, int) or max_store_bytes <= 0:
            raise ValueError("checkpoint cache byte limit must be a positive integer")
        if isinstance(min_free_bytes, bool) or not isinstance(min_free_bytes, int) or min_free_bytes < 0:
            raise ValueError("checkpoint free-space reserve must be a non-negative integer")
        self.max_store_bytes = max_store_bytes
        self.min_free_bytes = min_free_bytes

    def checkpoint_path(self, key: ExactChunkCheckpointKey) -> Path:
        """Return the deterministic path for diagnostics and focused tests."""
        return self.root / key.exact_request_sha256 / key.filename

    def load(self, key: ExactChunkCheckpointKey) -> ExactChunkCheckpoint | None:
        """Load a checkpoint only after validating its complete contract."""
        with _CHECKPOINT_FILESYSTEM_LOCK:
            return self._load_locked(key)

    def _load_locked(self, key: ExactChunkCheckpointKey) -> ExactChunkCheckpoint | None:
        if self._directory_state(self.root) is not True:
            return None
        path = self.checkpoint_path(key)
        if self._directory_state(path.parent) is not True:
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW

        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            return None
        except OSError:
            # A symlink, permission error, or other unexpected entry is never
            # accepted as cached model output.
            logger.warning("Ignoring unreadable exact chunk checkpoint: %s", path)
            return None

        try:
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                return None
            maximum_file_bytes = len(_MAGIC) + _HEADER_LENGTH.size + _MAX_HEADER_BYTES + _MAX_PCM_BYTES
            if file_stat.st_size <= len(_MAGIC) + _HEADER_LENGTH.size:
                raise ValueError("checkpoint is truncated")
            if file_stat.st_size > maximum_file_bytes:
                raise ValueError("checkpoint exceeds the bounded file size")

            with os.fdopen(fd, "rb", closefd=True) as handle:
                fd = -1
                if handle.read(len(_MAGIC)) != _MAGIC:
                    raise ValueError("checkpoint magic is invalid")
                encoded_header_length = handle.read(_HEADER_LENGTH.size)
                if len(encoded_header_length) != _HEADER_LENGTH.size:
                    raise ValueError("checkpoint header length is truncated")
                (header_length,) = _HEADER_LENGTH.unpack(encoded_header_length)
                if not 1 <= header_length <= _MAX_HEADER_BYTES:
                    raise ValueError("checkpoint header length is invalid")
                encoded_header = handle.read(header_length)
                if len(encoded_header) != header_length:
                    raise ValueError("checkpoint header is truncated")
                header = json.loads(encoded_header.decode("utf-8"))
                self._validate_header(header, key)

                pcm_bytes = int(header["pcm_bytes"])
                expected_file_size = len(_MAGIC) + _HEADER_LENGTH.size + header_length + pcm_bytes
                if file_stat.st_size != expected_file_size:
                    raise ValueError("checkpoint size does not match its header")
                raw_pcm = handle.read(pcm_bytes)
                if len(raw_pcm) != pcm_bytes:
                    raise ValueError("checkpoint PCM is truncated")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, KeyError):
            logger.warning("Ignoring corrupt exact chunk checkpoint: %s", path)
            self._discard_corrupt(path)
            return None
        finally:
            if fd >= 0:
                os.close(fd)

        if not hmac.compare_digest(
            hashlib.sha256(raw_pcm).hexdigest(),
            header["pcm_sha256"],
        ):
            logger.warning("Ignoring checksum-mismatched exact chunk checkpoint: %s", path)
            self._discard_corrupt(path)
            return None

        audio = np.frombuffer(raw_pcm, dtype="<f4").astype(np.float32, copy=True)
        if audio.ndim != 1 or len(audio) != header["sample_count"]:
            self._discard_corrupt(path)
            return None
        if not np.isfinite(audio).all():
            self._discard_corrupt(path)
            return None
        return ExactChunkCheckpoint(audio=audio, sample_rate=header["sample_rate"])

    def save(
        self,
        key: ExactChunkCheckpointKey,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Path:
        """Atomically persist one post-trim float32 logical chunk.

        The file and containing directory are fsynced before this method
        returns, so a later chunk is never started on the assumption that an
        earlier checkpoint merely reached the operating-system page cache.
        """
        with _CHECKPOINT_FILESYSTEM_LOCK:
            return self._save_locked(key, audio, sample_rate)

    def _save_locked(
        self,
        key: ExactChunkCheckpointKey,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Path:
        pcm = self._validated_pcm(audio, sample_rate)
        raw_pcm = memoryview(pcm).cast("B")
        pcm_sha256 = hashlib.sha256(raw_pcm).hexdigest()
        header = {
            "dtype": "<f4",
            "exact_request_sha256": key.exact_request_sha256,
            "logical_index": key.logical_index,
            "pcm_bytes": raw_pcm.nbytes,
            "pcm_sha256": pcm_sha256,
            "sample_count": len(pcm),
            "sample_rate": sample_rate,
            "schema_version": _SCHEMA_VERSION,
            "seed": key.seed,
            "text_sha256": key.text_sha256,
        }
        encoded_header = json.dumps(
            header,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        if len(encoded_header) > _MAX_HEADER_BYTES:
            raise ValueError("checkpoint header exceeds its bounded size")

        encoded_file_bytes = len(_MAGIC) + _HEADER_LENGTH.size + len(encoded_header) + raw_pcm.nbytes
        self._require_store_capacity(encoded_file_bytes)
        path = self.checkpoint_path(key)
        request_dir = path.parent
        volume_path = self.root
        while not volume_path.exists() and volume_path != volume_path.parent:
            volume_path = volume_path.parent
        try:
            reservation = reserve_disk_space(
                volume_path,
                encoded_file_bytes + _CHECKPOINT_FILESYSTEM_OVERHEAD_BYTES,
                min_free_bytes=self.min_free_bytes,
            )
        except DiskSpaceReservationError as exc:
            raise CheckpointCapacityError(
                "Insufficient free space for an exact chunk checkpoint while preserving the disk reserve"
            ) from exc

        temporary_path = request_dir / f".{path.name}.tmp-{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC

        fd = -1
        try:
            self._ensure_directory_is_durable(self.root)
            self._ensure_directory_is_durable(request_dir)
            fd = os.open(temporary_path, flags, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as handle:
                fd = -1
                handle.write(_MAGIC)
                handle.write(_HEADER_LENGTH.pack(len(encoded_header)))
                handle.write(encoded_header)
                handle.write(raw_pcm)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            self._fsync_directory(request_dir)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                temporary_path.unlink()
            raise
        finally:
            reservation.release()
        return path

    def _require_store_capacity(self, pending_bytes: int) -> None:
        """Fail before writing when the bounded checkpoint-cache quota is exhausted."""
        stored_bytes = self._accounted_storage_bytes()
        if stored_bytes + pending_bytes > self.max_store_bytes:
            raise CheckpointCapacityError(
                "Exact chunk checkpoint cache limit reached; resume or remove older work before retrying"
            )

    def _accounted_storage_bytes(self) -> int:
        """Boundedly count every entry below the private checkpoint cache."""
        state = self._directory_state(self.root)
        if state is None:
            return 0
        if state is not True:
            raise CheckpointCapacityError("Exact chunk checkpoint cache is not a real directory")

        total = 0
        scanned = 0
        try:
            request_entries = list(self.root.iterdir())
        except OSError as exc:
            raise CheckpointCapacityError("Could not inspect exact chunk checkpoint cache") from exc
        if len(request_entries) > _MAX_ACCOUNTING_ENTRIES:
            raise CheckpointCapacityError("Exact chunk checkpoint cache contains too many entries")

        for request_dir in request_entries:
            scanned += 1
            try:
                request_stat = request_dir.lstat()
            except OSError as exc:
                raise CheckpointCapacityError("Could not inspect exact chunk checkpoint entry") from exc
            if (
                not _SHA256_RE.fullmatch(request_dir.name)
                or not stat.S_ISDIR(request_stat.st_mode)
                or stat.S_ISLNK(request_stat.st_mode)
            ):
                raise CheckpointCapacityError("Exact chunk checkpoint cache contains an unexpected entry")
            try:
                files = list(request_dir.iterdir())
            except OSError as exc:
                raise CheckpointCapacityError("Could not inspect exact chunk checkpoint request") from exc
            scanned += len(files)
            if scanned > _MAX_ACCOUNTING_ENTRIES:
                raise CheckpointCapacityError("Exact chunk checkpoint cache contains too many entries")
            for entry in files:
                try:
                    entry_stat = entry.lstat()
                except OSError as exc:
                    raise CheckpointCapacityError("Could not inspect exact chunk checkpoint file") from exc
                if not stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                    raise CheckpointCapacityError("Exact chunk checkpoint cache contains an unsafe entry")
                total += entry_stat.st_size
                if total > self.max_store_bytes:
                    raise CheckpointCapacityError("Exact chunk checkpoint cache already exceeds its byte limit")
        return total

    def discard(self, key: ExactChunkCheckpointKey) -> None:
        """Discard one known-bad checkpoint without traversing other entries."""
        with _CHECKPOINT_FILESYSTEM_LOCK:
            self._discard_corrupt(self.checkpoint_path(key))

    def request_hashes(
        self,
        *,
        max_entries: int = _MAX_ACCOUNTING_ENTRIES,
    ) -> tuple[str, ...]:
        """Return every safely shaped request directory in one bounded scan.

        Startup collection uses the complete result rather than an arbitrary
        oldest-N window. Otherwise an 8 GiB cache made only of completed jobs
        could remain permanently full across restarts. Any unexpected root or
        child entry aborts the scan before a final checkpoint is removed.
        """
        with _CHECKPOINT_FILESYSTEM_LOCK:
            return self._request_hashes_locked(max_entries=max_entries)

    def _request_hashes_locked(
        self,
        *,
        max_entries: int,
    ) -> tuple[str, ...]:
        if max_entries <= 0:
            raise ValueError("checkpoint scan bound must be positive")
        root_state = self._directory_state(self.root)
        if root_state is None:
            return ()
        if root_state is not True:
            raise CheckpointGarbageCollectionError("Exact chunk checkpoint cache is not a real directory")

        try:
            request_entries = list(self.root.iterdir())
        except OSError as exc:
            raise CheckpointGarbageCollectionError("Could not inspect exact chunk checkpoint cache") from exc
        scanned = len(request_entries)
        if scanned > max_entries:
            raise CheckpointGarbageCollectionError("Exact chunk checkpoint cache exceeds the bounded GC scan")

        request_hashes: list[str] = []
        for request_dir in request_entries:
            try:
                request_stat = request_dir.lstat()
            except OSError as exc:
                raise CheckpointGarbageCollectionError("Could not inspect exact chunk checkpoint entry") from exc
            if (
                not _SHA256_RE.fullmatch(request_dir.name)
                or not stat.S_ISDIR(request_stat.st_mode)
                or stat.S_ISLNK(request_stat.st_mode)
            ):
                raise CheckpointGarbageCollectionError("Exact chunk checkpoint cache contains an unexpected entry")
            try:
                children = list(request_dir.iterdir())
            except OSError as exc:
                raise CheckpointGarbageCollectionError("Could not inspect exact chunk checkpoint request") from exc
            scanned += len(children)
            if scanned > max_entries:
                raise CheckpointGarbageCollectionError("Exact chunk checkpoint cache exceeds the bounded GC scan")
            if len(children) > _MAX_REQUEST_FILES:
                raise CheckpointGarbageCollectionError("Exact chunk checkpoint request exceeds the bounded GC scan")
            for child in children:
                if not self._recognized_checkpoint_entry(child):
                    raise CheckpointGarbageCollectionError("Exact chunk checkpoint cache contains an unsafe entry")
            request_hashes.append(request_dir.name)
        return tuple(sorted(request_hashes))

    def remove_request(
        self,
        exact_request_sha256: str,
        *,
        max_entries: int = _MAX_REQUEST_FILES,
    ) -> bool:
        """Remove only recognized files for one completed exact request.

        Returns ``True`` when the request directory is absent or was removed.
        Unexpected nested directories are never traversed.
        """
        with _CHECKPOINT_FILESYSTEM_LOCK:
            return self._remove_request_locked(
                exact_request_sha256,
                max_entries=max_entries,
            )

    def _remove_request_locked(
        self,
        exact_request_sha256: str,
        *,
        max_entries: int,
    ) -> bool:
        if not _SHA256_RE.fullmatch(exact_request_sha256):
            raise ValueError("exact_request_sha256 must be a lowercase SHA-256 digest")
        if self._supports_anchored_removal():
            return self._remove_request_anchored(
                exact_request_sha256,
                max_entries=max_entries,
            )
        return self._remove_request_path_fallback(
            exact_request_sha256,
            max_entries=max_entries,
        )

    def _remove_request_anchored(
        self,
        exact_request_sha256: str,
        *,
        max_entries: int,
    ) -> bool:
        """Remove via directory descriptors so parent swaps are never followed."""
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            root_fd = os.open(self.root, flags)
        except FileNotFoundError:
            return True
        except OSError:
            return False

        request_fd = -1
        try:
            if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
                return False
            try:
                request_fd = os.open(
                    exact_request_sha256,
                    flags,
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                return True
            except OSError:
                return False
            request_stat = os.fstat(request_fd)
            if not stat.S_ISDIR(request_stat.st_mode):
                return False
            try:
                names = os.listdir(request_fd)
            except OSError:
                return False
            if len(names) > max_entries:
                logger.warning(
                    "Refusing unbounded exact checkpoint cleanup with %d entries: %s",
                    len(names),
                    self.root / exact_request_sha256,
                )
                return False

            for name in names:
                if not self._recognized_checkpoint_name(name):
                    return False
                try:
                    entry_stat = os.stat(
                        name,
                        dir_fd=request_fd,
                        follow_symlinks=False,
                    )
                except OSError:
                    return False
                if not self._safe_checkpoint_stat(entry_stat):
                    return False

            for name in names:
                try:
                    os.unlink(name, dir_fd=request_fd)
                except FileNotFoundError:
                    pass
                except OSError:
                    return False

            try:
                current_stat = os.stat(
                    exact_request_sha256,
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
            except OSError:
                return False
            if (current_stat.st_dev, current_stat.st_ino) != (
                request_stat.st_dev,
                request_stat.st_ino,
            ):
                return False
            try:
                os.rmdir(exact_request_sha256, dir_fd=root_fd)
            except FileNotFoundError:
                return True
            except OSError:
                return False
            os.fsync(root_fd)
            return True
        finally:
            if request_fd >= 0:
                os.close(request_fd)
            os.close(root_fd)

    def _remove_request_path_fallback(
        self,
        exact_request_sha256: str,
        *,
        max_entries: int,
    ) -> bool:
        root_state = self._directory_state(self.root)
        if root_state is None:
            return True
        if root_state is not True:
            return False
        request_dir = self.root / exact_request_sha256
        try:
            directory_stat = request_dir.lstat()
        except FileNotFoundError:
            return True
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
            return False

        try:
            entries = list(request_dir.iterdir())
        except OSError:
            return False
        if len(entries) > max_entries:
            logger.warning(
                "Refusing unbounded exact checkpoint cleanup with %d entries: %s",
                len(entries),
                request_dir,
            )
            return False

        # Validate the complete directory before unlinking its first entry.
        # This makes an unexpected file, symlink, hard link, or device a
        # fail-closed refusal rather than a partially destructive cleanup.
        if not all(self._recognized_checkpoint_entry(entry) for entry in entries):
            logger.warning(
                "Refusing unsafe exact checkpoint cleanup: %s",
                request_dir,
            )
            return False

        for entry in entries:
            try:
                entry.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                return False

        try:
            request_dir.rmdir()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        self._fsync_directory(self.root)
        return True

    @staticmethod
    def _recognized_checkpoint_entry(entry: Path) -> bool:
        if not ExactChunkCheckpointStore._recognized_checkpoint_name(entry.name):
            return False
        try:
            entry_stat = entry.lstat()
        except OSError:
            return False
        return ExactChunkCheckpointStore._safe_checkpoint_stat(entry_stat)

    @staticmethod
    def _recognized_checkpoint_name(name: str) -> bool:
        return bool(_CHECKPOINT_NAME_RE.fullmatch(name)) or bool(_TEMP_CHECKPOINT_NAME_RE.fullmatch(name))

    @staticmethod
    def _safe_checkpoint_stat(entry_stat: os.stat_result) -> bool:
        return stat.S_ISREG(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode) and entry_stat.st_nlink == 1

    @staticmethod
    def _supports_anchored_removal() -> bool:
        return (
            hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
            and os.open in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.unlink in os.supports_dir_fd
            and os.rmdir in os.supports_dir_fd
            and os.listdir in os.supports_fd
        )

    def prune_stale(
        self,
        *,
        now: float | None = None,
        max_age_seconds: int = _DEFAULT_STALE_AGE_SECONDS,
        max_scan_entries: int = 4096,
        max_request_dirs: int = 64,
    ) -> int:
        """Boundedly remove abandoned atomic-write temporary files.

        Valid ``.vbc`` checkpoints never expire based on age: an audiobook may
        be intentionally paused for months.  Deleting those needs DB-aware
        proof that no resumable request references the exact contract.
        """
        if max_age_seconds <= 0 or max_scan_entries <= 0 or max_request_dirs <= 0:
            raise ValueError("stale cleanup bounds must be positive")
        cutoff = (time.time() if now is None else now) - max_age_seconds
        return self._prune_temporary_files(
            cutoff=cutoff,
            max_scan_entries=max_scan_entries,
            max_request_dirs=max_request_dirs,
        )

    def prune_abandoned_temporary_files(
        self,
        *,
        max_scan_entries: int = 4096,
        max_request_dirs: int = 64,
    ) -> int:
        """Remove bounded atomic-write temps before generation workers start.

        The caller must own the data-root lifetime lock. Final ``.vbc`` files
        are never removed by this operation.
        """
        if max_scan_entries <= 0 or max_request_dirs <= 0:
            raise ValueError("cleanup bounds must be positive")
        return self._prune_temporary_files(
            cutoff=float("inf"),
            max_scan_entries=max_scan_entries,
            max_request_dirs=max_request_dirs,
        )

    def _prune_temporary_files(
        self,
        *,
        cutoff: float,
        max_scan_entries: int,
        max_request_dirs: int,
    ) -> int:
        """Boundedly remove recognized temporary checkpoint files."""
        with _CHECKPOINT_FILESYSTEM_LOCK:
            return self._prune_temporary_files_locked(
                cutoff=cutoff,
                max_scan_entries=max_scan_entries,
                max_request_dirs=max_request_dirs,
            )

    def _prune_temporary_files_locked(
        self,
        *,
        cutoff: float,
        max_scan_entries: int,
        max_request_dirs: int,
    ) -> int:
        if self._directory_state(self.root) is not True:
            return 0
        try:
            iterator = os.scandir(self.root)
        except FileNotFoundError:
            return 0

        candidates: list[tuple[float, str]] = []
        with iterator:
            for index, entry in enumerate(iterator):
                if index >= max_scan_entries:
                    break
                if not _SHA256_RE.fullmatch(entry.name):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        candidates.append((entry.stat(follow_symlinks=False).st_mtime, entry.name))
                except OSError:
                    continue

        removed_files = 0
        for _, request_sha256 in sorted(candidates)[:max_request_dirs]:
            request_dir = self.root / request_sha256
            if self._directory_state(request_dir) is not True:
                continue
            try:
                entries = list(request_dir.iterdir())
            except OSError:
                continue
            if len(entries) > max_scan_entries:
                continue
            for entry in entries:
                if not _TEMP_CHECKPOINT_NAME_RE.fullmatch(entry.name):
                    continue
                try:
                    entry_stat = entry.lstat()
                    if entry_stat.st_mtime >= cutoff:
                        continue
                    if stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
                        entry.unlink()
                        removed_files += 1
                except OSError:
                    continue
            with contextlib.suppress(OSError):
                request_dir.rmdir()
        return removed_files

    @staticmethod
    def _validated_pcm(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise InvalidCheckpointAudioError("sample rate must be an integer")
        if not _MIN_SAMPLE_RATE <= sample_rate <= _MAX_SAMPLE_RATE:
            raise InvalidCheckpointAudioError("sample rate is outside the supported range")
        candidate = np.asarray(audio)
        if candidate.ndim != 1:
            raise InvalidCheckpointAudioError("checkpoint PCM must be one-dimensional")
        if candidate.size == 0:
            raise InvalidCheckpointAudioError("checkpoint PCM cannot be empty")
        pcm = np.ascontiguousarray(candidate, dtype="<f4")
        if pcm.nbytes > _MAX_PCM_BYTES:
            raise InvalidCheckpointAudioError("checkpoint PCM exceeds the bounded size")
        if not np.isfinite(pcm).all():
            raise InvalidCheckpointAudioError("checkpoint PCM must contain only finite samples")
        return pcm

    @staticmethod
    def _validate_header(header: object, key: ExactChunkCheckpointKey) -> None:
        expected_fields = {
            "dtype",
            "exact_request_sha256",
            "logical_index",
            "pcm_bytes",
            "pcm_sha256",
            "sample_count",
            "sample_rate",
            "schema_version",
            "seed",
            "text_sha256",
        }
        if not isinstance(header, dict) or set(header) != expected_fields:
            raise ValueError("checkpoint header schema is invalid")
        if header["schema_version"] != _SCHEMA_VERSION or header["dtype"] != "<f4":
            raise ValueError("checkpoint schema version or dtype is unsupported")
        if header["exact_request_sha256"] != key.exact_request_sha256:
            raise ValueError("checkpoint exact request identity does not match")
        if header["logical_index"] != key.logical_index:
            raise ValueError("checkpoint logical index does not match")
        if header["text_sha256"] != key.text_sha256:
            raise ValueError("checkpoint text identity does not match")
        if header["seed"] != key.seed:
            raise ValueError("checkpoint seed does not match")
        if (
            isinstance(header["sample_rate"], bool)
            or not isinstance(header["sample_rate"], int)
            or not _MIN_SAMPLE_RATE <= header["sample_rate"] <= _MAX_SAMPLE_RATE
        ):
            raise ValueError("checkpoint sample rate is invalid")
        if (
            isinstance(header["sample_count"], bool)
            or not isinstance(header["sample_count"], int)
            or header["sample_count"] <= 0
        ):
            raise ValueError("checkpoint sample count is invalid")
        if (
            isinstance(header["pcm_bytes"], bool)
            or not isinstance(header["pcm_bytes"], int)
            or header["pcm_bytes"] != header["sample_count"] * 4
            or not 0 < header["pcm_bytes"] <= _MAX_PCM_BYTES
        ):
            raise ValueError("checkpoint PCM size is invalid")
        if not isinstance(header["pcm_sha256"], str) or not _SHA256_RE.fullmatch(header["pcm_sha256"]):
            raise ValueError("checkpoint PCM digest is invalid")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            # Windows does not expose POSIX directory fsync. os.replace still
            # gives atomic visibility there; the production MLX path is macOS.
            return
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @classmethod
    def _ensure_directory_is_durable(cls, path: Path) -> None:
        """Create managed directories and persist every new parent entry."""
        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        for directory in reversed(missing):
            directory.mkdir(exist_ok=True, mode=0o700)
            cls._fsync_directory(directory)
            cls._fsync_directory(directory.parent)

        entry_stat = path.lstat()
        if not stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
            raise OSError(f"Exact checkpoint path is not a real directory: {path}")

    @staticmethod
    def _directory_state(path: Path) -> bool | None:
        """Return True only for a present, non-symlink directory."""
        try:
            entry_stat = path.lstat()
        except FileNotFoundError:
            return None
        except OSError:
            return False
        return stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode)

    @staticmethod
    def _discard_corrupt(path: Path) -> None:
        try:
            entry_stat = path.lstat()
            if stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
                path.unlink()
        except OSError:
            pass


def garbage_collect_exact_chunk_checkpoints(
    db: Session,
    *,
    request_hashes: Iterable[str] | None = None,
    store: ExactChunkCheckpointStore | None = None,
) -> CheckpointGarbageCollectionReport:
    """Remove checkpoints proven to have no resumable database owner.

    A fresh SQLite ``BEGIN IMMEDIATE`` transaction spans both the owner query
    and filesystem removal. A concurrently submitted exact attempt therefore
    either commits first and preserves the shared request hash, or waits until
    cleanup has completed and subsequently writes into a new request directory.

    Rows whose status is anything other than ``completed`` are resumable
    owners, including ``failed`` rows deliberately kept for a later retry.
    Historical NULL statuses use the application's existing completed
    semantics and therefore do not retain cache data.
    """
    checkpoint_store = store or ExactChunkCheckpointStore()
    if request_hashes is None:
        candidates = checkpoint_store.request_hashes()
    else:
        candidate_set = set(request_hashes)
        if len(candidate_set) > _MAX_ACCOUNTING_ENTRIES:
            raise CheckpointGarbageCollectionError("Exact chunk checkpoint GC has too many candidates")
        if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in candidate_set):
            raise CheckpointGarbageCollectionError("Exact chunk checkpoint GC received an invalid request hash")
        candidates = tuple(sorted(candidate_set))

    if not candidates:
        return CheckpointGarbageCollectionReport(
            candidates=0,
            removed=0,
            preserved=0,
            refused=0,
        )

    bind = db.get_bind()
    engine = getattr(bind, "engine", bind)
    url = getattr(engine, "url", None)
    if url is None or url.get_backend_name() != "sqlite" or url.database in {None, "", ":memory:"}:
        raise CheckpointGarbageCollectionError(
            "Exact checkpoint GC requires an independent file-backed SQLite transaction"
        )

    connection = engine.raw_connection()
    cursor = None
    removed = 0
    preserved = 0
    refused = 0
    try:
        cursor = connection.cursor()
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(f"DROP TABLE IF EXISTS temp.{_GC_TEMP_TABLE}")
        cursor.execute(f"CREATE TEMP TABLE {_GC_TEMP_TABLE} (exact_request_sha256 TEXT PRIMARY KEY) WITHOUT ROWID")
        cursor.executemany(
            f"INSERT INTO {_GC_TEMP_TABLE} (exact_request_sha256) VALUES (?)",
            ((candidate,) for candidate in candidates),
        )
        owner_rows = cursor.execute(
            f"SELECT DISTINCT candidate.exact_request_sha256 "
            f"FROM {_GC_TEMP_TABLE} AS candidate "
            "JOIN generations AS generation "
            "ON generation.exact_request_sha256 = candidate.exact_request_sha256 "
            "WHERE COALESCE(generation.status, 'completed') <> 'completed'"
        ).fetchall()
        owned = {str(row[0]) for row in owner_rows}

        with _CHECKPOINT_FILESYSTEM_LOCK:
            for candidate in candidates:
                if candidate in owned:
                    preserved += 1
                    continue
                if checkpoint_store.remove_request(candidate):
                    removed += 1
                else:
                    refused += 1

        cursor.execute(f"DROP TABLE {_GC_TEMP_TABLE}")
        connection.commit()
    except BaseException as exc:
        with contextlib.suppress(Exception):
            connection.rollback()
        if isinstance(exc, CheckpointGarbageCollectionError):
            raise
        raise CheckpointGarbageCollectionError("Could not establish durable exact-checkpoint ownership") from exc
    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    return CheckpointGarbageCollectionReport(
        candidates=len(candidates),
        removed=removed,
        preserved=preserved,
        refused=refused,
    )


class ExactChunkCheckpointSession:
    """Request-scoped adapter used only by exact singleton generation."""

    def __init__(
        self,
        exact_request_sha256: str,
        *,
        expected_sample_rate: int = 24_000,
        store: ExactChunkCheckpointStore | None = None,
    ) -> None:
        if not _SHA256_RE.fullmatch(exact_request_sha256):
            raise ValueError("exact_request_sha256 must be a lowercase SHA-256 digest")
        if (
            isinstance(expected_sample_rate, bool)
            or not isinstance(expected_sample_rate, int)
            or not _MIN_SAMPLE_RATE <= expected_sample_rate <= _MAX_SAMPLE_RATE
        ):
            raise ValueError("expected checkpoint sample rate is invalid")
        self.exact_request_sha256 = exact_request_sha256
        self.expected_sample_rate = expected_sample_rate
        self.store = store or ExactChunkCheckpointStore()

    def load(
        self,
        *,
        logical_index: int,
        text: str,
        seed: int,
    ) -> ExactChunkCheckpoint | None:
        key = self._key(logical_index=logical_index, text=text, seed=seed)
        checkpoint = self.store.load(key)
        if checkpoint is None:
            return None
        if checkpoint.sample_rate != self.expected_sample_rate:
            logger.warning(
                "Ignoring exact chunk checkpoint with unexpected sample rate: %s",
                self.store.checkpoint_path(key),
            )
            self.store.discard(key)
            return None
        return checkpoint

    def save(
        self,
        *,
        logical_index: int,
        text: str,
        seed: int,
        audio: np.ndarray,
        sample_rate: int,
    ) -> Path:
        if sample_rate != self.expected_sample_rate:
            raise InvalidCheckpointAudioError("exact Qwen chunk returned an unexpected sample rate")
        return self.store.save(
            self._key(logical_index=logical_index, text=text, seed=seed),
            audio,
            sample_rate,
        )

    def complete(self, db: Session) -> None:
        """Best-effort DB-aware cleanup after completed status is durable."""
        try:
            report = garbage_collect_exact_chunk_checkpoints(
                db,
                request_hashes=(self.exact_request_sha256,),
                store=self.store,
            )
        except Exception:
            logger.exception("Failed to remove completed exact chunk checkpoints")
            return
        if report.refused:
            logger.warning(
                "Refused unsafe completed exact chunk checkpoint cleanup: %s",
                self.exact_request_sha256,
            )

    def _key(self, *, logical_index: int, text: str, seed: int) -> ExactChunkCheckpointKey:
        return ExactChunkCheckpointKey.from_text(
            exact_request_sha256=self.exact_request_sha256,
            logical_index=logical_index,
            text=text,
            seed=seed,
        )
