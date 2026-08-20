"""Bounded, private voice-prompt caching utilities."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import stat
import sys
import threading
import uuid
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

import torch

from .. import config
from .disk_reservations import DiskSpaceReservationError, reserve_disk_space

logger = logging.getLogger(__name__)

VOICE_PROMPT_CACHE_ROOT_NAME = "voice-prompts-v1"
VOICE_PROMPT_MEMORY_MAX_ENTRIES = 16
VOICE_PROMPT_MEMORY_MAX_BYTES = 256 * 1024 * 1024
VOICE_PROMPT_DISK_MAX_ENTRIES = 128
VOICE_PROMPT_DISK_MAX_BYTES = 1024 * 1024 * 1024
VOICE_PROMPT_MAX_FILE_BYTES = 256 * 1024 * 1024
VOICE_PROMPT_MIN_FREE_BYTES = 1024 * 1024 * 1024
VOICE_PROMPT_IO_CHUNK_BYTES = 1024 * 1024

_CACHE_KEY_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_cache_lock = threading.RLock()
_memory_cache: OrderedDict[str, tuple[object, int]] = OrderedDict()
_memory_cache_bytes = 0


def _get_cache_dir() -> Path:
    """Return the shared Voicebox cache directory."""
    return config.get_cache_dir()


def _voice_prompt_cache_dir() -> Path:
    root = _get_cache_dir() / VOICE_PROMPT_CACHE_ROOT_NAME
    root.mkdir(mode=0o700, exist_ok=True)
    entry_stat = root.lstat()
    is_junction = getattr(root, "is_junction", None)
    if (
        not stat.S_ISDIR(entry_stat.st_mode)
        or stat.S_ISLNK(entry_stat.st_mode)
        or (is_junction is not None and is_junction())
    ):
        raise RuntimeError("Voice-prompt cache root is not a real directory")
    if os.name == "posix":
        os.chmod(root, 0o700, follow_symlinks=False)
    return root


def _validated_cache_key(cache_key: str) -> str:
    if not isinstance(cache_key, str) or not _CACHE_KEY_RE.fullmatch(cache_key):
        raise ValueError("Voice-prompt cache key is invalid")
    return cache_key


def get_cache_key(audio_path: str, reference_text: str) -> str:
    """Generate a collision-resistant key without buffering the WAV in RAM."""
    digest = hashlib.sha256()
    with open(audio_path, "rb") as audio_file:
        while chunk := audio_file.read(VOICE_PROMPT_IO_CHUNK_BYTES):
            digest.update(chunk)
    encoded_text = reference_text.encode("utf-8")
    digest.update(len(encoded_text).to_bytes(8, "big"))
    digest.update(encoded_text)
    return digest.hexdigest()


def _estimated_object_bytes(value: object, seen: set[int] | None = None) -> int:
    """Conservatively estimate retained prompt memory, including tensors."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        return max(sys.getsizeof(value), value.numel() * value.element_size())
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            _estimated_object_bytes(key, seen) + _estimated_object_bytes(item, seen) for key, item in value.items()
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return sys.getsizeof(value) + sum(_estimated_object_bytes(item, seen) for item in value)
    if isinstance(value, str):
        return sys.getsizeof(value) + len(value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return sys.getsizeof(value) + len(value)
    return sys.getsizeof(value)


def _remember_locked(cache_key: str, prompt: object) -> None:
    global _memory_cache_bytes
    old = _memory_cache.pop(cache_key, None)
    if old is not None:
        _memory_cache_bytes -= old[1]
    size = _estimated_object_bytes(prompt)
    if size > VOICE_PROMPT_MEMORY_MAX_BYTES:
        return
    while _memory_cache and (
        len(_memory_cache) >= VOICE_PROMPT_MEMORY_MAX_ENTRIES
        or _memory_cache_bytes + size > VOICE_PROMPT_MEMORY_MAX_BYTES
    ):
        _old_key, (_old_prompt, old_size) = _memory_cache.popitem(last=False)
        _memory_cache_bytes -= old_size
    _memory_cache[cache_key] = (prompt, size)
    _memory_cache_bytes += size


def _forget_memory_locked() -> None:
    global _memory_cache_bytes
    _memory_cache.clear()
    _memory_cache_bytes = 0


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_cache_leaf(candidate: Path) -> bool:
    """Unlink one non-directory cache leaf without following it."""
    try:
        entry_stat = candidate.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISDIR(entry_stat.st_mode):
        logger.warning("Refused unexpected directory in the voice-prompt cache")
        return False
    try:
        candidate.unlink()
    except FileNotFoundError:
        return False
    return True


def _prompt_entries_locked(
    root: Path,
    *,
    preserved_temporary: Path | None = None,
) -> tuple[list[tuple[int, Path, int]], int]:
    entries: list[tuple[int, Path, int]] = []
    removed = 0
    for candidate in root.iterdir():
        try:
            entry_stat = candidate.lstat()
        except FileNotFoundError:
            continue
        if candidate.name.startswith(".tmp-"):
            if preserved_temporary is not None and candidate == preserved_temporary:
                continue
            removed += int(_unlink_cache_leaf(candidate))
            continue
        if not candidate.name.endswith(".prompt"):
            continue
        key = candidate.name[: -len(".prompt")]
        if not _CACHE_KEY_RE.fullmatch(key) or not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
            removed += int(_unlink_cache_leaf(candidate))
            continue
        entries.append((entry_stat.st_mtime_ns, candidate, entry_stat.st_size))
    return entries, removed


def _prune_disk_locked() -> int:
    """Bring finalized entries back under the persistent cache bounds."""
    root = _voice_prompt_cache_dir()
    entries, removed = _prompt_entries_locked(root)
    entries.sort(key=lambda item: (item[0], item[1].name))
    total_bytes = sum(item[2] for item in entries)
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError as exc:
        raise RuntimeError("Could not determine voice-prompt cache capacity") from exc
    while entries and (
        len(entries) > VOICE_PROMPT_DISK_MAX_ENTRIES
        or total_bytes > VOICE_PROMPT_DISK_MAX_BYTES
        or free_bytes < VOICE_PROMPT_MIN_FREE_BYTES
    ):
        _mtime, candidate, size = entries.pop(0)
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        else:
            total_bytes -= size
            free_bytes += size
            removed += 1
    if (
        len(entries) > VOICE_PROMPT_DISK_MAX_ENTRIES
        or total_bytes > VOICE_PROMPT_DISK_MAX_BYTES
        or free_bytes < VOICE_PROMPT_MIN_FREE_BYTES
    ):
        raise RuntimeError("Voice-prompt cache capacity is unavailable")
    if removed:
        _fsync_directory(root)
    return removed


def _reserve_entry_capacity_locked(
    *,
    target_name: str,
    new_size: int,
    temporary_path: Path | None = None,
    write_reservation: int = 0,
) -> None:
    """Evict LRU entries until one replacement fits every cache bound.

    Before serialization, ``write_reservation`` models the temporary file's
    peak disk use. After serialization, ``temporary_path`` names the already
    allocated inode, so replacing the old target will recover its size.
    """
    root = _voice_prompt_cache_dir()
    entries, cleaned = _prompt_entries_locked(root, preserved_temporary=temporary_path)
    if cleaned:
        _fsync_directory(root)
    target_size = 0
    other_entries: list[tuple[int, Path, int]] = []
    for entry in entries:
        if entry[1].name == target_name:
            target_size = entry[2]
        else:
            other_entries.append(entry)
    other_entries.sort(key=lambda item: (item[0], item[1].name))

    total_bytes = sum(item[2] for item in other_entries) + new_size
    total_entries = len(other_entries) + 1
    try:
        free_bytes = shutil.disk_usage(root).free
    except OSError as exc:
        raise RuntimeError("Could not determine voice-prompt cache capacity") from exc
    projected_free = free_bytes + target_size if temporary_path is not None else free_bytes - write_reservation
    removed = False
    while other_entries and (
        total_entries > VOICE_PROMPT_DISK_MAX_ENTRIES
        or total_bytes > VOICE_PROMPT_DISK_MAX_BYTES
        or projected_free < VOICE_PROMPT_MIN_FREE_BYTES
    ):
        _mtime, candidate, size = other_entries.pop(0)
        if _unlink_cache_leaf(candidate):
            total_entries -= 1
            total_bytes -= size
            projected_free += size
            removed = True
    if (
        total_entries > VOICE_PROMPT_DISK_MAX_ENTRIES
        or total_bytes > VOICE_PROMPT_DISK_MAX_BYTES
        or projected_free < VOICE_PROMPT_MIN_FREE_BYTES
    ):
        raise RuntimeError("Voice-prompt cache capacity is unavailable")
    if removed:
        _fsync_directory(root)


def prune_voice_prompt_cache() -> int:
    """Reclaim unsafe, stale, and over-quota prompt files at startup."""
    with _cache_lock:
        removed = _prune_disk_locked()
        legacy_root = _get_cache_dir()
        for candidate in legacy_root.iterdir():
            if not candidate.name.endswith(".prompt"):
                continue
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue
            removed += 1
        if removed:
            _fsync_directory(legacy_root)
        return removed


def get_cached_voice_prompt(cache_key: str) -> object | None:
    """Return a cached voice prompt and update its bounded LRU position."""
    cache_key = _validated_cache_key(cache_key)
    with _cache_lock:
        memory_entry = _memory_cache.pop(cache_key, None)
        if memory_entry is not None:
            _memory_cache[cache_key] = memory_entry
            return memory_entry[0]

        cache_file = _voice_prompt_cache_dir() / f"{cache_key}.prompt"
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(cache_file, flags)
        except FileNotFoundError:
            return None
        except OSError:
            logger.warning("Refused unsafe voice-prompt cache entry")
            return None
        try:
            entry_stat = os.fstat(descriptor)
            if (
                not stat.S_ISREG(entry_stat.st_mode)
                or entry_stat.st_nlink != 1
                or not 0 < entry_stat.st_size <= VOICE_PROMPT_MAX_FILE_BYTES
            ):
                raise ValueError("Voice-prompt cache entry has an unsafe shape")
            with os.fdopen(descriptor, "rb") as cache_stream:
                descriptor = -1
                prompt = torch.load(cache_stream, weights_only=True)
        except Exception:
            logger.warning("Discarding an unreadable voice-prompt cache entry", exc_info=True)
            with suppress(OSError):
                cache_file.unlink()
            return None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        with suppress(OSError):
            os.utime(cache_file, None, follow_symlinks=False)
        _remember_locked(cache_key, prompt)
        return prompt


def cache_voice_prompt(cache_key: str, voice_prompt: object) -> None:
    """Cache a prompt with bounded memory and atomic, quota-bound disk use."""
    cache_key = _validated_cache_key(cache_key)
    estimated_bytes = _estimated_object_bytes(voice_prompt) + 1024 * 1024
    with _cache_lock:
        _remember_locked(cache_key, voice_prompt)
        if estimated_bytes > VOICE_PROMPT_MAX_FILE_BYTES:
            logger.warning("Skipping an oversized disk voice-prompt cache entry")
            return
        root = _voice_prompt_cache_dir()
        final_path = root / f"{cache_key}.prompt"
        try:
            _reserve_entry_capacity_locked(
                target_name=final_path.name,
                new_size=estimated_bytes,
                write_reservation=VOICE_PROMPT_MAX_FILE_BYTES,
            )
        except RuntimeError:
            logger.warning("Skipping disk voice-prompt cache because its bounded capacity is full")
            return
        try:
            disk_reservation = reserve_disk_space(
                root,
                VOICE_PROMPT_MAX_FILE_BYTES,
                min_free_bytes=VOICE_PROMPT_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError:
            logger.warning("Skipping disk voice-prompt cache because shared storage capacity is full")
            return

        temporary_path = root / f".tmp-{uuid.uuid4().hex}"
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_path, flags, 0o600)
            with os.fdopen(descriptor, "wb") as cache_stream:
                descriptor = None
                torch.save(voice_prompt, cache_stream)
                cache_stream.flush()
                os.fsync(cache_stream.fileno())
            entry_stat = temporary_path.lstat()
            if (
                not stat.S_ISREG(entry_stat.st_mode)
                or entry_stat.st_nlink != 1
                or not 0 < entry_stat.st_size <= VOICE_PROMPT_MAX_FILE_BYTES
            ):
                raise ValueError("Serialized voice prompt exceeds the safe file limit")
            _reserve_entry_capacity_locked(
                target_name=final_path.name,
                new_size=entry_stat.st_size,
                temporary_path=temporary_path,
            )
            os.replace(temporary_path, final_path)
            if os.name == "posix":
                os.chmod(final_path, 0o600, follow_symlinks=False)
            _fsync_directory(root)
        except Exception:
            logger.warning("Could not persist the bounded voice-prompt cache entry", exc_info=True)
            with suppress(OSError):
                temporary_path.unlink()
        finally:
            if descriptor is not None:
                os.close(descriptor)
            disk_reservation.release()


def _clear_prompt_entries_locked() -> int:
    _forget_memory_locked()
    deleted = 0
    for root in (_voice_prompt_cache_dir(), _get_cache_dir()):
        root_deleted = 0
        for candidate in root.iterdir():
            if not candidate.name.endswith(".prompt") and not candidate.name.startswith(".tmp-"):
                continue
            try:
                candidate.unlink()
            except OSError as exc:
                logger.warning("Failed to delete voice-prompt cache entry: %s", exc)
            else:
                deleted += 1
                root_deleted += 1
        if root_deleted:
            _fsync_directory(root)
    return deleted


def clear_voice_prompt_cache() -> int:
    """Clear all prompt entries and historical combined reference WAVs."""
    with _cache_lock:
        deleted_count = _clear_prompt_entries_locked()
        cache_dir = _get_cache_dir()
        for audio_file in cache_dir.iterdir():
            if not audio_file.name.startswith("combined_") or not audio_file.name.endswith(".wav"):
                continue
            try:
                audio_file.unlink()
            except OSError as exc:
                logger.warning("Failed to delete combined voice cache entry: %s", exc)
            else:
                deleted_count += 1
        return deleted_count


def clear_profile_cache(profile_id: str) -> int:
    """Invalidate only the selected profile's combined reference WAVs.

    Content-addressed prompt entries have no profile identifier and can be
    shared across profiles. Their global memory/disk LRU bounds reclaim stale
    content without penalizing unrelated voices on every profile mutation.
    """
    with _cache_lock:
        deleted_count = 0
        cache_dir = _get_cache_dir()
        prefix = f"combined_{profile_id}_"
        for audio_file in cache_dir.iterdir():
            if not audio_file.name.startswith(prefix) or not audio_file.name.endswith(".wav"):
                continue
            try:
                audio_file.unlink()
            except OSError as exc:
                logger.warning("Failed to delete combined profile cache entry: %s", exc)
            else:
                deleted_count += 1
        return deleted_count
