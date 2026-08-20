"""Voice profile management module."""

import errno
import hashlib
import json as _json
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config
from ..backends.mlx_tts_lifecycle import run_blocking_operation_cancellation_safe
from ..database import (
    Generation as DBGeneration,
    MCPClientBinding as DBMCPClientBinding,
    ProfileChannelMapping as DBProfileChannelMapping,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)
from ..models import (
    PROFILE_SAMPLE_REFERENCE_TEXT_MAX_CHARS,
    EffectConfig,
    ProfileSampleResponse,
    VoiceProfileCreate,
    VoiceProfileResponse,
)
from ..utils.audio import save_audio, validate_and_load_reference_audio
from ..utils.cache import _get_cache_dir, clear_profile_cache
from ..utils.disk_reservations import DiskSpaceReservationError, reserve_disk_space
from ..utils.images import process_avatar, validate_image
from . import deletion_journal

logger = logging.getLogger(__name__)

CLONING_ENGINES = {"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada"}
_SAMPLE_ORDINAL_CONFLICT = "UNIQUE constraint failed: profile_samples.profile_id, profile_samples.ordinal"
_SAMPLE_ORDINAL_RETRIES = 3
_PROFILE_STORAGE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ACTIVE_GENERATION_STATUSES = {
    "pending",
    "queued",
    "loading_model",
    "generating",
}

# Exact cloned-voice snapshots are immutable and content addressed, but profile
# edits can otherwise retain an ever-growing series of prefixes (one sample,
# then two, then three, ...).  Keep this store independently bounded so it can
# never consume the whole Voicebox data volume.  The limits are intentionally
# far above practical Qwen conditioning: API-created samples are at most 30 s
# of mono 24 kHz audio (about 1.4 MiB each).
EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES = 2 * 1024 * 1024 * 1024
EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES = 1024 * 1024 * 1024
EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES = 64
EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES = 64 * 1024 * 1024
EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES = 128 * 1024 * 1024
EXACT_VOICE_SNAPSHOT_MAX_DERIVED_AUDIO_BYTES = 256 * 1024 * 1024
EXACT_VOICE_SNAPSHOT_MAX_REFERENCE_TEXT_BYTES = 16 * 1024
EXACT_VOICE_SNAPSHOT_MAX_TOTAL_REFERENCE_TEXT_BYTES = 512 * 1024
EXACT_VOICE_SNAPSHOT_MAX_METADATA_BYTES = 1024 * 1024
EXACT_VOICE_SNAPSHOT_MAX_ROOT_ENTRIES = 8192
EXACT_VOICE_SNAPSHOT_MAX_FILES = 100_000
EXACT_VOICE_SNAPSHOT_MAX_OWNERS = 100_000

_EXACT_SNAPSHOT_MUTATION_LOCK = threading.RLock()
_exact_snapshot_reserved_bytes = 0
_exact_snapshot_reserved_entries = 0


class ExactVoiceSnapshotCapacityError(ValueError):
    """Raised before an exact snapshot write would exceed a hard limit."""


class ExactVoiceSnapshotGarbageCollectionError(RuntimeError):
    """Raised when ownership is uncertain and finalized snapshots must stay."""


@dataclass(frozen=True)
class ExactVoiceSnapshotGCReport:
    pending_removed: int = 0
    finalized_removed: int = 0
    refused: int = 0


@dataclass(frozen=True)
class _ExactVoiceSnapshotUsage:
    bytes: int
    root_entries: int
    files: int


class ProfileGenerationActiveError(RuntimeError):
    """Raised when profile deletion would race a queued/shared generation job."""


def _ordered_profile_samples(profile_id: str, db: Session) -> list[DBProfileSample]:
    """Return samples in their immutable, profile-local conditioning order."""
    return db.query(DBProfileSample).filter_by(profile_id=profile_id).order_by(DBProfileSample.ordinal.asc()).all()


def _next_profile_sample_ordinal(profile_id: str, db: Session) -> int:
    current_max = db.query(func.max(DBProfileSample.ordinal)).filter_by(profile_id=profile_id).scalar()
    return 0 if current_max is None else int(current_max) + 1


def _stable_regular_file_identity_and_sha256(
    path: Path,
    *,
    max_bytes: int = EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES,
) -> tuple[tuple[int, int, int, int], str]:
    """Hash one bounded regular file through a no-follow descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Profile sample audio is not a regular file: {path}")
        if before.st_size < 0 or before.st_size > max_bytes:
            raise ValueError(f"Profile sample audio exceeds the exact snapshot limit: {path}")
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            bytes_read += len(block)
            if bytes_read > max_bytes:
                raise ValueError(f"Profile sample audio exceeds the exact snapshot limit: {path}")
            digest.update(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ValueError(f"Cannot read profile sample audio {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"Profile sample audio changed while it was being fingerprinted: {path}")
    if bytes_read != before.st_size:
        raise ValueError(f"Profile sample audio changed while it was being fingerprinted: {path}")
    return before_identity, digest.hexdigest()


def _stable_file_sha256(
    path: Path,
    *,
    max_bytes: int = EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES,
) -> str:
    """Hash one sample while rejecting a concurrent replacement or rewrite."""
    _identity, digest = _stable_regular_file_identity_and_sha256(
        path,
        max_bytes=max_bytes,
    )
    return digest


def _voice_binding_sha256(payload: dict) -> str:
    canonical = _json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _raw_exact_snapshot_key(voice_binding_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", voice_binding_sha256):
        raise ValueError("Exact voice binding must be a lowercase SHA-256 digest")
    snapshot_hash = hashlib.sha256(f"exact-raw-voice-snapshot-v1\0{voice_binding_sha256}".encode()).hexdigest()
    return f"raw-{snapshot_hash}"


def _derived_exact_snapshot_key(
    voice_binding_sha256: str,
    engine: str,
    tts_implementation_revision: str,
) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", voice_binding_sha256):
        raise ValueError("Exact voice binding must be a lowercase SHA-256 digest")
    if not engine or not tts_implementation_revision:
        raise ValueError("Exact derived prompt identity is incomplete")
    derived_hash = hashlib.sha256(
        (f"exact-derived-voice-prompt-v1\0{voice_binding_sha256}\0{engine}\0{tts_implementation_revision}").encode()
    ).hexdigest()
    return f"prompt-{derived_hash}"


def compute_profile_voice_binding_sha256(profile_id: str, db: Session) -> str:
    """Bind exact work to ordered conditioning bytes and complete voice metadata."""
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if profile is None:
        raise ValueError(f"Profile not found: {profile_id}")
    voice_type = getattr(profile, "voice_type", None) or "cloned"
    payload: dict = {"version": 1, "voice_type": voice_type}
    if voice_type == "cloned":
        samples = _ordered_profile_samples(profile_id, db)
        if not samples:
            raise ValueError(f"No samples found for profile {profile_id}")
        _validate_exact_sample_count(samples)
        bound_samples = []
        reference_texts: list[str] = []
        total_audio_bytes = 0
        for sample in samples:
            audio_path = config.resolve_storage_path(sample.audio_path)
            if audio_path is None:
                raise ValueError(f"Sample audio not found for profile {profile_id}")
            reference_text = _validate_exact_reference_text(str(sample.reference_text or ""))
            reference_texts.append(reference_text)
            identity, audio_sha256 = _stable_regular_file_identity_and_sha256(audio_path)
            total_audio_bytes += identity[2]
            bound_samples.append(
                {
                    "ordinal": int(sample.ordinal),
                    "reference_text": reference_text,
                    "audio_sha256": audio_sha256,
                }
            )
        _validate_exact_total_reference_text(reference_texts)
        if total_audio_bytes > EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES:
            raise ValueError("Exact voice snapshot audio exceeds the safe aggregate size limit")
        payload["samples"] = bound_samples
    elif voice_type == "preset":
        payload.update(
            preset_engine=profile.preset_engine,
            preset_voice_id=profile.preset_voice_id,
        )
    elif voice_type == "designed":
        payload["design_prompt"] = profile.design_prompt
    else:
        raise ValueError(f"Unsupported profile voice type: {voice_type}")
    return _voice_binding_sha256(payload)


def _copy_stable_private_file(
    source: Path,
    destination: Path,
    *,
    expected_identity: tuple[int, int, int, int] | None = None,
    expected_sha256: str | None = None,
) -> str:
    """Copy one regular file from a single stable descriptor and return its SHA."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = None
    destination_fd = None
    try:
        source_fd = os.open(source, flags)
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Profile sample audio is not a regular file: {source}")
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if before.st_size < 0 or before.st_size > EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES:
            raise ValueError(f"Profile sample audio exceeds the exact snapshot limit: {source}")
        if expected_identity is not None and before_identity != expected_identity:
            raise ValueError(f"Profile sample audio changed while it was being snapshotted: {source}")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        digest = hashlib.sha256()
        bytes_copied = 0
        while True:
            block = os.read(source_fd, 1024 * 1024)
            if not block:
                break
            bytes_copied += len(block)
            if bytes_copied > EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES:
                raise ValueError(f"Profile sample audio exceeds the exact snapshot limit: {source}")
            digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
    except OSError as exc:
        raise ValueError(f"Cannot snapshot profile sample audio {source}: {exc}") from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or bytes_copied != before.st_size:
        raise ValueError(f"Profile sample audio changed while it was being snapshotted: {source}")
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(f"Profile sample audio changed while it was being snapshotted: {source}")
    return actual_sha256


def _populate_profile_sample_audio(
    descriptor: int,
    audio,
    sample_rate: int,
    intent: deletion_journal.DeletionIntent,
) -> None:
    """Encode a sample into the already journaled staging inode."""
    import soundfile as sf

    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not deletion_journal.entry_matches_intent(intent, before)
    ):
        raise OSError("Journaled profile-sample staging file identity changed")

    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    with open(descriptor, "r+b", closefd=False) as output:
        sf.write(output, audio, sample_rate, format="WAV")
        output.flush()
    os.fsync(descriptor)

    after = os.fstat(descriptor)
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or not deletion_journal.entry_matches_intent(intent, after)
    ):
        raise OSError("Journaled profile-sample WAV identity changed during encoding")


def _private_exact_snapshot_root() -> Path:
    root = config.get_data_dir() / "exact_voice_snapshots"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"Exact voice snapshot root is not a private directory: {root}")
    os.chmod(root, 0o700)
    return root


def _encode_private_snapshot_json(payload: dict) -> bytes:
    encoded = (_json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    if not encoded or len(encoded) > EXACT_VOICE_SNAPSHOT_MAX_METADATA_BYTES:
        raise ValueError("Exact voice snapshot metadata exceeds the safe size limit")
    return encoded


def _validate_exact_reference_text(
    text: object,
    *,
    max_bytes: int = EXACT_VOICE_SNAPSHOT_MAX_REFERENCE_TEXT_BYTES,
) -> str:
    if not isinstance(text, str):
        raise ValueError("Exact voice snapshot reference text is invalid")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError("Exact voice snapshot reference text exceeds the safe size limit")
    return text


def _validate_exact_sample_count(samples: list[object]) -> None:
    if not samples:
        raise ValueError("Exact voice snapshot has no samples")
    if len(samples) > EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES:
        raise ValueError(f"Exact voice snapshots support at most {EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES} samples")


def _validate_exact_total_reference_text(texts: list[str]) -> None:
    total = sum(len(text.encode("utf-8")) for text in texts)
    if total > EXACT_VOICE_SNAPSHOT_MAX_TOTAL_REFERENCE_TEXT_BYTES:
        raise ValueError("Exact voice snapshot transcripts exceed the safe aggregate size limit")


def _validate_profile_sample_reference_text(reference_text: str) -> str:
    if not isinstance(reference_text, str) or not reference_text:
        raise ValueError("Reference text must not be empty")
    if len(reference_text) > PROFILE_SAMPLE_REFERENCE_TEXT_MAX_CHARS:
        raise ValueError(f"Reference text exceeds the {PROFILE_SAMPLE_REFERENCE_TEXT_MAX_CHARS}-character limit")
    _validate_exact_reference_text(reference_text)
    return reference_text


def _validate_profile_sample_admission(
    profile_id: str,
    reference_text: str,
    new_sample_bytes: int,
    db: Session,
) -> None:
    """Enforce the conditioning limits under the profile writer transaction."""
    samples = (
        db.query(DBProfileSample)
        .filter_by(profile_id=profile_id)
        .order_by(DBProfileSample.ordinal, DBProfileSample.id)
        .all()
    )
    if len(samples) >= EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES:
        raise ValueError(f"Voice profiles support at most {EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES} reference samples")

    texts = [_validate_exact_reference_text(str(sample.reference_text or "")) for sample in samples]
    texts.append(reference_text)
    _validate_exact_total_reference_text(texts)

    aggregate_bytes = new_sample_bytes
    if new_sample_bytes <= 0 or new_sample_bytes > EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES:
        raise ValueError("Reference sample audio exceeds the safe size limit")
    expected_parent = Path("profiles") / profile_id
    for sample in samples:
        relative = config.managed_storage_relative_path(sample.audio_path)
        if relative is None or expected_parent not in relative.parents:
            raise ValueError("Existing reference sample is outside managed profile storage")
        entry_stat = deletion_journal.managed_entry_stat(relative)
        if (
            entry_stat is None
            or not stat.S_ISREG(entry_stat.st_mode)
            or entry_stat.st_nlink != 1
            or entry_stat.st_size <= 0
            or entry_stat.st_size > EXACT_VOICE_SNAPSHOT_MAX_SAMPLE_BYTES
        ):
            raise ValueError("Existing reference sample audio is unsafe or oversized")
        aggregate_bytes += entry_stat.st_size
        if aggregate_bytes > EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES:
            raise ValueError("Reference sample audio exceeds the safe aggregate size limit")


def _bounded_exact_snapshot_usage(root: Path) -> _ExactVoiceSnapshotUsage:
    """Account for every byte without following links or recursing unboundedly."""
    total_bytes = 0
    root_entries = 0
    files = 0
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                root_entries += 1
                if root_entries > EXACT_VOICE_SNAPSHOT_MAX_ROOT_ENTRIES:
                    raise ExactVoiceSnapshotCapacityError("Exact voice snapshot store contains too many entries")
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
                    raise ExactVoiceSnapshotCapacityError("Exact voice snapshot store contains an unsafe entry")
                with os.scandir(entry.path) as children:
                    for child in children:
                        files += 1
                        if files > EXACT_VOICE_SNAPSHOT_MAX_FILES:
                            raise ExactVoiceSnapshotCapacityError("Exact voice snapshot store contains too many files")
                        child_stat = child.stat(follow_symlinks=False)
                        if not stat.S_ISREG(child_stat.st_mode) or stat.S_ISLNK(child_stat.st_mode):
                            raise ExactVoiceSnapshotCapacityError("Exact voice snapshot store contains an unsafe file")
                        if child_stat.st_size < 0:
                            raise ExactVoiceSnapshotCapacityError("Exact voice snapshot store contains an invalid file")
                        total_bytes += child_stat.st_size
                        if total_bytes > EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES:
                            raise ExactVoiceSnapshotCapacityError(
                                "Exact voice snapshot store already exceeds its byte limit"
                            )
    except OSError as exc:
        raise ExactVoiceSnapshotCapacityError(f"Cannot safely account for exact voice snapshots: {exc}") from exc
    return _ExactVoiceSnapshotUsage(total_bytes, root_entries, files)


@contextmanager
def _reserve_exact_snapshot_capacity(
    required_bytes: int,
    *,
    required_entries: int,
) -> Iterator[None]:
    """Reserve store and filesystem capacity across concurrent snapshot writes."""
    if required_bytes < 0 or required_bytes > EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES:
        raise ExactVoiceSnapshotCapacityError("Exact voice snapshot is too large for the bounded store")
    if required_entries <= 0 or required_entries > EXACT_VOICE_SNAPSHOT_MAX_FILES:
        raise ExactVoiceSnapshotCapacityError("Exact voice snapshot has too many files")
    root = _private_exact_snapshot_root()
    global _exact_snapshot_reserved_bytes, _exact_snapshot_reserved_entries
    with _EXACT_SNAPSHOT_MUTATION_LOCK:
        usage = _bounded_exact_snapshot_usage(root)
        if usage.bytes + _exact_snapshot_reserved_bytes + required_bytes > EXACT_VOICE_SNAPSHOT_MAX_STORE_BYTES:
            raise ExactVoiceSnapshotCapacityError(
                "Exact voice snapshot store limit reached; restart Voicebox to reclaim completed snapshots, "
                "and resume or remove older failed exact work before restarting if it still owns the space"
            )
        if (
            usage.root_entries + _exact_snapshot_reserved_entries + 1 > EXACT_VOICE_SNAPSHOT_MAX_ROOT_ENTRIES
            or usage.files + _exact_snapshot_reserved_entries + required_entries > EXACT_VOICE_SNAPSHOT_MAX_FILES
        ):
            raise ExactVoiceSnapshotCapacityError("Exact voice snapshot store entry limit reached")
        try:
            disk_reservation = reserve_disk_space(
                root,
                required_bytes,
                min_free_bytes=EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES,
            )
        except DiskSpaceReservationError as exc:
            raise ExactVoiceSnapshotCapacityError(
                "Insufficient reserved free space for an exact voice snapshot"
            ) from exc
        _exact_snapshot_reserved_bytes += required_bytes
        _exact_snapshot_reserved_entries += required_entries
    try:
        yield
    finally:
        with _EXACT_SNAPSHOT_MUTATION_LOCK:
            _exact_snapshot_reserved_bytes -= required_bytes
            _exact_snapshot_reserved_entries -= required_entries
        disk_reservation.release()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_private_json(path: Path, payload: dict) -> None:
    encoded = _encode_private_snapshot_json(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while publishing exact voice metadata")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_private_snapshot_directory(pending: Path, destination: Path) -> bool:
    """Publish once, treating only a real pre-existing directory as deduplication.

    Linux commonly reports ``EEXIST`` while macOS reports ``ENOTEMPTY`` when an
    immutable content-addressed directory is already present.  Both are safe only
    after a no-follow type check; the caller still verifies the existing contents.
    """
    try:
        pending.rename(destination)
    except OSError as exc:
        if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
            raise
        try:
            destination_stat = destination.lstat()
        except OSError as inspect_exc:
            raise ValueError(f"Cannot verify existing exact voice snapshot {destination}: {inspect_exc}") from exc
        if not stat.S_ISDIR(destination_stat.st_mode) or stat.S_ISLNK(destination_stat.st_mode):
            raise ValueError(f"Existing exact voice snapshot is not a directory: {destination}") from exc
        return False
    return True


def _load_private_snapshot_metadata(snapshot_dir: Path) -> dict:
    metadata_fd = None
    try:
        directory_stat = snapshot_dir.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode) or stat.S_ISLNK(directory_stat.st_mode):
            raise ValueError(f"Exact voice snapshot is not a private directory: {snapshot_dir}")
        metadata_path = snapshot_dir / "snapshot.json"
        metadata_fd = os.open(
            metadata_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata_stat = os.fstat(metadata_fd)
        if not stat.S_ISREG(metadata_stat.st_mode) or stat.S_ISLNK(metadata_stat.st_mode):
            raise ValueError(f"Exact voice snapshot metadata is invalid: {metadata_path}")
        if metadata_stat.st_size <= 0 or metadata_stat.st_size > EXACT_VOICE_SNAPSHOT_MAX_METADATA_BYTES:
            raise ValueError(f"Exact voice snapshot metadata has invalid size: {metadata_path}")
        encoded = bytearray()
        while len(encoded) <= EXACT_VOICE_SNAPSHOT_MAX_METADATA_BYTES:
            block = os.read(metadata_fd, min(64 * 1024, EXACT_VOICE_SNAPSHOT_MAX_METADATA_BYTES + 1 - len(encoded)))
            if not block:
                break
            encoded.extend(block)
        if len(encoded) != metadata_stat.st_size:
            raise ValueError(f"Exact voice snapshot metadata changed while being read: {metadata_path}")
        metadata = _json.loads(encoded.decode("utf-8"))
    except (OSError, UnicodeDecodeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot verify exact voice snapshot {snapshot_dir}: {exc}") from exc
    finally:
        if metadata_fd is not None:
            os.close(metadata_fd)
    if not isinstance(metadata, dict):
        raise ValueError(f"Exact voice snapshot metadata is invalid: {snapshot_dir}")
    return metadata


def _verify_private_snapshot_entries(snapshot_dir: Path, expected_names: set[str]) -> None:
    try:
        actual_names: set[str] = set()
        with os.scandir(snapshot_dir) as entries:
            for entry in entries:
                if len(actual_names) >= EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES + 2:
                    raise ValueError(f"Exact voice snapshot contains too many files: {snapshot_dir}")
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
                    raise ValueError(f"Exact voice snapshot contains an unsafe file: {snapshot_dir}")
                actual_names.add(entry.name)
    except OSError as exc:
        raise ValueError(f"Cannot verify exact voice snapshot files {snapshot_dir}: {exc}") from exc
    if actual_names != expected_names:
        raise ValueError(f"Exact voice snapshot file set is invalid: {snapshot_dir}")


def _verify_raw_exact_snapshot(
    descriptor: dict,
    *,
    expected_binding_sha256: str,
    expected_samples: list[dict] | None = None,
) -> tuple[Path, dict]:
    snapshot_key = descriptor.get("snapshot_key")
    if not isinstance(snapshot_key, str) or not re.fullmatch(r"raw-[0-9a-f]{64}", snapshot_key):
        raise ValueError("Invalid exact voice snapshot descriptor")
    if descriptor.get("voice_binding_sha256") != expected_binding_sha256:
        raise ValueError("Exact voice snapshot descriptor binding does not match")
    snapshot_dir = _private_exact_snapshot_root() / snapshot_key
    metadata = _load_private_snapshot_metadata(snapshot_dir)
    if (
        set(metadata)
        != {
            "format_version",
            "kind",
            "voice_binding_sha256",
            "samples",
        }
        or metadata.get("format_version") != 1
        or metadata.get("kind") != "raw"
    ):
        raise ValueError("Unsupported exact voice snapshot format")
    if metadata.get("voice_binding_sha256") != expected_binding_sha256:
        raise ValueError("Exact voice snapshot binding metadata does not match")
    if snapshot_key != _raw_exact_snapshot_key(expected_binding_sha256):
        raise ValueError("Exact voice snapshot key does not match its binding")
    samples = metadata.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Exact voice snapshot has no samples")
    _validate_exact_sample_count(samples)
    if expected_samples is not None and samples != expected_samples:
        raise ValueError("Exact voice snapshot samples do not match verified input")
    expected_filenames = {"snapshot.json"}
    reference_texts: list[str] = []
    total_audio_bytes = 0
    previous_ordinal = -1
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or set(sample) != {
            "ordinal",
            "reference_text",
            "audio_sha256",
            "filename",
        }:
            raise ValueError("Exact voice snapshot sample metadata is invalid")
        ordinal = sample.get("ordinal")
        reference_text = sample.get("reference_text")
        filename = sample.get("filename")
        digest = sample.get("audio_sha256")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0 or ordinal <= previous_ordinal:
            raise ValueError("Exact voice snapshot sample identity is invalid")
        previous_ordinal = ordinal
        reference_text = _validate_exact_reference_text(reference_text)
        reference_texts.append(reference_text)
        if filename != f"sample-{index:04d}.wav":
            raise ValueError("Exact voice snapshot sample filename is invalid")
        expected_filenames.add(filename)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Exact voice snapshot sample digest is invalid")
        audio_path = snapshot_dir / filename
        audio_identity, actual_digest = _stable_regular_file_identity_and_sha256(audio_path)
        total_audio_bytes += audio_identity[2]
        if actual_digest != digest:
            raise ValueError(f"Exact voice snapshot audio was modified: {audio_path}")
    _validate_exact_total_reference_text(reference_texts)
    if total_audio_bytes > EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES:
        raise ValueError("Exact voice snapshot audio exceeds the safe aggregate size limit")
    _verify_private_snapshot_entries(snapshot_dir, expected_filenames)
    recomputed_binding = _voice_binding_sha256(
        {
            "version": 1,
            "voice_type": "cloned",
            "samples": [
                {
                    "ordinal": sample["ordinal"],
                    "reference_text": sample["reference_text"],
                    "audio_sha256": sample["audio_sha256"],
                }
                for sample in samples
            ],
        }
    )
    if recomputed_binding != expected_binding_sha256:
        raise ValueError("Exact voice snapshot contents do not match their binding")
    return snapshot_dir, metadata


def freeze_exact_voice_profile(
    profile_id: str,
    db: Session,
    *,
    engine: str = "qwen",
) -> dict:
    """Durably freeze exact raw reference bytes before accepting queue work."""
    db.expire_all()
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if profile is None:
        raise ValueError(f"Profile not found: {profile_id}")
    if (getattr(profile, "voice_type", None) or "cloned") != "cloned":
        raise ValueError("Exact pinned Qwen generation requires a cloned profile")
    validate_profile_engine(profile, engine)
    samples = _ordered_profile_samples(profile_id, db)
    if not samples:
        raise ValueError(f"No samples found for profile {profile_id}")
    _validate_exact_sample_count(samples)

    snapshot_root = _private_exact_snapshot_root()
    sample_metadata: list[dict] = []
    sources: list[tuple[Path, tuple[int, int, int, int], str]] = []
    reference_texts: list[str] = []
    for index, sample in enumerate(samples):
        source = config.resolve_storage_path(sample.audio_path)
        if source is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        reference_text = _validate_exact_reference_text(str(sample.reference_text or ""))
        reference_texts.append(reference_text)
        identity, audio_sha256 = _stable_regular_file_identity_and_sha256(source)
        sources.append((source, identity, audio_sha256))
        sample_metadata.append(
            {
                "ordinal": int(sample.ordinal),
                "reference_text": reference_text,
                "audio_sha256": audio_sha256,
                "filename": f"sample-{index:04d}.wav",
            }
        )
    _validate_exact_total_reference_text(reference_texts)
    if sum(identity[2] for _source, identity, _digest in sources) > (EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES):
        raise ValueError("Exact voice snapshot audio exceeds the safe aggregate size limit")
    binding_payload = {
        "version": 1,
        "voice_type": "cloned",
        "samples": [
            {
                "ordinal": sample["ordinal"],
                "reference_text": sample["reference_text"],
                "audio_sha256": sample["audio_sha256"],
            }
            for sample in sample_metadata
        ],
    }
    binding_sha256 = _voice_binding_sha256(binding_payload)
    descriptor = {
        "format_version": 1,
        "snapshot_key": _raw_exact_snapshot_key(binding_sha256),
        "voice_binding_sha256": binding_sha256,
    }
    metadata = {
        "format_version": 1,
        "kind": "raw",
        "voice_binding_sha256": binding_sha256,
        "samples": sample_metadata,
    }
    metadata_bytes = _encode_private_snapshot_json(metadata)
    snapshot_dir = snapshot_root / descriptor["snapshot_key"]
    try:
        snapshot_dir.lstat()
    except FileNotFoundError:
        pass
    else:
        _verify_raw_exact_snapshot(
            descriptor,
            expected_binding_sha256=binding_sha256,
            expected_samples=sample_metadata,
        )
        return descriptor

    required_bytes = sum(identity[2] for _source, identity, _digest in sources) + len(metadata_bytes)
    with _reserve_exact_snapshot_capacity(
        required_bytes,
        required_entries=len(sources) + 1,
    ):
        pending_dir: Path | None = Path(tempfile.mkdtemp(prefix=".pending-", dir=snapshot_root))
        os.chmod(pending_dir, 0o700)
        try:
            for sample, (source, identity, digest) in zip(sample_metadata, sources, strict=True):
                _copy_stable_private_file(
                    source,
                    pending_dir / sample["filename"],
                    expected_identity=identity,
                    expected_sha256=digest,
                )
            _write_private_json(pending_dir / "snapshot.json", metadata)
            _fsync_directory(pending_dir)
            if _publish_private_snapshot_directory(pending_dir, snapshot_dir):
                pending_dir = None
                _fsync_directory(snapshot_root)
            _verify_raw_exact_snapshot(
                descriptor,
                expected_binding_sha256=binding_sha256,
                expected_samples=sample_metadata,
            )
            return descriptor
        finally:
            if pending_dir is not None:
                shutil.rmtree(pending_dir, ignore_errors=True)


def _require_exact_tts_revision(expected_tts_implementation_revision: str) -> None:
    if not expected_tts_implementation_revision:
        raise ValueError("Exact TTS implementation revision is required")
    from ..backends import get_tts_implementation_revision

    actual = get_tts_implementation_revision()
    if actual != expected_tts_implementation_revision:
        raise RuntimeError(
            "TTS implementation revision changed while exact generation was queued: "
            f"expected {expected_tts_implementation_revision!r}, running {actual!r}"
        )


def _verify_derived_exact_snapshot(
    snapshot_dir: Path,
    *,
    expected_voice_binding_sha256: str | None = None,
    expected_engine: str | None = None,
    expected_tts_implementation_revision: str | None = None,
    expected_reference_text: str | None = None,
) -> dict:
    """Strictly validate one finalized combined-reference snapshot."""
    if not re.fullmatch(r"prompt-[0-9a-f]{64}", snapshot_dir.name):
        raise ValueError("Exact derived voice prompt key is invalid")
    metadata = _load_private_snapshot_metadata(snapshot_dir)
    binding = metadata.get("voice_binding_sha256")
    engine = metadata.get("engine")
    revision = metadata.get("tts_implementation_revision")
    reference_text = _validate_exact_reference_text(
        metadata.get("reference_text"),
        max_bytes=EXACT_VOICE_SNAPSHOT_MAX_TOTAL_REFERENCE_TEXT_BYTES,
    )
    digest = metadata.get("combined_sha256")
    if (
        set(metadata)
        != {
            "format_version",
            "kind",
            "voice_binding_sha256",
            "engine",
            "tts_implementation_revision",
            "combined_sha256",
            "reference_text",
        }
        or metadata.get("format_version") != 1
        or metadata.get("kind") != "derived"
        or not isinstance(binding, str)
        or not re.fullmatch(r"[0-9a-f]{64}", binding)
        or not isinstance(engine, str)
        or not engine
        or not isinstance(revision, str)
        or not revision
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("Exact derived voice prompt metadata is invalid")
    if snapshot_dir.name != _derived_exact_snapshot_key(binding, engine, revision):
        raise ValueError("Exact derived voice prompt key does not match its metadata")
    _verify_private_snapshot_entries(snapshot_dir, {"combined.wav", "snapshot.json"})
    if (
        (expected_voice_binding_sha256 is not None and binding != expected_voice_binding_sha256)
        or (expected_engine is not None and engine != expected_engine)
        or (expected_tts_implementation_revision is not None and revision != expected_tts_implementation_revision)
        or (expected_reference_text is not None and reference_text != expected_reference_text)
    ):
        raise ValueError("Exact derived voice prompt metadata is invalid")
    prompt_path = snapshot_dir / "combined.wav"
    if (
        _stable_file_sha256(
            prompt_path,
            max_bytes=EXACT_VOICE_SNAPSHOT_MAX_DERIVED_AUDIO_BYTES,
        )
        != digest
    ):
        raise ValueError("Exact derived voice prompt audio was modified")
    return metadata


async def create_exact_voice_prompt_from_snapshot(
    descriptor: dict,
    *,
    expected_voice_binding_sha256: str,
    expected_tts_implementation_revision: str,
    engine: str = "qwen",
) -> dict:
    """Create model conditioning only from the raw snapshot accepted by the route."""
    _require_exact_tts_revision(expected_tts_implementation_revision)
    snapshot_dir, metadata = _verify_raw_exact_snapshot(
        descriptor,
        expected_binding_sha256=expected_voice_binding_sha256,
    )
    samples = metadata["samples"]
    from ..backends import get_tts_backend_for_engine

    tts_model = get_tts_backend_for_engine(engine)
    if len(samples) == 1:
        prompt_path = snapshot_dir / samples[0]["filename"]
        prompt_text = samples[0]["reference_text"]
    else:
        snapshot_root = _private_exact_snapshot_root()
        derived_dir = snapshot_root / _derived_exact_snapshot_key(
            expected_voice_binding_sha256,
            engine,
            expected_tts_implementation_revision,
        )
        expected_prompt_text = " ".join(sample["reference_text"] for sample in samples)
        _validate_exact_total_reference_text([sample["reference_text"] for sample in samples])
        # The placeholder digest has the same encoded length as the final one,
        # so metadata is rejected before combine/publish rather than afterwards.
        placeholder_metadata = {
            "format_version": 1,
            "kind": "derived",
            "voice_binding_sha256": expected_voice_binding_sha256,
            "engine": engine,
            "tts_implementation_revision": expected_tts_implementation_revision,
            "combined_sha256": "0" * 64,
            "reference_text": expected_prompt_text,
        }
        placeholder_size = len(_encode_private_snapshot_json(placeholder_metadata))
        try:
            derived_dir.lstat()
        except FileNotFoundError:
            create_derived = True
        else:
            create_derived = False
        if create_derived:
            with _reserve_exact_snapshot_capacity(
                EXACT_VOICE_SNAPSHOT_MAX_DERIVED_AUDIO_BYTES + placeholder_size,
                required_entries=2,
            ):
                combined_audio, prompt_text = await tts_model.combine_voice_prompts(
                    [str(snapshot_dir / sample["filename"]) for sample in samples],
                    [sample["reference_text"] for sample in samples],
                )
                if prompt_text != expected_prompt_text:
                    raise ValueError("Combined exact voice prompt text does not match its immutable samples")
                combined_size = getattr(combined_audio, "size", None)
                if isinstance(combined_size, bool) or not isinstance(combined_size, int):
                    raise ValueError("Combined exact voice prompt audio has an invalid shape")
                estimated_wav_bytes = combined_size * 2 + 4096
                if estimated_wav_bytes > EXACT_VOICE_SNAPSHOT_MAX_DERIVED_AUDIO_BYTES:
                    raise ExactVoiceSnapshotCapacityError(
                        "Combined exact voice prompt exceeds the safe audio size limit"
                    )
                pending_dir: Path | None = Path(tempfile.mkdtemp(prefix=".pending-prompt-", dir=snapshot_root))
                os.chmod(pending_dir, 0o700)
                try:
                    combined_path = pending_dir / "combined.wav"
                    save_audio(combined_audio, str(combined_path), 24000)
                    os.chmod(combined_path, 0o600)
                    combined_identity, combined_sha256 = _stable_regular_file_identity_and_sha256(
                        combined_path,
                        max_bytes=EXACT_VOICE_SNAPSHOT_MAX_DERIVED_AUDIO_BYTES,
                    )
                    if combined_identity[2] > estimated_wav_bytes:
                        raise ExactVoiceSnapshotCapacityError("Combined exact voice prompt exceeded its reserved size")
                    final_metadata = dict(placeholder_metadata)
                    final_metadata["combined_sha256"] = combined_sha256
                    _encode_private_snapshot_json(final_metadata)
                    _write_private_json(pending_dir / "snapshot.json", final_metadata)
                    _fsync_directory(pending_dir)
                    if _publish_private_snapshot_directory(pending_dir, derived_dir):
                        pending_dir = None
                        _fsync_directory(snapshot_root)
                finally:
                    if pending_dir is not None:
                        shutil.rmtree(pending_dir, ignore_errors=True)
        _verify_derived_exact_snapshot(
            derived_dir,
            expected_voice_binding_sha256=expected_voice_binding_sha256,
            expected_engine=engine,
            expected_tts_implementation_revision=expected_tts_implementation_revision,
            expected_reference_text=expected_prompt_text,
        )
        prompt_path = derived_dir / "combined.wav"
        prompt_text = expected_prompt_text
    _require_exact_tts_revision(expected_tts_implementation_revision)
    voice_prompt, _ = await tts_model.create_voice_prompt(
        str(prompt_path),
        prompt_text,
        use_cache=True,
    )
    return voice_prompt


def _safe_remove_exact_snapshot_directory(path: Path, root: Path) -> None:
    """Atomically hide and remove one already-validated flat directory."""
    path_stat = path.lstat()
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"Exact voice snapshot is not a removable directory: {path}")
    hidden = root / f".gc-{uuid.uuid4().hex}"
    path.rename(hidden)
    _fsync_directory(root)
    shutil.rmtree(hidden)
    _fsync_directory(root)


def _safe_remove_abandoned_exact_snapshot_directory(path: Path, root: Path) -> None:
    """Remove a private unpublished directory only when every child is safe."""
    path_stat = path.lstat()
    if not stat.S_ISDIR(path_stat.st_mode) or stat.S_ISLNK(path_stat.st_mode):
        raise ValueError(f"Abandoned exact voice snapshot is unsafe: {path}")
    child_count = 0
    with os.scandir(path) as children:
        for child in children:
            child_count += 1
            if child_count > EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES + 3:
                raise ValueError(f"Abandoned exact voice snapshot contains too many files: {path}")
            child_stat = child.stat(follow_symlinks=False)
            if not stat.S_ISREG(child_stat.st_mode) or stat.S_ISLNK(child_stat.st_mode):
                raise ValueError(f"Abandoned exact voice snapshot contains an unsafe file: {path}")
    shutil.rmtree(path)
    _fsync_directory(root)


def _prune_abandoned_exact_voice_snapshot_directories(root: Path) -> tuple[int, int]:
    removed = 0
    refused = 0
    root_entries = 0
    with os.scandir(root) as entries:
        candidates = []
        for entry in entries:
            root_entries += 1
            if root_entries > EXACT_VOICE_SNAPSHOT_MAX_ROOT_ENTRIES:
                raise ExactVoiceSnapshotGarbageCollectionError("Exact voice snapshot cleanup exceeds its bounded scan")
            if entry.name.startswith((".pending-", ".pending-prompt-", ".gc-")):
                candidates.append(Path(entry.path))
    for candidate in candidates:
        try:
            _safe_remove_abandoned_exact_snapshot_directory(candidate, root)
        except (OSError, ValueError):
            refused += 1
            logger.warning("Refused unsafe abandoned exact voice snapshot %s", candidate)
        else:
            removed += 1
    return removed, refused


def _exact_voice_snapshot_owners(db: Session, root: Path) -> tuple[set[str], set[str]]:
    """Inventory ownership completely before any finalized directory is removed."""
    keep_keys: set[str] = set()
    keep_bindings: set[str] = set()
    try:
        exact_filter = or_(
            DBGeneration.exact_request_sha256.is_not(None),
            DBGeneration.exact_voice_snapshot_json.is_not(None),
            DBGeneration.voice_binding_sha256.is_not(None),
        )
        incomplete_filter = or_(
            DBGeneration.status.is_(None),
            DBGeneration.status != "completed",
        )
        owner_query = db.query(
            DBGeneration.id,
            DBGeneration.exact_voice_snapshot_json,
            DBGeneration.voice_binding_sha256,
        ).filter(exact_filter, incomplete_filter)
        if owner_query.count() > EXACT_VOICE_SNAPSHOT_MAX_OWNERS:
            raise ExactVoiceSnapshotGarbageCollectionError("Exact voice snapshot ownership exceeds its bounded scan")
        for generation_id, snapshot_json, binding in owner_query.all():
            if (
                not isinstance(snapshot_json, str)
                or not snapshot_json
                or len(snapshot_json.encode("utf-8")) > EXACT_VOICE_SNAPSHOT_MAX_METADATA_BYTES
                or not isinstance(binding, str)
                or not re.fullmatch(r"[0-9a-f]{64}", binding)
            ):
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Exact generation {generation_id} has uncertain snapshot ownership"
                )
            try:
                descriptor = _json.loads(snapshot_json)
            except _json.JSONDecodeError as exc:
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Exact generation {generation_id} has invalid snapshot ownership"
                ) from exc
            if not isinstance(descriptor, dict) or set(descriptor) != {
                "format_version",
                "snapshot_key",
                "voice_binding_sha256",
            }:
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Exact generation {generation_id} has invalid snapshot ownership"
                )
            if descriptor.get("format_version") != 1 or descriptor.get("voice_binding_sha256") != binding:
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Exact generation {generation_id} has mismatched snapshot ownership"
                )
            try:
                _verify_raw_exact_snapshot(
                    descriptor,
                    expected_binding_sha256=binding,
                )
            except ValueError as exc:
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Exact generation {generation_id} references an invalid snapshot"
                ) from exc
            keep_keys.add(descriptor["snapshot_key"])
            keep_bindings.add(binding)

        profile_query = db.query(DBVoiceProfile.id).filter(
            or_(DBVoiceProfile.voice_type.is_(None), DBVoiceProfile.voice_type == "cloned")
        )
        if profile_query.count() > EXACT_VOICE_SNAPSHOT_MAX_OWNERS:
            raise ExactVoiceSnapshotGarbageCollectionError("Exact voice profile ownership exceeds its bounded scan")
        for (profile_id,) in profile_query.all():
            if db.query(DBProfileSample.id).filter_by(profile_id=profile_id).first() is None:
                continue
            try:
                binding = compute_profile_voice_binding_sha256(profile_id, db)
                key = _raw_exact_snapshot_key(binding)
            except ValueError as exc:
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Profile {profile_id} has uncertain exact snapshot ownership"
                ) from exc
            keep_keys.add(key)
            keep_bindings.add(binding)
            current_path = root / key
            try:
                current_path.lstat()
            except FileNotFoundError:
                continue
            try:
                _verify_raw_exact_snapshot(
                    {
                        "format_version": 1,
                        "snapshot_key": key,
                        "voice_binding_sha256": binding,
                    },
                    expected_binding_sha256=binding,
                )
            except ValueError as exc:
                raise ExactVoiceSnapshotGarbageCollectionError(
                    f"Profile {profile_id} has an invalid current exact snapshot"
                ) from exc
    except ExactVoiceSnapshotGarbageCollectionError:
        raise
    except Exception as exc:
        raise ExactVoiceSnapshotGarbageCollectionError("Could not establish exact voice snapshot ownership") from exc
    # Derived prompts are pure caches: every one can be rebuilt from its raw
    # binding. Keep only the key usable by this exact runtime. Older runtime
    # revisions cannot satisfy _require_exact_tts_revision and retaining them
    # can otherwise exhaust the bounded store across upgrades. GC is startup-
    # only, before init_queue/routes, so no prompt can be actively leased here.
    try:
        from ..backends import get_tts_implementation_revision

        current_revision = get_tts_implementation_revision()
    except Exception as exc:
        raise ExactVoiceSnapshotGarbageCollectionError("Could not establish the current exact prompt revision") from exc
    keep_derived_keys = (
        {_derived_exact_snapshot_key(binding, "qwen", current_revision) for binding in keep_bindings}
        if current_revision
        else set()
    )
    return keep_keys, keep_derived_keys


def garbage_collect_exact_voice_snapshots(db: Session) -> ExactVoiceSnapshotGCReport:
    """Startup-only, reference-aware cleanup of immutable exact voice data.

    This must run before the task queue and HTTP routes accept work.  Runtime
    cleanup would need explicit leases spanning route freeze through prompt
    construction; keeping GC startup-only makes that race impossible.
    """
    root = _private_exact_snapshot_root()
    with _EXACT_SNAPSHOT_MUTATION_LOCK:
        if _exact_snapshot_reserved_bytes or _exact_snapshot_reserved_entries:
            raise ExactVoiceSnapshotGarbageCollectionError(
                "Cannot garbage-collect exact voice snapshots while a write is reserved"
            )
        pending_removed, refused = _prune_abandoned_exact_voice_snapshot_directories(root)
        keep_keys, keep_derived_keys = _exact_voice_snapshot_owners(db, root)

        root_entries = 0
        finalized_removed = 0
        with os.scandir(root) as entries:
            candidates = []
            for entry in entries:
                root_entries += 1
                if root_entries > EXACT_VOICE_SNAPSHOT_MAX_ROOT_ENTRIES:
                    raise ExactVoiceSnapshotGarbageCollectionError(
                        "Exact voice snapshot cleanup exceeds its bounded scan"
                    )
                candidates.append(Path(entry.path))
        for candidate in candidates:
            name = candidate.name
            if name.startswith((".pending-", ".pending-prompt-", ".gc-")):
                # Unsafe abandoned entries were already refused and retained.
                continue
            try:
                if re.fullmatch(r"raw-[0-9a-f]{64}", name):
                    metadata = _load_private_snapshot_metadata(candidate)
                    binding = metadata.get("voice_binding_sha256")
                    if not isinstance(binding, str) or not re.fullmatch(r"[0-9a-f]{64}", binding):
                        raise ValueError("Exact raw voice snapshot binding is invalid")
                    _verify_raw_exact_snapshot(
                        {
                            "format_version": 1,
                            "snapshot_key": name,
                            "voice_binding_sha256": binding,
                        },
                        expected_binding_sha256=binding,
                    )
                    retain = name in keep_keys
                elif re.fullmatch(r"prompt-[0-9a-f]{64}", name):
                    _verify_derived_exact_snapshot(candidate)
                    retain = name in keep_derived_keys
                else:
                    raise ValueError("Unknown exact voice snapshot entry")
                if not retain:
                    _safe_remove_exact_snapshot_directory(candidate, root)
                    finalized_removed += 1
            except (OSError, ValueError):
                refused += 1
                logger.warning("Refused unsafe exact voice snapshot %s", candidate)
        return ExactVoiceSnapshotGCReport(
            pending_removed=pending_removed,
            finalized_removed=finalized_removed,
            refused=refused,
        )


def _profile_to_response(
    profile: DBVoiceProfile,
    generation_count: int = 0,
    sample_count: int = 0,
) -> VoiceProfileResponse:
    """Convert a DB profile to a VoiceProfileResponse, deserializing effects_chain."""
    effects_chain = None
    if profile.effects_chain:
        try:
            raw = _json.loads(profile.effects_chain)
            effects_chain = [EffectConfig(**e) for e in raw]
        except Exception as e:
            import logging

            logging.warning(f"Failed to parse effects_chain for profile {profile.id}: {e}")
    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        avatar_path=profile.avatar_path,
        effects_chain=effects_chain,
        voice_type=getattr(profile, "voice_type", None) or "cloned",
        preset_engine=getattr(profile, "preset_engine", None),
        preset_voice_id=getattr(profile, "preset_voice_id", None),
        design_prompt=getattr(profile, "design_prompt", None),
        default_engine=getattr(profile, "default_engine", None),
        personality=getattr(profile, "personality", None),
        generation_count=generation_count,
        sample_count=sample_count,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _durable_profile_sample_response(
    db: Session,
    *,
    sample_id: str,
    profile_id: str,
    ordinal: int,
    audio_path: str,
    reference_text: str,
) -> ProfileSampleResponse | None:
    """Return an independently verified sample after an ambiguous commit."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        durable_sample = durable_db.query(DBProfileSample).filter_by(id=sample_id).one_or_none()
        if durable_sample is None:
            return None
        durable_fields = (
            durable_sample.profile_id,
            durable_sample.ordinal,
            durable_sample.audio_path,
            durable_sample.reference_text,
        )
        expected_fields = (
            profile_id,
            ordinal,
            audio_path,
            reference_text,
        )
        if durable_fields != expected_fields:
            raise RuntimeError("Profile sample ID is owned by different durable data")
        return ProfileSampleResponse.model_validate(durable_sample)


def _durable_created_profile_response(
    db: Session,
    *,
    profile_id: str,
    expected_fields: dict[str, object],
) -> VoiceProfileResponse | None:
    """Resolve whether a create-profile commit became durably visible."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        durable_profile = durable_db.query(DBVoiceProfile).filter_by(id=profile_id).one_or_none()
        if durable_profile is None:
            return None
        if any(getattr(durable_profile, field) != value for field, value in expected_fields.items()):
            raise RuntimeError("Profile ID is owned by different durable data")
        return _profile_to_response(durable_profile)


def _get_preset_voice_ids(engine: str) -> set[str]:
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        return {voice_id for voice_id, _name, _gender, _lang in KOKORO_VOICES}

    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        return {voice_id for voice_id, _name, _gender, _lang, _desc in QWEN_CUSTOM_VOICES}

    return set()


def _validate_profile_fields(
    *,
    voice_type: str,
    preset_engine: str | None,
    preset_voice_id: str | None,
    design_prompt: str | None,
    default_engine: str | None,
) -> str | None:
    if voice_type == "preset":
        if not preset_engine or not preset_voice_id:
            return "Preset profiles require both preset_engine and preset_voice_id"
        if default_engine and default_engine != preset_engine:
            return "Preset profiles must use their preset_engine as default_engine"

        available_voice_ids = _get_preset_voice_ids(preset_engine)
        if available_voice_ids and preset_voice_id not in available_voice_ids:
            return f"Preset voice '{preset_voice_id}' is not valid for engine '{preset_engine}'"
        return None

    if voice_type == "designed":
        if not design_prompt or not design_prompt.strip():
            return "Designed profiles require a design_prompt"
        if preset_engine or preset_voice_id:
            return "Designed profiles cannot set preset_engine or preset_voice_id"
        return None

    if preset_engine or preset_voice_id:
        return "Cloned profiles cannot set preset_engine or preset_voice_id"
    if design_prompt:
        return "Cloned profiles cannot set design_prompt"
    if default_engine and default_engine not in CLONING_ENGINES:
        return f"Cloned profiles cannot use default engine '{default_engine}'"
    return None


def validate_profile_engine(profile, engine: str) -> None:
    voice_type = getattr(profile, "voice_type", None) or "cloned"

    if voice_type == "preset":
        preset_engine = getattr(profile, "preset_engine", None)
        preset_voice_id = getattr(profile, "preset_voice_id", None)
        if not preset_engine or not preset_voice_id:
            raise ValueError(f"Preset profile {profile.id} is missing preset engine metadata")
        if preset_engine != engine:
            raise ValueError(f"Preset profile {profile.id} only supports engine '{preset_engine}', not '{engine}'")
        return

    if voice_type == "designed":
        design_prompt = getattr(profile, "design_prompt", None)
        if not design_prompt or not design_prompt.strip():
            raise ValueError(f"Designed profile {profile.id} is missing design_prompt")
        return

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")


async def create_profile(
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse:
    """
    Create a new voice profile.

    Args:
        data: Profile creation data
        db: Database session

    Returns:
        Created profile

    Raises:
        ValueError: If a profile with the same name already exists
    """
    existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
    if existing_profile:
        raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    # Auto-set default_engine for preset profiles
    default_engine = data.default_engine
    voice_type = data.voice_type or "cloned"
    if voice_type == "preset" and data.preset_engine and not default_engine:
        default_engine = data.preset_engine

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
    )
    if validation_error:
        raise ValueError(validation_error)

    profile_id = str(uuid.uuid4())
    expected_fields = {
        "name": data.name,
        "description": data.description,
        "language": data.language,
        "voice_type": voice_type,
        "preset_engine": data.preset_engine,
        "preset_voice_id": data.preset_voice_id,
        "design_prompt": data.design_prompt,
        "default_engine": default_engine,
        "personality": data.personality,
    }
    db_profile = DBVoiceProfile(
        id=profile_id,
        name=data.name,
        description=data.description,
        language=data.language,
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
        personality=data.personality,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(db_profile)
    try:
        db.commit()
        db.refresh(db_profile)
        response = _profile_to_response(db_profile)
    except BaseException as exc:
        try:
            db.rollback()
        except BaseException:
            logger.error("Profile creation rollback failed", exc_info=True)
        try:
            response = _durable_created_profile_response(
                db,
                profile_id=profile_id,
                expected_fields=expected_fields,
            )
        except BaseException as reconciliation_error:
            raise exc from reconciliation_error
        if response is None:
            if isinstance(exc, IntegrityError):
                raise ValueError(
                    f"A profile with the name '{data.name}' already exists. Please choose a different name."
                ) from exc
            raise

    profile_dir = config.get_profiles_dir() / profile_id
    try:
        profile_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        profile_stat = profile_dir.lstat()
        if not stat.S_ISDIR(profile_stat.st_mode) or stat.S_ISLNK(profile_stat.st_mode):
            raise OSError("Profile storage is not a real directory")
        if os.name == "posix":
            os.chmod(profile_dir, 0o700, follow_symlinks=False)
    except OSError:
        # The durable profile remains usable and sample upload will retry the
        # directory creation with its stricter managed-path validation. Do not
        # turn a committed random-ID create into an apparent failure/duplicate.
        logger.warning("Deferred storage creation for committed profile %s", profile_id, exc_info=True)

    return response


async def add_profile_sample(
    profile_id: str,
    audio_path: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse:
    """
    Add a sample to a voice profile.

    Args:
        profile_id: Profile ID
        audio_path: Path to temporary audio file
        reference_text: Transcript of audio
        db: Database session

    Returns:
        Created sample
    """
    reference_text = _validate_profile_sample_reference_text(reference_text)
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")
    if not _PROFILE_STORAGE_ID_RE.fullmatch(profile_id) or ".." in profile_id:
        raise ValueError("Profile ID is unsafe for managed sample storage")
    if (
        db.query(func.count(DBProfileSample.id)).filter(DBProfileSample.profile_id == profile_id).scalar()
        >= EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES
    ):
        raise ValueError(f"Voice profiles support at most {EXACT_VOICE_SNAPSHOT_MAX_PROFILE_SAMPLES} reference samples")

    # Validate and load audio in a single pass, off the event loop
    is_valid, error_msg, audio, sr = await run_blocking_operation_cancellation_safe(
        validate_and_load_reference_audio,
        audio_path,
    )
    if not is_valid:
        raise ValueError(f"Invalid reference audio: {error_msg}")

    sample_id = str(uuid.uuid4())
    profiles_dir = config.get_profiles_dir()
    profile_dir = profiles_dir / profile_id
    profile_dir.mkdir(mode=0o700, exist_ok=True)
    profile_relative = Path("profiles") / profile_id
    profile_dir_stat = deletion_journal.managed_entry_stat(profile_relative)
    if profile_dir_stat is None or not stat.S_ISDIR(profile_dir_stat.st_mode):
        raise ValueError("Profile sample directory is not a real managed directory")
    os.chmod(profile_dir, 0o700)

    dest_relative = profile_relative / f"{sample_id}.wav"
    pending_relative = profile_relative / f".voicebox-delete-sample-new-{sample_id}.wav"
    dest_path = config.get_data_dir() / dest_relative
    pending_path = config.get_data_dir() / pending_relative
    publish_intent: deletion_journal.DeletionIntent | None = None
    published = False
    db_sample: DBProfileSample | None = None
    sample_response: ProfileSampleResponse | None = None
    pending_fd: int | None = None

    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        pending_fd = os.open(pending_path, flags, 0o600)
        os.fsync(pending_fd)
        pending_stat = os.fstat(pending_fd)
        if not stat.S_ISREG(pending_stat.st_mode) or pending_stat.st_nlink != 1:
            raise ValueError("Profile sample staging path is not a private regular file")

        publish_intent = deletion_journal.prepare_deletion_intent(
            kind=deletion_journal.PROFILE_SAMPLE,
            original=dest_relative,
            staged=pending_relative,
            entry_stat=pending_stat,
            owner_id=profile_id,
        )
        try:
            await run_blocking_operation_cancellation_safe(
                _populate_profile_sample_audio,
                pending_fd,
                audio,
                sr,
                publish_intent,
            )
        finally:
            descriptor_to_close = pending_fd
            pending_fd = None
            os.close(descriptor_to_close)

        pending_stat = deletion_journal.managed_entry_stat(pending_relative)
        if pending_stat is None or not deletion_journal.entry_matches_intent(
            publish_intent,
            pending_stat,
        ):
            raise OSError("Journaled profile-sample WAV was replaced before publication")

        for attempt in range(_SAMPLE_ORDINAL_RETRIES):
            # Validation and audio persistence both await worker threads. End
            # the old read transaction and force a fresh parent lookup before
            # publishing anything that the database may own.
            db.rollback()
            db.expire_all()
            profile = db.query(DBVoiceProfile).filter_by(id=profile_id).populate_existing().one_or_none()
            if profile is None:
                raise ValueError(f"Profile {profile_id} not found")

            # Flush the parent update first. On SQLite this acquires the writer
            # lock and makes a concurrent profile delete either precede this
            # revalidation or wait until the sample transaction is committed.
            profile.updated_at = datetime.utcnow()
            db.flush()

            _validate_profile_sample_admission(
                profile_id,
                reference_text,
                pending_stat.st_size,
                db,
            )

            if not published:
                if deletion_journal.managed_entry_stat(dest_relative) is not None:
                    raise FileExistsError("Profile sample destination already exists")
                deletion_journal.rename_managed_entry(
                    pending_relative,
                    dest_relative,
                )
                published = True
            stored_audio_path = config.to_storage_path(dest_path)
            sample_ordinal = _next_profile_sample_ordinal(profile_id, db)
            db_sample = DBProfileSample(
                id=sample_id,
                profile_id=profile_id,
                ordinal=sample_ordinal,
                audio_path=stored_audio_path,
                reference_text=reference_text,
            )
            db.add(db_sample)

            try:
                db.commit()
                db.refresh(db_sample)
                sample_response = ProfileSampleResponse.model_validate(db_sample)
                break
            except BaseException as exc:
                try:
                    db.rollback()
                except BaseException:
                    logger.error("Profile-sample addition rollback failed", exc_info=True)

                durable_response = _durable_profile_sample_response(
                    db,
                    sample_id=sample_id,
                    profile_id=profile_id,
                    ordinal=sample_ordinal,
                    audio_path=stored_audio_path,
                    reference_text=reference_text,
                )
                if durable_response is not None:
                    sample_response = durable_response
                    break

                if not isinstance(exc, IntegrityError):
                    raise
                ordinal_conflict = _SAMPLE_ORDINAL_CONFLICT in str(exc.orig)
                if not ordinal_conflict or attempt == _SAMPLE_ORDINAL_RETRIES - 1:
                    raise

                # A driver can report an error after its commit became
                # durable. Retry only when an independent session proves this
                # path has no committed owner.
                try:
                    with deletion_journal.durable_reconciliation_session(db) as durable_db:
                        if deletion_journal.database_owns_managed_path(
                            dest_relative,
                            durable_db,
                        ):
                            raise
                except IntegrityError:
                    raise
                except BaseException:
                    # Losing the ability to establish durable ownership makes
                    # the outcome ambiguous; retain the file and its intent.
                    raise exc from None

        if sample_response is None:
            raise RuntimeError("Profile sample transaction did not run")
    except BaseException:
        if pending_fd is not None:
            descriptor_to_close = pending_fd
            pending_fd = None
            os.close(descriptor_to_close)
        try:
            db.rollback()
        except BaseException:
            logger.error("Profile-sample addition rollback failed", exc_info=True)

        if publish_intent is not None:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    profile_exists = (
                        durable_db.query(DBVoiceProfile.id).filter_by(id=profile_id).one_or_none() is not None
                    )
                    audio_owned = deletion_journal.database_owns_managed_path(
                        dest_relative,
                        durable_db,
                    )
                    try:
                        profile_storage_stat = deletion_journal.managed_entry_stat(
                            profile_relative,
                        )
                    except FileNotFoundError:
                        profile_storage_stat = None
                    if profile_storage_stat is None:
                        if profile_exists or audio_owned:
                            raise RuntimeError("Profile sample storage disappeared during database reconciliation")
                        deletion_journal.finish_deletion_intent(publish_intent)
                    else:
                        deletion_journal.reconcile_deletion_intent(
                            publish_intent,
                            durable_db,
                        )
                if not profile_exists and not audio_owned:
                    try:
                        profile_dir.rmdir()
                        _fsync_directory(profiles_dir)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        if cleanup_error.errno not in {
                            errno.EEXIST,
                            errno.ENOTEMPTY,
                            errno.ENOTDIR,
                        }:
                            logger.warning(
                                "Could not durably remove an empty deleted-profile directory",
                                exc_info=True,
                            )
            except BaseException:
                logger.warning(
                    "Retaining an interrupted profile-sample publish intent for startup recovery",
                    exc_info=True,
                )
        else:
            try:
                deletion_journal.discard_managed_entry(pending_relative)
            except (FileNotFoundError, OSError):
                logger.warning("Deferred cleanup of an interrupted profile-sample write")
        try:
            # A commit-then-raise outcome may have durably changed the sample
            # set. Conservative invalidation is harmless when it did not.
            clear_profile_cache(profile_id)
        except BaseException:
            logger.warning("Deferred profile cache invalidation after sample-add failure")
        raise

    try:
        deletion_journal.finish_deletion_intent(publish_intent)
    except OSError:
        logger.warning("Deferred cleanup of a committed profile-sample publish intent")

    # Invalidate combined audio cache for this profile
    # Since a new sample was added, any cached combined audio is now stale
    try:
        clear_profile_cache(profile_id)
    except BaseException:
        # Once the database row and WAV are durable, cache cleanup must not
        # turn an acknowledged success into a retry that duplicates the voice.
        logger.warning("Deferred profile cache invalidation after sample-add success", exc_info=True)

    return sample_response


async def get_profile(
    profile_id: str,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Get a voice profile by ID.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        Profile or None if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    return _profile_to_response(profile)


def get_profile_orm_by_name_or_id(
    name_or_id: str,
    db: Session,
) -> DBVoiceProfile | None:
    """Resolve a profile from a user-supplied string that may be either id or name.

    Id is tried first (fast path, matches UUIDs). Name fallback is
    case-insensitive so agents can say "Morgan" regardless of casing.
    """
    if not name_or_id:
        return None
    row = db.query(DBVoiceProfile).filter(DBVoiceProfile.id == name_or_id).first()
    if row is not None:
        return row
    return db.query(DBVoiceProfile).filter(func.lower(DBVoiceProfile.name) == name_or_id.lower()).first()


async def get_profile_samples(
    profile_id: str,
    db: Session,
) -> list[ProfileSampleResponse]:
    """
    Get all samples for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        List of samples
    """
    samples = _ordered_profile_samples(profile_id, db)
    return [ProfileSampleResponse.model_validate(s) for s in samples]


async def list_profiles(db: Session) -> list[VoiceProfileResponse]:
    """
    List all voice profiles with generation and sample counts.

    Args:
        db: Database session

    Returns:
        List of profiles
    """
    profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()

    if not profiles:
        return []

    # Batch-fetch generation counts
    gen_counts_rows = (
        db.query(DBGeneration.profile_id, func.count(DBGeneration.id)).group_by(DBGeneration.profile_id).all()
    )
    gen_counts = {row[0]: row[1] for row in gen_counts_rows}

    # Batch-fetch sample counts
    sample_counts_rows = (
        db.query(DBProfileSample.profile_id, func.count(DBProfileSample.id)).group_by(DBProfileSample.profile_id).all()
    )
    sample_counts = {row[0]: row[1] for row in sample_counts_rows}

    return [
        _profile_to_response(
            p,
            generation_count=gen_counts.get(p.id, 0),
            sample_count=sample_counts.get(p.id, 0),
        )
        for p in profiles
    ]


async def update_profile(
    profile_id: str,
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Update a voice profile.

    Args:
        profile_id: Profile ID
        data: Updated profile data
        db: Database session

    Returns:
        Updated profile or None if not found

    Raises:
        ValueError: If a profile with the same name already exists (different profile)
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    if profile.name != data.name:
        existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
        if existing_profile:
            raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    preset_engine = getattr(profile, "preset_engine", None)
    preset_voice_id = getattr(profile, "preset_voice_id", None)
    design_prompt = getattr(profile, "design_prompt", None)
    default_engine = (
        data.default_engine if data.default_engine is not None else getattr(profile, "default_engine", None)
    )

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=preset_engine,
        preset_voice_id=preset_voice_id,
        design_prompt=design_prompt,
        default_engine=default_engine,
    )
    if validation_error:
        raise ValueError(validation_error)

    profile.name = data.name
    profile.description = data.description
    profile.language = data.language
    profile.personality = data.personality
    if data.default_engine is not None:
        profile.default_engine = data.default_engine or None  # empty string → NULL
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


@dataclass(frozen=True)
class _StagedProfileStorage:
    original: Path
    staged: Path
    intent: deletion_journal.DeletionIntent


@dataclass(frozen=True)
class _StagedProfileSample:
    original: Path
    staged: Path
    intent: deletion_journal.DeletionIntent


@dataclass(frozen=True)
class _StagedProfileAvatar:
    original: Path
    staged: Path
    intent: deletion_journal.DeletionIntent


def _stage_profile_storage(profile_id: str) -> _StagedProfileStorage | None:
    """Atomically hide one profile directory until its DB transaction commits."""
    if not _PROFILE_STORAGE_ID_RE.fullmatch(profile_id) or ".." in profile_id:
        raise ValueError("Profile ID is unsafe for managed storage cleanup")
    profile_root = config.get_profiles_dir()
    profile_dir = profile_root / profile_id
    original_relative = Path("profiles") / profile_id
    profile_stat = deletion_journal.managed_entry_stat(original_relative)
    if profile_stat is None:
        return None
    if not (stat.S_ISDIR(profile_stat.st_mode) or stat.S_ISLNK(profile_stat.st_mode)):
        raise ValueError("Profile storage entry is not a directory")
    staged_dir = profile_root / f".voicebox-delete-{profile_id}-{uuid.uuid4().hex}"
    staged_relative = Path("profiles") / staged_dir.name
    intent = deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.PROFILE_STORAGE,
        original=original_relative,
        staged=staged_relative,
        entry_stat=profile_stat,
        owner_id=profile_id,
    )
    try:
        deletion_journal.rename_managed_entry(original_relative, staged_relative)
    except BaseException:
        if deletion_journal.managed_entry_stat(staged_relative) is not None:
            deletion_journal.rename_managed_entry(staged_relative, original_relative)
        restored_stat = deletion_journal.managed_entry_stat(original_relative)
        if restored_stat is not None and deletion_journal.entry_matches_intent(intent, restored_stat):
            deletion_journal.finish_deletion_intent(intent)
        raise
    return _StagedProfileStorage(profile_dir, staged_dir, intent)


def _restore_profile_storage(staged: _StagedProfileStorage | None) -> None:
    if staged is None:
        return
    profile_dir, staged_dir = staged.original, staged.staged
    staged_relative = Path("profiles") / staged_dir.name
    original_relative = Path("profiles") / profile_dir.name
    if deletion_journal.managed_entry_stat(staged_relative) is None:
        return
    if deletion_journal.managed_entry_stat(original_relative) is not None:
        raise RuntimeError("Cannot restore profile storage because its original path was replaced")
    deletion_journal.rename_managed_entry(staged_relative, original_relative)
    deletion_journal.finish_deletion_intent(staged.intent)


def _discard_profile_storage(staged: _StagedProfileStorage | None) -> None:
    if staged is None:
        return
    staged_dir = staged.staged
    try:
        staged_relative = Path("profiles") / staged_dir.name
        staged_stat = deletion_journal.managed_entry_stat(staged_relative)
        if staged_stat is not None and not (stat.S_ISLNK(staged_stat.st_mode) or stat.S_ISDIR(staged_stat.st_mode)):
            logger.warning("Deferred cleanup of unexpected staged profile storage")
            return
        deletion_journal.discard_managed_entry(staged_relative)
        deletion_journal.finish_deletion_intent(staged.intent)
    except OSError as exc:
        logger.warning("Deferred cleanup of committed profile storage: %s", exc)


def _reconcile_profile_storage(
    staged: _StagedProfileStorage | None,
    db: Session,
) -> None:
    if staged is None:
        return
    try:
        deletion_journal.reconcile_deletion_intent(staged.intent, db)
    except BaseException:
        logger.error(
            "Retaining one profile-storage deletion intent for startup recovery",
            exc_info=True,
        )


def _managed_profile_sample_relative_path(
    stored_path: str | None,
    profile_id: str,
) -> Path | None:
    """Return a lexical sample path below this profile, or leave it unmanaged."""
    if not stored_path:
        return None
    if not _PROFILE_STORAGE_ID_RE.fullmatch(profile_id) or ".." in profile_id:
        raise ValueError("Profile ID is unsafe for managed sample cleanup")

    relative = _managed_data_relative_path(stored_path)
    if relative is None:
        logger.warning("Refusing to delete profile sample audio outside managed storage")
        return None
    if len(relative.parts) < 3 or relative.parts[0] != "profiles" or relative.parts[1] != profile_id:
        logger.warning("Refusing to delete profile sample audio outside its profile")
        return None
    return relative


def _managed_data_relative_path(stored_path: str | None) -> Path | None:
    """Canonicalize a stored path lexically without following filesystem links."""
    return config.managed_storage_relative_path(stored_path)


def _profile_storage_has_surviving_owner(profile_id: str, db: Session) -> bool:
    prefix = Path("profiles") / profile_id
    return deletion_journal.database_owns_below(prefix, db)


def _stage_profile_sample_audio(
    stored_path: str | None,
    profile_id: str,
) -> _StagedProfileSample | None:
    """Hide one managed sample until its database deletion commits."""
    relative = _managed_profile_sample_relative_path(stored_path, profile_id)
    if relative is None:
        return None

    entry_stat = deletion_journal.managed_entry_stat(relative)
    if entry_stat is None:
        return None
    if not (stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode)):
        raise ValueError("Refusing to delete a non-file profile sample entry")
    staged_name = f".voicebox-delete-sample-{uuid.uuid4().hex}.tmp"
    hidden = relative.with_name(staged_name)
    intent = deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.PROFILE_SAMPLE,
        original=relative,
        staged=hidden,
        entry_stat=entry_stat,
        owner_id=profile_id,
    )
    try:
        deletion_journal.rename_managed_entry(relative, hidden)
    except BaseException:
        if deletion_journal.managed_entry_stat(hidden) is not None:
            deletion_journal.rename_managed_entry(hidden, relative)
        restored_stat = deletion_journal.managed_entry_stat(relative)
        if restored_stat is not None and deletion_journal.entry_matches_intent(intent, restored_stat):
            deletion_journal.finish_deletion_intent(intent)
        raise
    return _StagedProfileSample(relative, hidden, intent)


def _restore_profile_sample_audio(staged: _StagedProfileSample | None) -> None:
    if staged is None:
        return
    original, hidden = staged.original, staged.staged
    if deletion_journal.managed_entry_stat(hidden) is None:
        return
    if deletion_journal.managed_entry_stat(original) is not None:
        raise RuntimeError("Cannot restore profile sample because its original path was replaced")
    deletion_journal.rename_managed_entry(hidden, original)
    deletion_journal.finish_deletion_intent(staged.intent)


def _discard_profile_sample_audio(staged: _StagedProfileSample | None) -> None:
    if staged is None:
        return
    hidden = staged.staged
    try:
        deletion_journal.discard_managed_entry(hidden)
        deletion_journal.finish_deletion_intent(staged.intent)
    except OSError as exc:
        logger.warning("Deferred cleanup of committed profile sample: %s", exc)


def _reconcile_profile_sample_audio(
    staged: _StagedProfileSample | None,
    db: Session,
) -> None:
    if staged is None:
        return
    try:
        deletion_journal.reconcile_deletion_intent(staged.intent, db)
    except BaseException:
        logger.error(
            "Retaining one profile-sample deletion intent for startup recovery",
            exc_info=True,
        )


def _managed_profile_avatar_relative_path(
    stored_path: str | None,
    profile_id: str,
) -> Path | None:
    """Return a managed avatar path owned by exactly this profile."""
    if not stored_path:
        return None
    if not _PROFILE_STORAGE_ID_RE.fullmatch(profile_id) or ".." in profile_id:
        raise ValueError("Profile ID is unsafe for managed avatar cleanup")
    relative = _managed_data_relative_path(stored_path)
    if relative is None or len(relative.parts) != 3 or relative.parts[:2] != ("profiles", profile_id):
        logger.warning("Refusing to delete profile avatar outside its profile")
        return None
    return relative


def _stage_profile_avatar(
    stored_path: str | None,
    profile_id: str,
) -> _StagedProfileAvatar | None:
    """Hide one managed avatar behind a durable database-aware intent."""
    relative = _managed_profile_avatar_relative_path(stored_path, profile_id)
    if relative is None:
        return None
    entry_stat = deletion_journal.managed_entry_stat(relative)
    if entry_stat is None:
        return None
    if not (stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode)):
        raise ValueError("Refusing to delete a non-file profile avatar entry")
    hidden = relative.with_name(f".voicebox-delete-avatar-{uuid.uuid4().hex}.tmp")
    intent = deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.PROFILE_AVATAR,
        original=relative,
        staged=hidden,
        entry_stat=entry_stat,
        owner_id=profile_id,
    )
    try:
        deletion_journal.rename_managed_entry(relative, hidden)
    except BaseException:
        if deletion_journal.managed_entry_stat(hidden) is not None:
            deletion_journal.rename_managed_entry(hidden, relative)
        restored_stat = deletion_journal.managed_entry_stat(relative)
        if restored_stat is not None and deletion_journal.entry_matches_intent(
            intent,
            restored_stat,
        ):
            deletion_journal.finish_deletion_intent(intent)
        raise
    return _StagedProfileAvatar(relative, hidden, intent)


def _restore_profile_avatar(staged: _StagedProfileAvatar | None) -> None:
    if staged is None or deletion_journal.managed_entry_stat(staged.staged) is None:
        return
    if deletion_journal.managed_entry_stat(staged.original) is not None:
        raise RuntimeError("Cannot restore profile avatar because its path was replaced")
    deletion_journal.rename_managed_entry(staged.staged, staged.original)
    deletion_journal.finish_deletion_intent(staged.intent)


def _discard_profile_avatar(staged: _StagedProfileAvatar | None) -> None:
    if staged is None:
        return
    try:
        deletion_journal.discard_managed_entry(staged.staged)
        deletion_journal.finish_deletion_intent(staged.intent)
    except OSError as exc:
        logger.warning("Deferred cleanup of committed profile avatar: %s", exc)


def _reconcile_profile_avatar(
    staged: _StagedProfileAvatar | None,
    db: Session,
) -> None:
    if staged is None:
        return
    try:
        deletion_journal.reconcile_deletion_intent(staged.intent, db)
    except BaseException:
        logger.error(
            "Retaining one profile-avatar deletion intent for startup recovery",
            exc_info=True,
        )


async def delete_profile(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete a voice profile, its managed samples, and generation history.

    Content-addressed exact-voice snapshots are intentionally retained because
    another profile or active request may share the same immutable bytes.  They
    require a separate reference-aware garbage collector.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return False
    if not _PROFILE_STORAGE_ID_RE.fullmatch(profile_id) or ".." in profile_id:
        raise ValueError("Profile ID is unsafe for managed storage cleanup")

    generations = db.query(DBGeneration).filter_by(profile_id=profile_id).all()
    from .task_queue import generation_job_is_active

    if any(
        generation_job_is_active(generation.id) or (generation.status or "completed") in _ACTIVE_GENERATION_STATUSES
        for generation in generations
    ):
        raise ProfileGenerationActiveError(
            "Cancel every active generation for this profile and wait for a terminal state before deleting it"
        )

    # Delete every managed generation/version file and dependent row first. SQLite
    # foreign keys are not enabled in existing installs, so profile deletion cannot
    # rely on cascading behavior.
    from . import history

    profile_samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    stored_avatar_path = profile.avatar_path
    staged_audio: list[history._StagedGenerationAudio] = []
    staged_profile: _StagedProfileStorage | None = None
    staged_samples: list[_StagedProfileSample] = []
    staged_avatars: list[_StagedProfileAvatar] = []
    exact_checkpoint_candidates: set[str] = set()
    try:
        await history.delete_generations_by_profile(
            profile_id,
            db,
            commit=False,
            staged_audio=staged_audio,
            exact_checkpoint_candidates=exact_checkpoint_candidates,
        )
        db.query(DBProfileSample).filter_by(profile_id=profile_id).delete()
        db.query(DBProfileChannelMapping).filter_by(profile_id=profile_id).delete()
        db.query(DBMCPClientBinding).filter_by(profile_id=profile_id).update(
            {"profile_id": None},
            synchronize_session=False,
        )
        db.delete(profile)
        db.flush()
        if _profile_storage_has_surviving_owner(profile_id, db):
            staged_relatives: set[Path] = set()
            for sample in profile_samples:
                relative = _managed_profile_sample_relative_path(
                    sample.audio_path,
                    profile_id,
                )
                if (
                    relative is None
                    or relative in staged_relatives
                    or deletion_journal.database_owns_managed_path(relative, db)
                ):
                    continue
                staged = _stage_profile_sample_audio(sample.audio_path, profile_id)
                if staged is not None:
                    staged_samples.append(staged)
                    staged_relatives.add(relative)
            avatar_relative = _managed_profile_avatar_relative_path(
                stored_avatar_path,
                profile_id,
            )
            if (
                avatar_relative is not None
                and avatar_relative not in staged_relatives
                and not deletion_journal.database_owns_managed_path(
                    avatar_relative,
                    db,
                )
            ):
                staged_avatar = _stage_profile_avatar(
                    stored_avatar_path,
                    profile_id,
                )
                if staged_avatar is not None:
                    staged_avatars.append(staged_avatar)
            logger.warning("Retaining only profile storage still referenced by another database row")
        else:
            staged_profile = _stage_profile_storage(profile_id)
        db.commit()
    except BaseException:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Profile deletion rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_profile_storage(staged_profile, db)
            for staged in staged_samples:
                _reconcile_profile_sample_audio(staged, db)
            for staged in staged_avatars:
                _reconcile_profile_avatar(staged, db)
            history._reconcile_staged_generation_audio(staged_audio, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_profile_storage(staged_profile, durable_db)
                    for staged in staged_samples:
                        _reconcile_profile_sample_audio(staged, durable_db)
                    for staged in staged_avatars:
                        _reconcile_profile_avatar(staged, durable_db)
                    history._reconcile_staged_generation_audio(
                        staged_audio,
                        durable_db,
                    )
            except BaseException:
                logger.error("Deferred profile deletion reconciliation", exc_info=True)
            raise rollback_error from None
        raise

    history._discard_staged_generation_audio(staged_audio)
    for staged in staged_samples:
        _discard_profile_sample_audio(staged)
    for staged in staged_avatars:
        _discard_profile_avatar(staged)
    _discard_profile_storage(staged_profile)
    history._reclaim_exact_checkpoint_candidates(
        exact_checkpoint_candidates,
        db,
    )

    # Clean up combined audio cache files for this profile
    clear_profile_cache(profile_id)

    return True


async def delete_profile_sample(
    sample_id: str,
    db: Session,
) -> bool:
    """
    Delete a profile sample.

    Args:
        sample_id: Sample ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return False

    # Store profile_id before deleting
    profile_id = sample.profile_id

    sample_relative = _managed_profile_sample_relative_path(
        sample.audio_path,
        profile_id,
    )
    stored_audio_path = sample.audio_path
    staged_audio: _StagedProfileSample | None = None
    try:
        db.delete(sample)
        db.flush()
        if sample_relative is not None and not deletion_journal.database_owns_managed_path(
            sample_relative,
            db,
        ):
            staged_audio = _stage_profile_sample_audio(
                stored_audio_path,
                profile_id,
            )
        db.commit()
    except BaseException:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Profile-sample deletion rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_profile_sample_audio(staged_audio, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_profile_sample_audio(staged_audio, durable_db)
            except BaseException:
                logger.error("Deferred profile-sample reconciliation", exc_info=True)
            raise rollback_error from None
        raise

    _discard_profile_sample_audio(staged_audio)

    # Invalidate combined audio cache for this profile
    # Since the sample set changed, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return True


async def update_profile_sample(
    sample_id: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse | None:
    """
    Update a profile sample's reference text.

    Args:
        sample_id: Sample ID
        reference_text: Updated reference text
        db: Database session

    Returns:
        Updated sample or None if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return None

    # Store profile_id before updating
    profile_id = sample.profile_id

    sample.reference_text = reference_text
    db.commit()
    db.refresh(sample)

    # Invalidate combined audio cache for this profile
    # Since the reference text changed, cache keys and combined text are now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(sample)


async def create_voice_prompt_for_profile(
    profile_id: str,
    db: Session,
    use_cache: bool = True,
    engine: str = "qwen",
) -> dict:
    """
    Create a voice prompt from a profile.

    For cloned profiles: combines all audio samples into a voice prompt.
    For preset profiles: returns the engine-specific preset voice reference.
    For designed profiles: returns the text design prompt (future).

    Args:
        profile_id: Profile ID
        db: Database session
        use_cache: Whether to use cached prompts
        engine: TTS engine to create prompt for

    Returns:
        Voice prompt dictionary
    """
    from ..backends import get_tts_backend_for_engine

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile not found: {profile_id}")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    validate_profile_engine(profile, engine)

    # ── Preset profiles: return engine-specific voice reference ──
    if voice_type == "preset":
        if not profile.preset_engine or not profile.preset_voice_id:
            raise ValueError(f"Preset profile {profile_id} is missing preset engine metadata")
        if profile.preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile_id} only supports engine '{profile.preset_engine}', not '{engine}'"
            )
        return {
            "voice_type": "preset",
            "preset_engine": profile.preset_engine,
            "preset_voice_id": profile.preset_voice_id,
        }

    # ── Designed profiles: return text description (future) ──
    if voice_type == "designed":
        if not profile.design_prompt or not profile.design_prompt.strip():
            raise ValueError(f"Designed profile {profile_id} is missing design_prompt")
        return {
            "voice_type": "designed",
            "design_prompt": profile.design_prompt,
        }

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")

    # ── Cloned profiles: create from audio samples ──
    samples = _ordered_profile_samples(profile_id, db)

    if not samples:
        raise ValueError(f"No samples found for profile {profile_id}")
    _validate_exact_sample_count(samples)

    audio_paths: list[str] = []
    reference_texts: list[str] = []
    total_audio_bytes = 0
    for sample in samples:
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        identity, _digest = _stable_regular_file_identity_and_sha256(sample_audio_path)
        total_audio_bytes += identity[2]
        audio_paths.append(str(sample_audio_path))
        reference_texts.append(_validate_exact_reference_text(str(sample.reference_text or "")))
    _validate_exact_total_reference_text(reference_texts)
    if total_audio_bytes > EXACT_VOICE_SNAPSHOT_MAX_AGGREGATE_SAMPLE_BYTES:
        raise ValueError("Cloned voice samples exceed the safe aggregate audio size limit")

    tts_model = get_tts_backend_for_engine(engine)

    if len(samples) == 1:
        sample = samples[0]
        voice_prompt, _ = await tts_model.create_voice_prompt(
            audio_paths[0],
            reference_texts[0],
            use_cache=use_cache,
        )
        return voice_prompt

    combined_audio, combined_text = await tts_model.combine_voice_prompts(
        audio_paths,
        reference_texts,
    )

    # Save combined audio to cache directory (persistent)
    # Create a hash of sample IDs to identify this specific combination
    import hashlib

    sample_ids_str = "-".join(sorted([s.id for s in samples]))
    combination_hash = hashlib.md5(sample_ids_str.encode()).hexdigest()[:12]

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_path = cache_dir / f"combined_{profile_id}_{combination_hash}.wav"
    combined_size = getattr(combined_audio, "size", None)
    if isinstance(combined_size, bool) or not isinstance(combined_size, int) or combined_size < 0:
        raise ValueError("Combined voice prompt has an invalid audio shape")
    estimated_wav_bytes = combined_size * 2 + 4096
    if estimated_wav_bytes > EXACT_VOICE_SNAPSHOT_MAX_DERIVED_AUDIO_BYTES:
        raise ValueError("Combined voice prompt exceeds the safe derived-audio size limit")
    try:
        combined_reservation = reserve_disk_space(
            cache_dir,
            estimated_wav_bytes,
            min_free_bytes=EXACT_VOICE_SNAPSHOT_MIN_FREE_BYTES,
        )
    except DiskSpaceReservationError as exc:
        raise ValueError("Insufficient reserved free space for a combined voice prompt") from exc
    try:
        save_audio(combined_audio, str(combined_path), 24000)
    finally:
        combined_reservation.release()

    voice_prompt, _ = await tts_model.create_voice_prompt(
        str(combined_path),
        combined_text,
        use_cache=use_cache,
    )
    return voice_prompt


async def upload_avatar(
    profile_id: str,
    image_path: str,
    db: Session,
) -> VoiceProfileResponse:
    """
    Upload and process avatar image for a profile.

    Args:
        profile_id: Profile ID
        image_path: Path to uploaded image file
        db: Database session

    Returns:
        Updated profile
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(error_msg)

    old_avatar_path = profile.avatar_path

    # Determine file extension from uploaded file
    from PIL import Image

    with Image.open(image_path) as img:
        # Normalize JPEG variants (MPO is multi-picture format from some cameras)
        img_format = img.format
        if img_format in ("MPO", "JPG"):
            img_format = "JPEG"

        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
        ext = ext_map.get(img_format, ".png")

    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_dir_stat = deletion_journal.managed_entry_stat(Path("profiles") / profile_id)
    if profile_dir_stat is None or not stat.S_ISDIR(profile_dir_stat.st_mode):
        raise ValueError("Profile avatar directory is not a real managed directory")

    upload_id = uuid.uuid4().hex
    output_relative = Path("profiles") / profile_id / f"avatar-{upload_id}{ext}"
    pending_relative = output_relative.with_name(f".voicebox-delete-avatar-new-{upload_id}{ext}")
    output_path = config.get_data_dir() / output_relative
    pending_path = config.get_data_dir() / pending_relative
    publish_intent: deletion_journal.DeletionIntent | None = None

    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        pending_fd = os.open(pending_path, flags, 0o600)
        try:
            pending_stat = os.fstat(pending_fd)
            publish_intent = deletion_journal.prepare_deletion_intent(
                kind=deletion_journal.PROFILE_AVATAR,
                original=output_relative,
                staged=pending_relative,
                entry_stat=pending_stat,
                owner_id=profile_id,
            )
        finally:
            os.close(pending_fd)
        process_avatar(image_path, str(pending_path))
        pending_fd = os.open(pending_path, os.O_RDONLY)
        try:
            os.fsync(pending_fd)
        finally:
            os.close(pending_fd)
        pending_stat = deletion_journal.managed_entry_stat(pending_relative)
        if pending_stat is None or not stat.S_ISREG(pending_stat.st_mode):
            raise ValueError("Processed profile avatar is not a regular file")
        if not deletion_journal.entry_matches_intent(publish_intent, pending_stat):
            raise ValueError("Processed profile avatar replaced its journaled inode")
        deletion_journal.rename_managed_entry(pending_relative, output_relative)
        profile.avatar_path = config.to_storage_path(output_path)
        profile.updated_at = datetime.utcnow()
        db.commit()
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Profile-avatar upload rollback failed", exc_info=True)
        if publish_intent is None:
            try:
                deletion_journal.discard_managed_entry(pending_relative)
            except OSError:
                logger.warning("Deferred cleanup of an interrupted avatar upload")
        else:
            # A commit error can be ambiguous. Reconcile against an
            # independent durable view when rollback itself failed; the
            # request Session may still expose uncommitted ownership.
            try:
                if rollback_error is None:
                    deletion_journal.reconcile_deletion_intent(
                        publish_intent,
                        db,
                    )
                else:
                    with deletion_journal.durable_reconciliation_session(db) as durable_db:
                        deletion_journal.reconcile_deletion_intent(
                            publish_intent,
                            durable_db,
                        )
            except BaseException:
                logger.warning(
                    "Retaining an interrupted avatar publish intent for startup recovery",
                    exc_info=True,
                )
        if rollback_error is not None:
            raise rollback_error from operation_error
        raise

    try:
        deletion_journal.finish_deletion_intent(publish_intent)
    except OSError:
        logger.warning("Deferred cleanup of a committed avatar publish intent")

    # The old avatar is retired only after the new pointer is committed. A
    # crash before this cleanup leaves an extra private file, never a broken
    # profile; a crash during it is recovered by the durable intent.
    old_relative = _managed_profile_avatar_relative_path(
        old_avatar_path,
        profile_id,
    )
    if (
        old_relative is not None
        and old_relative != output_relative
        and not deletion_journal.database_owns_managed_path(old_relative, db)
    ):
        try:
            _discard_profile_avatar(_stage_profile_avatar(old_avatar_path, profile_id))
        except BaseException:
            logger.warning("Deferred cleanup of the previous profile avatar", exc_info=True)

    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_avatar(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete avatar image for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found or no avatar
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile or not profile.avatar_path:
        return False

    stored_avatar = profile.avatar_path
    avatar_relative = _managed_profile_avatar_relative_path(
        stored_avatar,
        profile_id,
    )
    staged_avatar: _StagedProfileAvatar | None = None
    profile.avatar_path = None
    profile.updated_at = datetime.utcnow()
    try:
        db.flush()
        if avatar_relative is not None and not deletion_journal.database_owns_managed_path(
            avatar_relative,
            db,
        ):
            staged_avatar = _stage_profile_avatar(stored_avatar, profile_id)
        db.commit()
    except BaseException:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Profile-avatar deletion rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_profile_avatar(staged_avatar, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_profile_avatar(staged_avatar, durable_db)
            except BaseException:
                logger.error("Deferred profile-avatar reconciliation", exc_info=True)
            raise rollback_error from None
        raise

    _discard_profile_avatar(staged_avatar)

    return True
