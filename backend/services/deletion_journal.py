"""Crash recovery for database-backed deletion of managed Voicebox files.

SQLite commits and filesystem renames cannot share one atomic transaction.  A
small, fsynced intent file therefore records every managed entry before it is
hidden.  Startup can then use the committed database state to restore a file
that is still owned, or discard a file whose owning row was committed away.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from .. import config
from ..database import (
    Capture as DBCapture,
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)

logger = logging.getLogger(__name__)

JOURNAL_SCHEMA_VERSION = 1
GENERATION_AUDIO = "generation_audio"
GENERATION_AUDIO_PUBLICATION = "generation_audio_publication"
CAPTURE_AUDIO = "capture_audio"
PROFILE_AVATAR = "profile_avatar"
PROFILE_SAMPLE = "profile_sample"
PROFILE_STORAGE = "profile_storage"
_KINDS = frozenset(
    {
        CAPTURE_AUDIO,
        GENERATION_AUDIO,
        GENERATION_AUDIO_PUBLICATION,
        PROFILE_AVATAR,
        PROFILE_SAMPLE,
        PROFILE_STORAGE,
    }
)
_MAX_JOURNAL_BYTES = 16 * 1024


@dataclass(frozen=True)
class DeletionIntent:
    """One durable file-deletion intent."""

    journal_name: str
    kind: str
    original: Path
    staged: Path
    expected_dev: int
    expected_ino: int
    expected_type: int
    owner_id: str | None = None


@dataclass
class RecoveryReport:
    """Path-free aggregate startup recovery result."""

    restored: int = 0
    discarded: int = 0
    cleared: int = 0
    unresolved: int = 0
    malformed: int = 0


def _directory_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def secure_dir_fd_supported() -> bool:
    """Return whether this platform supports the no-follow descriptor protocol."""
    return os.name == "posix" and all(
        function in os.supports_dir_fd for function in (os.open, os.stat, os.rename, os.unlink)
    )


def _file_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    return flags


def _validate_relative_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Deletion journal paths must be safe relative paths")
    return path


def _validate_intent_paths(kind: str, original: Path, staged: Path, owner_id: str | None) -> None:
    if kind not in _KINDS:
        raise ValueError("Unsupported deletion journal kind")
    if original.parent != staged.parent or original == staged:
        raise ValueError("Staged deletion paths must share one parent")
    if not staged.name.startswith(".voicebox-delete-"):
        raise ValueError("Staged deletion path has an invalid name")
    if kind in {GENERATION_AUDIO, GENERATION_AUDIO_PUBLICATION} and original.parts[0] != "generations":
        raise ValueError("Generation deletion must stay below generations")
    if kind == CAPTURE_AUDIO and original.parts[0] != "captures":
        raise ValueError("Capture deletion must stay below captures")
    if kind in {PROFILE_AVATAR, PROFILE_SAMPLE, PROFILE_STORAGE} and original.parts[0] != "profiles":
        raise ValueError("Profile deletion must stay below profiles")
    if kind == PROFILE_SAMPLE and len(original.parts) < 3:
        raise ValueError("Profile sample deletion path is incomplete")
    if kind == PROFILE_AVATAR and (len(original.parts) != 3 or not owner_id or original.parts[1] != owner_id):
        raise ValueError("Profile avatar intent does not match its owner")
    if kind == CAPTURE_AUDIO and (len(original.parts) != 2 or not owner_id):
        raise ValueError("Capture audio intent does not match its owner")
    if kind == PROFILE_STORAGE and (len(original.parts) != 2 or not owner_id or original.name != owner_id):
        raise ValueError("Profile storage intent does not match its owner")


def _open_journal_dir() -> int:
    return os.open(config.get_deletion_journal_dir(), _directory_flags())


def _open_managed_parent(relative: Path) -> int:
    current_fd = os.open(config.get_data_dir(), _directory_flags())
    try:
        for component in relative.parts[:-1]:
            next_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _fallback_managed_path(relative: Path) -> Path:
    """Return a confined path while rejecting linked parents on non-POSIX hosts."""
    relative = _validate_relative_path(relative)
    root = config.get_data_dir()
    current = root
    root_stat = current.lstat()
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError("Configured Voicebox data root is not a real directory")
    for component in relative.parts[:-1]:
        current = current / component
        entry_stat = current.lstat()
        if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
            raise RuntimeError("Managed deletion parent is not a real directory")
    return root / relative


def _fsync_directory_path(path: Path) -> None:
    """Best-effort directory durability for platforms that permit directory fsync."""
    try:
        directory_fd = os.open(path, _directory_flags())
    except (NotImplementedError, OSError):
        return
    try:
        os.fsync(directory_fd)
    except OSError:
        # Windows does not expose a portable directory-flush primitive.  File
        # replaces remain atomic, and startup reconciliation is still safe.
        pass
    finally:
        os.close(directory_fd)


def managed_entry_stat(relative: str | Path) -> os.stat_result | None:
    """lstat one confined entry without following its final symbolic link."""
    safe = _validate_relative_path(relative)
    if secure_dir_fd_supported():
        parent_fd = _open_managed_parent(safe)
        try:
            try:
                return os.stat(safe.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None
        finally:
            os.close(parent_fd)
    try:
        return _fallback_managed_path(safe).lstat()
    except FileNotFoundError:
        return None


def rename_managed_entry(source: str | Path, destination: str | Path) -> None:
    """Atomically rename two confined entries in one managed directory."""
    source_path = _validate_relative_path(source)
    destination_path = _validate_relative_path(destination)
    if source_path.parent != destination_path.parent:
        raise ValueError("Managed rename must remain in one directory")
    if secure_dir_fd_supported():
        parent_fd = _open_managed_parent(source_path)
        try:
            os.rename(
                source_path.name,
                destination_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return
    source_absolute = _fallback_managed_path(source_path)
    destination_absolute = _fallback_managed_path(destination_path)
    os.rename(source_absolute, destination_absolute)
    _fsync_directory_path(source_absolute.parent)


def replace_managed_entry(source: str | Path, destination: str | Path) -> None:
    """Atomically replace one confined entry for a journaled publication."""
    source_path = _validate_relative_path(source)
    destination_path = _validate_relative_path(destination)
    if source_path.parent != destination_path.parent:
        raise ValueError("Managed replacement must remain in one directory")
    if secure_dir_fd_supported():
        parent_fd = _open_managed_parent(source_path)
        try:
            os.replace(
                source_path.name,
                destination_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return
    source_absolute = _fallback_managed_path(source_path)
    destination_absolute = _fallback_managed_path(destination_path)
    os.replace(source_absolute, destination_absolute)
    _fsync_directory_path(source_absolute.parent)


def discard_managed_entry(relative: str | Path) -> None:
    """Remove one confined staged file, link, or directory and flush its parent."""
    safe = _validate_relative_path(relative)
    entry_stat = managed_entry_stat(safe)
    if entry_stat is None:
        return
    if stat.S_ISDIR(entry_stat.st_mode):
        absolute = config.get_data_dir() / safe
        shutil.rmtree(absolute)
        _fsync_directory_path(absolute.parent)
        return
    if secure_dir_fd_supported():
        parent_fd = _open_managed_parent(safe)
        try:
            os.unlink(safe.name, dir_fd=parent_fd)
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return
    absolute = _fallback_managed_path(safe)
    absolute.unlink()
    _fsync_directory_path(absolute.parent)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(fd, payload[offset:])
        if written <= 0:
            raise OSError("Could not write deletion journal")
        offset += written


def prepare_deletion_intent(
    *,
    kind: str,
    original: str | Path,
    staged: str | Path,
    entry_stat: os.stat_result,
    owner_id: str | None = None,
) -> DeletionIntent:
    """Durably record an intent before the corresponding filesystem rename."""
    original_path = _validate_relative_path(original)
    staged_path = _validate_relative_path(staged)
    _validate_intent_paths(kind, original_path, staged_path, owner_id)
    operation_id = uuid.uuid4().hex
    journal_name = f"{operation_id}.json"
    temporary_name = f".tmp-{operation_id}"
    expected_type = stat.S_IFMT(entry_stat.st_mode)
    payload = json.dumps(
        {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "operation_id": operation_id,
            "kind": kind,
            "original": original_path.as_posix(),
            "staged": staged_path.as_posix(),
            "expected_dev": entry_stat.st_dev,
            "expected_ino": entry_stat.st_ino,
            "expected_type": expected_type,
            "owner_id": owner_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    if not secure_dir_fd_supported():
        journal_dir = config.get_deletion_journal_dir()
        temporary_path = journal_dir / temporary_name
        journal_path = journal_dir / journal_name
        file_fd: int | None = None
        try:
            file_fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            _write_all(file_fd, payload)
            os.fsync(file_fd)
            os.close(file_fd)
            file_fd = None
            os.replace(temporary_path, journal_path)
            _fsync_directory_path(journal_dir)
        except BaseException:
            if file_fd is not None:
                os.close(file_fd)
            with suppress(FileNotFoundError):
                temporary_path.unlink()
            raise
        return DeletionIntent(
            journal_name=journal_name,
            kind=kind,
            original=original_path,
            staged=staged_path,
            expected_dev=entry_stat.st_dev,
            expected_ino=entry_stat.st_ino,
            expected_type=expected_type,
            owner_id=owner_id,
        )

    journal_fd = _open_journal_dir()
    file_fd: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        file_fd = os.open(temporary_name, flags, 0o600, dir_fd=journal_fd)
        _write_all(file_fd, payload)
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = None
        os.rename(temporary_name, journal_name, src_dir_fd=journal_fd, dst_dir_fd=journal_fd)
        os.fsync(journal_fd)
    except BaseException:
        if file_fd is not None:
            os.close(file_fd)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=journal_fd)
        raise
    finally:
        os.close(journal_fd)

    return DeletionIntent(
        journal_name=journal_name,
        kind=kind,
        original=original_path,
        staged=staged_path,
        expected_dev=entry_stat.st_dev,
        expected_ino=entry_stat.st_ino,
        expected_type=expected_type,
        owner_id=owner_id,
    )


def finish_deletion_intent(intent: DeletionIntent) -> None:
    """Remove and fsync one intent after its filesystem outcome is durable."""
    if Path(intent.journal_name).name != intent.journal_name or not intent.journal_name.endswith(".json"):
        raise ValueError("Deletion journal filename is unsafe")
    if not secure_dir_fd_supported():
        journal_dir = config.get_deletion_journal_dir()
        try:
            (journal_dir / intent.journal_name).unlink()
        except FileNotFoundError:
            return
        _fsync_directory_path(journal_dir)
        return

    journal_fd = _open_journal_dir()
    try:
        try:
            os.unlink(intent.journal_name, dir_fd=journal_fd)
        except FileNotFoundError:
            return
        os.fsync(journal_fd)
    finally:
        os.close(journal_fd)


def _intent_from_payload(journal_name: str, payload: bytes) -> DeletionIntent:
    if len(payload) > _MAX_JOURNAL_BYTES:
        raise ValueError("Deletion journal entry is too large")
    raw = json.loads(payload.decode("utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != JOURNAL_SCHEMA_VERSION:
        raise ValueError("Unsupported deletion journal schema")
    operation_id = raw.get("operation_id")
    if not isinstance(operation_id, str) or journal_name != f"{operation_id}.json":
        raise ValueError("Deletion journal identity mismatch")
    kind = raw.get("kind")
    owner_id = raw.get("owner_id")
    if not isinstance(kind, str) or (owner_id is not None and not isinstance(owner_id, str)):
        raise ValueError("Deletion journal metadata is malformed")
    original = _validate_relative_path(raw.get("original", ""))
    staged = _validate_relative_path(raw.get("staged", ""))
    _validate_intent_paths(kind, original, staged, owner_id)
    expected_values = (raw.get("expected_dev"), raw.get("expected_ino"), raw.get("expected_type"))
    if any(type(value) is not int or value < 0 for value in expected_values):
        raise ValueError("Deletion journal stat identity is malformed")
    return DeletionIntent(
        journal_name=journal_name,
        kind=kind,
        original=original,
        staged=staged,
        expected_dev=expected_values[0],
        expected_ino=expected_values[1],
        expected_type=expected_values[2],
        owner_id=owner_id,
    )


def _read_intents() -> tuple[list[DeletionIntent], int]:
    intents: list[DeletionIntent] = []
    invalid = 0
    if not secure_dir_fd_supported():
        journal_dir = config.get_deletion_journal_dir()
        for path in sorted(journal_dir.glob("*.json")):
            try:
                entry_stat = path.lstat()
                if not stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                    raise ValueError("Deletion journal entry is not a private regular file")
                with path.open("rb") as source:
                    payload = source.read(_MAX_JOURNAL_BYTES + 1)
                intents.append(_intent_from_payload(path.name, payload))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                invalid += 1
                logger.error("Retaining one malformed deletion journal entry for manual inspection")
        return intents, invalid

    journal_fd = _open_journal_dir()
    try:
        with os.scandir(journal_fd) as entries:
            names = sorted(entry.name for entry in entries if entry.name.endswith(".json"))
        for name in names:
            file_fd: int | None = None
            try:
                file_fd = os.open(name, _file_flags(), dir_fd=journal_fd)
                entry_stat = os.fstat(file_fd)
                if not stat.S_ISREG(entry_stat.st_mode) or entry_stat.st_nlink != 1:
                    raise ValueError("Deletion journal entry is not a private regular file")
                payload = os.read(file_fd, _MAX_JOURNAL_BYTES + 1)
                intents.append(_intent_from_payload(name, payload))
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                invalid += 1
                logger.error("Retaining one malformed deletion journal entry for manual inspection")
            finally:
                if file_fd is not None:
                    os.close(file_fd)
    finally:
        os.close(journal_fd)
    return intents, invalid


def _relative_stat(relative: Path) -> os.stat_result | None:
    return managed_entry_stat(relative)


def entry_matches_intent(intent: DeletionIntent, entry_stat: os.stat_result) -> bool:
    return (
        entry_stat.st_dev == intent.expected_dev
        and entry_stat.st_ino == intent.expected_ino
        and stat.S_IFMT(entry_stat.st_mode) == intent.expected_type
    )


def _canonical_managed_path(stored_path: str | None) -> Path | None:
    return config.managed_storage_relative_path(stored_path)


def _database_file_paths(db: Session):
    """Yield every DB-owned managed-file reference, including legacy aliases."""
    queries = (
        db.query(DBGeneration.audio_path).all(),
        db.query(DBGenerationVersion.audio_path).all(),
        db.query(DBProfileSample.audio_path).all(),
        db.query(DBVoiceProfile.avatar_path).all(),
        db.query(DBCapture.audio_path).all(),
    )
    for rows in queries:
        for (stored_path,) in rows:
            if stored_path:
                yield stored_path


def database_owns_managed_path(relative: str | Path, db: Session) -> bool:
    """Return whether any live database row references one canonical file."""
    expected = _validate_relative_path(relative)
    return any(_canonical_managed_path(stored_path) == expected for stored_path in _database_file_paths(db))


def database_owns_below(relative: str | Path, db: Session) -> bool:
    """Return whether a live file reference is nested below one managed dir."""
    prefix = _validate_relative_path(relative)
    return any(
        (canonical := _canonical_managed_path(stored_path)) is not None
        and canonical != prefix
        and prefix in canonical.parents
        for stored_path in _database_file_paths(db)
    )


def _database_owns(intent: DeletionIntent, db: Session) -> bool:
    if intent.kind == PROFILE_STORAGE:
        if db.query(DBVoiceProfile.id).filter_by(id=intent.owner_id).first() is not None:
            return True
        return database_owns_below(intent.original, db)
    return database_owns_managed_path(intent.original, db)


def _restore(intent: DeletionIntent) -> None:
    rename_managed_entry(intent.staged, intent.original)


def _discard(intent: DeletionIntent) -> None:
    discard_managed_entry(intent.staged)


def reconcile_deletion_intent(intent: DeletionIntent, db: Session) -> str:
    """Finish one staged operation from durable database ownership.

    This is safe after an outcome-ambiguous commit: a surviving owner restores
    hidden data, while a committed deletion discards it. Any malformed or
    ambiguous filesystem state raises without removing the journal.
    """
    original_stat = _relative_stat(intent.original)
    staged_stat = _relative_stat(intent.staged)
    if intent.kind == GENERATION_AUDIO_PUBLICATION:
        # Publication always populates ``staged`` completely before renaming
        # it over ``original``, and only commits database ownership after that
        # rename. If both paths survive, publication did not happen: discard
        # only our inode and leave any predecessor at the public path intact.
        if original_stat is not None and staged_stat is not None:
            if not entry_matches_intent(intent, staged_stat):
                raise RuntimeError("Staged publication entry identity changed")
            discard_managed_entry(intent.staged)
            finish_deletion_intent(intent)
            return "discarded"
        if original_stat is not None:
            if not entry_matches_intent(intent, original_stat):
                raise RuntimeError("Published generation entry identity changed")
            if _database_owns(intent, db):
                finish_deletion_intent(intent)
                return "cleared"
            discard_managed_entry(intent.original)
            finish_deletion_intent(intent)
            return "discarded"
        if staged_stat is not None:
            if not entry_matches_intent(intent, staged_stat):
                raise RuntimeError("Staged publication entry identity changed")
            # A database owner cannot be created before the rename in the
            # publication protocol. A staged-only file is therefore always an
            # incomplete payload, even when replacing an already-owned path.
            discard_managed_entry(intent.staged)
            finish_deletion_intent(intent)
            return "discarded"
        finish_deletion_intent(intent)
        return "cleared"
    if original_stat is not None and staged_stat is not None:
        raise RuntimeError("Both original and staged deletion entries exist")
    if original_stat is not None:
        if not entry_matches_intent(intent, original_stat):
            raise RuntimeError("Original deletion entry identity changed")
        if not _database_owns(intent, db):
            discard_managed_entry(intent.original)
            finish_deletion_intent(intent)
            return "discarded"
        finish_deletion_intent(intent)
        return "cleared"
    if staged_stat is None:
        finish_deletion_intent(intent)
        return "cleared"
    if not entry_matches_intent(intent, staged_stat):
        raise RuntimeError("Staged deletion entry identity changed")
    if _database_owns(intent, db):
        _restore(intent)
        action = "restored"
    else:
        _discard(intent)
        action = "discarded"
    finish_deletion_intent(intent)
    return action


@contextmanager
def durable_reconciliation_session(db: Session):
    """Open an independent view of durable DB state after rollback failure."""
    bind = db.get_bind()
    engine = getattr(bind, "engine", bind)
    url = getattr(engine, "url", None)
    if url is not None and url.get_backend_name() == "sqlite" and url.database in {None, "", ":memory:"}:
        raise RuntimeError("Cannot independently reconcile an in-memory database")
    durable_db = Session(bind=engine)
    try:
        yield durable_db
    finally:
        durable_db.close()


def recover_interrupted_deletions(db: Session) -> RecoveryReport:
    """Reconcile durable deletion intents before the application accepts work."""
    intents, invalid = _read_intents()
    # Whole-profile staging can temporarily contain nested sample staging.
    # Restore/discard the shallow container first so child intents are
    # actionable in this same startup rather than requiring a second restart.
    intents.sort(
        key=lambda intent: (
            0 if intent.kind == PROFILE_STORAGE else 1,
            len(intent.original.parts),
            intent.journal_name,
        )
    )
    report = RecoveryReport(malformed=invalid)
    for intent in intents:
        try:
            action = reconcile_deletion_intent(intent, db)
            setattr(report, action, getattr(report, action) + 1)
        except BaseException as exc:
            report.unresolved += 1
            logger.error("Retaining one unresolved deletion intent: %s", exc)
    return report
