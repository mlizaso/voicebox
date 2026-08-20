"""
Captures service — persists raw audio alongside its STT transcript and,
optionally, an LLM-refined version.

A capture is a single voice input event (dictation, long-form recording, or
uploaded file). Storage mirrors the generations flow: audio lives under
``data/captures/<id>.wav`` and rows live in the ``captures`` table.
"""

import contextlib
import json
import logging
import os
import stat
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import librosa
import soundfile as sf
from sqlalchemy.orm import Session

from .. import config
from ..backends.mlx_tts_lifecycle import run_blocking_operation_cancellation_safe
from ..database import Capture as DBCapture
from ..models import CaptureResponse, RefinementFlagsModel
from ..utils.upload_limits import (
    AUDIO_UPLOAD_MAX_BYTES,
    AUDIO_UPLOAD_MAX_DURATION_SECONDS,
    UploadDurationLimitError,
    UploadSizeLimitError,
)
from . import deletion_journal
from .refinement import RefinementFlags, refine_transcript
from .transcribe import get_whisper_model

logger = logging.getLogger(__name__)


VALID_SOURCES = {"dictation", "recording", "file"}


class CaptureTranscriptChangedError(RuntimeError):
    """The raw transcript changed while an LLM refinement was running."""


@dataclass(frozen=True)
class _StagedCaptureAudio:
    original: Path
    staged: Path
    intent: deletion_journal.DeletionIntent


def _create_private_capture_placeholder(path: Path) -> os.stat_result:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fsync(descriptor)
        entry_stat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return entry_stat


def _populate_private_capture(
    path: Path,
    source: bytes | bytearray | memoryview | Path,
    *,
    expected_stat: os.stat_result,
) -> None:
    flags = os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    source_descriptor: int | None = None
    total_bytes = 0
    try:
        entry_stat = os.fstat(descriptor)
        if (
            not stat.S_ISREG(entry_stat.st_mode)
            or entry_stat.st_dev != expected_stat.st_dev
            or entry_stat.st_ino != expected_stat.st_ino
        ):
            raise OSError("Journaled capture staging file identity changed")

        if isinstance(source, Path):
            source_flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                source_flags |= os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                source_flags |= os.O_CLOEXEC
            source_descriptor = os.open(source, source_flags)
            source_stat = os.fstat(source_descriptor)
            if not stat.S_ISREG(source_stat.st_mode):
                raise ValueError("Capture upload source is not a regular file")
            while chunk := os.read(source_descriptor, 1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > AUDIO_UPLOAD_MAX_BYTES:
                    raise UploadSizeLimitError(AUDIO_UPLOAD_MAX_BYTES)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("short write while saving capture audio")
                    view = view[written:]
        else:
            view = memoryview(source)
            total_bytes = len(view)
            if total_bytes > AUDIO_UPLOAD_MAX_BYTES:
                raise UploadSizeLimitError(AUDIO_UPLOAD_MAX_BYTES)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while saving capture audio")
                view = view[written:]

        if total_bytes == 0:
            raise ValueError("Capture upload is empty")
        os.fsync(descriptor)
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(descriptor)


def _fsync_capture(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_capture_publication(
    pending: Path,
    target: Path,
    capture_id: str,
) -> _StagedCaptureAudio:
    pending_relative = Path("captures") / pending.name
    target_relative = Path("captures") / target.name
    entry_stat = deletion_journal.managed_entry_stat(pending_relative)
    if entry_stat is None or not stat.S_ISREG(entry_stat.st_mode):
        raise OSError("Pending capture audio is not a private regular file")
    intent = deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.CAPTURE_AUDIO,
        original=target_relative,
        staged=pending_relative,
        entry_stat=entry_stat,
        owner_id=capture_id,
    )
    return _StagedCaptureAudio(target_relative, pending_relative, intent)


def _managed_capture_relative_path(stored_path: str | None) -> Path | None:
    relative = config.managed_storage_relative_path(stored_path)
    if relative is None or len(relative.parts) != 2 or relative.parts[0] != "captures":
        logger.warning("Refusing to delete capture audio outside managed storage")
        return None
    return relative


def _stage_capture_audio(
    stored_path: str | None,
    capture_id: str,
) -> _StagedCaptureAudio | None:
    relative = _managed_capture_relative_path(stored_path)
    if relative is None:
        return None
    entry_stat = deletion_journal.managed_entry_stat(relative)
    if entry_stat is None:
        return None
    if not (stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode)):
        raise ValueError("Refusing to delete a non-file capture audio entry")
    hidden = relative.with_name(f".voicebox-delete-capture-{uuid.uuid4().hex}.tmp")
    intent = deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.CAPTURE_AUDIO,
        original=relative,
        staged=hidden,
        entry_stat=entry_stat,
        owner_id=capture_id,
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
    return _StagedCaptureAudio(relative, hidden, intent)


def _restore_capture_audio(staged: _StagedCaptureAudio | None) -> None:
    if staged is None or deletion_journal.managed_entry_stat(staged.staged) is None:
        return
    if deletion_journal.managed_entry_stat(staged.original) is not None:
        raise RuntimeError("Cannot restore capture audio because its path was replaced")
    deletion_journal.rename_managed_entry(staged.staged, staged.original)
    deletion_journal.finish_deletion_intent(staged.intent)


def _discard_capture_audio(staged: _StagedCaptureAudio | None) -> None:
    if staged is None:
        return
    try:
        deletion_journal.discard_managed_entry(staged.staged)
        deletion_journal.finish_deletion_intent(staged.intent)
    except OSError as exc:
        logger.warning("Deferred cleanup of committed capture audio: %s", exc)


def _reconcile_capture_audio(
    staged: _StagedCaptureAudio | None,
    db: Session,
) -> None:
    if staged is None:
        return
    try:
        deletion_journal.reconcile_deletion_intent(staged.intent, db)
    except BaseException:
        logger.error(
            "Retaining one capture deletion intent for startup recovery",
            exc_info=True,
        )


def _to_response(row: DBCapture) -> CaptureResponse:
    flags_model: RefinementFlagsModel | None = None
    if row.refinement_flags:
        try:
            flags_model = RefinementFlagsModel(**json.loads(row.refinement_flags))
        except (ValueError, TypeError):
            flags_model = None

    return CaptureResponse(
        id=row.id,
        audio_path=row.audio_path,
        source=row.source,
        language=row.language,
        duration_ms=row.duration_ms,
        transcript_raw=row.transcript_raw or "",
        transcript_refined=row.transcript_refined,
        stt_model=row.stt_model,
        llm_model=row.llm_model,
        refinement_flags=flags_model,
        created_at=row.created_at,
    )


def _durable_created_capture_response(
    db: Session,
    *,
    capture_id: str,
    expected_fields: dict[str, object],
    publication: _StagedCaptureAudio,
) -> CaptureResponse | None:
    """Return a capture only after independently proving its row and audio."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        row = durable_db.query(DBCapture).filter_by(id=capture_id).one_or_none()
        if row is None:
            return None
        if any(getattr(row, field) != value for field, value in expected_fields.items()):
            raise RuntimeError("Capture ID is owned by different durable data")
        audio_stat = deletion_journal.managed_entry_stat(publication.original)
        if (
            audio_stat is None
            or not deletion_journal.entry_matches_intent(publication.intent, audio_stat)
            or not stat.S_ISREG(audio_stat.st_mode)
            or audio_stat.st_nlink != 1
        ):
            raise RuntimeError("Capture audio identity changed after commit")
        return _to_response(row)


def _durable_updated_capture_response(
    db: Session,
    *,
    capture_id: str,
    expected_fields: dict[str, object],
) -> CaptureResponse | None:
    """Return an independently proven capture update after commit ambiguity."""
    with deletion_journal.durable_reconciliation_session(db) as durable_db:
        row = durable_db.query(DBCapture).filter_by(id=capture_id).one_or_none()
        if row is None:
            return None
        if any(getattr(row, field) != value for field, value in expected_fields.items()):
            return None
        return _to_response(row)


def _commit_capture_update(
    db: Session,
    *,
    capture_id: str,
    expected_fields: dict[str, object],
) -> CaptureResponse:
    """Commit an update and acknowledge a proven durable post-commit result."""
    try:
        db.commit()
        row = db.query(DBCapture).filter_by(id=capture_id).one_or_none()
        if row is None or any(getattr(row, field) != value for field, value in expected_fields.items()):
            raise RuntimeError("Capture changed while acknowledging its update")
        return _to_response(row)
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Capture update rollback failed", exc_info=True)
        try:
            durable_response = _durable_updated_capture_response(
                db,
                capture_id=capture_id,
                expected_fields=expected_fields,
            )
        except BaseException:
            durable_response = None
            logger.warning("Could not prove an ambiguous capture update", exc_info=True)
        if durable_response is not None:
            return durable_response
        if rollback_error is not None:
            raise rollback_error from operation_error
        raise


def _finish_created_capture_intent(
    intent: deletion_journal.DeletionIntent,
) -> None:
    """Best-effort journal retirement after capture ownership is durable."""
    try:
        deletion_journal.finish_deletion_intent(intent)
    except Exception:
        logger.warning(
            "Deferred cleanup of a committed capture publication intent",
            exc_info=True,
        )


async def create_capture(
    *,
    audio_bytes: bytes | bytearray | memoryview | Path,
    filename: str,
    source: str,
    language: str | None,
    stt_model: str | None,
    db: Session,
) -> CaptureResponse:
    """Persist raw audio, run STT, store the row."""
    if source not in VALID_SOURCES:
        raise ValueError(f"Invalid source '{source}'. Must be one of {sorted(VALID_SOURCES)}")

    capture_id = str(uuid.uuid4())
    suffix = Path(filename).suffix.lower() or ".wav"
    if suffix not in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"):
        suffix = ".wav"

    captures_dir = config.get_captures_dir()
    raw_target = captures_dir / f"{capture_id}{suffix}"
    raw_pending = captures_dir / f".voicebox-delete-capture-new-{capture_id}.part{suffix}"
    untracked_pending: set[Path] = {raw_pending}
    active_publications: list[_StagedCaptureAudio] = []
    transcoded_temp_path: Path | None = None
    commit_started = False
    expected_capture_fields: dict[str, object] = {}
    response: CaptureResponse | None = None

    try:
        raw_pending_stat = _create_private_capture_placeholder(raw_pending)
        raw_publication = _prepare_capture_publication(
            raw_pending,
            raw_target,
            capture_id,
        )
        untracked_pending.remove(raw_pending)
        active_publications.append(raw_publication)
        await run_blocking_operation_cancellation_safe(
            _populate_private_capture,
            raw_pending,
            audio_bytes,
            expected_stat=raw_pending_stat,
        )

        # Decode once with librosa — its audioread fallback handles webm/opus
        # via ffmpeg, which miniaudio (used inside mlx-audio's whisper) can't.
        # The decoded array gives us an accurate duration and becomes the
        # canonical WAV we hand to whisper.
        try:
            audio, sr = await run_blocking_operation_cancellation_safe(
                librosa.load,
                str(raw_pending),
                sr=24_000,
                mono=True,
                duration=AUDIO_UPLOAD_MAX_DURATION_SECONDS + 1,
            )
            duration_ms = int((len(audio) / sr) * 1000) if sr else None
            if duration_ms is not None and duration_ms > AUDIO_UPLOAD_MAX_DURATION_SECONDS * 1000:
                raise UploadDurationLimitError(AUDIO_UPLOAD_MAX_DURATION_SECONDS)
        except UploadDurationLimitError:
            raise
        except Exception as decode_err:
            logger.warning("Could not decode capture %s (%s): %r", capture_id, suffix, decode_err)
            audio, sr = None, None
            duration_ms = None

        if audio is None or sr is None:
            # Never hand an unmeasured upload to another decoder: without a
            # successful bounded decode we cannot enforce the duration cap.
            raise ValueError(f"Could not decode {suffix} audio — the recording may be empty or corrupt")
        if suffix == ".wav":
            audio_pending = raw_pending
            audio_target = raw_target
            publication = raw_publication
        else:
            # Transcode to WAV so downstream loaders (miniaudio, soundfile) work
            # regardless of what format the client shipped.
            audio_target = captures_dir / f"{capture_id}.wav"
            audio_pending = captures_dir / f".voicebox-delete-capture-new-{capture_id}.part.wav"
            transcode_descriptor, transcode_path = tempfile.mkstemp(
                prefix="voicebox-capture-transcode-",
                suffix=".wav",
            )
            os.close(transcode_descriptor)
            transcoded_temp_path = Path(transcode_path)
            await run_blocking_operation_cancellation_safe(
                sf.write,
                str(transcoded_temp_path),
                audio,
                sr,
                format="WAV",
            )
            _fsync_capture(transcoded_temp_path)

            untracked_pending.add(audio_pending)
            audio_pending_stat = _create_private_capture_placeholder(audio_pending)
            publication = _prepare_capture_publication(
                audio_pending,
                audio_target,
                capture_id,
            )
            untracked_pending.remove(audio_pending)
            active_publications.append(publication)
            await run_blocking_operation_cancellation_safe(
                _populate_private_capture,
                audio_pending,
                transcoded_temp_path,
                expected_stat=audio_pending_stat,
            )
            deletion_journal.discard_managed_entry(raw_publication.staged)
            deletion_journal.finish_deletion_intent(raw_publication.intent)
            active_publications.remove(raw_publication)

        whisper = get_whisper_model()
        resolved_stt = stt_model or whisper.model_size
        transcript = await whisper.transcribe(str(audio_pending), language, resolved_stt)

        deletion_journal.rename_managed_entry(
            publication.staged,
            publication.original,
        )

        created_at = datetime.utcnow()
        row = DBCapture(
            id=capture_id,
            audio_path=config.to_storage_path(audio_target),
            source=source,
            language=language,
            duration_ms=duration_ms,
            transcript_raw=transcript,
            stt_model=resolved_stt,
            transcript_refined=None,
            llm_model=None,
            refinement_flags=None,
            created_at=created_at,
        )
        expected_capture_fields = {
            field: getattr(row, field)
            for field in (
                "id",
                "audio_path",
                "source",
                "language",
                "duration_ms",
                "transcript_raw",
                "transcript_refined",
                "stt_model",
                "llm_model",
                "refinement_flags",
                "created_at",
            )
        }
        db.add(row)
        commit_started = True
        db.commit()
        db.refresh(row)
        response = _to_response(row)
        _finish_created_capture_intent(publication.intent)
        active_publications.remove(publication)
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Capture creation rollback failed", exc_info=True)
        durable_response: CaptureResponse | None = None
        if commit_started:
            try:
                durable_response = _durable_created_capture_response(
                    db,
                    capture_id=capture_id,
                    expected_fields=expected_capture_fields,
                    publication=publication,
                )
            except Exception:
                logger.warning(
                    "Could not prove an ambiguous capture creation",
                    exc_info=True,
                )
        if durable_response is not None:
            _finish_created_capture_intent(publication.intent)
            active_publications.remove(publication)
            return durable_response
        if rollback_error is None:
            for prepared in reversed(active_publications):
                _reconcile_capture_audio(prepared, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    for prepared in reversed(active_publications):
                        _reconcile_capture_audio(prepared, durable_db)
            except BaseException:
                logger.error("Deferred capture creation reconciliation", exc_info=True)
        for path in untracked_pending:
            with contextlib.suppress(OSError, ValueError):
                deletion_journal.discard_managed_entry(Path("captures") / path.name)
        if rollback_error is not None:
            raise rollback_error from operation_error
        raise
    finally:
        if transcoded_temp_path is not None:
            transcoded_temp_path.unlink(missing_ok=True)

    if response is None:
        raise RuntimeError("Capture transaction completed without a response")
    return response


def list_captures(db: Session, limit: int = 50, offset: int = 0) -> tuple[list[CaptureResponse], int]:
    total = db.query(DBCapture).count()
    rows = db.query(DBCapture).order_by(DBCapture.created_at.desc()).limit(limit).offset(offset).all()
    return [_to_response(r) for r in rows], total


def get_capture(capture_id: str, db: Session) -> CaptureResponse | None:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    return _to_response(row) if row else None


def delete_capture(capture_id: str, db: Session) -> bool:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return False

    stored_audio_path = row.audio_path
    relative = _managed_capture_relative_path(stored_audio_path)
    staged_audio: _StagedCaptureAudio | None = None
    try:
        db.delete(row)
        db.flush()
        if relative is not None and not deletion_journal.database_owns_managed_path(relative, db):
            staged_audio = _stage_capture_audio(stored_audio_path, capture_id)
        db.commit()
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Capture deletion rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_capture_audio(staged_audio, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_capture_audio(staged_audio, durable_db)
            except BaseException:
                logger.error("Deferred capture deletion reconciliation", exc_info=True)
            raise rollback_error from operation_error
        raise

    _discard_capture_audio(staged_audio)
    return True


async def refine_capture(
    capture_id: str,
    flags: RefinementFlags,
    model_size: str | None,
    db: Session,
) -> CaptureResponse | None:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return None

    source_transcript = row.transcript_raw or ""
    # Do not retain a SQLite transaction/connection while the serialized LLM
    # request may wait behind hours of accelerator work.
    db.close()

    refined, llm_size = await refine_transcript(
        source_transcript,
        flags,
        model_size=model_size,
    )

    refinement_flags = json.dumps(flags.to_dict())
    updated = (
        db.query(DBCapture)
        .filter(
            DBCapture.id == capture_id,
            DBCapture.transcript_raw == source_transcript,
        )
        .update(
            {
                DBCapture.transcript_refined: refined,
                DBCapture.llm_model: llm_size,
                DBCapture.refinement_flags: refinement_flags,
            },
            synchronize_session=False,
        )
    )
    if updated != 1:
        db.rollback()
        current = db.query(DBCapture).filter(DBCapture.id == capture_id).one_or_none()
        if current is None:
            return None
        raise CaptureTranscriptChangedError(
            "Capture transcript changed while refinement was running; retry the refinement"
        )

    return _commit_capture_update(
        db,
        capture_id=capture_id,
        expected_fields={
            "transcript_raw": source_transcript,
            "transcript_refined": refined,
            "llm_model": llm_size,
            "refinement_flags": refinement_flags,
        },
    )


async def retranscribe_capture(
    capture_id: str,
    stt_model: str | None,
    language: str | None,
    db: Session,
) -> CaptureResponse | None:
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        return None

    resolved = config.resolve_storage_path(row.audio_path)
    if not resolved or not resolved.exists():
        raise FileNotFoundError(f"Audio for capture {capture_id} is missing")

    whisper = get_whisper_model()
    resolved_stt = stt_model or whisper.model_size
    # Release the request's read transaction while the process-wide inference
    # guard is awaited. The row is re-selected atomically before mutation.
    db.close()
    transcript = await whisper.transcribe(str(resolved), language, resolved_stt)

    update_fields: dict = {
        DBCapture.transcript_raw: transcript,
        DBCapture.stt_model: resolved_stt,
        # Refined text is stale after a fresh STT pass — force a re-refine.
        DBCapture.transcript_refined: None,
        DBCapture.llm_model: None,
        DBCapture.refinement_flags: None,
    }
    expected_fields: dict[str, object] = {
        "transcript_raw": transcript,
        "stt_model": resolved_stt,
        "transcript_refined": None,
        "llm_model": None,
        "refinement_flags": None,
    }
    if language:
        update_fields[DBCapture.language] = language
        expected_fields["language"] = language
    updated = db.query(DBCapture).filter(DBCapture.id == capture_id).update(update_fields, synchronize_session=False)
    if updated != 1:
        db.rollback()
        return None
    return _commit_capture_update(
        db,
        capture_id=capture_id,
        expected_fields=expected_fields,
    )
