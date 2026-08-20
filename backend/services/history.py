"""Generation history management module."""

import logging
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import config
from ..database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    StoryItem as DBStoryItem,
    VoiceProfile as DBVoiceProfile,
)
from ..models import (
    EffectConfig,
    GenerationResponse,
    GenerationVersionResponse,
    HistoryListResponse,
    HistoryQuery,
    HistoryResponse,
)
from . import deletion_journal

logger = logging.getLogger(__name__)


class GenerationInUseError(ValueError):
    """Raised when deleting history would silently remove a Story item."""


def _managed_generation_relative_path(stored_path: str | None) -> Path | None:
    """Return a lexical path below ``data/generations`` or reject it as unmanaged."""
    relative = config.managed_storage_relative_path(stored_path)
    if relative is None:
        logger.warning("Refusing to delete generation audio outside the managed data root")
        return None
    if not relative.parts or relative.parts[0] != "generations":
        logger.warning("Refusing to delete a non-generation managed data path")
        return None
    return relative


@dataclass(frozen=True)
class _StagedGenerationAudio:
    original: Path
    staged: Path
    intent: deletion_journal.DeletionIntent


def _stage_managed_generation_audio(
    stored_path: str | None,
) -> _StagedGenerationAudio | None:
    """Atomically hide one managed file so a failed DB transaction can restore it."""
    relative = _managed_generation_relative_path(stored_path)
    if relative is None:
        return None

    entry_stat = deletion_journal.managed_entry_stat(relative)
    if entry_stat is None:
        return None
    if not (stat.S_ISREG(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode)):
        raise ValueError("Refusing to delete a non-file generation audio entry")
    staged_name = f".voicebox-delete-{uuid.uuid4().hex}.tmp"
    original = relative
    staged = relative.with_name(staged_name)
    intent = deletion_journal.prepare_deletion_intent(
        kind=deletion_journal.GENERATION_AUDIO,
        original=original,
        staged=staged,
        entry_stat=entry_stat,
    )
    try:
        deletion_journal.rename_managed_entry(original, staged)
    except BaseException:
        if deletion_journal.managed_entry_stat(staged) is not None:
            deletion_journal.rename_managed_entry(staged, original)
        original_stat = deletion_journal.managed_entry_stat(original)
        if original_stat is not None and deletion_journal.entry_matches_intent(intent, original_stat):
            deletion_journal.finish_deletion_intent(intent)
        raise
    return _StagedGenerationAudio(
        original=original,
        staged=staged,
        intent=intent,
    )


def _restore_staged_generation_audio(staged_audio: list[_StagedGenerationAudio]) -> None:
    first_error: BaseException | None = None
    for item in reversed(staged_audio):
        try:
            if deletion_journal.managed_entry_stat(item.staged) is None:
                continue
            if deletion_journal.managed_entry_stat(item.original) is not None:
                raise RuntimeError("Cannot restore generation audio because its original path was replaced")
            deletion_journal.rename_managed_entry(item.staged, item.original)
            deletion_journal.finish_deletion_intent(item.intent)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise RuntimeError("Failed to restore one or more staged generation files") from first_error


def _discard_staged_generation_audio(staged_audio: list[_StagedGenerationAudio]) -> None:
    for item in staged_audio:
        try:
            deletion_journal.discard_managed_entry(item.staged)
            deletion_journal.finish_deletion_intent(item.intent)
        except OSError as exc:
            logger.warning("Deferred cleanup of committed generation audio: %s", exc)


def _reconcile_staged_generation_audio(
    staged_audio: list[_StagedGenerationAudio],
    db: Session,
) -> None:
    """Resolve staged files from durable ownership after an ambiguous commit."""
    for item in reversed(staged_audio):
        try:
            deletion_journal.reconcile_deletion_intent(item.intent, db)
        except BaseException:
            logger.error(
                "Retaining one generation deletion intent for startup recovery",
                exc_info=True,
            )


def _delete_generation_records(
    generations: list[DBGeneration],
    db: Session,
    staged_audio: list[_StagedGenerationAudio],
    exact_checkpoint_candidates: set[str] | None = None,
) -> int:
    """Stage owned audio and delete dependent rows without committing."""
    if not generations:
        return 0
    if exact_checkpoint_candidates is not None:
        exact_checkpoint_candidates.update(
            generation.exact_request_sha256 for generation in generations if generation.exact_request_sha256
        )
    generation_ids = [generation.id for generation in generations]
    if db.query(DBStoryItem.id).filter(DBStoryItem.generation_id.in_(generation_ids)).first() is not None:
        raise GenerationInUseError(
            "One or more generations are used by a Story; remove them from every Story before deleting them"
        )
    versions = db.query(DBGenerationVersion).filter(DBGenerationVersion.generation_id.in_(generation_ids)).all()
    deleting_version_ids = {version.id for version in versions}
    candidates: dict[Path, str] = {}
    for audio_path in [version.audio_path for version in versions] + [
        generation.audio_path for generation in generations
    ]:
        canonical = _managed_generation_relative_path(audio_path)
        if canonical is not None:
            candidates.setdefault(canonical, audio_path)

    referenced: set[Path] = set()
    for generation_id, audio_path in db.query(DBGeneration.id, DBGeneration.audio_path).all():
        if generation_id in generation_ids:
            continue
        canonical = _managed_generation_relative_path(audio_path)
        if canonical in candidates:
            referenced.add(canonical)
    for version_id, audio_path in db.query(DBGenerationVersion.id, DBGenerationVersion.audio_path).all():
        if version_id in deleting_version_ids:
            continue
        canonical = _managed_generation_relative_path(audio_path)
        if canonical in candidates:
            referenced.add(canonical)

    for canonical, audio_path in candidates.items():
        if canonical in referenced:
            continue
        staged = _stage_managed_generation_audio(audio_path)
        if staged is not None:
            staged_audio.append(staged)

    for version in versions:
        db.delete(version)
    for generation in generations:
        db.delete(generation)
    db.flush()
    return len(generations)


def _reclaim_exact_checkpoint_candidates(
    candidates: set[str],
    db: Session,
) -> None:
    """Best-effort reclaim after the generation deletion commit is durable."""
    if not candidates:
        return
    try:
        from .exact_chunk_checkpoints import (
            garbage_collect_exact_chunk_checkpoints,
        )

        report = garbage_collect_exact_chunk_checkpoints(
            db,
            request_hashes=candidates,
        )
    except BaseException:
        # Deleting history is authoritative; a cache collector failure must
        # retain bytes for the next startup pass, not roll back user data.
        logger.warning(
            "Deferred exact chunk checkpoint cleanup after history deletion",
            exc_info=True,
        )
        return
    if report.refused:
        logger.warning(
            "Refused cleanup of %d unsafe exact checkpoint request %s",
            report.refused,
            "directory" if report.refused == 1 else "directories",
        )


def _get_versions_for_generation(generation_id: str, db: Session) -> tuple:
    """Get versions list and active version ID for a generation."""
    import json

    versions_rows = (
        db.query(DBGenerationVersion)
        .filter_by(generation_id=generation_id)
        .order_by(DBGenerationVersion.created_at)
        .all()
    )
    if not versions_rows:
        return None, None

    versions = []
    active_version_id = None
    for v in versions_rows:
        effects_chain = None
        if v.effects_chain:
            try:
                raw = json.loads(v.effects_chain)
                effects_chain = [EffectConfig(**e) for e in raw]
            except Exception:
                pass
        versions.append(
            GenerationVersionResponse(
                id=v.id,
                generation_id=v.generation_id,
                label=v.label,
                audio_path=v.audio_path,
                effects_chain=effects_chain,
                is_default=v.is_default,
                created_at=v.created_at,
            )
        )
        if v.is_default:
            active_version_id = v.id

    return versions, active_version_id


async def create_generation(
    profile_id: str,
    text: str,
    language: str,
    audio_path: str,
    duration: float,
    seed: int | None,
    db: Session,
    instruct: str | None = None,
    generation_id: str | None = None,
    status: str = "completed",
    engine: str | None = "qwen",
    model_size: str | None = None,
    source: str = "manual",
    exact_request_sha256: str | None = None,
    exact_envelope_sha256: str | None = None,
    exact_effects_json: str | None = None,
    exact_voice_snapshot_json: str | None = None,
    voice_binding_sha256: str | None = None,
) -> GenerationResponse:
    """
    Create a new generation history entry.

    Args:
        profile_id: Profile ID used for generation
        text: Generated text
        language: Language code
        audio_path: Path where audio was saved
        duration: Audio duration in seconds
        seed: Random seed used (if any)
        db: Database session
        instruct: Natural language instruction used (if any)
        generation_id: Pre-assigned ID (for async generation flow)
        status: Generation status (generating, completed, failed)
        engine: TTS engine used (qwen, luxtts, chatterbox, chatterbox_turbo)
        model_size: Model size variant (1.7B, 0.6B) — only relevant for qwen
        source: Origin marker stored on the row. ``"manual"`` for regular
            /generate calls; ``"personality_speak"`` for rows created
            by the /profiles/{id}/speak endpoint. Enables filtering the
            history view for personality-driven output.

    Returns:
        Created generation entry
    """
    db_generation = DBGeneration(
        id=generation_id or str(uuid.uuid4()),
        profile_id=profile_id,
        text=text,
        language=language,
        audio_path=audio_path,
        duration=duration,
        seed=seed,
        instruct=instruct,
        engine=engine,
        model_size=model_size,
        status=status,
        source=source,
        exact_request_sha256=exact_request_sha256,
        exact_envelope_sha256=exact_envelope_sha256,
        exact_effects_json=exact_effects_json,
        exact_voice_snapshot_json=exact_voice_snapshot_json,
        voice_binding_sha256=voice_binding_sha256,
        created_at=datetime.utcnow(),
    )

    db.add(db_generation)
    expected = {
        column: getattr(db_generation, column)
        for column in (
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
            "source",
            "exact_request_sha256",
            "exact_envelope_sha256",
            "exact_effects_json",
            "exact_voice_snapshot_json",
            "voice_binding_sha256",
        )
    }
    try:
        db.commit()
        db.refresh(db_generation)
        return GenerationResponse.model_validate(db_generation)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Generation creation rollback failed", exc_info=True)
        if not isinstance(operation_error, IntegrityError):
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    durable = durable_db.query(DBGeneration).filter_by(id=db_generation.id).one_or_none()
                    if durable is not None and all(
                        getattr(durable, field) == value for field, value in expected.items()
                    ):
                        return GenerationResponse.model_validate(durable)
            except BaseException:
                logger.error("Could not reconcile ambiguous generation creation", exc_info=True)
        raise operation_error


async def create_generation_batch(
    entries: list[dict],
    db: Session,
) -> list[GenerationResponse]:
    """Create all rows for a model batch in one database transaction."""
    if not 1 <= len(entries) <= 2:
        raise ValueError("exact generation envelopes require one or two entries")

    created_at = datetime.utcnow()
    rows = [DBGeneration(created_at=created_at, **entry) for entry in entries]
    try:
        db.add_all(rows)
        db.commit()
        for row in rows:
            db.refresh(row)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Generation batch creation rollback failed", exc_info=True)
        if not isinstance(operation_error, IntegrityError):
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    durable_rows = (
                        durable_db.query(DBGeneration).filter(DBGeneration.id.in_([row.id for row in rows])).all()
                    )
                    durable_by_id = {row.id: row for row in durable_rows}
                    if len(durable_by_id) == len(entries) and all(
                        all(getattr(durable_by_id[entry["id"]], field) == value for field, value in entry.items())
                        for entry in entries
                    ):
                        return [GenerationResponse.model_validate(durable_by_id[entry["id"]]) for entry in entries]
            except BaseException:
                logger.error("Could not reconcile ambiguous generation batch creation", exc_info=True)
        raise operation_error
    return [GenerationResponse.model_validate(row) for row in rows]


async def update_generation_status(
    generation_id: str,
    status: str,
    db: Session,
    audio_path: str | None = None,
    duration: float | None = None,
    error: str | None = None,
    clear_error: bool = False,
) -> GenerationResponse | None:
    """Update the status of a generation (used by async generation flow)."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        return None

    generation.status = status
    if audio_path is not None:
        generation.audio_path = audio_path
    if duration is not None:
        generation.duration = duration
    if error is not None or clear_error:
        generation.error = error

    expected = {"status": status}
    if audio_path is not None:
        expected["audio_path"] = audio_path
    if duration is not None:
        expected["duration"] = duration
    if error is not None or clear_error:
        expected["error"] = error

    try:
        db.commit()
        db.refresh(generation)
        return GenerationResponse.model_validate(generation)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Generation status rollback failed", exc_info=True)

        # COMMIT can become durable before a connection/refresh failure is
        # reported. Treat the requested state as successful only when a fresh
        # independent session proves every written field is already durable;
        # otherwise let the caller's normal failure path run.
        try:
            with deletion_journal.durable_reconciliation_session(db) as durable_db:
                durable = durable_db.query(DBGeneration).filter_by(id=generation_id).one_or_none()
                if durable is not None and all(getattr(durable, field) == value for field, value in expected.items()):
                    return GenerationResponse.model_validate(durable)
        except BaseException:
            logger.error("Could not reconcile ambiguous generation status update", exc_info=True)
        raise operation_error


async def get_generation(
    generation_id: str,
    db: Session,
) -> GenerationResponse | None:
    """
    Get a generation by ID.

    Args:
        generation_id: Generation ID
        db: Database session

    Returns:
        Generation or None if not found
    """
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        return None

    return GenerationResponse.model_validate(generation)


async def list_generations(
    query: HistoryQuery,
    db: Session,
) -> HistoryListResponse:
    """
    List generations with optional filters.

    Args:
        query: Query parameters (filters, pagination)
        db: Database session

    Returns:
        HistoryListResponse with items and total count
    """
    # Build base query with join to get profile name
    q = db.query(DBGeneration, DBVoiceProfile.name.label("profile_name")).join(
        DBVoiceProfile, DBGeneration.profile_id == DBVoiceProfile.id
    )

    # Apply profile filter
    if query.profile_id:
        q = q.filter(DBGeneration.profile_id == query.profile_id)

    # Apply search filter (searches in text content)
    if query.search:
        search_pattern = f"%{query.search}%"
        q = q.filter(DBGeneration.text.like(search_pattern))

    # Get total count before pagination
    total_count = q.count()

    # Apply ordering (newest first)
    q = q.order_by(DBGeneration.created_at.desc())

    # Apply pagination
    q = q.offset(query.offset).limit(query.limit)

    # Execute query
    results = q.all()

    # Convert to HistoryResponse with profile_name
    items = []
    for generation, profile_name in results:
        versions, active_version_id = _get_versions_for_generation(generation.id, db)
        items.append(
            HistoryResponse(
                id=generation.id,
                profile_id=generation.profile_id,
                profile_name=profile_name,
                text=generation.text,
                language=generation.language,
                audio_path=generation.audio_path,
                duration=generation.duration,
                seed=generation.seed,
                instruct=generation.instruct,
                engine=generation.engine or "qwen",
                model_size=generation.model_size,
                status=generation.status or "completed",
                error=generation.error,
                is_favorited=bool(generation.is_favorited),
                exact_request_sha256=generation.exact_request_sha256,
                exact_envelope_sha256=generation.exact_envelope_sha256,
                exact_effects_json=generation.exact_effects_json,
                exact_voice_snapshot_json=generation.exact_voice_snapshot_json,
                voice_binding_sha256=generation.voice_binding_sha256,
                created_at=generation.created_at,
                versions=versions,
                active_version_id=active_version_id,
            )
        )

    return HistoryListResponse(
        items=items,
        total=total_count,
    )


async def delete_generation(
    generation_id: str,
    db: Session,
) -> bool:
    """
    Delete a generation.

    Args:
        generation_id: Generation ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        return False

    staged_audio: list[_StagedGenerationAudio] = []
    exact_checkpoint_candidates: set[str] = set()
    try:
        _delete_generation_records(
            [generation],
            db,
            staged_audio,
            exact_checkpoint_candidates,
        )
        db.commit()
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Generation deletion rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_staged_generation_audio(staged_audio, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_staged_generation_audio(staged_audio, durable_db)
            except BaseException:
                logger.error("Deferred generation deletion reconciliation", exc_info=True)
            raise rollback_error from operation_error
        raise
    _discard_staged_generation_audio(staged_audio)
    _reclaim_exact_checkpoint_candidates(exact_checkpoint_candidates, db)

    return True


async def delete_failed_generations(db: Session) -> int:
    """
    Delete every generation whose status is 'failed'.

    Used by the "Clear failed" action in the UI so users can tidy up
    history after the model wasn't loaded, the app was closed mid-run,
    or a generation otherwise errored out (see issue #410).

    Returns:
        Number of generations deleted.
    """
    from .task_queue import generation_job_is_active

    failed = [
        generation
        for generation in db.query(DBGeneration).filter(DBGeneration.status == "failed").all()
        if not generation_job_is_active(generation.id)
    ]
    staged_audio: list[_StagedGenerationAudio] = []
    exact_checkpoint_candidates: set[str] = set()
    try:
        count = _delete_generation_records(
            failed,
            db,
            staged_audio,
            exact_checkpoint_candidates,
        )
        db.commit()
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Failed-generation deletion rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_staged_generation_audio(staged_audio, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_staged_generation_audio(staged_audio, durable_db)
            except BaseException:
                logger.error("Deferred failed-generation reconciliation", exc_info=True)
            raise rollback_error from operation_error
        raise
    _discard_staged_generation_audio(staged_audio)
    _reclaim_exact_checkpoint_candidates(exact_checkpoint_candidates, db)
    return count


async def delete_generations_by_profile(
    profile_id: str,
    db: Session,
    *,
    commit: bool = True,
    staged_audio: list[_StagedGenerationAudio] | None = None,
    exact_checkpoint_candidates: set[str] | None = None,
) -> int:
    """
    Delete all generations for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        Number of generations deleted
    """
    if not commit and staged_audio is None:
        raise ValueError("uncommitted generation deletion requires staged-audio ownership")
    if not commit and exact_checkpoint_candidates is None:
        raise ValueError("uncommitted generation deletion requires exact-checkpoint ownership")
    owned_stages = staged_audio if staged_audio is not None else []
    owned_checkpoint_candidates = exact_checkpoint_candidates if exact_checkpoint_candidates is not None else set()
    generations = db.query(DBGeneration).filter_by(profile_id=profile_id).all()
    try:
        count = _delete_generation_records(
            generations,
            db,
            owned_stages,
            owned_checkpoint_candidates,
        )
        if commit:
            db.commit()
    except BaseException as operation_error:
        if commit:
            rollback_error: BaseException | None = None
            try:
                db.rollback()
            except BaseException as exc:
                rollback_error = exc
                logger.error("Profile-generation deletion rollback failed", exc_info=True)
            if rollback_error is None:
                _reconcile_staged_generation_audio(owned_stages, db)
            else:
                try:
                    with deletion_journal.durable_reconciliation_session(db) as durable_db:
                        _reconcile_staged_generation_audio(owned_stages, durable_db)
                except BaseException:
                    logger.error("Deferred profile-generation reconciliation", exc_info=True)
                raise rollback_error from operation_error
        raise
    if commit:
        _discard_staged_generation_audio(owned_stages)
        _reclaim_exact_checkpoint_candidates(owned_checkpoint_candidates, db)
    return count


async def get_generation_stats(db: Session) -> dict:
    """
    Get generation statistics.

    Args:
        db: Database session

    Returns:
        Statistics dictionary
    """
    from sqlalchemy import func

    total = db.query(func.count(DBGeneration.id)).scalar()

    total_duration = db.query(func.sum(DBGeneration.duration)).scalar() or 0

    # Get generations by profile
    by_profile = (
        db.query(DBGeneration.profile_id, func.count(DBGeneration.id).label("count"))
        .group_by(DBGeneration.profile_id)
        .all()
    )

    return {
        "total_generations": total,
        "total_duration_seconds": total_duration,
        "generations_by_profile": {profile_id: count for profile_id, count in by_profile},
    }
