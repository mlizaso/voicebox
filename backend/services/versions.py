"""
Generation versions management module.

Each generation can have multiple audio versions: a clean (unprocessed)
version and any number of processed versions with different effects chains.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    StoryItem as DBStoryItem,
)
from ..models import EffectConfig, GenerationVersionResponse
from . import deletion_journal, history as history_service

logger = logging.getLogger(__name__)


class VersionInUseError(ValueError):
    """Raised when durable provenance or a Story still references a version."""


def _stage_version_audio(
    audio_paths: list[str],
) -> list[history_service._StagedGenerationAudio]:
    """Stage unique managed files, restoring earlier entries if staging fails."""
    staged_audio: list[history_service._StagedGenerationAudio] = []
    try:
        for audio_path in dict.fromkeys(audio_paths):
            staged = history_service._stage_managed_generation_audio(audio_path)
            if staged is not None:
                staged_audio.append(staged)
    except BaseException:
        history_service._restore_staged_generation_audio(staged_audio)
        raise
    return staged_audio


def _unreferenced_managed_audio_paths(
    audio_paths: list[str],
    db: Session,
    *,
    deleting_version_ids: set[str],
    changing_generation_ids: set[str],
) -> list[str]:
    """Return canonical managed files with no database owner after this transaction."""
    candidates = {}
    for audio_path in audio_paths:
        canonical = history_service._managed_generation_relative_path(audio_path)
        if canonical is not None:
            candidates.setdefault(canonical, audio_path)
    if not candidates:
        return []

    referenced = set()
    version_paths = db.query(DBGenerationVersion.id, DBGenerationVersion.audio_path).all()
    for version_id, audio_path in version_paths:
        if version_id in deleting_version_ids:
            continue
        canonical = history_service._managed_generation_relative_path(audio_path)
        if canonical in candidates:
            referenced.add(canonical)

    generation_paths = db.query(DBGeneration.id, DBGeneration.audio_path).all()
    for generation_id, audio_path in generation_paths:
        if generation_id in changing_generation_ids:
            continue
        canonical = history_service._managed_generation_relative_path(audio_path)
        if canonical in candidates:
            referenced.add(canonical)

    return [stored_path for canonical, stored_path in candidates.items() if canonical not in referenced]


@contextmanager
def _version_deletion_transaction(
    db: Session,
    audio_paths: list[str],
) -> Iterator[None]:
    """Commit row changes before permanently discarding their staged audio."""
    staged_audio = _stage_version_audio(audio_paths)
    try:
        yield
        db.flush()
        db.commit()
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Generation-version deletion rollback failed", exc_info=True)
        if rollback_error is None:
            history_service._reconcile_staged_generation_audio(staged_audio, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    history_service._reconcile_staged_generation_audio(
                        staged_audio,
                        durable_db,
                    )
            except BaseException:
                logger.error("Deferred generation-version reconciliation", exc_info=True)
            raise rollback_error from operation_error
        raise
    history_service._discard_staged_generation_audio(staged_audio)


def _version_response(v: DBGenerationVersion) -> GenerationVersionResponse:
    """Convert a DB version row to a Pydantic response."""
    effects_chain = None
    if v.effects_chain:
        raw = json.loads(v.effects_chain)
        effects_chain = [EffectConfig(**e) for e in raw]
    return GenerationVersionResponse(
        id=v.id,
        generation_id=v.generation_id,
        label=v.label,
        audio_path=v.audio_path,
        effects_chain=effects_chain,
        source_version_id=v.source_version_id,
        is_default=v.is_default,
        created_at=v.created_at,
    )


def _durable_version_write(
    *,
    db: Session,
    version_id: str,
    expected: dict[str, object],
    expected_generation_audio_path: str | None,
    require_unique_default: bool,
) -> GenerationVersionResponse | None:
    """Return a committed version after an ambiguous commit/refresh failure."""
    try:
        with deletion_journal.durable_reconciliation_session(db) as durable_db:
            durable = durable_db.query(DBGenerationVersion).filter_by(id=version_id).one_or_none()
            if durable is None or any(getattr(durable, field) != value for field, value in expected.items()):
                return None
            if expected_generation_audio_path is not None:
                generation = durable_db.query(DBGeneration).filter_by(id=durable.generation_id).one_or_none()
                if generation is None or generation.audio_path != expected_generation_audio_path:
                    return None
            if require_unique_default:
                competing_default = (
                    durable_db.query(DBGenerationVersion.id)
                    .filter(
                        DBGenerationVersion.generation_id == durable.generation_id,
                        DBGenerationVersion.is_default.is_(True),
                        DBGenerationVersion.id != durable.id,
                    )
                    .first()
                )
                if competing_default is not None:
                    return None
            return _version_response(durable)
    except BaseException:
        logger.error("Could not reconcile ambiguous generation-version write", exc_info=True)
        return None


def list_versions(generation_id: str, db: Session) -> list[GenerationVersionResponse]:
    """List all versions for a generation."""
    versions = (
        db.query(DBGenerationVersion)
        .filter_by(generation_id=generation_id)
        .order_by(DBGenerationVersion.created_at)
        .all()
    )
    return [_version_response(v) for v in versions]


def get_version(version_id: str, db: Session) -> GenerationVersionResponse | None:
    """Get a specific version by ID."""
    v = db.query(DBGenerationVersion).filter_by(id=version_id).first()
    if not v:
        return None
    return _version_response(v)


def get_default_version(generation_id: str, db: Session) -> GenerationVersionResponse | None:
    """Get the default version for a generation."""
    v = db.query(DBGenerationVersion).filter_by(generation_id=generation_id, is_default=True).first()
    if not v:
        # Fallback: return the first version
        v = (
            db.query(DBGenerationVersion)
            .filter_by(generation_id=generation_id)
            .order_by(DBGenerationVersion.created_at)
            .first()
        )
    if not v:
        return None
    return _version_response(v)


def create_version(
    generation_id: str,
    label: str,
    audio_path: str,
    db: Session,
    effects_chain: list[dict] | None = None,
    is_default: bool = False,
    source_version_id: str | None = None,
) -> GenerationVersionResponse:
    """Create a new version for a generation.

    If ``is_default`` is True, all other versions for this generation
    are un-defaulted first.
    """
    version_id = str(uuid.uuid4())
    encoded_effects = json.dumps(effects_chain) if effects_chain else None
    version = DBGenerationVersion(
        id=version_id,
        generation_id=generation_id,
        label=label,
        audio_path=audio_path,
        effects_chain=encoded_effects,
        source_version_id=source_version_id,
        is_default=is_default,
    )
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    expected_generation_audio_path = audio_path if is_default and generation else None
    expected = {
        "generation_id": generation_id,
        "label": label,
        "audio_path": audio_path,
        "effects_chain": encoded_effects,
        "source_version_id": source_version_id,
        "is_default": is_default,
    }
    try:
        if is_default:
            _clear_defaults(generation_id, db)
        db.add(version)
        if expected_generation_audio_path is not None:
            generation.audio_path = audio_path
        db.commit()
        db.refresh(version)
        return _version_response(version)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Generation-version creation rollback failed", exc_info=True)
        if not isinstance(operation_error, IntegrityError):
            durable = _durable_version_write(
                db=db,
                version_id=version_id,
                expected=expected,
                expected_generation_audio_path=expected_generation_audio_path,
                require_unique_default=is_default,
            )
            if durable is not None:
                return durable
        raise operation_error


def set_default_version(version_id: str, db: Session) -> GenerationVersionResponse | None:
    """Set a version as the default for its generation."""
    version = db.query(DBGenerationVersion).filter_by(id=version_id).first()
    if not version:
        return None

    generation = db.query(DBGeneration).filter_by(id=version.generation_id).first()
    expected_generation_audio_path = version.audio_path if generation else None
    expected = {
        "generation_id": version.generation_id,
        "label": version.label,
        "audio_path": version.audio_path,
        "effects_chain": version.effects_chain,
        "source_version_id": version.source_version_id,
        "is_default": True,
    }
    try:
        _clear_defaults(version.generation_id, db)
        version.is_default = True
        if generation is not None:
            generation.audio_path = version.audio_path
        db.commit()
        db.refresh(version)
        return _version_response(version)
    except BaseException as operation_error:
        try:
            db.rollback()
        except BaseException:
            logger.error("Generation-version default rollback failed", exc_info=True)
        durable = _durable_version_write(
            db=db,
            version_id=version_id,
            expected=expected,
            expected_generation_audio_path=expected_generation_audio_path,
            require_unique_default=True,
        )
        if durable is not None:
            return durable
        raise operation_error


def delete_version(version_id: str, db: Session) -> bool:
    """Delete a version. Cannot delete the last remaining version."""
    version = db.query(DBGenerationVersion).filter_by(id=version_id).first()
    if not version:
        return False

    if db.query(DBStoryItem.id).filter_by(version_id=version_id).first() is not None:
        raise VersionInUseError(
            "This version is pinned by a Story item; select another Story version before deleting it"
        )
    if db.query(DBGenerationVersion.id).filter(DBGenerationVersion.source_version_id == version_id).first() is not None:
        raise VersionInUseError("This version is the source of another version; delete the derived version first")

    remaining = (
        db.query(DBGenerationVersion)
        .filter(
            DBGenerationVersion.generation_id == version.generation_id,
            DBGenerationVersion.id != version.id,
        )
        .order_by(DBGenerationVersion.created_at, DBGenerationVersion.id)
        .all()
    )
    if not remaining:
        return False

    was_default = version.is_default
    gen_id = version.generation_id
    generation = db.query(DBGeneration).filter_by(id=gen_id).first()

    # Historical rows can use different storage aliases for the same file. Keep
    # it while any surviving version or generation still owns its canonical path.
    audio_paths = _unreferenced_managed_audio_paths(
        [version.audio_path],
        db,
        deleting_version_ids={version.id},
        changing_generation_ids={gen_id} if was_default and generation else set(),
    )

    with _version_deletion_transaction(db, audio_paths):
        db.delete(version)

        # Delete, deterministic default promotion, and the public generation
        # pointer are one transaction. A failed commit restores the staged WAV.
        if was_default:
            first = remaining[0]
            for candidate in remaining:
                candidate.is_default = candidate.id == first.id
            if generation:
                generation.audio_path = first.audio_path

    return True


def delete_versions_for_generation(generation_id: str, db: Session) -> int:
    """Delete all versions for a generation (used when deleting a generation)."""
    versions = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).all()
    if not versions:
        return 0

    audio_paths = _unreferenced_managed_audio_paths(
        [version.audio_path for version in versions],
        db,
        deleting_version_ids={version.id for version in versions},
        changing_generation_ids=set(),
    )
    with _version_deletion_transaction(
        db,
        audio_paths,
    ):
        for version in versions:
            db.delete(version)
    return len(versions)


def _clear_defaults(generation_id: str, db: Session) -> None:
    """Clear the is_default flag on all versions for a generation."""
    db.query(DBGenerationVersion).filter_by(generation_id=generation_id, is_default=True).update({"is_default": False})
    db.flush()
