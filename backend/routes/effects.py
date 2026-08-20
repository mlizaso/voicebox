"""Effects presets and generation version endpoints."""

import logging
import os
import stat
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import config, models
from ..database import (
    Generation as DBGeneration,
    GenerationVersion as DBGenerationVersion,
    get_db,
)
from ..services import deletion_journal, effects_processing, history
from ..utils.responses import CleanupFileResponse

router = APIRouter()
logger = logging.getLogger(__name__)


def _managed_effects_source(stored_path: str | Path | None) -> Path | None:
    """Resolve one DB audio path lexically without following storage links."""
    relative = config.managed_storage_relative_path(stored_path)
    if relative is None or relative.parts[0] != "generations":
        return None
    candidate = config.get_data_dir() / relative
    try:
        candidate.lstat()
    except OSError:
        return None
    return candidate


def _reconcile_effects_publication(
    intent: deletion_journal.DeletionIntent,
    db: Session,
) -> None:
    try:
        deletion_journal.reconcile_deletion_intent(intent, db)
    except BaseException:
        logger.error(
            "Retaining one effects-version publication intent for startup recovery",
            exc_info=True,
        )


@router.post("/effects/preview/{generation_id}")
async def preview_effects(
    generation_id: str,
    data: models.ApplyEffectsRequest,
    db: Session = Depends(get_db),
):
    """Apply effects to a generation's clean audio and stream back without saving."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation is not completed")

    from ..services import versions as versions_mod
    from ..utils.effects import validate_effects_chain

    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise HTTPException(status_code=400, detail=error)

    all_versions = versions_mod.list_versions(generation_id, db)
    source_version_id = data.source_version_id
    if source_version_id:
        source_version = next((v for v in all_versions if v.id == source_version_id), None)
        if source_version is None:
            raise HTTPException(status_code=404, detail="Source version not found")
        source_path = source_version.audio_path
    else:
        clean_version = next((v for v in all_versions if v.effects_chain is None), None)
        source_path = clean_version.audio_path if clean_version else gen.audio_path
    resolved_source_path = _managed_effects_source(source_path)
    if resolved_source_path is None:
        raise HTTPException(status_code=404, detail="Source audio file not found")
    db.close()

    preview = None
    handed_to_response = False
    try:
        preview = await effects_processing.create_effects_preview(resolved_source_path, chain_dicts)
        safe_generation_id = "".join(
            character for character in generation_id if character.isalnum() or character in ("-", "_")
        )[:80]
        if not safe_generation_id:
            safe_generation_id = "generation"
        response = CleanupFileResponse(
            preview.path,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f'inline; filename="preview_{safe_generation_id}.wav"',
                "Cache-Control": "no-cache, no-store",
            },
            cleanup=preview.cleanup,
        )
        handed_to_response = True
        return response
    except effects_processing.EffectsProcessingLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except effects_processing.EffectsProcessingStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except effects_processing.EffectsProcessingBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except effects_processing.EffectsProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if preview is not None and not handed_to_response:
            preview.cleanup()


@router.get("/effects/available", response_model=models.AvailableEffectsResponse)
async def get_available_effects():
    """List all available effect types with parameter definitions."""
    from ..utils.effects import get_available_effects as _get_effects

    return models.AvailableEffectsResponse(effects=[models.AvailableEffect(**e) for e in _get_effects()])


@router.get("/effects/presets", response_model=list[models.EffectPresetResponse])
async def list_effect_presets(db: Session = Depends(get_db)):
    """List all effect presets (built-in + user-created)."""
    from ..services import effects as effects_mod

    return effects_mod.list_presets(db)


@router.get("/effects/presets/{preset_id}", response_model=models.EffectPresetResponse)
async def get_effect_preset(preset_id: str, db: Session = Depends(get_db)):
    """Get a specific effect preset."""
    from ..services import effects as effects_mod

    preset = effects_mod.get_preset(preset_id, db)
    if not preset:
        raise HTTPException(status_code=404, detail="Preset not found")
    return preset


@router.post("/effects/presets", response_model=models.EffectPresetResponse)
async def create_effect_preset(
    data: models.EffectPresetCreate,
    db: Session = Depends(get_db),
):
    """Create a new effect preset."""
    from ..services import effects as effects_mod

    try:
        return effects_mod.create_preset(data, db)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.put("/effects/presets/{preset_id}", response_model=models.EffectPresetResponse)
async def update_effect_preset(
    preset_id: str,
    data: models.EffectPresetUpdate,
    db: Session = Depends(get_db),
):
    """Update an effect preset."""
    from ..services import effects as effects_mod

    try:
        result = effects_mod.update_preset(preset_id, data, db)
        if not result:
            raise HTTPException(status_code=404, detail="Preset not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/effects/presets/{preset_id}")
async def delete_effect_preset(preset_id: str, db: Session = Depends(get_db)):
    """Delete a user effect preset."""
    from ..services import effects as effects_mod

    try:
        if not effects_mod.delete_preset(preset_id, db):
            raise HTTPException(status_code=404, detail="Preset not found")
        return {"status": "deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get(
    "/generations/{generation_id}/versions",
    response_model=list[models.GenerationVersionResponse],
)
async def list_generation_versions(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """List all versions for a generation."""
    gen = await history.get_generation(generation_id, db)
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")

    from ..services import versions as versions_mod

    return versions_mod.list_versions(generation_id, db)


@router.post(
    "/generations/{generation_id}/versions/apply-effects",
    response_model=models.GenerationVersionResponse,
)
async def apply_effects_to_generation(
    generation_id: str,
    data: models.ApplyEffectsRequest,
    db: Session = Depends(get_db),
):
    """Apply an effects chain to an existing generation, creating a new version."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    if (gen.status or "completed") != "completed":
        raise HTTPException(status_code=400, detail="Generation is not completed")

    from ..services import versions as versions_mod
    from ..utils.effects import validate_effects_chain

    chain_dicts = [e.model_dump() for e in data.effects_chain]
    error = validate_effects_chain(chain_dicts)
    if error:
        raise HTTPException(status_code=400, detail=error)

    all_versions = versions_mod.list_versions(generation_id, db)
    source_version_id = data.source_version_id
    if source_version_id:
        source_version = next((v for v in all_versions if v.id == source_version_id), None)
        if not source_version:
            raise HTTPException(status_code=404, detail="Source version not found")
        source_path = source_version.audio_path
    else:
        clean_version = next((v for v in all_versions if v.effects_chain is None), None)
        if not clean_version:
            source_path = gen.audio_path
        else:
            source_path = clean_version.audio_path
            source_version_id = clean_version.id

    resolved_source_path = _managed_effects_source(source_path)
    if resolved_source_path is None:
        raise HTTPException(status_code=404, detail="Source audio file not found")

    version_id = str(uuid.uuid4())
    generations_dir = config.get_generations_dir()
    processed_path = generations_dir / f"{generation_id}_{version_id[:8]}.wav"
    pending_path = generations_dir / f".voicebox-delete-effects-{version_id}.part.wav"
    processed_relative = Path("generations") / processed_path.name
    pending_relative = Path("generations") / pending_path.name
    publication_intent: deletion_journal.DeletionIntent | None = None
    pending_descriptor: int | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        pending_descriptor = os.open(pending_path, flags, 0o600)
        pending_stat = os.fstat(pending_descriptor)
        publication_intent = deletion_journal.prepare_deletion_intent(
            kind=deletion_journal.GENERATION_AUDIO_PUBLICATION,
            original=processed_relative,
            staged=pending_relative,
            entry_stat=pending_stat,
            owner_id=version_id,
        )
        await effects_processing.render_effects_to_descriptor(
            resolved_source_path,
            pending_descriptor,
            generations_dir,
            chain_dicts,
        )
        rendered_stat = os.fstat(pending_descriptor)
        if (
            not stat.S_ISREG(rendered_stat.st_mode)
            or rendered_stat.st_dev != pending_stat.st_dev
            or rendered_stat.st_ino != pending_stat.st_ino
        ):
            raise OSError("Processed audio publication replaced its journaled inode")
        os.close(pending_descriptor)
        pending_descriptor = None
        pending_stat = deletion_journal.managed_entry_stat(pending_relative)
        if pending_stat is None or not stat.S_ISREG(pending_stat.st_mode):
            raise OSError("Processed audio publication is not a regular file")
        if not deletion_journal.entry_matches_intent(publication_intent, pending_stat):
            raise OSError("Processed audio publication replaced its journaled inode")
    except BaseException as processing_error:
        if pending_descriptor is not None:
            os.close(pending_descriptor)
        if publication_intent is None:
            try:
                deletion_journal.discard_managed_entry(pending_relative)
            except BaseException:
                logger.error("Deferred cleanup of unpublished effects audio", exc_info=True)
        else:
            _reconcile_effects_publication(publication_intent, db)
        if isinstance(processing_error, effects_processing.EffectsProcessingLimitError):
            raise HTTPException(status_code=413, detail=str(processing_error)) from processing_error
        if isinstance(processing_error, effects_processing.EffectsProcessingStorageError):
            raise HTTPException(status_code=507, detail=str(processing_error)) from processing_error
        if isinstance(processing_error, effects_processing.EffectsProcessingBusyError):
            raise HTTPException(
                status_code=409,
                detail=str(processing_error),
                headers={"Retry-After": "1"},
            ) from processing_error
        if isinstance(processing_error, effects_processing.EffectsProcessingError):
            raise HTTPException(status_code=400, detail=str(processing_error)) from processing_error
        raise

    # Both expensive operations above yield to other requests. Re-read the
    # parent only after the final await; SQLite foreign keys are not guaranteed.
    db.expire_all()
    current_generation = db.query(DBGeneration).populate_existing().filter_by(id=generation_id).first()
    if current_generation is None:
        _reconcile_effects_publication(publication_intent, db)
        raise HTTPException(status_code=404, detail="Generation was deleted while applying effects")
    if source_version_id is not None:
        current_source_version = (
            db.query(DBGenerationVersion)
            .populate_existing()
            .filter_by(id=source_version_id, generation_id=generation_id)
            .first()
        )
        if current_source_version is None:
            _reconcile_effects_publication(publication_intent, db)
            raise HTTPException(
                status_code=404,
                detail="Source version was deleted while applying effects",
            )

    try:
        deletion_journal.rename_managed_entry(pending_relative, processed_relative)
    except BaseException:
        # The rename and its directory fsync cannot be one atomic syscall. If
        # the flush fails after the rename, reconcile both possible locations
        # now instead of retaining an orphan until the next process startup.
        _reconcile_effects_publication(publication_intent, db)
        raise

    label = data.label or f"version-{len(all_versions) + 1}"

    try:
        version = versions_mod.create_version(
            generation_id=generation_id,
            label=label,
            audio_path=config.to_storage_path(processed_path),
            db=db,
            effects_chain=chain_dicts,
            is_default=data.set_as_default,
            source_version_id=source_version_id,
        )
    except BaseException as operation_error:
        rollback_error: BaseException | None = None
        try:
            db.rollback()
        except BaseException as exc:
            rollback_error = exc
            logger.error("Effects-version rollback failed", exc_info=True)
        if rollback_error is None:
            _reconcile_effects_publication(publication_intent, db)
        else:
            try:
                with deletion_journal.durable_reconciliation_session(db) as durable_db:
                    _reconcile_effects_publication(publication_intent, durable_db)
            except BaseException:
                logger.error("Deferred effects-version reconciliation", exc_info=True)
            raise rollback_error from operation_error
        raise

    try:
        deletion_journal.finish_deletion_intent(publication_intent)
    except OSError:
        # The version row and WAV are already durable. A stale intent is safe
        # for startup recovery and must not turn success into a retryable 500.
        logger.warning("Deferred cleanup of a committed effects-version publication intent")

    return version


@router.put(
    "/generations/{generation_id}/versions/{version_id}/set-default",
    response_model=models.GenerationVersionResponse,
)
async def set_default_version(
    generation_id: str,
    version_id: str,
    db: Session = Depends(get_db),
):
    """Set a specific version as the default for a generation."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version or version.generation_id != generation_id:
        raise HTTPException(status_code=404, detail="Version not found")

    result = versions_mod.set_default_version(version_id, db)
    if not result:
        raise HTTPException(status_code=404, detail="Version not found")
    return result


@router.delete("/generations/{generation_id}/versions/{version_id}")
async def delete_generation_version(
    generation_id: str,
    version_id: str,
    db: Session = Depends(get_db),
):
    """Delete a version. Cannot delete the last remaining version."""
    from ..services import versions as versions_mod

    version = versions_mod.get_version(version_id, db)
    if not version or version.generation_id != generation_id:
        raise HTTPException(status_code=404, detail="Version not found")

    try:
        deleted = versions_mod.delete_version(version_id, db)
    except versions_mod.VersionInUseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete the last remaining version",
        )
    return {"status": "deleted"}
