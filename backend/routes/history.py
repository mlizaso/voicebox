"""Generation history endpoints."""

import mimetypes

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..app import safe_content_disposition
from ..database import Generation as DBGeneration, VoiceProfile as DBVoiceProfile, get_db
from ..services import export_import, history
from ..services.task_queue import generation_job_is_active
from ..utils.responses import CleanupFileResponse
from ..utils.upload_limits import UploadSizeLimitError, spool_upload_bounded

router = APIRouter()
GENERATION_ARCHIVE_MAX_BYTES = (
    export_import.GENERATION_ARCHIVE_MAX_TOTAL_BYTES + export_import.ARCHIVE_EXPORT_OVERHEAD_BYTES
)


@router.get("/history", response_model=models.HistoryListResponse)
async def list_history(
    profile_id: str | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """List generation history with optional filters."""
    query = models.HistoryQuery(
        profile_id=profile_id,
        search=search,
        limit=limit,
        offset=offset,
    )
    return await history.list_generations(query, db)


@router.get("/history/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get generation statistics."""
    return await history.get_generation_stats(db)


@router.post("/history/import")
async def import_generation(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Import a generation from a ZIP archive."""
    archive_path = None
    try:
        archive_path = await spool_upload_bounded(
            file,
            max_bytes=GENERATION_ARCHIVE_MAX_BYTES,
            suffix=".zip",
        )
        result = await export_import.import_generation_from_zip(archive_path, db)
        return result
    except UploadSizeLimitError as exc:
        raise HTTPException(
            status_code=413,
            detail=(f"Generation archive is too large (max {exc.max_bytes // (1024 * 1024)} MB)"),
        ) from exc
    except export_import.ArchiveImportBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except export_import.ArchiveImportStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to import generation archive",
        ) from exc
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)


@router.delete("/history/failed")
async def clear_failed_generations(db: Session = Depends(get_db)):
    """Delete every generation with status='failed'. Used by the UI's 'Clear failed' button (#410)."""
    try:
        count = await history.delete_failed_generations(db)
    except history.GenerationInUseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"deleted": count}


@router.get("/history/{generation_id}", response_model=models.HistoryResponse)
async def get_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Get a generation by ID."""
    result = (
        db.query(DBGeneration, DBVoiceProfile.name.label("profile_name"))
        .join(DBVoiceProfile, DBGeneration.profile_id == DBVoiceProfile.id)
        .filter(DBGeneration.id == generation_id)
        .first()
    )

    if not result:
        raise HTTPException(status_code=404, detail="Generation not found")

    gen, profile_name = result
    versions, active_version_id = history._get_versions_for_generation(gen.id, db)
    return models.HistoryResponse(
        id=gen.id,
        profile_id=gen.profile_id,
        profile_name=profile_name,
        text=gen.text,
        language=gen.language,
        audio_path=gen.audio_path,
        duration=gen.duration,
        seed=gen.seed,
        instruct=gen.instruct,
        engine=gen.engine or "qwen",
        model_size=gen.model_size,
        status=gen.status or "completed",
        error=gen.error,
        is_favorited=bool(gen.is_favorited),
        exact_request_sha256=gen.exact_request_sha256,
        exact_envelope_sha256=gen.exact_envelope_sha256,
        exact_effects_json=gen.exact_effects_json,
        exact_voice_snapshot_json=gen.exact_voice_snapshot_json,
        voice_binding_sha256=gen.voice_binding_sha256,
        created_at=gen.created_at,
        versions=versions,
        active_version_id=active_version_id,
    )


@router.post("/history/{generation_id}/favorite")
async def toggle_favorite(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Toggle the favorite status of a generation."""
    gen = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not gen:
        raise HTTPException(status_code=404, detail="Generation not found")
    gen.is_favorited = not gen.is_favorited
    db.commit()
    return {"is_favorited": gen.is_favorited}


@router.delete("/history/{generation_id}")
async def delete_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Delete a generation."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if generation is None:
        raise HTTPException(status_code=404, detail="Generation not found")
    if generation_job_is_active(generation_id) or (generation.status or "completed") in {
        "pending",
        "queued",
        "loading_model",
        "generating",
    }:
        raise HTTPException(
            status_code=409,
            detail="Cancel this generation and wait for a terminal state before deleting it",
        )
    try:
        success = await history.delete_generation(generation_id, db)
    except history.GenerationInUseError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not success:
        raise HTTPException(status_code=404, detail="Generation not found")
    return {"message": "Generation deleted successfully"}


@router.get("/history/{generation_id}/export")
async def export_generation(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Export a generation as a ZIP archive."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")
    generation_text = generation.text

    archive_export = None
    handed_to_response = False
    try:
        archive_export = await export_import.export_generation_to_zip(generation_id, db)

        safe_text = "".join(c for c in generation_text[:30] if c.isalnum() or c in (" ", "-", "_")).strip()
        if not safe_text:
            safe_text = "generation"
        # Append a short id so exports of similarly-worded generations don't collide
        # on the same filename (the first 30 chars are frequently identical).
        filename = f"generation-{safe_text}-{generation_id[:8]}.voicebox.zip"
        db.close()

        response = CleanupFileResponse(
            archive_export.path,
            media_type="application/zip",
            headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
            cleanup=archive_export.cleanup,
        )
        handed_to_response = True
        return response
    except export_import.ArchiveExportLimitError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except export_import.ArchiveExportStorageError as exc:
        raise HTTPException(status_code=507, detail=str(exc)) from exc
    except export_import.ArchiveExportBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to export generation") from exc
    finally:
        if archive_export is not None and not handed_to_response:
            archive_export.cleanup()


@router.get("/history/{generation_id}/export-audio")
async def export_generation_audio(
    generation_id: str,
    db: Session = Depends(get_db),
):
    """Export only the audio file from a generation."""
    generation = db.query(DBGeneration).filter_by(id=generation_id).first()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    if not generation.audio_path:
        raise HTTPException(status_code=404, detail="Generation has no audio file")

    audio_path = config.resolve_storage_path(generation.audio_path)
    if audio_path is None or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")

    safe_text = "".join(c for c in generation.text[:30] if c.isalnum() or c in (" ", "-", "_")).strip()
    if not safe_text:
        safe_text = "generation"
    # Append a short id so exports of similarly-worded generations don't collide
    # on the same filename (the first 30 chars are frequently identical).
    audio_suffix = audio_path.suffix or ".wav"
    filename = f"{safe_text}-{generation_id[:8]}{audio_suffix}"
    media_type, _encoding = mimetypes.guess_type(audio_path.name)
    db.close()

    return FileResponse(
        audio_path,
        media_type=media_type or "application/octet-stream",
        headers={"Content-Disposition": safe_content_disposition("attachment", filename)},
    )
