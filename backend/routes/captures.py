"""Capture (voice input) endpoints."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .. import config, models
from ..backends import get_llm_model_configs, get_stt_model_configs
from ..backends.base import is_model_cached
from ..database import Capture as DBCapture, get_db
from ..services import captures as captures_service, settings as settings_service
from ..services.refinement import RefinementFlags
from ..utils.upload_limits import (
    AUDIO_UPLOAD_MAX_BYTES,
    UploadDurationLimitError,
    UploadSizeLimitError,
    spool_upload_bounded,
)

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB
CAPTURE_MAX_FILE_BYTES = AUDIO_UPLOAD_MAX_BYTES
CAPTURE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}


@router.post("/captures", response_model=models.CaptureCreateResponse)
async def create_capture_endpoint(
    file: UploadFile = File(...),
    source: str = Form("file"),
    language: str | None = Form(None),
    stt_model: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """Upload audio, run STT, persist the capture."""
    upload_path = None
    try:
        try:
            uploaded_suffix = Path(file.filename or "").suffix.lower()
            upload_path = await spool_upload_bounded(
                file,
                max_bytes=CAPTURE_MAX_FILE_BYTES,
                suffix=(uploaded_suffix if uploaded_suffix in CAPTURE_AUDIO_EXTENSIONS else ".wav"),
                chunk_bytes=UPLOAD_CHUNK_SIZE,
            )
        except UploadSizeLimitError as exc:
            raise HTTPException(
                status_code=413,
                detail=(f"Capture is too large (max {exc.max_bytes // (1024 * 1024)} MB)"),
            ) from exc

        if upload_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        saved = settings_service.get_capture_settings(db)
        resolved_stt = stt_model or saved.stt_model
        if language is None:
            resolved_language = None if saved.language == "auto" else saved.language
        else:
            resolved_language = None if language == "auto" else language

        # The capture service does not need a DB connection until it publishes
        # the final row. Release the settings read before STT waits on the
        # process-wide inference guard.
        db.close()

        try:
            capture = await captures_service.create_capture(
                audio_bytes=upload_path,
                filename=file.filename or "capture.wav",
                source=source,
                language=resolved_language,
                stt_model=resolved_stt,
                db=db,
            )
        except UploadSizeLimitError as exc:
            raise HTTPException(
                status_code=413,
                detail=(f"Capture is too large (max {exc.max_bytes // (1024 * 1024)} MB)"),
            ) from exc
        except UploadDurationLimitError as exc:
            raise HTTPException(
                status_code=413,
                detail=(f"Capture duration is too long (max {exc.max_seconds // 60} minutes)"),
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to create capture")
            raise HTTPException(status_code=500, detail="Failed to create capture") from exc

        return models.CaptureCreateResponse(
            **capture.model_dump(),
            auto_refine=bool(saved.auto_refine),
            allow_auto_paste=bool(saved.allow_auto_paste),
        )
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)


@router.get("/captures", response_model=models.CaptureListResponse)
async def list_captures_endpoint(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    items, total = captures_service.list_captures(db, limit=limit, offset=offset)
    return models.CaptureListResponse(items=items, total=total)


@router.get("/captures/{capture_id}", response_model=models.CaptureResponse)
async def get_capture_endpoint(capture_id: str, db: Session = Depends(get_db)):
    capture = captures_service.get_capture(capture_id, db)
    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture


@router.get("/captures/{capture_id}/audio")
async def get_capture_audio_endpoint(capture_id: str, db: Session = Depends(get_db)):
    """Stream the original capture audio file."""
    row = db.query(DBCapture).filter(DBCapture.id == capture_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Capture not found")

    audio_path = config.resolve_storage_path(row.audio_path)
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    db.close()

    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"capture_{capture_id}.wav",
    )


@router.delete("/captures/{capture_id}")
async def delete_capture_endpoint(capture_id: str, db: Session = Depends(get_db)):
    deleted = captures_service.delete_capture(capture_id, db)
    if not deleted:
        raise HTTPException(status_code=404, detail="Capture not found")
    return {"message": f"Capture {capture_id} deleted"}


@router.post("/captures/{capture_id}/refine", response_model=models.CaptureResponse)
async def refine_capture_endpoint(
    capture_id: str,
    request: models.CaptureRefineRequest,
    db: Session = Depends(get_db),
):
    saved = settings_service.get_capture_settings(db)
    if request.flags is not None:
        flags = RefinementFlags(
            smart_cleanup=request.flags.smart_cleanup,
            self_correction=request.flags.self_correction,
            preserve_technical=request.flags.preserve_technical,
        )
    else:
        flags = RefinementFlags(
            smart_cleanup=saved.smart_cleanup,
            self_correction=saved.self_correction,
            preserve_technical=saved.preserve_technical,
        )

    resolved_model = request.model_size or saved.llm_model
    db.close()

    try:
        capture = await captures_service.refine_capture(
            capture_id=capture_id,
            flags=flags,
            model_size=resolved_model,
            db=db,
        )
    except captures_service.CaptureTranscriptChangedError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception("Refinement failed for capture %s", capture_id)
        raise HTTPException(status_code=500, detail="Failed to refine capture") from e

    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture


@router.get("/capture/readiness", response_model=models.CaptureReadinessResponse)
async def capture_readiness_endpoint(db: Session = Depends(get_db)):
    """Whether the STT and LLM models the user has selected are downloaded.

    The frontend gates the global hotkey on this — pressing the chord with
    a missing model would otherwise produce a stuck "transcribing" pill that
    waits forever for a download to finish. Checks on-disk cache, not RAM
    load, so the answer survives backend restarts.
    """
    saved = settings_service.get_capture_settings(db)

    stt_cfg = next(
        (c for c in get_stt_model_configs() if c.model_size == saved.stt_model),
        None,
    )
    llm_cfg = next(
        (c for c in get_llm_model_configs() if c.model_size == saved.llm_model),
        None,
    )

    if stt_cfg is None or llm_cfg is None:
        # Should be impossible — both fields are pattern-validated against
        # known sizes — but bail loudly rather than return half a response.
        raise HTTPException(
            status_code=500,
            detail=f"No model config for stt={saved.stt_model} or llm={saved.llm_model}",
        )

    return models.CaptureReadinessResponse(
        stt=models.ModelReadiness(
            ready=is_model_cached(stt_cfg.hf_repo_id),
            model_name=stt_cfg.model_name,
            display_name=stt_cfg.display_name,
            size=stt_cfg.model_size,
            size_mb=stt_cfg.size_mb or None,
        ),
        llm=models.ModelReadiness(
            ready=is_model_cached(llm_cfg.hf_repo_id),
            model_name=llm_cfg.model_name,
            display_name=llm_cfg.display_name,
            size=llm_cfg.model_size,
            size_mb=llm_cfg.size_mb or None,
        ),
    )


@router.post("/captures/{capture_id}/retranscribe", response_model=models.CaptureResponse)
async def retranscribe_capture_endpoint(
    capture_id: str,
    request: models.CaptureRetranscribeRequest,
    db: Session = Depends(get_db),
):
    saved = settings_service.get_capture_settings(db)
    resolved_stt = request.model or saved.stt_model
    if request.language is None:
        resolved_language = None if saved.language == "auto" else saved.language
    else:
        resolved_language = request.language

    db.close()

    try:
        capture = await captures_service.retranscribe_capture(
            capture_id=capture_id,
            stt_model=resolved_stt,
            language=resolved_language,
            db=db,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=410, detail=str(e)) from e
    except Exception as e:
        logger.exception("Retranscribe failed for capture %s", capture_id)
        raise HTTPException(status_code=500, detail="Failed to retranscribe capture") from e

    if not capture:
        raise HTTPException(status_code=404, detail="Capture not found")
    return capture
