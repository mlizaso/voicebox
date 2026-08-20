"""Transcription endpoints."""

from contextlib import suppress
from pathlib import Path

import librosa
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .. import models
from ..backends.mlx_tts_lifecycle import (
    run_blocking_operation_cancellation_safe,
    run_tts_operation_cancellation_safe,
)
from ..services import transcribe
from ..services.task_queue import create_background_task
from ..utils.progress import get_progress_manager
from ..utils.tasks import get_task_manager
from ..utils.upload_limits import (
    AUDIO_UPLOAD_MAX_DURATION_SECONDS,
    UploadSizeLimitError,
    spool_upload_bounded,
)

router = APIRouter()

UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB
TRANSCRIPTION_MAX_FILE_BYTES = 100 * 1024 * 1024

# Same set profiles.py accepts for voice samples. librosa picks its decoder from the
# file extension, so the temp file has to keep the uploaded one.
ALLOWED_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}


@router.post("/transcribe", response_model=models.TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Form(None),
    model: str | None = Form(None),
):
    """Transcribe audio file to text."""
    uploaded_ext = Path(file.filename or "").suffix.lower()
    file_suffix = uploaded_ext if uploaded_ext in ALLOWED_AUDIO_EXTS else ".wav"

    tmp_path = None
    stt_path = None
    try:
        tmp_path = await spool_upload_bounded(
            file,
            max_bytes=TRANSCRIPTION_MAX_FILE_BYTES,
            suffix=file_suffix,
            chunk_bytes=UPLOAD_CHUNK_SIZE,
        )
        stt_path = tmp_path
        from ..backends import WHISPER_HF_REPOS
        from ..utils.audio import save_audio

        try:
            audio, sr = await run_blocking_operation_cancellation_safe(
                librosa.load,
                str(tmp_path),
                sr=24_000,
                mono=True,
                duration=AUDIO_UPLOAD_MAX_DURATION_SECONDS + 1,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio could not be decoded",
            ) from exc
        duration = len(audio) / sr
        if duration > AUDIO_UPLOAD_MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=413,
                detail=(f"Audio duration is too long (max {AUDIO_UPLOAD_MAX_DURATION_SECONDS // 60} minutes)"),
            )

        # The STT backend (mlx_audio.stt -> miniaudio) only decodes
        # WAV/FLAC/MP3/Vorbis, so browser recordings uploaded as WebM/Opus
        # fail with "unsupported file format" (issue: web-mode dictation).
        # librosa already decoded the file above (it falls back to
        # audioread/ffmpeg for exotic containers), so re-encode that PCM to a
        # temp WAV and hand *that* to Whisper. WAV inputs pass through
        # unchanged.
        if file_suffix != ".wav":
            stt_path = f"{tmp_path}.stt.wav"
            await run_blocking_operation_cancellation_safe(
                save_audio,
                audio,
                stt_path,
                sr,
            )

        whisper_model = transcribe.get_whisper_model()
        model_size = model if model else whisper_model.model_size

        valid_sizes = list(WHISPER_HF_REPOS.keys())
        if model_size not in valid_sizes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model size '{model_size}'. Must be one of: {', '.join(valid_sizes)}",
            )

        already_loaded = whisper_model.is_loaded() and whisper_model.model_size == model_size
        if not already_loaded and not whisper_model._is_model_cached(model_size):
            progress_model_name = f"whisper-{model_size}"
            task_manager = get_task_manager()
            progress_manager = get_progress_manager()
            from .models import (
                ModelDownloadAlreadyActiveError,
                start_owned_model_download_task,
            )

            async def download_whisper_background():
                try:
                    await run_tts_operation_cancellation_safe(
                        whisper_model,
                        whisper_model.load_model_async(model_size),
                    )
                    task_manager.complete_download(progress_model_name)
                except Exception as e:
                    task_manager.error_download(progress_model_name, str(e))

            with suppress(ModelDownloadAlreadyActiveError):
                start_owned_model_download_task(
                    progress_model_name,
                    download_whisper_background(),
                    task_manager=task_manager,
                    progress_manager=progress_manager,
                    task_factory=create_background_task,
                )

            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"Whisper model {model_size} is being downloaded. Please wait and try again.",
                    "model_name": progress_model_name,
                    "downloading": True,
                },
            )

        text = await whisper_model.transcribe(stt_path, language, model_size)

        return models.TranscriptionResponse(
            text=text,
            duration=duration,
        )

    except UploadSizeLimitError as exc:
        raise HTTPException(
            status_code=413,
            detail=(f"Audio file is too large (max {exc.max_bytes // (1024 * 1024)} MB)"),
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded audio",
        ) from exc
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if stt_path is not None and stt_path != tmp_path:
            Path(stt_path).unlink(missing_ok=True)
