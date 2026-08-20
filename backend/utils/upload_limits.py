"""Bounded streaming helpers for untrusted multipart uploads."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

from fastapi import UploadFile

UPLOAD_CHUNK_BYTES = 1024 * 1024
AUDIO_UPLOAD_MAX_BYTES = 100 * 1024 * 1024
AUDIO_UPLOAD_MAX_DURATION_SECONDS = 30 * 60


class UploadSizeLimitError(ValueError):
    """Raised after an upload exceeds its endpoint's byte budget."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Upload exceeds the {max_bytes}-byte limit")


class UploadDurationLimitError(ValueError):
    """Raised when bounded decoding proves an audio upload is too long."""

    def __init__(self, max_seconds: int) -> None:
        self.max_seconds = max_seconds
        super().__init__(f"Audio exceeds the {max_seconds}-second duration limit")


async def read_upload_bounded(
    upload: UploadFile,
    *,
    max_bytes: int,
    chunk_bytes: int = UPLOAD_CHUNK_BYTES,
) -> bytes:
    """Read at most ``max_bytes`` without first materializing the whole body."""
    if max_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("Upload bounds must be positive")

    output = io.BytesIO()
    total = 0
    while chunk := await upload.read(chunk_bytes):
        total += len(chunk)
        if total > max_bytes:
            raise UploadSizeLimitError(max_bytes)
        output.write(chunk)
    return output.getvalue()


async def spool_upload_bounded(
    upload: UploadFile,
    *,
    max_bytes: int,
    suffix: str = "",
    chunk_bytes: int = UPLOAD_CHUNK_BYTES,
) -> Path:
    """Copy an upload to a private temp file while enforcing a hard byte cap.

    The caller owns the returned path and must unlink it. Failed and oversized
    uploads are removed here before the exception escapes.
    """
    if max_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("Upload bounds must be positive")
    if suffix and (Path(suffix).name != suffix or not suffix.startswith(".")):
        raise ValueError("Temporary upload suffix must be one safe extension")

    descriptor, raw_path = tempfile.mkstemp(prefix="voicebox-upload-", suffix=suffix)
    path = Path(raw_path)
    total = 0
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as destination:
            while chunk := await upload.read(chunk_bytes):
                total += len(chunk)
                if total > max_bytes:
                    raise UploadSizeLimitError(max_bytes)
                destination.write(chunk)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise
