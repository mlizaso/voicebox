"""Reject invalid upload metadata before decoding or changing model state."""

import io

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient

from backend.database import get_db
from backend.routes import captures, profiles, transcription
from backend.utils.upload_limits import spool_upload_bounded


@pytest.mark.parametrize(
    ("path", "data"),
    [
        ("/captures", {"stt_model": "attacker-model"}),
        ("/captures", {"language": "x" * 1024}),
        ("/transcribe", {"language": "invalid"}),
        ("/transcribe", {"model": "attacker-model"}),
    ],
)
def test_invalid_transcription_metadata_never_reads_audio(monkeypatch, path, data):
    app = FastAPI()
    app.include_router(captures.router)
    app.include_router(transcription.router)
    app.dependency_overrides[get_db] = lambda: None

    async def unexpected_spool(*_args, **_kwargs):
        pytest.fail("Invalid metadata must be rejected before reading audio")

    monkeypatch.setattr(captures, "spool_upload_bounded", unexpected_spool)
    monkeypatch.setattr(transcription, "spool_upload_bounded", unexpected_spool)
    response = TestClient(app).post(path, data=data, files={"file": ("sample.wav", b"unused")})
    assert response.status_code in {400, 422}


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["." + "x" * 256, ".png\0", ".a\\b"])
async def test_unsafe_upload_suffix_is_rejected_before_tempfile(monkeypatch, suffix):
    monkeypatch.setattr(
        "backend.utils.upload_limits.tempfile.mkstemp",
        lambda **_kwargs: pytest.fail("Unsafe suffix reached the filesystem"),
    )
    with pytest.raises(ValueError, match="suffix"):
        await spool_upload_bounded(UploadFile(file=io.BytesIO(b"unused")), max_bytes=100, suffix=suffix)


def test_avatar_rejects_overlong_extension_as_client_error():
    app = FastAPI()
    app.include_router(profiles.router)
    app.dependency_overrides[get_db] = lambda: None
    response = TestClient(app).post("/profiles/profile-id/avatar", files={"file": ("avatar." + "x" * 1024, b"unused")})
    assert response.status_code == 400
