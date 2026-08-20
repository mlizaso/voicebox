"""Resource bounds for MCP audio transcription."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.mcp_server import tools
from backend.services import transcribe as transcribe_service


class _Whisper:
    model_size = "small"

    def __init__(self) -> None:
        self.transcribed = False

    def is_loaded(self) -> bool:
        return True

    def _is_model_cached(self, _model_size: str) -> bool:
        return True

    async def transcribe(self, _path: str, _language: str | None, _model_size: str) -> str:
        self.transcribed = True
        return "bounded"


def test_base64_rejects_encoded_payload_before_decode(monkeypatch):
    monkeypatch.setattr(tools, "MAX_TRANSCRIBE_BASE64_CHARS", 4)
    monkeypatch.setattr(
        tools.b64,
        "b64decode",
        lambda *_args, **_kwargs: pytest.fail("oversized input was decoded"),
    )

    with pytest.raises(ValueError, match="exceeds"):
        tools._decode_audio_base64_bounded("A" * 5)


@pytest.mark.asyncio
async def test_transcribe_probes_duration_without_decoding_pcm(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"container")
    whisper = _Whisper()
    monkeypatch.setattr(transcribe_service, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(tools.librosa, "get_duration", lambda *, path: 12.5)

    result = await tools._transcribe_file(audio_path, None, "small")

    assert result["duration"] == 12.5
    assert result["text"] == "bounded"
    assert whisper.transcribed is True


@pytest.mark.asyncio
async def test_transcribe_rejects_overlong_audio_before_whisper(tmp_path, monkeypatch):
    audio_path = tmp_path / "audio.mp3"
    audio_path.write_bytes(b"container")
    whisper = _Whisper()
    monkeypatch.setattr(transcribe_service, "get_whisper_model", lambda: whisper)
    monkeypatch.setattr(tools, "AUDIO_UPLOAD_MAX_DURATION_SECONDS", 30)
    monkeypatch.setattr(tools.librosa, "get_duration", lambda *, path: 31.0)

    with pytest.raises(ValueError, match="too long"):
        await tools._transcribe_file(Path(audio_path), None, "small")

    assert whisper.transcribed is False
