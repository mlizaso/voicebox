"""Focused contracts for reusable HTTP response helpers."""

from pathlib import Path

import pytest

from backend.utils.responses import CleanupFileResponse, safe_content_disposition


def test_content_disposition_keeps_ascii_fallback_header_safe() -> None:
    header = safe_content_disposition("attachment", 'résumé "draft"\r\n.wav')

    assert 'filename="rsum draft.wav"' in header
    assert "filename*=UTF-8''r%C3%A9sum%C3%A9%20%22draft%22%0D%0A.wav" in header
    assert "\r" not in header
    assert "\n" not in header


def _scope() -> dict:
    return {"type": "http", "method": "GET", "headers": []}


async def _receive() -> dict:
    return {"type": "http.disconnect"}


@pytest.mark.asyncio
async def test_cleanup_file_response_runs_cleanup_once_after_success(tmp_path: Path) -> None:
    source = tmp_path / "download.txt"
    source.write_text("download")
    calls: list[Path] = []
    response = CleanupFileResponse(source, cleanup=lambda: calls.append(source))

    async def send(_message: dict) -> None:
        return None

    await response(_scope(), _receive, send)

    assert calls == [source]


@pytest.mark.asyncio
async def test_cleanup_file_response_runs_cleanup_once_when_send_fails(tmp_path: Path) -> None:
    source = tmp_path / "download.txt"
    source.write_text("download")
    calls: list[Path] = []
    response = CleanupFileResponse(source, cleanup=lambda: calls.append(source))

    async def send(message: dict) -> None:
        if message["type"] == "http.response.body":
            raise ConnectionError("client disconnected")

    with pytest.raises(ConnectionError, match="client disconnected"):
        await response(_scope(), _receive, send)

    assert calls == [source]


@pytest.mark.asyncio
async def test_cleanup_file_response_does_not_retry_failed_cleanup(tmp_path: Path) -> None:
    source = tmp_path / "download.txt"
    source.write_text("download")
    calls: list[Path] = []

    def cleanup() -> None:
        calls.append(source)
        raise RuntimeError("cleanup failed")

    response = CleanupFileResponse(source, cleanup=cleanup)

    async def send(_message: dict) -> None:
        return None

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await response(_scope(), _receive, send)

    assert calls == [source]
