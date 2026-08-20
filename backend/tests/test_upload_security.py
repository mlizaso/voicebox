"""Bounded multipart and image-decompression regressions."""

import asyncio
import io
import os
from types import SimpleNamespace

import pytest
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.testclient import TestClient

from backend.request_limits import RequestBodyLimitMiddleware, request_body_limit
from backend.utils import images, upload_limits


def _multipart_body(boundary: str, payload: bytes) -> bytes:
    return (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )


async def _call_asgi(
    app,
    *,
    body: bytes,
    content_length: int | bytes | None,
    chunk_bytes: int,
) -> list[dict]:
    boundary = "voicebox-boundary"
    headers = [(b"host", b"testserver"), (b"content-type", f"multipart/form-data; boundary={boundary}".encode())]
    if content_length is not None:
        encoded_content_length = content_length if isinstance(content_length, bytes) else str(content_length).encode()
        headers.append((b"content-length", encoded_content_length))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/upload",
        "raw_path": b"/upload",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    offset = 0

    async def receive():
        nonlocal offset
        chunk = body[offset : offset + chunk_bytes]
        offset += len(chunk)
        return {
            "type": "http.request",
            "body": chunk,
            "more_body": offset < len(body),
        }

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


def _multipart_app(*, max_body_bytes: int) -> tuple[FastAPI, dict[str, int]]:
    app = FastAPI()
    calls = {"routes": 0, "file_bytes": 0}
    app.add_middleware(
        RequestBodyLimitMiddleware,
        exact_path_limits={("POST", "/upload"): max_body_bytes},
        admission_threshold_bytes=0,
        temp_storage_reserve_bytes=0,
    )

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        calls["routes"] += 1
        calls["file_bytes"] = len(await file.read())
        return {"ok": True}

    return app, calls


@pytest.mark.asyncio
async def test_second_concurrent_large_multipart_is_rejected_without_consuming_body(monkeypatch):
    boundary = "voicebox-boundary"
    body = _multipart_body(boundary, b"payload")
    first_receive_started = asyncio.Event()
    release_first_receive = asyncio.Event()
    second_receive_calls = 0

    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        exact_path_limits={("POST", "/upload"): len(body) + 1},
        admission_threshold_bytes=0,
        max_concurrent_large_requests=1,
        temp_storage_reserve_bytes=0,
    )
    monkeypatch.setattr(
        "backend.utils.disk_reservations.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10**12),
    )

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        return {"size": len(await file.read())}

    def scope():
        return {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/upload",
            "raw_path": b"/upload",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

    first_sent: list[dict] = []
    first_delivered = False

    async def first_receive():
        nonlocal first_delivered
        if not first_delivered:
            first_receive_started.set()
            await release_first_receive.wait()
            first_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def first_send(message):
        first_sent.append(message)

    first_task = asyncio.create_task(app(scope(), first_receive, first_send))
    await asyncio.wait_for(first_receive_started.wait(), timeout=1)

    second_sent: list[dict] = []

    async def second_receive():
        nonlocal second_receive_calls
        second_receive_calls += 1
        return {"type": "http.request", "body": body, "more_body": False}

    async def second_send(message):
        second_sent.append(message)

    await app(scope(), second_receive, second_send)

    assert second_sent[0]["status"] == 429
    assert second_receive_calls == 0

    release_first_receive.set()
    await first_task
    assert first_sent[0]["status"] == 200


@pytest.mark.asyncio
async def test_second_concurrent_large_mcp_json_is_rejected_before_body_parse(monkeypatch):
    body = b'{"audio_base64":"' + (b"A" * 256) + b'"}'
    first_receive_started = asyncio.Event()
    release_first_receive = asyncio.Event()
    second_receive_calls = 0

    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        exact_path_limits={("POST", "/mcp"): len(body) + 1},
        admission_threshold_bytes=64,
        max_concurrent_large_mcp_requests=1,
    )
    monkeypatch.setattr(
        "backend.utils.disk_reservations.shutil.disk_usage",
        lambda _path: pytest.fail("MCP JSON admission must not reserve spool disk"),
    )

    @app.post("/mcp")
    async def mcp(payload: dict = Body(...)):
        return {"size": len(payload["audio_base64"])}

    def scope():
        return {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        }

    first_sent: list[dict] = []
    first_delivered = False

    async def first_receive():
        nonlocal first_delivered
        if not first_delivered:
            first_receive_started.set()
            await release_first_receive.wait()
            first_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def first_send(message):
        first_sent.append(message)

    first_task = asyncio.create_task(app(scope(), first_receive, first_send))
    await asyncio.wait_for(first_receive_started.wait(), timeout=1)

    second_sent: list[dict] = []

    async def second_receive():
        nonlocal second_receive_calls
        second_receive_calls += 1
        return {"type": "http.request", "body": body, "more_body": False}

    async def second_send(message):
        second_sent.append(message)

    await app(scope(), second_receive, second_send)

    assert second_sent[0]["status"] == 429
    assert (b"retry-after", b"1") in second_sent[0]["headers"]
    assert second_receive_calls == 0

    release_first_receive.set()
    await first_task
    assert first_sent[0]["status"] == 200


@pytest.mark.asyncio
async def test_mcp_body_cannot_bypass_memory_admission_with_false_small_content_length():
    body = b'{"audio_base64":"' + (b"A" * 256) + b'"}'
    route_calls = 0
    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        exact_path_limits={("POST", "/mcp"): len(body) + 1},
        admission_threshold_bytes=64,
        max_concurrent_large_mcp_requests=1,
    )

    @app.post("/mcp")
    async def mcp(payload: dict = Body(...)):
        nonlocal route_calls
        route_calls += 1
        return payload

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/mcp",
        "raw_path": b"/mcp",
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", b"32"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    assert sent[0]["status"] == 400
    assert route_calls == 0


@pytest.mark.asyncio
async def test_large_multipart_is_rejected_when_temp_storage_cannot_be_reserved(monkeypatch):
    boundary = "voicebox-boundary"
    body = _multipart_body(boundary, b"payload")
    receive_calls = 0

    app = FastAPI()
    app.add_middleware(
        RequestBodyLimitMiddleware,
        exact_path_limits={("POST", "/upload"): len(body) + 1},
        admission_threshold_bytes=0,
        temp_storage_reserve_bytes=100,
    )

    @app.post("/upload")
    async def upload(file: UploadFile = File(...)):
        return {"size": len(await file.read())}

    monkeypatch.setattr(
        "backend.utils.disk_reservations.shutil.disk_usage",
        # One framework spool would fit. The simultaneous bounded endpoint
        # copy plus reserve would not, so admission must fail before receive.
        lambda _path: SimpleNamespace(free=(2 * len(body)) + 99),
    )

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": body, "more_body": False}

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/upload",
            "raw_path": b"/upload",
            "query_string": b"",
            "headers": [
                (b"host", b"testserver"),
                (b"content-type", f"multipart/form-data; boundary={boundary}".encode()),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 507
    assert receive_calls == 0


@pytest.mark.asyncio
async def test_declared_oversized_multipart_is_rejected_before_receive_or_route():
    boundary = "voicebox-boundary"
    body = _multipart_body(boundary, b"small")
    app, calls = _multipart_app(max_body_bytes=len(body) - 1)

    sent = await _call_asgi(
        app,
        body=body,
        content_length=len(body),
        chunk_bytes=3,
    )

    assert sent[0]["status"] == 413
    assert calls == {"routes": 0, "file_bytes": 0}


@pytest.mark.asyncio
async def test_pathological_content_length_is_rejected_without_integer_parser_failure():
    boundary = "voicebox-boundary"
    body = _multipart_body(boundary, b"small")
    app, calls = _multipart_app(max_body_bytes=len(body))

    sent = await _call_asgi(
        app,
        body=body,
        content_length=b"9" * 5000,
        chunk_bytes=3,
    )

    assert sent[0]["status"] == 400
    assert calls == {"routes": 0, "file_bytes": 0}


@pytest.mark.asyncio
async def test_chunked_oversized_multipart_is_stopped_before_route_runs():
    boundary = "voicebox-boundary"
    body = _multipart_body(boundary, b"streamed payload")
    app, calls = _multipart_app(max_body_bytes=len(body) - 2)

    sent = await _call_asgi(
        app,
        body=body,
        content_length=None,
        chunk_bytes=4,
    )

    assert sent[0]["status"] == 413
    assert calls == {"routes": 0, "file_bytes": 0}


@pytest.mark.asyncio
async def test_exact_limit_chunked_multipart_reaches_real_parser_and_route():
    boundary = "voicebox-boundary"
    payload = b"accepted payload"
    body = _multipart_body(boundary, payload)
    app, calls = _multipart_app(max_body_bytes=len(body))

    sent = await _call_asgi(
        app,
        body=body,
        content_length=None,
        chunk_bytes=5,
    )

    assert sent[0]["status"] == 200
    assert calls == {"routes": 1, "file_bytes": len(payload)}


def test_production_request_limits_cover_largest_legitimate_uploads():
    cases = {
        "/profiles/import": 516 * 1024 * 1024,
        "/history/import": 516 * 1024 * 1024,
        "/generate/import": 200 * 1024 * 1024,
        "/transcribe": 100 * 1024 * 1024,
        "/captures": 100 * 1024 * 1024,
        "/profiles/profile-id/samples": 50 * 1024 * 1024,
        "/profiles/profile-id/avatar": 5 * 1024 * 1024,
    }
    for path, endpoint_file_limit in cases.items():
        for accepted_path in (path, f"{path}/"):
            limit = request_body_limit({"method": "POST", "path": accepted_path})
            assert endpoint_file_limit < limit <= endpoint_file_limit + 1024 * 1024


def test_large_mcp_json_limit_does_not_consume_multipart_disk_admission(monkeypatch):
    def forbidden_disk_check(_path):
        raise AssertionError("non-multipart MCP JSON must not reserve upload spool storage")

    monkeypatch.setattr("backend.utils.disk_reservations.shutil.disk_usage", forbidden_disk_check)
    app = FastAPI()
    app.add_middleware(RequestBodyLimitMiddleware)

    @app.post("/mcp")
    async def mcp():
        return {"ok": True}

    with TestClient(app) as client:
        response = client.post("/mcp", json={"jsonrpc": "2.0"})

    assert response.status_code == 200


def test_profile_sample_oversized_reference_text_is_rejected_before_database_or_service_work(monkeypatch):
    # Import the application first because profiles uses its shared download
    # header helper and the application registers this router during import.
    from backend import (
        app as _application,  # noqa: F401
        models,
    )
    from backend.database import get_db
    from backend.routes import profiles as profile_routes

    calls = {"service": 0}

    class PoisonDatabase:
        def __getattr__(self, name):
            raise AssertionError(f"database operation {name} must not run for invalid form data")

    def forbidden_db():
        # FastAPI opens yield dependencies before consolidating body-validation
        # errors. The poison session proves no query, write, commit, or rollback
        # is attempted before the 422 response.
        yield PoisonDatabase()

    async def forbidden_service(*_args, **_kwargs):
        calls["service"] += 1
        raise AssertionError("profile service must not run for invalid form data")

    app = FastAPI()
    app.include_router(profile_routes.router)
    app.dependency_overrides[get_db] = forbidden_db
    monkeypatch.setattr(profile_routes.profiles, "add_profile_sample", forbidden_service)

    with TestClient(app) as client:
        response = client.post(
            "/profiles/profile-id/samples",
            files={"file": ("sample.wav", b"not decoded", "audio/wav")},
            data={
                "reference_text": "x" * (models.PROFILE_SAMPLE_REFERENCE_TEXT_MAX_CHARS + 1),
            },
        )

    assert response.status_code == 422
    assert calls == {"service": 0}


@pytest.mark.asyncio
async def test_bounded_reader_accepts_exact_limit_and_rejects_one_byte_over():
    exact = UploadFile(file=io.BytesIO(b"1234"), filename="audio.wav")
    assert (
        await upload_limits.read_upload_bounded(
            exact,
            max_bytes=4,
            chunk_bytes=2,
        )
        == b"1234"
    )

    oversized = UploadFile(file=io.BytesIO(b"12345"), filename="audio.wav")
    with pytest.raises(upload_limits.UploadSizeLimitError):
        await upload_limits.read_upload_bounded(
            oversized,
            max_bytes=4,
            chunk_bytes=2,
        )


@pytest.mark.asyncio
async def test_failed_spool_removes_private_partial_file(tmp_path, monkeypatch):
    partial_path = tmp_path / "partial.upload"

    def fake_mkstemp(*, prefix, suffix):
        assert prefix == "voicebox-upload-"
        descriptor = os.open(
            partial_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        return descriptor, str(partial_path)

    monkeypatch.setattr(upload_limits.tempfile, "mkstemp", fake_mkstemp)
    upload = UploadFile(file=io.BytesIO(b"too large"), filename="archive.zip")
    with pytest.raises(upload_limits.UploadSizeLimitError):
        await upload_limits.spool_upload_bounded(
            upload,
            max_bytes=3,
            suffix=".zip",
            chunk_bytes=2,
        )
    assert not partial_path.exists()


@pytest.mark.asyncio
async def test_cancelled_spool_removes_private_partial_file(tmp_path, monkeypatch):
    partial_path = tmp_path / "cancelled.upload"

    def fake_mkstemp(*, prefix, suffix):
        descriptor = os.open(
            partial_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        return descriptor, str(partial_path)

    class CancelledUpload:
        calls = 0

        async def read(self, _chunk_bytes):
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            raise asyncio.CancelledError

    monkeypatch.setattr(upload_limits.tempfile, "mkstemp", fake_mkstemp)
    with pytest.raises(asyncio.CancelledError):
        await upload_limits.spool_upload_bounded(
            CancelledUpload(),
            max_bytes=100,
            suffix=".zip",
        )
    assert not partial_path.exists()


@pytest.mark.asyncio
async def test_successful_spool_is_private_on_posix(tmp_path, monkeypatch):
    upload_path = tmp_path / "complete.upload"

    def fake_mkstemp(*, prefix, suffix):
        descriptor = os.open(
            upload_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        return descriptor, str(upload_path)

    monkeypatch.setattr(upload_limits.tempfile, "mkstemp", fake_mkstemp)
    upload = UploadFile(file=io.BytesIO(b"ok"), filename="archive.zip")
    result = await upload_limits.spool_upload_bounded(
        upload,
        max_bytes=2,
        suffix=".zip",
    )
    try:
        assert result.read_bytes() == b"ok"
        if os.name == "posix":
            assert result.stat().st_mode & 0o777 == 0o600
    finally:
        result.unlink(missing_ok=True)


def test_avatar_pixel_limit_runs_before_image_decode(tmp_path, monkeypatch):
    avatar = tmp_path / "avatar.png"
    avatar.write_bytes(b"small encoded input")
    loaded = False

    class OversizedImage:
        width = 5000
        height = 5000

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load(self):
            nonlocal loaded
            loaded = True

    monkeypatch.setattr(images.Image, "open", lambda _path: OversizedImage())
    valid, error = images.validate_image(str(avatar))
    assert valid is False
    assert error == "Image dimensions are too large"
    assert loaded is False
