"""Transport-security regressions for the stdio MCP shim."""

import io
import json

import httpx
import pytest

from backend.mcp_shim import __main__ as mcp_shim


@pytest.fixture(autouse=True)
def _clear_shim_environment(monkeypatch):
    for name in (
        "VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP",
        "VOICEBOX_HOST",
        "VOICEBOX_PORT",
        "VOICEBOX_REMOTE_API_TOKEN",
        "VOICEBOX_SCHEME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_loopback_shim_keeps_http_default():
    assert mcp_shim._base_url() == (
        "http://127.0.0.1:17493/mcp/",
        "http://127.0.0.1:17493/health",
    )


def test_remote_shim_refuses_to_send_bearer_over_plain_http(monkeypatch):
    monkeypatch.setenv("VOICEBOX_HOST", "voicebox.example")
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "x" * 32)

    with pytest.raises(ValueError, match="refusing to send"):
        mcp_shim._base_url()


def test_remote_shim_accepts_https_and_formats_ipv6(monkeypatch):
    monkeypatch.setenv("VOICEBOX_HOST", "::1")
    monkeypatch.setenv("VOICEBOX_SCHEME", "https")
    monkeypatch.setenv("VOICEBOX_PORT", "443")
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "x" * 32)

    assert mcp_shim._base_url() == (
        "https://[::1]:443/mcp/",
        "https://[::1]:443/health",
    )


def test_remote_shim_plain_http_requires_explicit_insecure_opt_in(monkeypatch):
    monkeypatch.setenv("VOICEBOX_HOST", "voicebox.lan")
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "x" * 32)
    monkeypatch.setenv("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP", "1")

    assert mcp_shim._base_url()[0] == "http://voicebox.lan:17493/mcp/"


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.read_count = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.read_count += 1
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_health_probe_reads_headers_without_consuming_the_body(monkeypatch):
    stream = _ResponseStream([AssertionError("the health body is not needed")])
    monkeypatch.setattr(mcp_shim, "HEALTH_TIMEOUT_S", 0.01)
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, stream=stream))

    async with httpx.AsyncClient(transport=transport) as client:
        healthy = await mcp_shim._wait_for_backend(client, "http://localhost/health")

    assert healthy
    assert stream.read_count == 0
    assert stream.closed


@pytest.mark.asyncio
async def test_error_diagnostics_stop_reading_after_the_short_prefix(monkeypatch, capsys):
    stream = _ResponseStream([b"upstream failure " * 300, AssertionError("unbounded diagnostic read")])
    replies = []
    monkeypatch.setattr(mcp_shim, "_write_stdout", replies.append)
    transport = httpx.MockTransport(lambda _request: httpx.Response(500, stream=stream))

    async with httpx.AsyncClient(transport=transport) as client:
        await mcp_shim._handle_request(client, "http://localhost/mcp/", '{"id":7}', {}, [None])

    assert stream.read_count == 1
    assert stream.closed
    assert replies == [
        {"jsonrpc": "2.0", "id": 7, "error": {"code": -32000, "message": "Voicebox MCP proxy got HTTP 500"}}
    ]
    assert "upstream failure" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["application/json", "text/event-stream"])
async def test_oversized_responses_stop_before_consuming_the_whole_stream(monkeypatch, content_type):
    monkeypatch.setattr(mcp_shim, "MAX_RESPONSE_BYTES", 1024, raising=False)
    payload = json.dumps({"jsonrpc": "2.0", "id": 7, "result": "x" * 8192}).encode()
    if content_type == "text/event-stream":
        payload = b"data: " + payload + b"\n\n"
    stream = _ResponseStream([payload[i : i + 512] for i in range(0, len(payload), 512)])
    replies = []
    monkeypatch.setattr(mcp_shim, "_write_stdout", replies.append)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": content_type}, stream=stream)
    )

    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="MCP response exceeds"):
            await mcp_shim._handle_request(client, "http://localhost/mcp/", '{"id":7}', {}, [None])

    assert stream.read_count < len(stream.chunks)
    assert stream.closed
    assert replies == []


@pytest.mark.asyncio
@pytest.mark.parametrize("content_type", ["text/event-stream", "Text/Event-Stream; charset=utf-8"])
async def test_sse_preserves_multiline_json_unicode_and_split_newlines(monkeypatch, content_type):
    payload = 'data: {"jsonrpc":"2.0",\r\ndata: "id":7,"result":"¡Hola!"}\r\n\r\n'
    stream = _ResponseStream([bytes([byte]) for byte in payload.encode()])
    replies = []
    monkeypatch.setattr(mcp_shim, "_write_stdout", replies.append)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": content_type}, stream=stream)
    )

    async with httpx.AsyncClient(transport=transport) as client:
        await mcp_shim._handle_request(client, "http://localhost/mcp/", '{"id":7}', {}, [None])

    assert replies == [{"jsonrpc": "2.0", "id": 7, "result": "¡Hola!"}]
    assert stream.closed


@pytest.mark.asyncio
async def test_stdio_limit_matches_server_and_stops_before_reading_unlimited_input(monkeypatch):
    from backend.request_limits import MCP_REQUEST_BODY_MAX_BYTES

    assert mcp_shim.MAX_REQUEST_BYTES == MCP_REQUEST_BODY_MAX_BYTES
    monkeypatch.setattr(mcp_shim, "MAX_REQUEST_BYTES", 100)
    source = io.StringIO("x" * 10_000)
    monkeypatch.setattr(mcp_shim.sys, "stdin", source)
    with pytest.raises(ValueError, match="MCP request exceeds"):
        await mcp_shim._read_stdin_line()
    assert source.tell() == 101


@pytest.mark.asyncio
async def test_sse_limits_the_aggregate_event_not_just_each_line(monkeypatch):
    monkeypatch.setattr(mcp_shim, "MAX_RESPONSE_BYTES", 100)
    stream = _ResponseStream([b"data: " + b"x" * 30 + b"\n"] * 10)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match="MCP response exceeds"):
            await mcp_shim._handle_request(client, "http://localhost/mcp/", '{"id":7}', {}, [None])
    assert stream.read_count < 10
    assert stream.closed
