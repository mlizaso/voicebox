"""voicebox-mcp — stdio ↔ Streamable-HTTP MCP proxy.

Some MCP clients only speak stdio. They spawn this binary, we pipe each
JSON-RPC message to ``http://127.0.0.1:<port>/mcp/``, and stream the
server's response back. The Voicebox server does all the real work.

Environment variables:
  VOICEBOX_PORT       Voicebox server port (default 17493).
  VOICEBOX_HOST       Host (default 127.0.0.1).
  VOICEBOX_SCHEME     http for loopback (default), https for remote servers.
  VOICEBOX_CLIENT_ID  Forwarded as X-Voicebox-Client-Id on every request.
  VOICEBOX_REMOTE_API_TOKEN  Bearer capability for a non-loopback server.

Stdout is JSON-RPC only. Diagnostics go to stderr.
Exit 0 on clean EOF, 1 on transport error, 2 if backend never answers.
"""

from __future__ import annotations

import asyncio
import codecs
import io
import ipaddress
import json
import os
import sys
from typing import Any

import httpx

CLIENT_ID_HEADER = "X-Voicebox-Client-Id"
SESSION_HEADER = "mcp-session-id"
HEALTH_TIMEOUT_S = 30.0
DEFAULT_PORT = 17493
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
# Match the server's 200 MiB base64 transcription upload plus JSON framing.
MAX_REQUEST_BYTES = 4 * ((200 * 1024 * 1024 + 2) // 3) + 1024 * 1024


def _err(msg: str) -> None:
    print(f"voicebox-mcp: {msg}", file=sys.stderr, flush=True)


def _base_url() -> tuple[str, str]:
    host = os.environ.get("VOICEBOX_HOST", "127.0.0.1")
    if not host or any(character.isspace() or character in "/?#@" for character in host):
        raise ValueError("VOICEBOX_HOST must contain only a hostname or IP address")
    scheme = os.environ.get("VOICEBOX_SCHEME", "http").lower()
    if scheme not in {"http", "https"}:
        raise ValueError("VOICEBOX_SCHEME must be either http or https")
    port = int(os.environ.get("VOICEBOX_PORT", str(DEFAULT_PORT)))
    if not 1 <= port <= 65535:
        raise ValueError("VOICEBOX_PORT must be between 1 and 65535")

    normalized_host = host.removeprefix("[").removesuffix("]")
    try:
        address = ipaddress.ip_address(normalized_host)
        loopback = address.is_loopback
    except ValueError:
        if ":" in normalized_host:
            raise ValueError("VOICEBOX_HOST contains an invalid IP address") from None
        loopback = normalized_host.rstrip(".").lower() in {"localhost", "tauri.localhost"}
    remote_api_token = os.environ.get("VOICEBOX_REMOTE_API_TOKEN")
    if (
        remote_api_token
        and scheme != "https"
        and not loopback
        and os.environ.get("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP") != "1"
    ):
        raise ValueError("refusing to send VOICEBOX_REMOTE_API_TOKEN over plaintext HTTP; set VOICEBOX_SCHEME=https")

    authority_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    base = f"{scheme}://{authority_host}:{port}"
    return f"{base}/mcp/", f"{base}/health"


async def _wait_for_backend(client: httpx.AsyncClient, health_url: str) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + HEALTH_TIMEOUT_S
    while loop.time() < deadline:
        try:
            async with client.stream("GET", health_url, timeout=2.0) as response:
                if response.status_code == 200:
                    return True
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.5)
    return False


async def _read_stdin_line() -> str | None:
    """Async-read a single line from stdin. Returns None on EOF."""
    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline, MAX_REQUEST_BYTES + 1)
    if len(line) > MAX_REQUEST_BYTES or len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise ValueError("MCP request exceeds the size limit")
    if not line:
        return None
    return line


def _write_stdout(obj: Any) -> None:
    """Write a JSON object to stdout as one line, flushed."""
    sys.stdout.write(json.dumps(obj, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


async def _response_body(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
            raise ValueError("MCP response exceeds the size limit")
        body.extend(chunk)
    return bytes(body)


async def _response_lines(response: httpx.Response):
    # Bound incomplete lines before HTTPX's line iterator could accumulate an
    # unlimited response. Decode UTF-8 and CR/LF across network chunk boundaries.
    decoder = io.IncrementalNewlineDecoder(codecs.getincrementaldecoder("utf-8-sig")("replace"), translate=True)
    pending = io.StringIO()
    pending_bytes = 0
    async for chunk in response.aiter_bytes():
        # StringIO avoids repeatedly copying a long incomplete line when a
        # peer sends tiny fragments. Slice large transport chunks before decode.
        for offset in range(0, len(chunk), 64 * 1024):
            lines = decoder.decode(chunk[offset : offset + 64 * 1024]).split("\n")
            for index, line in enumerate(lines):
                pending_bytes += len(line.encode("utf-8"))
                if pending_bytes > MAX_RESPONSE_BYTES:
                    raise ValueError("MCP response exceeds the size limit")
                pending.write(line)
                if index < len(lines) - 1:
                    yield pending.getvalue()
                    pending = io.StringIO()
                    pending_bytes = 0
    for line in (pending.getvalue() + decoder.decode(b"", final=True)).split("\n")[:-1]:
        if len(line.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("MCP response exceeds the size limit")
        yield line


async def _sse_payloads(response: httpx.Response):
    fields: list[str] = []
    event_bytes = 0
    async for line in _response_lines(response):
        if not line:
            if fields:
                yield "\n".join(fields)
            fields.clear()
            event_bytes = 0
        elif line == "data" or line.startswith("data:"):
            value = line[5:].removeprefix(" ")
            event_bytes += len(value.encode("utf-8")) + 1
            if event_bytes > MAX_RESPONSE_BYTES:
                raise ValueError("MCP response exceeds the size limit")
            fields.append(value)


async def _handle_request(
    client: httpx.AsyncClient,
    url: str,
    raw: str,
    headers: dict[str, str],
    session_id: list[str | None],
) -> None:
    """Forward one JSON-RPC payload to the server and relay the response."""
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        _err(f"invalid JSON on stdin: {exc}")
        return

    req_headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        **headers,
    }
    if session_id[0]:
        req_headers[SESSION_HEADER] = session_id[0]

    # Notifications (no "id") don't expect a response body. Server returns
    # 202 Accepted and we stay quiet.
    is_notification = isinstance(message, dict) and "id" not in message

    async with client.stream("POST", url, headers=req_headers, content=raw.encode("utf-8")) as response:
        # Capture session id on initialize.
        if session_id[0] is None:
            sid = response.headers.get(SESSION_HEADER)
            if sid:
                session_id[0] = sid

        if response.status_code == 202:
            return  # notification acknowledged
        if response.status_code >= 400:
            body = b""
            async for chunk in response.aiter_bytes(chunk_size=4096):
                body = chunk
                break
            _err(f"server {response.status_code}: {body.decode('utf-8', errors='replace')[:400]}")
            if is_notification:
                return
            _write_stdout(
                {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {
                        "code": -32000,
                        "message": (f"Voicebox MCP proxy got HTTP {response.status_code}"),
                    },
                }
            )
            return

        ctype = response.headers.get("content-type", "").lower()
        if "text/event-stream" in ctype:
            async for payload in _sse_payloads(response):
                if not payload.strip():
                    continue
                try:
                    _write_stdout(json.loads(payload))
                except json.JSONDecodeError:
                    _err(f"malformed SSE payload: {payload[:200]}")
        else:
            body = await _response_body(response)
            try:
                _write_stdout(json.loads(body))
            except json.JSONDecodeError:
                _err(f"non-JSON response ({ctype}): {body.decode('utf-8', errors='replace')[:200]}")


async def _run() -> int:
    url, health_url = _base_url()
    forward_headers: dict[str, str] = {}
    client_id = os.environ.get("VOICEBOX_CLIENT_ID")
    if client_id:
        forward_headers[CLIENT_ID_HEADER] = client_id
    remote_api_token = os.environ.get("VOICEBOX_REMOTE_API_TOKEN")
    if remote_api_token:
        forward_headers["Authorization"] = f"Bearer {remote_api_token}"

    session_id: list[str | None] = [None]

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),
        headers=forward_headers,
    ) as client:
        if not await _wait_for_backend(client, health_url):
            _err(f"timed out waiting for Voicebox at {health_url} — is the app open?")
            return 2

        try:
            while True:
                line = await _read_stdin_line()
                if line is None:
                    return 0
                line = line.strip()
                if not line:
                    continue
                await _handle_request(client, url, line, forward_headers, session_id)
        except (KeyboardInterrupt, SystemExit):
            return 0
        except Exception as exc:
            _err(f"proxy failed: {exc!r}")
            return 1


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0
    except ValueError as exc:
        _err(f"invalid configuration: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
