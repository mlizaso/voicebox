# Voicebox MCP server

MCP runs inside the Voicebox backend at `/mcp` using Streamable HTTP. The stdio
shim connects to that running backend; it does not load models or launch the full
studio itself. See the [project README](../../README.md) for backend startup.

## Connect locally

Use your MCP client's HTTP server configuration:

```json
{
  "mcpServers": {
    "voicebox": {
      "url": "http://127.0.0.1:17493/mcp",
      "headers": { "X-Voicebox-Client-Id": "my-agent" }
    }
  }
}
```

Keep the client ID stable to retain its voice binding. Settings → MCP manages
bindings; tools can also select a profile by name or ID.

For a stdio client, use the installed `voicebox-mcp` binary or this source setup:

```json
{
  "mcpServers": {
    "voicebox": {
      "command": "/absolute/path/to/voicebox/backend/venv/bin/python",
      "args": ["-m", "backend.mcp_shim"],
      "cwd": "/absolute/path/to/voicebox",
      "env": { "VOICEBOX_CLIENT_ID": "my-agent" }
    }
  }
}
```

Client support for `cwd` varies; run from the checkout or use the bundled binary
when needed. The shim defaults to host 127.0.0.1, port 17493, scheme HTTP, and waits
up to 30 seconds for health. The backend can be a standalone server or a desktop
sidecar configured to remain running.

## Tools

| Tool | Inputs / result |
| --- | --- |
| `voicebox.speak` | Required `text`; optional `profile`, `engine`, `model_size`, `language`, `personality`. Returns generation ID and status/poll URL. |
| `voicebox.transcribe` | Exactly one of `audio_base64` or server-local `audio_path`; optional model/language. Returns transcript metadata. |
| `voicebox.list_profiles` | Lists profile IDs, names, types, languages, and personality availability. |
| `voicebox.list_captures` | `limit` 1–200, nonnegative `offset`; returns captures and total. Invalid bounds are rejected. |

Speak profile precedence is explicit name/ID → per-client binding →
`capture_settings.default_playback_voice_id`. A missing/invalid explicit profile
is an error, not a fallback voice. Qwen variants accept 0.6B/1.7B and TADA 1B/3B;
other engines ignore irrelevant sizes. Personality rewriting is optional and
changes the spoken text before TTS.

The same speech path is available without MCP:

```bash
curl --fail http://127.0.0.1:17493/speak \
  -H 'Content-Type: application/json' \
  -H 'X-Voicebox-Client-Id: my-agent' \
  -d '{"text":"Build complete.","profile":"Narrator","personality":false}'
```

The connected desktop consumes speak events to display/play agent speech.
History stores the generation; a headless backend does not create a native
speaker or desktop pill on its own.

## Remote authentication and bounds

Use an HTTPS MCP URL with `Authorization: Bearer <server token>` whenever peer
or Host is remote. Set the server's trusted hosts/origins as appropriate.
For the shim, set `VOICEBOX_HOST`, `VOICEBOX_PORT`, `VOICEBOX_SCHEME=https`, and
`VOICEBOX_REMOTE_API_TOKEN`. It authenticates both health and MCP calls and refuses
to send a remote token over plain HTTP by default. Never place tokens in a URL.

Server-local `audio_path` requires both direct peer and Host to be local.
Authenticated reverse-proxy calls remain remote and must upload base64 audio.
Transcription bytes are limited to 200 MiB; duration/decoded-size checks also apply.
The stdio request bound accommodates that base64 payload plus framing. JSON
responses, incomplete SSE lines, and complete SSE events are capped at 32 MiB;
health reads no body and HTTP error diagnostics read only a bounded prefix.

## Implementation

| File | Role |
| --- | --- |
| `backend/mcp_server/server.py` | FastMCP server and app mount/lifespan |
| `tools.py`, `resolve.py` | Tools and profile selection |
| `context.py` | Client identity and local-request context |
| `events.py` | Speak event interface |
| `backend/mcp_shim/__main__.py` | Bounded stdio/HTTP bridge |

The package is named `mcp_server` to avoid shadowing the installed MCP SDK.
For diagnostics, start with health and `voicebox.list_profiles`, then try a short
speak request. See the generated REST reference for bindings and status routes.
