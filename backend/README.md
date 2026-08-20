# Voicebox Backend

FastAPI server powering voice cloning, speech generation, and audio processing. Runs locally as a Tauri sidecar or standalone via `python -m backend.main`.

## Running

```bash
# Via justfile (recommended)
just dev:server

# Standalone
python -m backend.main --host 127.0.0.1 --port 17493

# With custom data directory
python -m backend.main --data-dir /path/to/data
```

The server auto-initializes the SQLite database on first startup. Models are downloaded from HuggingFace on first use.

## Architecture

```
backend/
  app.py                  # FastAPI app factory, CORS, lifecycle events
  main.py                 # Entry point (imports app, runs uvicorn)
  config.py               # Data directory paths and configuration
  models.py               # Pydantic request/response schemas
  server.py               # Tauri sidecar launcher, parent-pid watchdog

  routes/                 # Thin HTTP handlers — validation, delegation, response formatting
  services/               # Business logic, CRUD, orchestration
  backends/               # TTS/STT engine implementations (MLX, PyTorch, etc.)
  database/               # ORM models, session management, migrations, seed data
  utils/                  # Shared utilities (audio, effects, caching, progress tracking)
```

### Request flow

```
HTTP request
  -> routes/        (validate input, parse params)
  -> services/      (business logic, database queries, orchestration)
  -> backends/      (TTS/STT inference)
  -> utils/         (audio processing, effects, caching)
```

Route handlers are intentionally thin. They validate input, delegate to a service function, and format the response. All business logic lives in `services/`.

### Key modules

**services/generation.py** -- Single `run_generation()` function that handles all three generation modes (generate, retry, regenerate). Manages model loading, voice prompt creation, chunked inference, normalization, effects, and version persistence.

**services/task_queue.py** -- Serial generation queue. Ensures only one GPU inference runs at a time. Background tasks are tracked to prevent garbage collection.

**backends/__init__.py** -- Protocol definitions (`TTSBackend`, `STTBackend`), model config registry, and factory functions. Adding a new engine means implementing the protocol and registering a config entry.

**backends/base.py** -- Shared utilities used across all engine implementations: HuggingFace cache checks, device detection, voice prompt combination, progress tracking.

**database/** -- SQLAlchemy ORM models with a re-exporting `__init__.py` for backward compatibility. Migrations run automatically on startup.

### Backend selection

The server detects the best inference backend at startup:

| Platform | Backend | Acceleration |
|----------|---------|-------------|
| macOS (Apple Silicon) | MLX | Metal / Neural Engine |
| Windows / Linux (NVIDIA) | PyTorch | CUDA |
| Linux (AMD) | PyTorch | ROCm |
| Intel Arc | PyTorch | IPEX / XPU |
| Windows (any GPU) | PyTorch | DirectML |
| Any | PyTorch | CPU fallback |

Detection is handled by `utils/platform_detect.py`. Both backends implement the same `TTSBackend` protocol, so the API layer is engine-agnostic.

## API

90 endpoints organized by domain. Full interactive documentation available at `http://localhost:17493/docs` when the server is running.

| Domain | Prefix | Description |
|--------|--------|-------------|
| Health | `/`, `/health` | Server status, GPU info, filesystem checks |
| Profiles | `/profiles` | Voice profile CRUD, samples, avatars, import/export |
| Channels | `/channels` | Audio channel management and voice assignment |
| Generation | `/generate` | TTS generation, retry, regenerate, status SSE |
| History | `/history` | Generation history, search, favorites, export |
| Transcription | `/transcribe` | Whisper-based audio-to-text |
| Stories | `/stories` | Multi-track timeline editor, audio export |
| Effects | `/effects` | Effect presets, preview, version management |
| Audio | `/audio`, `/samples` | Audio file serving |
| Models | `/models` | Load, unload, download, migrate, status |
| Tasks | `/tasks`, `/cache` | Active task tracking, cache management |
| CUDA | `/backend/cuda-*` | CUDA binary download and management |

Story WAV exports are streamed from private, disk-backed scratch rather than
assembled in RAM. They are limited to 1,000 items and a 24-hour, 24 kHz mono
timeline (the classic PCM16 WAV size ceiling), with bounded channel, sample-rate,
decoded-sample, and temporary-disk budgets. Source clips may remain long-form;
new direct audio uploads retain their separate 30-minute limit. The desktop app
and browsers with the File System Access API stream downloads directly to disk;
other browsers cap the in-memory compatibility download at 64 MiB and reject
larger exports with an actionable error instead of risking tab exhaustion.

### Quick examples

```bash
# Generate speech
curl -X POST http://localhost:17493/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "profile_id": "...", "language": "en"}'

# List profiles
curl http://localhost:17493/profiles

# Stream generation status (SSE)
curl http://localhost:17493/generate/{id}/status
```

## Data directory

```
{data_dir}/
  voicebox.db             # SQLite database
  profiles/{id}/          # Voice samples per profile
  generations/            # Generated audio files
  cache/                  # Voice prompt cache (memory + disk)
  backends/               # Downloaded CUDA binary (if applicable)
```

Default location is the OS-specific app data directory. Override with `--data-dir` or the `VOICEBOX_DATA_DIR` environment variable.

On POSIX systems, Voicebox protects app-owned data directories with mode `0700`
and files with mode `0600`, and repairs existing owned data at startup without
following symbolic links. The configured data root and managed top-level
directories must therefore be real directories, not symlinks. Generated audio
uses the same private policy by default. For an intentional host-facing export
or Docker bind mount, set
`VOICEBOX_SHARED_GENERATIONS=1`. The data root then permits traversal without
listing (`0711`), while `generations/` uses `0755` directories and `0644`
files. The database, voice profiles, cache, captures, and logs remain private.
The provided `docker-compose.yml` enables this mode for its `./output` bind
mount.

### Local API security

The API accepts `Host` values for loopback and the Tauri webview by default and
rejects other authorities, which prevents browser DNS-rebinding. Browser
requests must also use the same origin as the API or one of the built-in local
development/Tauri origins. Originless clients such as `curl`, the audiobook
launcher, and health checks continue to work.

Remote access is an explicit opt-in. Add every remote API hostname or address
to the comma-separated `VOICEBOX_TRUSTED_HOSTS` setting. If a browser UI is
hosted on a different origin, add each full origin to
`VOICEBOX_CORS_ORIGINS` as well. Wildcards are intentionally rejected for both
settings; reverse proxies must preserve the public `Host` header.

Credential-free access is allowed only when both the actual network peer and
the requested Host authority are local. Every other request—including a public
Host forwarded by a same-machine reverse proxy—must authenticate with
`Authorization: Bearer $VOICEBOX_REMOTE_API_TOKEN`. Set a randomly generated,
URL-safe token of at least 32 characters before binding beyond loopback, for
example:

```bash
export VOICEBOX_REMOTE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export VOICEBOX_TRUSTED_HOSTS="voicebox.example"
uvicorn backend.main:app --host 127.0.0.1 --port 17493
curl -H "Authorization: Bearer $VOICEBOX_REMOTE_API_TOKEN" \
  https://voicebox.example/health
```

Publish the loopback listener through an HTTPS reverse proxy. Preserve the
public Host and HTTPS request scheme, and trust forwarded headers only from
the proxy address. Direct `--host 0.0.0.0` serves plain HTTP and its protected
routes return 426 by default. The compatibility escape hatch
`VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP=1` must be set explicitly; it makes the
bearer replayable to anyone who can observe the connection and is not suitable
for ordinary remote deployment.

Loopback CLI clients remain credential-free. The web/desktop client has a
**Remote API token** field under Settings → Server and never sends that token
to non-Voicebox origins. Successful bearer authentication establishes an
HttpOnly session cookie for same-origin media and event streams. Use HTTPS for
remote deployments or connect through a loopback SSH tunnel. The browser and
desktop client refuse non-loopback `http://` server URLs. Host and Origin checks
remain active in addition to the token.

The ASGI receive boundary also rejects oversized declared and chunked bodies
before Starlette's multipart parser can spool file parts. Limits are matched to
each upload/import endpoint (plus bounded multipart framing); ordinary JSON is
capped at 2 MiB and the MCP endpoint retains its documented bounded base64
transcription allowance. Large multipart admission is capped at two concurrent
requests and reserves temporary-disk capacity before consuming the body. MCP
control calls remain concurrent, while a declared or streaming MCP body above
5 MiB uses a separate one-request memory admission gate. An overloaded server
returns 429 and low temporary storage returns 507.

## Code quality

Linting and formatting are enforced by [ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`. See `STYLE_GUIDE.md` for conventions.

```bash
just check-python       # lint + format check
just fix-python         # auto-fix lint issues + reformat
just test               # run pytest
```

## Dependencies

Runtime dependencies are in `requirements.txt`. macOS-only MLX dependencies are in `requirements-mlx.txt`. Dev tools (ruff, pytest) are installed automatically by `just setup-python`.
