# Voicebox backend

FastAPI serves the shared voice studio, native desktop client, MCP clients, and
the private audiobook renderer. Start with the [project README](../README.md)
for the complete workflow and storage map.

## Run from the repository root

```bash
just setup-python
just dev-backend
```

The development recipe uses reload mode on port 17493. For stable generation:

```bash
backend/venv/bin/python -m backend.main \
  --host 127.0.0.1 --port 17493 --data-dir /absolute/path/to/voicebox-data
```

On Windows use `backend\venv\Scripts\python.exe`. Python 3.12+ is required;
the setup recipe handles the special engine and Apple Silicon dependency pins.
The CLI defaults to port 8000 if `--port` is omitted. Without `--data-dir`, the
root is `data/` relative to the process working directory. Packaged Tauri supplies
its OS app-data directory explicitly. `VOICEBOX_DATA_DIR` is not a backend CLI
environment override; the private audiobook launcher translates it to `--data-dir`.

Database creation, additive migrations, data-permission repair, and recovery run
at startup. Models download/load on demand and remain cached. Avoid reload mode
while producing a real book. The private wizard normally owns a backend on 17494.

## Structure and flow

| Path | Responsibility |
| --- | --- |
| `main.py` / `server.py` | Python module / frozen sidecar entry points |
| `app.py` | App composition, middleware, routers, lifecycle |
| `config.py`, `data_permissions.py` | Data paths, ownership and permissions |
| `api_security.py`, `request_limits.py` | Host/origin/authentication and request admission |
| `models.py` | Pydantic schemas; `HistoryResponse` extends `GenerationResponse` |
| `routes/` | Validate and coordinate HTTP operations |
| `services/` | Profiles, history, queue, generation, archives, effects, recovery |
| `backends/` | Engine protocols/registry, MLX/PyTorch implementations, runtime guards |
| `database/` | SQLAlchemy models, sessions, migrations, seed data |
| `utils/` | Audio, chunking, conditioning cache, bounded uploads, progress |
| `mcp_server/`, `mcp_shim/` | MCP tools and bounded stdio/HTTP adapter |
| `tests/` | Synthetic regression tests and opt-in frozen-model runner |

Durable generation enters `routes/generations.py`, is admitted to the serial
queue, and runs through `services/generation.py`. The selected backend produces
audio; publication and history/version updates retain ownership across failures.
MLX lifecycle guards coordinate generation, model loading, unloading, and migration.
Some exact-generation validation intentionally remains in the route layer.

## API entry points

The running server's `/docs` and `/openapi.json` are authoritative.

| Area | Entry points |
| --- | --- |
| Health / storage | `GET /health`, `GET /health/filesystem` |
| Voices | `/profiles`, samples, avatars, ZIP import/export, exact snapshots |
| Durable generation | `POST /generate`, `/generate/{id}/status`, retry, regenerate |
| Exact generation | `/generate/exact`, `/generate/batch/exact`, `/generate/stream/exact` |
| Streaming audio | `POST /generate/stream` for supported backends |
| History / audio | `/history`, `/audio/{id}`, generation export and versions |
| Transcription / capture | `POST /transcribe`, `/captures` |
| Composition | `/stories`, `/channels`, `/effects/available`, `/effects/presets` |
| Model management | `/models/status`, download/load/unload/delete/migrate operations |
| Agent voice | `POST /speak`, `/events/speak`, `/mcp`, `/mcp/bindings` |

```bash
curl --fail http://127.0.0.1:17493/profiles
curl --fail http://127.0.0.1:17493/generate \
  -H 'Content-Type: application/json' \
  -d '{"profile_id":"PROFILE_ID","text":"Hello world.","language":"en","engine":"qwen","model_size":"1.7B"}'
curl --fail --no-buffer http://127.0.0.1:17493/generate/GENERATION_ID/status
curl --fail http://127.0.0.1:17493/transcribe \
  -F 'file=@recording.wav' -F 'model=turbo' -F 'language=en'
```

Wait for a completed durable generation before downloading `/audio/{id}`.
Transcription uses multipart `file`, `model` (`base`, `small`, `medium`, `large`,
`turbo`), and optional `language`. Explicit language hints are `en`, `zh`, `ja`,
`ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`; omit the hint for detection.
Capture settings additionally accept `auto`.

## Generation preservation

Normal defaults are Qwen 1.7B, 800-character chunks, 50 ms crossfade, and
normalization enabled. New rows store `max_chunk_chars`, `crossfade_ms`, and
`normalize_audio` in the existing insert. Retry/regenerate reuse them; legacy
NULL values keep the previous defaults. Regeneration creates a new take rather
than promising the same waveform. Engine/model-size validation rejects invalid
combinations before inference; old TADA fallback records remain replayable.

Archives preserve original audio bytes and generation metadata/settings.
Direct audio imports cannot be regenerated as invented TTS requests. Exact
endpoints also bind request, voice snapshot, and implementation revision; a
changed runtime fails closed before incompatible output can be adopted.

## Storage and backups

See the [complete storage table](../README.md#where-your-files-are-saved).
`GET /health/filesystem` identifies the live data root. Profile metadata and
transcripts are in `voicebox.db`; sample WAVs and avatars are in `profiles/{id}/`.
History/version audio is under `generations/`, recordings under `captures/`,
and immutable exact references under `exact_voice_snapshots/`.

Back up the entire data root while stopped, or use item ZIP exports. Do not
delete SQLite WAL/SHM files to fix a lock error. Model weights use the Hugging Face
cache (or `VOICEBOX_MODELS_DIR`), separately from application data. Model migration
preflights conflicts and refuses linked roots or existing destination models.

On POSIX, managed directories/files are private by default (0700/0600); symlink
data roots are refused. `VOICEBOX_SHARED_GENERATIONS=1` makes only generated
audio host-readable (0755/0644 with root traversal), retaining private database,
profiles, captures, cache, and logs. Compose uses this for its host output mount.

## Authentication and limits

Only requests whose actual peer and Host are both local can omit credentials.
Remote deployment requires an explicit `VOICEBOX_TRUSTED_HOSTS` allowlist, any
additional `VOICEBOX_CORS_ORIGINS`, and `VOICEBOX_REMOTE_API_TOKEN` (32–512
URL-safe ASCII characters). Wildcards are rejected. Serve through HTTPS, preserve
the public Host and HTTPS scheme, and trust forwarded headers only from the proxy.

The app's **Settings → Server** accepts the token. Its API, avatars, audio, and
event streams use bearer headers and omit cookies; the token stays in memory.
The server retains a same-origin cookie compatibility path, but the app does not
depend on it. Non-loopback HTTP URLs are refused by the app. The server's explicit
insecure HTTP compatibility override is not a substitute for TLS.

Host/origin checks apply alongside authentication. The receive boundary limits
ordinary JSON and endpoint-specific uploads before multipart parsing, with
temporary-space and concurrency admission. MCP base64 transcription has its own
bounded allowance; the stdio shim caps requests, responses, SSE lines, and events.
Bad input fails before inference; overload and low temporary capacity return
429 and 507. Server-local MCP transcription paths require a direct local request.

Story WAV exports use bounded disk-backed mixing rather than loading the entire
timeline into memory. They support up to 1,000 items and a 24-hour mono 24 kHz
timeline; direct audio imports have their separate 30-minute limit. Desktop and
compatible browsers stream exports to disk. The browser memory fallback caps
downloads at 64 MiB and reports larger exports instead of exhausting the tab.

## Validation

```bash
backend/venv/bin/python -m pytest backend/tests -q
backend/venv/bin/ruff check backend
backend/venv/bin/ruff format --check backend
```

Use the host's actual accelerator/native runtime for relevant tests. Existing
whole-tree lint debt is recorded in the [security audit](../docs/SECURITY_AUDIT_2026-09-05.md).
See [tests](tests/README.md), [style](STYLE_GUIDE.md), and
[contribution guidelines](../CONTRIBUTING.md). Pinned MLX dependency upgrades
require exact-runtime and audio-equivalence validation, not only a passing import.
