# Voicebox

Voicebox is a local voice studio for cloned and preset voices, speech generation,
stories, transcription, dictation, and MCP voice tools. This checkout also has an
optional private audiobook workspace with chaptered output and durable resume.

**Start here:** this file explains how to run the project, find your profiles and
audio, generate a book, and navigate the code.

This is [mlizaso/voicebox](https://github.com/mlizaso/voicebox), based on
[jamiepine/voicebox](https://github.com/jamiepine/voicebox). Upstream installers,
the landing site, and the configured updater are separate from building this
checkout; upstream downloads do not necessarily contain this fork's changes.

## Quick navigation

- [Where your files are saved](#where-your-files-are-saved)
- [Run the project](#run-the-project)
- [Create a voice and generate audio](#create-a-voice-and-generate-audio)
- [Generate or resume an audiobook](#generate-or-resume-an-audiobook)
- [Audiobook command card](AUDIOBOOK_COMMANDS.md)
- [API and MCP](#api-and-mcp)
- [Project structure](#project-structure)
- [Backup, relocation, and troubleshooting](#backup-and-relocation)
- [Checks and further documentation](#development-checks)

## Where your files are saved

**A voice profile is database metadata plus reference audio.** Its name, language,
transcripts, voice type, preset choice, personality, and effects live in
`voicebox.db`. Cloned samples and avatars live under `profiles/<profile-id>/`.
Copying that folder alone does not restore a profile: export a profile ZIP or
back up the database and files together.

| How the backend is started | Data root |
| --- | --- |
| Development commands from the repository root | `./data/` |
| This checkout / default local audiobook wizard | `/Users/manexlizaso/Developer/manex/voicebox/data/` |
| Packaged macOS app | `~/Library/Application Support/sh.voicebox.app/` |
| Packaged Windows app | `%APPDATA%\sh.voicebox.app\` |
| Packaged Linux app | `$XDG_DATA_HOME/sh.voicebox.app/`, normally `~/.local/share/sh.voicebox.app/` |
| Standalone backend with `--data-dir` | The directory you pass |
| Docker Compose | `/app/data` in the data volume; generated files also appear in host `./output/` |
| Remote connection | The configured data directory on the server |

There is **no extra `data/` suffix** beneath the packaged app paths. A desktop
client connected to an already running backend uses that backend's data.

Ask the running server for its actual paths and disk space:

```bash
curl --fail --silent --show-error http://127.0.0.1:17493/health/filesystem \
  | python3 -m json.tool
```

The audiobook backend defaults to **17494**; the launcher can reuse a compatible
backend on **17493** and prints the selected URL. The response names `data`,
`profiles`, `generations`, and `captures`. Remote calls need a bearer token.
Settings folder buttons work only for the local desktop connection.

```text
<data-root>/
├── voicebox.db              Profiles, transcripts, history, stories, settings
├── profiles/<profile-id>/   Processed reference WAVs and avatars
├── generations/             Generated/imported audio and version artifacts
├── captures/                Saved recordings and uploaded capture audio
├── cache/                   Derived voice-conditioning cache
├── exact_voice_snapshots/   Immutable references for exact generation
├── deletion_journal/        Interrupted-file-operation recovery records
├── logs/                    Backend-managed logs
└── backends/                Downloaded CUDA/ROCm sidecars, when used
```

**Model weights are separate:** Hugging Face normally uses
`~/.cache/huggingface/hub/`, subject to `HF_HOME` or `HF_HUB_CACHE`.
Set `VOICEBOX_MODELS_DIR` before startup to override the hub cache location.
Unloading a model releases memory; deleting it removes downloaded weights.

**The private `voice-profile/` folder is different from `data/profiles/`.** It
contains source samples, narrator research, voice definitions, and audiobook
tools. It is ignored by the main repository; `voice-profile/build/` is a separate
local Git repository. A fresh clone does not contain these private assets.

## Run the project

Run commands from the repository root. The backend requires **Python 3.12+**;
setup prefers 3.12/3.13 for ML dependency compatibility. The frontend uses Bun.
Desktop development also requires Rust and native build tools; the audiobook
workspace needs FFmpeg/FFprobe and Python with Tk support.
See [CONTRIBUTING.md](CONTRIBUTING.md) for installation and platform details.

### First setup

```bash
git clone https://github.com/mlizaso/voicebox.git
cd voicebox
just setup
```

For this existing checkout, enter
`/Users/manexlizaso/Developer/manex/voicebox` instead of cloning again.
`just setup` installs the backend environment and Bun workspace dependencies.
It handles platform-specific ML packages; installing only
`backend/requirements.txt` is not equivalent. It does not install the private
audiobook workspace or download every model in advance.

| Task | Command from the repository root |
| --- | --- |
| Backend and desktop development | `just dev` |
| Backend and browser development | `just dev-web` |
| Backend development only | `just dev-backend` |
| Browser UI against an existing backend | `bun run dev:web` |
| Desktop UI against an existing backend | `bun run dev` |
| Build web client | `bun run build:web` |
| Build server and desktop installer | `just build` |

The backend uses **17493**. Open the Vite URL printed by the browser dev server,
normally `http://localhost:5173`. The browser shares the UI but lacks native
hotkeys, system capture, auto-paste, and local folder opening. If the desktop
finds an occupied port, use **Connect to my running server** only for a server
you started.

### Stable backend for a long generation

With the existing environment, no `just` command is needed:

```bash
backend/venv/bin/python -m backend.main \
  --host 127.0.0.1 --port 17493 \
  --data-dir /Users/manexlizaso/Developer/manex/voicebox/data
```

Keep that terminal open. On Windows use `backend\venv\Scripts\python.exe`;
on another checkout substitute your own absolute data path. Always specify the
port: the module alone defaults to **8000**. The backend CLI uses `--data-dir`;
`VOICEBOX_DATA_DIR` is a setting of the private audiobook launcher, not a general
backend environment override. Avoid `--reload` during real renders: source edits
can restart the backend and interrupt generation.

Only one backend can own a data directory, even on different ports. The CLI checks
this before loading audio dependencies: a repeated local launch reuses the
verified healthy backend and prints its actual URL, even if you requested another
port. Add `--strict-port` when the exact requested address is required. The
audiobook launcher discovers a compatible backend for its data folder, including
custom ports, and retries startup if a competing owner exits while it waits. An
explicit `VOICEBOX_URL` remains fixed. Leftover lock files after a crash are
harmless; ownership is released by the operating system when the process exits.

```bash
curl --fail http://127.0.0.1:17493/health
```

Health reports the runtime and exact TTS revision. Interactive API docs are at
`http://127.0.0.1:17493/docs`. First use may include model download and loading.

## Create a voice and generate audio

1. Open **Voices** and create a profile. For cloning, upload or record clean
   speech and supply its exact transcript. A sample can be at most 30 seconds;
   its transcript can be at most 1,000 characters. Preset profiles need no sample.
2. Choose a compatible engine. Qwen cloning offers `0.6B` and `1.7B`; TADA offers
   `1B` and `3B`. Other engines include LuxTTS, Chatterbox Multilingual,
   Chatterbox Turbo, Qwen CustomVoice, and Kokoro. Language and delivery controls
   depend on the engine.
3. Select the voice in the generation form, enter text and language, and review
   instructions and effects. Keep personality rewriting off when the spoken
   words must match a book or script exactly.
4. Generate, follow progress in history, and listen to a short sample first.
   Durable requests run through a serial queue.
5. Download the active audio, export a generation ZIP with metadata, or add the
   clip to a story. ZIP import/export preserves original audio bytes.

Normal requests accept up to **50,000 characters**. Defaults are **800 characters
per chunk**, **50 ms crossfade**, and normalization enabled. Configurable bounds
are 100–5,000 characters and 0–500 ms. Changing chunking, model, seed, references,
or effects can change the sound.

New records preserve chunking, crossfade, and normalization across retries and
regeneration. **Retry** repeats a failed request with its stored seed;
**Regenerate** makes a new take and may sound different. Older rows keep legacy
defaults. Directly imported audio has no TTS request to retry or regenerate.

The Stories editor arranges clips with tracks, trims, splits, and pinned versions,
then exports a mixed WAV. The private book wizard below adds book ingestion,
metadata, final containers, and book-wide resume; it is not a built-in audiobook tab.

## Generate or resume an audiobook

For the short copy/paste list, use the [audiobook command card](AUDIOBOOK_COMMANDS.md).

For this machine's existing private workspace:

```bash
cd /Users/manexlizaso/Developer/manex/voicebox
python3 voice-profile/build/make_audio.py
```

The launcher needs Tk; render workers use the private interpreter under
`voice-profile/.venv/`, the backend uses `backend/venv/`, and the launcher expects
FFmpeg/FFprobe under `/opt/homebrew/bin/`. These local assets and tools are not
provided by a fresh clone.

1. Choose **New audiobook** or **Resume** a saved job.
2. Select voices, source and chapters, output folder, format, and metadata.
   Supported sources include EPUB, repaired JSON, Markdown, and plain text.
   Review which text is selected when a repaired sibling file accompanies an EPUB.
3. Use **Demo rendering (5 min/voice)** to compare the same excerpt across voices,
   then choose the narrator before rendering the full book.
4. Start rendering. The wizard checks disk capacity and starts its backend at
   `http://127.0.0.1:17494` when needed.
5. Use **Save progress** or **Save & close**. Reopen and resume later; completed
   audio is reused only when saved identity and checksum checks pass.

| Audiobook item | Location |
| --- | --- |
| Narrator definitions | `voice-profile/build/voices/` |
| Original reference assets | `voice-profile/samples/` |
| Imported backend profiles | `<data-root>/profiles/` plus `voicebox.db` |
| Saved jobs and frozen inputs | `~/Library/Application Support/Fabian Audiobook Maker/progress/` |
| Partial renders and shared phrase cache | Hidden work directories in the chosen output folder, recorded in the job |
| Final `.m4b`, `.m4a`, `.mp3`, or `.wav` | The folder chosen in the wizard |
| Book progress log | Beside the output, named with `[progress <job-id>]` |
| Launcher-started backend log | `/tmp/voicebox-backend.log` by default |

Your existing book, for example, is under
`/Users/manexlizaso/Developer/manex/ebook/ebook/la-balada-de-soi-cowboy/`.
Final books do not need to live inside the Voicebox repository.

### Quality and generation time

The audiobook contract uses Qwen `1.7B` for Spanish with the pinned MLX runtime
on Apple Silicon. Jobs freeze text, reference assets, seeds, synthesis and
mastering settings, and runtime revision. A revision mismatch blocks exact
resume; do not edit hashes to combine incompatible output.

Keep one model worker, avoid backend reloads, and use demos before producing
multiple complete versions. The renderer reuses verified phrase audio, reads
compatible PCM WAVs directly without starting FFmpeg for each read, and writes
compact checkpoints without reducing their durability. These I/O improvements
preserve bytes. Changing model size, precision, sampling, chunking, or mastering
is not an equivalent quality-preserving shortcut.

Long books still require synthesis for each new phrase, then assembly and
mastering. The recent audit added no extra per-phrase inference or encoding,
but did not benchmark a complete 13-hour rerender.
See [the audiobook operating guide](docs/AUDIOBOOKS.md) for recovery details.

## API and MCP

These examples use a direct loopback backend. Replace `PROFILE_ID` and
`GENERATION_ID` with IDs from the responses. Generation is asynchronous:

```bash
curl --fail http://127.0.0.1:17493/profiles

curl --fail http://127.0.0.1:17493/generate \
  -H 'Content-Type: application/json' \
  -d '{"profile_id":"PROFILE_ID","text":"Hola, esta es una prueba.","language":"es","engine":"qwen","model_size":"1.7B","normalize":true,"personality":false}'

curl --fail --no-buffer http://127.0.0.1:17493/generate/GENERATION_ID/status

curl --fail http://127.0.0.1:17493/audio/GENERATION_ID -o speech.wav

curl --fail http://127.0.0.1:17493/transcribe \
  -F 'file=@recording.wav' -F 'model=turbo' -F 'language=es'
```

Wait for `completed` before fetching audio. `GET /history/GENERATION_ID` also
returns status, metadata, versions, and saved replay settings.

MCP is mounted at `/mcp`. Use `python -m backend.mcp_shim` from the backend
environment or the bundled `voicebox-mcp` for stdio clients. Tools are
`voicebox.speak`, `voicebox.transcribe`, `voicebox.list_profiles`, and
`voicebox.list_captures`. A stable `X-Voicebox-Client-Id` identifies per-client
voice bindings. See [MCP configuration](backend/mcp_server/README.md).

Remote deployments require explicit `VOICEBOX_TRUSTED_HOSTS`, any additional
`VOICEBOX_CORS_ORIGINS`, and a URL-safe `VOICEBOX_REMOTE_API_TOKEN` of 32–512 ASCII
characters. Use HTTPS and `Authorization: Bearer <token>`. Enter the URL and token
in **Settings → Server**. The app authenticates media with bearer headers and
keeps tokens in memory, so a restart may require entering the token again.
See [remote setup](docs/content/docs/overview/remote-mode.mdx).

## Project structure

| Path | Responsibility |
| --- | --- |
| `app/src/components/` | Shared UI: voices, generation, history, stories, captures, server settings |
| `app/src/lib/api/` | Handwritten API client, types, authenticated/bounded media requests |
| `app/src/lib/hooks/`, `app/src/stores/` | Queries, recording/playback ownership, settings, connection state |
| `app/src/platform/` | Platform contract used by shared UI |
| `web/` | Browser entry point and web platform adapter |
| `tauri/src/platform/` | Desktop implementation of the platform contract |
| `tauri/src-tauri/src/` | Rust sidecar, audio capture/output, hotkeys, paste, native security |
| `backend/main.py`, `backend/app.py` | CLI, FastAPI composition, startup/shutdown |
| `backend/routes/` | HTTP validation, endpoint coordination, responses |
| `backend/services/` | Generation queue, profiles/history/stories, archives, effects, recovery |
| `backend/backends/` | Engine registry, TTS/STT/LLM implementations, MLX runtime guards |
| `backend/database/` | SQLite models, sessions, startup migrations |
| `backend/config.py`, `backend/api_security.py` | Data paths and HTTP security |
| `backend/utils/` | Audio processing, chunking, cache, progress, upload bounds |
| `backend/mcp_server/`, `backend/mcp_shim/` | MCP tools and stdio transport |
| `backend/tests/`, `app/tests/` | Regression tests |
| `landing/` | Next.js public landing site, separate from the studio |
| `docs/` | Fumadocs site, API snapshot, operating guides, audit/design records |
| `scripts/`, `.github/workflows/`, `justfile` | Setup, packaging, CI, release commands |
| `voice-profile/` | Optional ignored private narrator/audiobook workspace |

Durable generation flows from the shared API client to
`backend/routes/generations.py`, through the queue and `services/generation.py`,
into an engine backend, then publishes audio and history/version records.
Shared React code uses the platform contract instead of importing Tauri APIs.

## Backup and relocation

Export profile/generation ZIPs for individual portable items. For a full backup,
stop backends using that data root and copy the **whole root**, including the
database and audio. Resumable books also need the progress directory, partial
output directories, private assets, and matching runtime. A final file is enough
for playback, but not for resuming synthesis.

To move standalone application data, stop the backend, copy the complete root
to a real directory, and restart with `--data-dir /absolute/new/path`. Model
migration is separate: use settings or configure `VOICEBOX_MODELS_DIR` before
startup. Migration refuses to overwrite model directories at the destination.

POSIX application data is private by default. Optional
`VOICEBOX_SHARED_GENERATIONS=1` makes generated audio host-readable while keeping
profiles, database, cache, captures, and logs private; Compose uses it for `output/`.

| Symptom | First check |
| --- | --- |
| Profiles/history seem missing | Compare server URL, port, and `/health/filesystem`; data roots may differ. |
| Data directory already in use | Use the reported running backend, or stop it in its terminal/app before restarting. Changing only the port does not help. |
| First generation is slow | Check model download/load status, then `/health` for the accelerator. |
| Book stops after a source edit | Use a stable backend without reload and resume the saved job. |
| Exact-runtime mismatch | Restore the matching runtime or create a new job; retain existing progress/WAVs. |
| Remote media fails | Check HTTPS, token, trusted host, and browser origin. |
| Folder controls disabled | They work only on the local desktop connection. |
| HTTP 507 / insufficient space | Check both backend data and audiobook output volumes. |

## Development checks

```bash
bun run ci
backend/venv/bin/python -m pytest backend/tests -q
just check-python
bun run check
```

`bun run ci` runs shared-app tests, import-boundary linting, TypeScript checks,
and web/desktop frontend builds. Repository-wide Ruff/Biome still have documented
baseline findings; do not suppress rules to hide them. Native and documentation
checks are described in [CONTRIBUTING.md](CONTRIBUTING.md).

| Document | Purpose |
| --- | --- |
| [Audiobook guide](docs/AUDIOBOOKS.md) | Private wizard, resume, quality, work files |
| [Backend guide](backend/README.md) | Server operation, routes, security, storage |
| [Contribution guide](CONTRIBUTING.md) | Setup, validation, build boundaries |
| [Docs guide](docs/README.md) | Documentation site and generated API pages |
| [Project status](docs/PROJECT_STATUS.md) | Current functionality and dated roadmap context |
| [Changelog](CHANGELOG.md) | Unreleased changes and release history |
| [Security policy](SECURITY.md) / [audit](docs/SECURITY_AUDIT_2026-09-05.md) | Deployment boundaries and dependency limits |
| [Run notes](RUN_NOTES.md) | Historical implementation and benchmark evidence |

See [LICENSE](LICENSE) and [responsible use](RESPONSIBLE_USE.md).
