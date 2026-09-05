# Contributing to Voicebox

Start with the [README](README.md) for daily operation, data locations, and the
[project structure](README.md#project-structure). This guide covers developing
and validating this fork. Be respectful, explain concrete behavior, and keep
changes focused and reviewable.

## Prerequisites and setup

- Python 3.12+; the setup recipe prefers 3.12/3.13 for ML compatibility.
- Bun for the shared `app`, `web`, `tauri`, and `landing` workspace.
- `just` for repository recipes. On macOS it can be installed with Homebrew;
  `cargo install just` is another supported installation route.
- Rust plus native build tools for Tauri. Rust is not installed automatically
  by the Tauri CLI. macOS builds need Xcode, including the tools used by `actool`.
  Windows needs the MSVC/native desktop prerequisites; Linux needs the WebKitGTK,
  GTK, audio, and desktop packages used by its build configuration.

```bash
git clone https://github.com/mlizaso/voicebox.git
cd voicebox
just setup
just dev
```

`just setup-python` creates `backend/venv` and handles engine dependencies with
conflicting upstream pins. It detects relevant GPU packages, installs the pinned
Apple Silicon MLX stack, and installs pytest/Ruff/build tools. `just setup-js`
runs the main Bun workspace install. The separate `docs/` workspace needs its own
install. The private ignored `voice-profile/` tree is not supplied by cloning.

For reproducible JavaScript installs, use `bun install --frozen-lockfile` at the
repository root and separately in `docs/`. Do not blindly upgrade Transformers,
MLX, or engine packages: generation revision checks intentionally bind their
compatibility and numerical behavior.

## Development commands

| Command | Behavior |
| --- | --- |
| `just dev` | Backend reload server and desktop development UI |
| `just dev-web` | Backend reload server and browser UI |
| `just dev-backend` | Backend only on port 17493 |
| `just dev-frontend` / `bun run dev` | Desktop dev UI against an existing backend |
| `bun run dev:web` | Browser client only |
| `bun run dev:landing` | Public landing site, normally on port 3000 |
| `just --list` | Full available recipe list |

An occupied backend port requires an explicit desktop connection action. Source
development launched from the root uses `./data`; a packaged sidecar uses the
OS app-data path. For real generation, use the stable `python -m backend.main`
command in the README, with explicit `--port` and `--data-dir`, without reload.

## Ownership boundaries

- Shared React code is in `app/`. Use `app/src/platform/` contracts for filesystem,
  capture, playback, dialogs, and native operations; import Tauri APIs only in the
  desktop adapter. Biome's restricted-import gate checks this boundary.
- `app/src/lib/api/` is the handwritten client. API/media calls must preserve the
  selected connection, bearer authentication, cancellation, and bounded downloads.
  `ServerImage` is the common private-avatar component.
- Backend routes validate and coordinate requests; services own reusable business
  operations. The engine registry describes supported engines and model sizes.
- Database changes need additive, repeatable migrations and compatibility with
  old rows. Keep audio publication and recovery consistent with database ownership.
- Audio/model changes need checks appropriate to the engine. Never weaken exact
  identity guards, discard completed audio, or alter quality settings just to make
  an optimization or a test pass.

See [backend architecture](backend/README.md), [Python style](backend/STYLE_GUIDE.md),
and [adding an engine](docs/content/docs/developer/tts-engines.mdx).

## Validation

Use the narrow checks for the changed area first, then the relevant integration
gates. These are separate checks: `just check` does not replace TypeScript checks
or test execution.

```bash
# Shared tests, import boundary, TypeScript, web + desktop frontend builds
bun run ci

# Python regressions, lint, and formatting
backend/venv/bin/python -m pytest backend/tests -q
backend/venv/bin/ruff check backend
backend/venv/bin/ruff format --check backend

# Full frontend lint/format checks (existing baseline debt remains)
bun run check

# Native desktop
cargo test --manifest-path tauri/src-tauri/Cargo.toml --bin voicebox
cargo check --manifest-path tauri/src-tauri/Cargo.toml --all-targets
cargo fmt --manifest-path tauri/src-tauri/Cargo.toml --check

# Landing
bun run --cwd landing test
bun run build:landing
```

On this Mac, native builds may need
`DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` so `actool` uses Xcode.
Metal/native checks need access to the host runtime. Synthetic tests do not
replace microphone/paste/output checks or frozen-build/model testing on each OS.
Whole-tree Ruff/Biome/rustfmt have known pre-existing findings; report the baseline
and keep changed code clean rather than adding suppressions or reformatting unrelated code.

Documentation has its own [build and API-generation workflow](docs/README.md).
Real model tests are opt-in; see [backend/tests/README.md](backend/tests/README.md).

## Builds and release work

| Command | Output |
| --- | --- |
| `just build` | Server sidecars and Tauri installer |
| `just build-server` | Platform server/MCP sidecars |
| `just build-tauri` | Desktop bundle using prepared sidecars |
| `just build-web` | Browser build in `web/dist/` |
| `just build-local` | Windows CPU/CUDA sidecars and installer |
| `just build-server-cuda` | Windows CUDA sidecar for local testing |

Installers are placed under `tauri/src-tauri/target/release/bundle/`. `scripts/`
and `.github/workflows/release.yml` define packaging details; use
[the build guide](docs/content/docs/developer/building.mdx) for frozen dependencies.
The updater remains configured for upstream releases unless deliberately changed.

Update only `[Unreleased]` in `CHANGELOG.md` during ordinary work. The local
`draft-release-notes` skill helps collect real changes. Version stamping/tagging
belongs to the separate release workflow. Never publish signing material, private
voice samples, database files, or tokens.

## Submit a change

Create a focused branch, explain the concrete before/after behavior, and include
the checks actually run plus any limitations. Follow recent commit style, for
example `fix(history): preserve generation settings on retry`. Avoid unrelated
formatting and dependencies. Describe a regression with a failing scenario and a
test that would fail without the fix.

Keep the README current when commands, storage, or user workflows change. Update
the relevant MDX guide and regenerate the API snapshot when request/response
schemas change. Dated audits, benchmarks, release history, and design proposals
are evidence for their stated revision, not substitutes for current usage docs.
