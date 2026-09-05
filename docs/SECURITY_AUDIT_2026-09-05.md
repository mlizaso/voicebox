# Codebase audit — 2026-09-05

This audit covered the shared app, web and Tauri adapters, native desktop code,
backend routes/services/model loaders, MCP integration, build scripts, landing
site, documentation, and installed dependency graphs. Three independent
read-only reviews covered backend services, frontend state/media, and desktop
security/lifecycle; their findings were rechecked before implementation.

## Simplification and fixes

- Removed six unused legacy screens/components, the unused generated app API
  client, and its generation/patching scripts. The handwritten authenticated
  client remains the app boundary; documentation OpenAPI generation is retained.
- Replaced manual image-orientation branches with Pillow's complete EXIF
  operation, including mirrored orientations.
- Consolidated microphone ownership in the recording hook. Late permission
  responses and cancelled recordings release their streams; completed captures
  retain their own paste target and refinement preferences.
- Story playback loads active clips instead of predecoding the complete story.
  Remote media uses bounded bearer-authenticated downloads and revocable object
  URLs; local playback can stream. Remote media is capped at 32 MiB per clip and
  story playback at 32 simultaneous clips. Timeline waveform previews remain a
  separate loading path.
- Export responses stream through the platform file-saving boundary with size
  limits instead of eagerly building unrestricted blobs.
- Connection changes reset private query/player/draft state. Changes wait while
  a mutation or its callbacks are pending, preventing an old write from populating
  the next server's state. URL changes clear the old bearer token atomically before
  active queries refetch, preventing disclosure to the newly selected server.
  Requests and event streams use explicit bearer
  authentication instead of retained cookies; remote credentials are no longer
  written to browser storage. Event-stream parsing is capped at 1 MiB.
- The separate dictation window receives credentials through native memory and
  keeps pending work on its original connection. Capture events carry a temporary
  connection identity so late results cannot seed a different server's cache.
  Changing a connection during microphone recording cancels the eventual upload.
- Restricted desktop remote-origin, filesystem, and shell capabilities; added a
  CSP and exact origin checks. Removed unused diagnostic clipboard/focus commands.
  Paste targets are native, bounded, expiring, one-use identifiers, and clipboard
  snapshots have a 32 MiB limit before copying or replacing clipboard contents.
- Removed implicit trust in an unrelated server on the local port. Reuse requires
  an explicit connection action. Managed-child exit/failure clears ownership and
  stops its event monitor. Windows sidecars retain piped startup diagnostics.
- Native captures own their buffers, timer and stop signal. Startup failures are
  reported before accepting a recording; packet sizes and stored samples are
  bounded. Linux system capture never falls back to recording a microphone.
- Native outputs report readiness when streams start, reject cancelled starts,
  disambiguate duplicate device names, and bound input, decoded and resampled audio.
  A routing failure is surfaced instead of falling back to other speakers.
- Hardened deletion-journal recovery against malformed JSON paths, rejected
  missing Chatterbox voice references and invalid chunk sizes, and rolled back
  duplicate effect-preset updates cleanly.
- Repaired the corrupted repository ignore rule for local assistant settings.
  Download redirects use the canonical site origin instead of forwarding headers.

## Dependencies

- The main Bun workspace audit reports zero advisories after targeted framework
  and transitive upgrades. Tauri was upgraded beyond the IPC security fix in
  2.11.1; compatible vulnerable Rust transitive dependencies were upgraded.
- Documentation `image-size` 2.0.2 has no patched release for the ICNS/JXL/HEIF
  parser loops. A checked-in Bun patch fixes the loop bounds, with isolated
  timeout regression tests. Version-based scanners still flag the package.
  See [ICNS](https://github.com/advisories/GHSA-w3rx-r6r6-pgpr),
  [JXL/HEIF](https://github.com/advisories/GHSA-5p2g-fcmc-qvqq), and
  [patch maintenance](patches/README.md).
- The docs scanner's `fast-xml-parser <5.7.0` range also matches installed 4.5.7,
  but the [upstream advisory](https://github.com/NaturalIntelligence/fast-xml-parser/security/advisories/GHSA-gh4j-gqv2-49f6)
  explicitly lists 4.5.7 as fixed. No incompatible major override was added.
- **Remaining Linux dependency issue:** Tauri's GTK3 graph retains `glib` 0.18.5,
  affected by [VariantStrIter unsoundness](https://rustsec.org/advisories/RUSTSEC-2024-0429.html).
  The fixed 0.20 series is incompatible with that graph. GTK3, `fxhash`,
  `proc-macro-error`, and `unic-*` also carry maintenance advisories. These need
  an upstream desktop dependency migration; they are not represented as fixed.
- `rand` 0.7.3 remains transitively installed. Its
  [reentrant custom-logger issue](https://rustsec.org/advisories/RUSTSEC-2026-0097.html)
  requires a logger that calls the thread RNG; the application does not install
  such a logger. Compatible newer rand branches were updated.
- **Remaining Python constraints:** the audited environment uses Transformers
  4.57.3, pinned by the exact MLX runtime. Advisory fixes require incompatible
  5.x versions. Trainer checkpoint loading, X-CLIP, LightGlue and tokenizer
  `save_pretrained` paths are not used. Model repositories are fixed allowlists,
  the optional `kernels` package is absent, and the Qwen LLM loader now explicitly
  selects SDPA to prevent serialized configuration from selecting a remote kernel.
  This is a reachability assessment, not a claim that Transformers has no CVEs.
- Installed setuptools 80.10.2 has an sdist-generation advisory fixed in 83.0.0;
  this repository does not build sdists or use `MANIFEST.in`. Upgrade the build
  environment before introducing that workflow. Packages absent from PyPI and
  uninstalled platform-specific engine dependencies could not be fully audited.

## Validation and limits

Checks include the full backend suite; shared-app Bun tests and TypeScript
checks; web, Tauri frontend, landing and docs production builds; native macOS
compilation and synthetic Rust unit tests; touched-file Ruff/Biome/rustfmt checks;
and frozen-lockfile installation of both Bun workspaces. New regressions exercise
cloud callback state/replay/origin checks, malformed journals, mirrored EXIF,
invalid audio references, preset rollback, chunk bounds, media lifecycle,
connection isolation, bearer-authenticated media/events, parser loops, redirects,
native origin checks, capture isolation, output cancellation, and allocation limits.

The completed suites passed 840 backend tests (four platform skips), 24 shared-app
tests, 10 synthetic native tests, and three tests each for docs parsers and landing
redirects. The main `bun run ci`, native `cargo check --all-targets`, and separate
docs and landing production builds passed.

The initial full-repository lint baseline contained 410 Ruff errors and 113 Biome
errors, plus formatting debt. This audit did not reformat or suppress unrelated
files. Existing compiler/deprecation warnings and Vite's bundle-size warning remain.
Landing builds used Next's webpack builder because Turbopack's worker startup
was blocked in the local environment.

Tests use synthetic media and mocks. Actual microphone/system-audio recording,
clipboard pasting, browser autoplay behavior, Windows/Linux native execution,
GPU sidecars and frozen application startup still require platform validation.
Canceled or unused native focus snapshots expire rather than being discarded
immediately; retention is capped at 32 pending targets and one hour.
The exact MLX source fingerprint was refreshed because executable generation
code changed; pending exact-resume jobs from a different fingerprint correctly
remain incompatible. This audit is not a guarantee that no vulnerabilities remain.

## Second pass: security, modularity, and generation preservation

The follow-up review checked backend ingress and filesystem operations, desktop
and client boundaries, and generation persistence/replay. Three independent
read-only reviews covered these areas; confirmed findings were reproduced and
rechecked against the integrated changes.

- Bounded MCP stdin, JSON responses, SSE lines and complete SSE events. Health
  checks no longer read response bodies, error diagnostics read a bounded prefix,
  and fragmented SSE data is accumulated without repeated whole-line copying.
- Validated multipart transcription parameters before audio processing, bounded
  temporary upload suffixes, and sanitized imported display filenames. Archive
  imports now reuse live request validation for generation and profile metadata.
- Prevented model-cache migration from deleting existing destination models.
  Conflicts and linked model roots are rejected before moving any model.
- Restricted native directory opening to the current local server connection
  and main window. Network/device paths are rejected before canonicalization;
  stale filesystem responses are discarded after changing servers.
- Added one authenticated avatar component shared by four callers. It bounds
  downloads, omits cookies, releases blob URLs, and isolates connection changes.
- Made history responses extend the generation response model and removed unused
  regeneration version-ID plumbing. Added shared engine/model-size validation,
  retaining the actual TADA fallback used by old records.
- Persisted chunk size, crossfade, and normalization in the existing generation
  transaction. Retry and regeneration now reuse those settings. Nullable columns
  retain legacy defaults for old rows. Archive round trips preserve engine,
  model, seed, timestamp, source, settings, and the original audio bytes.
  Directly imported audio cannot be treated as a missing TTS request for replay.
- Coalesced concurrent release and star lookups and bounded their fetch duration.
  Failed pagination retains the previous complete result. In a controlled check,
  20 simultaneous release lookups used three upstream requests instead of 60,
  with the same returned download total. This measures duplicate work, not
  production network latency or audiobook generation time.

Final validation passed 884 backend tests (four platform skips), 26 shared-app
tests, four landing tests, and 11 native unit tests. The shared `bun run ci`,
landing webpack production build, and native `cargo check --all-targets` passed.
The separate audiobook renderer passed 76 focused I/O, phrasing, and recovery
tests plus 19 subtests, including audio-byte checks. Disabling persistence of
the new replay settings in a temporary mutation made all four replay regression
cases fail, confirming that those tests detect the original loss of settings.

Touched-file Python and TypeScript formatting checks passed. Remaining touched
file lint diagnostics match the baseline: 24 Ruff errors in the backend registry,
five Biome accessibility errors in the history table, and three existing Biome
warnings across the history table and model settings. Whole-project formatting
and compiler warnings described above remain; no lint rules were weakened.

Twelve waveform, conditioning, chunking, cache, queue, and MLX lifecycle modules
are byte-for-byte unchanged in this pass. Model weights, precision, sampling,
audio processing algorithms, and normal generation defaults are unchanged.
The added settings use existing database writes; no additional per-phrase
inference, encoding, or checkpoint flush was introduced. The exact source
fingerprint was refreshed because it also covers request and schema code;
cross-revision exact resumes continue to fail closed. Existing audiobook files
were not modified. A complete 13-hour book was not regenerated, so end-to-end
generation time and listening quality were not newly benchmarked.

A fresh main-workspace Bun audit returned no advisories. The installed Python
audit still reports the Transformers and setuptools constraints described above;
the pinned MLX toolchain was retained. The additional Transformers custom-generation
cache issue documented by [CERT VU#456290](https://www.kb.cert.org/vuls/id/456290)
was assessed against the application's fixed model allowlists. Those boundaries
reduce exposure but do not remove the upstream dependency issues. Linux/Windows
native runtime checks and full model-generation equivalence remain outside this
macOS validation pass.
