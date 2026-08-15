# Audiobook resume checkpoints — run notes

## Contract

Implement durable audiobook-generation progress for
`/Users/manexlizaso/Developer/manex/ebook/audiobook/voice-profile/build/make_audio.py`.

In scope:

- Persist an exact job snapshot at least every five minutes and on meaningful state changes.
- Provide an explicit **Save progress** action while rendering.
- Show saved jobs in the initial window, including book, selected voice(s), saved time, and progress.
- Allow each saved job to resume or be removed.
- Resume with the original normalized book text, voice configuration, output folder, format, seeds,
  render parameters, and completed audio artifacts.
- Never discard completed audio solely because a backend import produced a new profile UUID for the
  same voice assets.
- Survive backend/laptop restarts by reconnecting or restarting the backend and continuing from the
  durable renderer manifest.
- Delete a saved job automatically only after every requested voice has produced and validated its
  final audiobook file.
- Keep manual removal explicit and recoverable where practical; never delete produced audiobook
  files as part of removing progress metadata.
- Verify that the selected TTS model remains the highest-quality practical local cloned-voice model
  for Spanish, and apply safe upstream quality/performance fixes without silently changing voice
  identity.

Out of scope:

- Publishing or distributing cloned-voice audio.
- Cloud TTS services or paid APIs.
- Running multiple competing inference workers on the same Apple GPU.
- Deleting existing render artifacts without an explicit user action.

## Acceptance checks

1. A job saved to disk can be loaded by a fresh process without reading the original source again.
2. Automatic checkpoint writes are atomic and occur on a five-minute timer plus state transitions.
3. Manual save writes immediately and reports success/failure in the UI.
4. The initial screen lists multiple jobs and supports resume and remove independently.
5. Resume recreates missing backend profile IDs from the stored voice snapshot while retaining
   already-completed renderer artifacts through a stable voice fingerprint.
6. A backend connection loss is bounded, the backend is relaunched, and the current request retries
   without invalidating prior completed requests.
7. Failed or interrupted final assembly retains progress; only a fully validated final output clears
   the saved job.
8. Unit tests cover persistence, atomic writes, corrupt checkpoints, resume identity, removal,
   success-only cleanup, and non-ASCII book/voice names.
9. Existing relevant tests, syntax checks, and repository-native quality gates pass or any baseline
   failures are documented precisely.

## Baseline

- Runtime: host-native macOS, not an AppTec appliance or container.
- Git: clean `main` at `51f49de`; work moved to
  `feature/audiobook-resume-checkpoints`.
- Python compile check: passed.
- Repository `just` command was unavailable.
- The declared dev tools were missing from `backend/venv`; installed `pyinstaller`, `ruff`,
  `pytest`, and `pytest-asyncio` into that untracked virtual environment.
- Full Ruff baseline is already red across legacy backend files (thousands of findings; 57 files
  also fail format check).
- Full pytest baseline stops during collection in the pre-existing
  `test_profile_duplicate_names.py` import layout (`attempted relative import beyond top-level
  package`). Targeted and all-other test runs will be used to prove this change does not add
  failures.

## Decisions

- Saved jobs will use a versioned, atomic JSON registry plus immutable normalized-input snapshots.
  This is portable, inspectable, and needs no migration of the Voicebox database.
- A voice-content fingerprint, not a backend profile UUID, owns renderer cache identity. Backend
  UUIDs are runtime handles and may legitimately change after a restart or import.
- The current high-quality Qwen 1.7B cloned-voice configuration remains the default until a
  controlled quality benchmark proves a replacement is better.

## Implementation findings and choices

- The interrupted `La balada de Soi Cowboy` render still contains 17 of 33 current-contract WAV
  units, totalling 13,418.279 seconds (3 h 43 min 38 s). Two older WAV records are present but
  have a different parameter hash and are deliberately not mixed into the resumed book.
- The old UUID-era manifest did not record reference-audio hashes. Its currently attached two
  backend sample WAVs were therefore independently regenerated from the frozen `original`
  references through Voicebox's real preprocessing/storage code and compared byte-for-byte; both
  pairs match exactly (981,164 and 902,924 bytes). Resume now performs this read-only verification
  automatically and fails closed if any sample, transcript, count, or order differs.
- The apparent reset was caused by hashing the volatile Voicebox profile UUID into renderer cache
  identity. Re-importing the same voice under another database UUID invalidated otherwise good
  WAVs. New jobs hash immutable voice content; legacy jobs retain their original UUID contract so
  their compatible WAVs remain usable.
- New jobs use one 1,200-character REST request per 1,200-character model chunk. Observed model
  chunks on this machine were under five minutes, so a hard restart repeats at most the current
  small unit. Existing legacy jobs keep their frozen 45,000-character plan to avoid changing its
  parameter hash.
- Saved jobs live outside any book/output folder in a versioned atomic bundle under
  `~/Library/Application Support/Fabian Audiobook Maker/progress`. Each bundle contains immutable
  normalized documents, voice definitions, reference audio/transcripts, controlling configs, and
  a mutable `job.json` checkpoint.
- One global cross-process Voicebox session lock is held from frozen-profile activation through
  final rendering. A per-job and per-work lock additionally prevent stale checkpoint and manifest
  writers. Long renderer children inherit the lock descriptors so a parent crash cannot release
  protection while synthesis continues.
- Completed outputs are staged, decoded/probed, checked for chapter count, duration and content
  identity, fsynced, and revalidated before a job can enter `completed_verified`. Progress cleanup
  is the final operation and happens only when every selected voice passes that gate.
- The installed `mlx-audio==0.4.1` has upstream issue #874: float32 speaker embeddings promote a
  BF16 Qwen talker/KV cache to float32. Voicebox applies the exact dtype correction only to the
  affected version, expected Qwen interface, and BF16 talker. A direct upgrade to mlx-audio 0.4.8
  was rejected because it requires Transformers 5.14+, while Voicebox intentionally caps
  Transformers at 4.57.6.
- Clone errors and missing reference WAVs now fail closed. Falling back to an unconditioned generic
  voice could otherwise create hours of valid-looking audio in the wrong voice.
- Voicebox advertises the exact guarded Qwen implementation in `/health`. Audiobook generation
  uses `/generate/exact`, which checks the frozen revision before creating a history row or queue
  task. The distinct endpoint is intentional: an older backend cannot silently ignore an unknown
  JSON field during the health-check-to-POST race; it returns 404 and no audio is adopted.
- The startup list treats malformed central bundles and malformed renderer durations/totals as
  isolated corrupt progress rather than allowing one bad record to crash the entire initial UI.
- Qwen3-TTS 12 Hz 1.7B Base BF16 remains the production choice: it is the locally validated,
  Apache-2.0 Spanish clone, and Qwen's published Spanish results favor 1.7B over 0.6B. Higgs TTS 3
  4B is a strong quality challenger but its creator/non-commercial license is unsuitable for
  silent product integration. Apache-2.0 MOSS-TTS v1.5 is the most relevant long-form challenger,
  but its Spanish quality and Apple-Silicon throughput still need a controlled local A/B.

## Known limits

- The legacy UUID-era manifest cannot cryptographically prove which historical reference bytes or
  numerical dtype path produced its old WAVs because those fields were not recorded at the time.
  The current backend's two processed samples independently match the frozen originals byte for
  byte, so the 16 current-contract units are conservatively salvaged. New jobs record and enforce
  the missing fingerprints and implementation revision.
- A hard power loss can repeat the one request that was in flight. New requests are limited to one
  measured sub-five-minute model chunk; completed requests, normalized text, voice assets and all
  output/mastering settings are durable. The autoregressive decoder's in-RAM KV state itself is
  not serializable.
- No complete 20-hour audiobook or real Metal inference benchmark was run in the automated test
  environment. The speed expectation comes from upstream mlx-audio issue #874/PR #879 and the
  exact backport; a short real-book A/B remains the appropriate acceptance check for perceived
  voice quality and end-to-end wall time.

## Validation record

- Real isolated automatic startup: the launcher started Voicebox on loopback port 18494, reached
  healthy state, and reported the complete pinned package/model runtime fingerprint; the test
  process and its temporary data were stopped and removed without touching port 17494 or its data.
- Voicebox backend regression set: 169 passed, 4 skipped, 1 known flaky progress test deselected;
  the two pre-existing collection/Metal smoke-test blockers were excluded explicitly.
- Focused exact-generation, MLX backport and clone correctness tests pass, including rejection
  before profile/history/task/queue creation on a runtime mismatch.
- External launcher recovery, persistence, renderer/assembler, real FFmpeg and stub profile-import
  suites pass. Shell syntax and Python 3.9 compilation pass.

## Follow-up contract — resolve the audiobook “Main findings”

The follow-up request is interpreted as every concrete finding in the four existing
`makeaudio_audit_{data,pipeline,gui,failure}.json` reports. These reports overlap and predate the
checkpoint work, so each finding must first be re-tested against the current implementation. A
finding already fixed is closed with current evidence; a finding that remains reproducible is in
scope for implementation and regression coverage.

Acceptance checks:

1. EPUB extraction follows the package spine, excludes non-content metadata, preserves headings,
   and handles short legitimate chapters, lists, tables, drop caps, and Markdown input correctly.
2. Every output format uses the same declared pitch and loudness contract and remains safe for
   arbitrary valid paths.
3. Phrased and chunked renders preserve the exact document/chapter structure and never assemble
   stale or orphan artifacts.
4. Voice activation failures, cancellation, logging, saved-job discovery, overwrite behavior,
   estimates, title/author metadata, Back navigation, and intermediate-storage visibility are
   truthful in the wizard.
5. Disk-space checks discover the real backend data directory and prevent a long job from starting
   when its frozen storage estimate cannot fit safely.
6. Existing resume/checkpoint, exact-runtime, voice-integrity, and success-only-cleanup guarantees
   remain green after all fixes.

Out of scope for this follow-up: the separate acoustic QA/calibration findings in
`findings_consensus.json`; those concern `qa_generated.py` and `VOICE-PROFILE.md`, not the
`make_audio.py` “Main findings” reports or the requested audiobook execution workflow.

## Follow-up implementation result

- Every reproducible Main finding is closed. Source ingestion now has one normalized contract for
  EPUB, repaired JSON, Markdown and plain text; it follows EPUB spine order, excludes structural
  metadata/footnotes/URLs, preserves legitimate short chapters and repairs headings without
  corrupting normal Spanish prose. The wizard exposes the source/chapter choice as a real step.
- All containers now share one mastering topology. Multipart chapters are joined in one linear
  FFmpeg graph with the frozen crossfade, chapter gap, pitch and loudness settings; numeric chapter
  ordering is correct beyond 999 and apostrophes/UTF-8 output names are safe.
- Renderer output is assembled only from the current plan and checksum-valid mono 24 kHz PCM16
  files. Stale, malformed, wrong-version or corrupt records cannot inflate displayed progress or
  reduce resume storage estimates. Existing final audiobooks are never silently replaced.
- Storage preflight uses Voicebox's named, writable generations directory, accounts for remaining
  durable work on resume, and keeps assembly scratch on the preflighted output volume. Launcher,
  activation, rendering and assembly diagnostics are durable in the saved job.
- Standalone and wizard voice activation use immutable content-addressed snapshots. History detail
  now returns clean/original audio versions, so renderers download pre-effects audio before the
  single final mastering pass.
- The MLX numerical contract is pinned end to end: package versions, the BF16 speaker-embedding
  correction, and immutable Hugging Face commits for both supported Qwen model sizes are hashed
  into `/health` and enforced atomically by `/generate/exact` before any history row or task exists.
- The only imported UUID-era job is migrated only by its deterministic saved job id and known old
  contract. Unknown historical caches remain visible but fail closed rather than being mislabeled
  as current-runtime audio.

## Final validation record

- Current recoverable real work: `legacy-8d6ff27a0cfe383c61ea8837` is paused at 17/33 units
  (51.5%, 13,418.279 seconds) with its partial WAVs preserved.
- Focused Voicebox backend matrix: 59 passed, covering macOS ROCm startup, custom storage paths,
  concurrent filesystem probes, history versions, exact-generation rejection, pinned MLX loading,
  dtype identity and failure paths.
- External model-free/FFmpeg matrix: launcher 46, renderer/assembler 27, source parser 11, capacity
  5, job log 2, phrased version/disk 5, voice activation 8, plus every standalone progress and
  profile-import check. Localhost stub tests passed outside the restricted socket sandbox.
- The isolated auto-start check published the required backend on port 18494 and verified the exact
  runtime revision
  `qwen3-mlx-audio-0.4.1-bf16-speaker-v1-runtime-sha256-4e83c1b0dc7882c70bfc14054f5436c657bfb9e4d73eb496e0c1b7388e04a46a`.
- Bash syntax, ShellCheck, Python AST parsing, targeted Ruff checks/format checks and Git diff
  whitespace checks pass. The repository-wide legacy Ruff/pytest baselines remain as documented
  above and were not widened into this audiobook fix.
