# Audiobook resume checkpoints — run notes

> **Historical implementation record.** For current commands, storage locations,
> and project layout, read [README.md](README.md). For the private book wizard,
> read [docs/AUDIOBOOKS.md](docs/AUDIOBOOKS.md). Paths, hashes, test counts, and
> benchmarks below describe their original revision and may be superseded.
> Current security/replay work through `8c6a3cf` is documented in the
> [2026-09-05 audit](docs/SECURITY_AUDIT_2026-09-05.md).

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

## Inference-performance follow-up contract

The second follow-up names five inference findings explicitly. This pass treats all five as
acceptance requirements, while preserving the recovery and voice-integrity guarantees above:

1. Prove the historical Qwen RTF and where wall time is spent from the durable render logs; do not
   claim FFmpeg or HTTP work is the dominant optimization target.
2. Keep the controlled mlx-audio 0.4.1 BF16 dtype backport, but bind every numerical optimization
   and immutable model/runtime dependency into the exact `/health` and `/generate/exact` identity.
3. Preserve completed work across backend profile UUID changes, bound backend-outage waits, restart
   safely, and checkpoint every independently reusable synthesis unit.
4. Add cloned-voice model-level batching beginning at batch size two only if the actual ICL path is
   supported and quality/recovery semantics can fail closed. Do not fake batching with concurrent
   HTTP requests or silently mix batched and serial contracts in an existing saved job.
5. Cache the expensive, content-addressed reference conditioning itself (speaker embedding,
   reference codec and reference-text preparation), not merely its WAV path. Invalidate the cache
   on any reference/model/language change and on model unload.
6. Provide a reproducible 3/6/10/20/current-reference benchmark with wall-time plus intelligibility,
   speaker-similarity and prosody gates. A shorter production reference may be selected only from
   real acoustic evidence; otherwise the current reference remains frozen.

The current 17/33 legacy job must remain resumable under its frozen serial render contract. New
throughput settings belong in the semantic renderer/runtime identity so an upgrade cannot
silently reinterpret its completed artifacts.

## Generation-memory and shared-storage follow-up

Contract: keep legal 50,000-character / multi-hour TTS generation duration-independent in RAM,
preserve the exact legacy crossfade and normalize-before-effects samples, enforce the 24-hour and
1 GiB-reserve bounds, and drain cancellation before deleting private scratch or journaled output.

Implementation result:

- Multi-chunk and runaway-retried audio now accumulates in an anonymous/private float32 mapping;
  callers release its mapping and temporary-file handle deterministically.
- Normalization, WAV publication, streaming responses, and long effects render blockwise or through
  bounded disk scratch. Foreground and background both normalize the generated input before effects.
- A process-wide per-filesystem reservation ledger protects the same free bytes from concurrent
  generation, effects, normalization, publication, checkpoints, stories, uploads, archives, caches,
  profiles, and accelerator staging. Leases preserve the strongest active reserve floor and can be
  atomically resized between phases; non-growth cleanup cannot fail merely because free space fell.
- Exact checkpoints and journaled clean/processed publication retain their existing fsync and
  recovery ordering. The legacy byte-returning service is capped at ten minutes; the HTTP stream
  path remains disk-backed for the full legal request size.

Validation record:

- The focused generation/effects/checkpoint/stream matrix passes 171 tests, including byte-identical
  legacy crossfades, bounded retention, cancellation cleanup, reserve overlap, and short plus forced
  disk-pipeline foreground/background +6 dB parity.
- The focused shared-reservation lifecycle tests pass, including failed growth preserving the old
  claim and resize-to-zero succeeding after simulated external free-space exhaustion.
- Targeted Ruff, Ruff format, Python compilation, and diff whitespace checks pass. Expected
  Pydantic/SQLAlchemy deprecation warnings and the headless Metal atexit warning remain unchanged.

## Audiobook runtime identity repair (2026-08-15)

User goal: after stopping the stale backend on `127.0.0.1:17494`, make the audiobook launcher start
against the current Voicebox checkout without weakening the exact-generation identity gate.

Scope and decisions:

- Treat `tts_implementation_revision=null` as an attestation failure, not a launcher condition to
  bypass. The repair must refresh the embedded AST fingerprint from the settled numerical source
  and derive a new runtime identity from that fingerprint plus the existing package/model pins.
- Remove the launcher's duplicate manually copied current hash. It will load the reviewed embedded
  identity from Voicebox's lightweight `mlx_runtime.py`; live startup still accepts it only when
  `/health` reports the same value, so edited/unattested source continues to fail closed.
- Preserve every existing dirty-worktree change. Only the attestation constant, the launcher pin,
  their focused tests, and this run record are in scope.
- Do not load a model or resume the preserved audiobook job while validating. Acceptance is a
  model-free source-integrity test, matching live `/health`, a passing launcher compatibility check,
  and successful GUI/backend process startup on port 17494.
- Do not reuse the old c5 identity for changed executable sources; exact resume must fail closed
  across numerical source changes.

Acceptance checks:

1. The current source fingerprint equals the embedded fingerprint under the backend venv.
2. `/health` reports a non-null identity derived from the attested source and pinned runtime/model
   inputs.
3. `make_audio.TTS_IMPLEMENTATION_REVISION` is loaded from Voicebox's reviewed runtime metadata and
   equals that exact backend identity, with regressions for missing or malformed metadata.
4. Focused backend identity/exact-generation and launcher recovery tests pass; touched files pass
   Ruff/format/compile checks.
5. The old launcher/backend processes are replaced once and the audiobook maker prints that the
   backend is ready at `http://127.0.0.1:17494`.

Implementation and validation result:

- The attested local source inventory now includes effects processing, the bounded voice-prompt
  cache, and the shared disk-reservation implementation. Its embedded AST fingerprint matches the
  settled source, yielding
  `qwen3-mlx-audio-0.4.1-bf16-b2-icl-v3-runtime-sha256-9883b936782e3a234eb7c7e3fa1aaf2347410f7be4eed0d5c3c861819670f34f`.
- The launcher loads that reviewed identity directly from Voicebox runtime metadata. Regressions
  prove source-of-truth parity and fail closed for missing, malformed, or execution-failing
  metadata, so a future manually copied launcher hash cannot drift from the backend again.
- The model-free backend exact/audio matrix passed twice at 188/188 on the final source. The full
  audiobook suite passed twice at 152/152, and its focused recovery/identity file passed 59/59
  after the final loader hardening.
- Backend Ruff and format checks, external Python compilation, targeted external Ruff F/E9/I
  checks, and repository diff whitespace checks pass. The external scripts retain their older
  whole-file formatting/lint baseline; this repair did not mechanically rewrite unrelated code.
- The replacement backend is healthy on port 17494 and advertises the identity above; the live
  launcher reports the startup contract as compatible and remains open. No model was loaded and
  no saved audiobook work was resumed or mutated.

## Five-minute narrator demos and transient-health recovery (2026-08-15)

User goal: add a separate “demo rendering” action that renders the same representative five-minute
excerpt in every selected voice, then lets the user choose one narrator before starting the full
book. Also prevent a transient `/health` timeout from being misreported as a runtime change and
terminating an otherwise valid renderer.

Scope and decisions:

- Keep the exact runtime gate fail-closed before new work starts and on a positively observed
  revision mismatch. During an already-running render, an unreadable health response is only
  absence of evidence: retain the renderer and retry until a healthy response can prove a match or
  mismatch.
- Build one deterministic excerpt from the middle of the selected normalized text, using complete
  paragraphs/sentences and the shared 158-wpm estimate. Every selected voice receives identical
  text, capped at the approximately five-minute word budget.
- Expose “Demo rendering (5 min/voice)” beside “Start full rendering” on the review screen. Demo outputs
  are lossless WAV files with explicit demo titles, use the same frozen voice/runtime/pitch/effects
  pipeline, remain resumable, and never overwrite or reinterpret a full-book job.
- After a successful in-session demo, show every output with Play/Reveal controls and require one
  selected narrator before returning to the full-book review. A resumed demo remains recoverable
  even if its original wizard draft no longer exists.
- Do not restart or cancel the live backend. The orphaned exact generation accepted before the
  timeout completed durably and must be recovered by normal deterministic resume.

Acceptance checks:

1. A timeout or connection error in the supervisory revision probe does not signal or fail the
   renderer; a healthy response with another revision still does.
2. Demo extraction is deterministic, non-empty, identical across voices, and bounded to the
   five-minute word budget for short, long, Unicode, and oversized-paragraph sources.
3. Demo start freezes a distinct resumable WAV job without mutating the full-book draft; success
   restores that draft and narrows the final render to the narrator chosen on the result screen.
4. Focused launcher/demo regressions and the complete audiobook test suite pass twice; targeted
   lint, compilation, documentation, and diff audits are clean.

Implementation and validation result:

- The active-render supervisor now distinguishes an unreadable health probe from a positively
  observed identity mismatch. The former keeps the renderer attached and retries; only the latter
  terminates it. Initial/new-work admission remains fail-closed.
- Review now offers a separate five-minute-per-voice demo action. It freezes one deterministic,
  centered excerpt (790-word budget), uses the same excerpt for every selected voice, renders
  sequential lossless WAVs through the production pipeline, and offers Play/Reveal plus a required
  narrator choice before returning to full-book review in the same wizard session.
- Saved progress records and validates `book` versus `demo` mode and the 300-second planning target;
  pre-feature schema-1 jobs load as ordinary full-book work. Failed and restarted demos retain the
  same immutable snapshot/resume behavior as normal jobs.
- The timeout/mismatch and demo workflow regression file passes 65/65. The standalone progress,
  integrity, locking, legacy-import, and corruption suite passes, including demo restart and invalid
  metadata cases. The complete audiobook-maker suite passes twice at 158/158.
- Targeted Ruff F/E9/I/W293 and Python compilation pass. Whole-file Ruff formatting remains the
  external scripts' pre-existing baseline and was deliberately not used to rewrite unrelated code.
- Live verification found the supposedly changed backend still healthy at the exact saved v3
  revision. Generation `ad801966-cdf9-577f-8f8f-9d9b8d898a74` had completed durably; the relaunched
  maker reattached, downloaded it without regeneration, advanced the book from 17/33 to 19/33
  verified units, and continues rendering with the backend left running.

## Demo assembly recovery and pitch tolerance (2026-08-15)

- The resumed five-minute demo initially failed only while mastering the `podcast` voice: its
  measured correction was -0.67 semitones, just outside the old 0.5-semitone guard. The saved
  chunks were complete; no TTS output was lost.
- Demo assembly now carries an explicit per-voice lower-pitch tolerance. The podcast profile uses
  the measured 1.0-semitone tolerance, and older saved demo jobs without that field receive the
  same bounded demo-only fallback. Full-book renders keep the strict default unless a voice
  explicitly configures a tolerance.
- Resume skipped all four already verified voices, assembled the saved podcast chunks, and produced
  all five lossless narrator-demo WAVs. The maker now shows the narrator-selection screen.
- The complete audiobook-maker regression suite passes 158/158 after this fix. Focused recovery,
  lint, compilation, shell-syntax, and manual saved-chunk assembly checks also pass.

## Natural-pitch voice family (2026-08-15)

- Feedback identified the original, podcast, and expressive variants as poor matches. Their old
  definitions changed reference corpus, chunk renderer, and/or pitch correction, which changed the
  voice rather than merely changing delivery.
- All comparison variants now inherit natural-pitch's book reference pair, phrase renderer, no
  pitch correction, chunk size, and crossfade. They differ only in explicit pause scale: original
  0.60, expressive 0.65, podcast 0.75; natural-pitch remains the 0.50 canonical base.
- New audiobook selection defaults to natural-pitch. A regression contract verifies the shared base,
  bounded pause scales, and the default selection; the full audiobook suite passes 161/161.

## Audiobook bit-identical speedup continuation (2026-08-22)

### Contract

Resume checkpoint 1 from `voicebox-audiobook-speedup` without re-deriving its measured budget or
re-litigating rejected optimizations. Preserve the already implemented synchronous streaming route,
reference-prefix vocoder skip, and cross-variant phrase sharing; close the remaining technical risks
before treating the speedups as ready.

In scope:

- Re-establish the checkpoint's full Voicebox and audiobook test baselines.
- Exercise the old exact/history/download route and the new exact-stream route against the real,
  warm backend, prove their returned WAV bytes match, and measure the actual routing overhead.
- Root-cause the deterministic zero-frame failure at `ch003/000353` (seed `20261666`) and implement
  the smallest deterministic recovery that does not weaken voice or runtime identity.
- Re-run focused and full validation, audit the complete diff, and leave a reviewable commit.

Out of scope:

- Re-running previously rejected batching, chunk-renderer, merged-phrase, deduplication, or talker
  quantization experiments.
- Changing the paused job's saved numerical identity or silently discarding its artifacts.
- Entering the user's administrator password or changing macOS Low Power Mode without them.
- Starting a full 15,151-phrase production render before the deterministic crash is closed.

### Acceptance checks

1. The live backend advertises the newly attested implementation revision.
2. Multiple real phrases returned by the old and streaming routes are SHA-256 identical.
3. Warm interleaved measurements quantify endpoint overhead without model-load bias.
4. The formerly crashing phrase either generates valid deterministic audio or fails in a durable,
   explicit way that lets the remaining book continue without changing its seed.
5. Focused regression tests and both checkpoint full suites add no failures over their documented
   baselines; touched files pass repository-native lint/format/compile checks.
6. No tests are skipped, weakened, or silenced to obtain a green result.

### W5 zero-frame root cause and decision

- The exact failed plan entry is chapter 3, phrase 353: the punctuation-only editorial omission
  marker `[…].`, with its frozen positional seed `20261666` and planned 0.692-second pause. It is
  the only recognized standalone omission marker among all 15,151 phrases in the saved book.
- Four persisted generation attempts were independently rechecked in `data/voicebox.db`: all use
  text `[…].`, seed `20261666`, Qwen 1.7B Spanish, and exact request hash
  `792f25069a19b52d0c0e6153fe708519fde6fe45d73cd63a55dd7cfdd4bc55dc`; they span the two
  runtime profile UUIDs from the audiobook variants. Every corresponding WAV is exactly 44 bytes,
  mono PCM16 at 24 kHz, with zero frames and zero duration.
- The pinned mlx-audio `_generate_icl` loop checks codec EOS before appending `all_codes` to
  `generated_codes`; an immediate EOS therefore reaches its `if not generated_codes: return`
  without calling the speech-tokenizer decoder. Voicebox's MLX adapter then deliberately returns
  an empty float32 array. This proves the old effects error was downstream detection and rules out
  the reference-window vocoder optimization as the cause.
- Treat recognized `[…]`, `[...]`, parenthesized, guillemet, and bare ellipsis markers as silent
  editorial structure. The renderer writes a valid one-frame PCM16 cache artifact, retains the
  marker's planned pause, and still consumes its positional seed; all following spoken phrases
  therefore retain their exact historical seeds. This avoids a pointless model call without
  suppressing any spoken content.
- Do not retry with another seed. Voicebox now rejects every genuine zero-frame model result at
  the `generate_chunked` boundary, before normalization/effects, with a bounded preview of the
  responsible text. The synchronous route returns this deterministic condition as HTTP 400, so
  the renderer does not mistake it for a transient backend outage. Any spoken phrase that returns
  zero frames still fails loudly and durably.
- The renderer algorithm identity is bumped from `phrased-v2` to `phrased-v4`. Besides the omission
  marker contract, v4 binds phrase and chapter caches to the ordered reference audio/transcripts,
  the attested backend implementation revision, and a full canonical SHA-256 phrase identity. This
  is an explicit semantic cache boundary; the already-required backend revision restart means the
  paused 854/15,151 job could not exact-resume under the speedup sources in any case.
- Post-benchmark batch hardening and fail-safe cleanup of the serial decoder's one-shot reference
  handoff changed the local executable-source attestation once more. The current backend identity
  is `qwen3-mlx-audio-0.4.1-bf16-b2-icl-v3-runtime-sha256-891f41cb4ee209e972a8faf54263b26f671b730f0a737767520cb0c174ddd268`.

Focused validation so far: the pre-hardening Voicebox slice passed 53/53 with Metal available.
After the batch and serial-handoff cleanup, the combined touched slice passes 108 model-free tests;
the five focused decode cases cannot acquire Metal in the restricted sandbox and are not treated
as product failures. The external full suite last reached 297 passes
plus the documented pre-existing staging-sweep failure before the final integrity/capacity edits.
Those later external edits still require their focused and full reruns before commit. The live A/B
below is the direct real-model evidence for the serial numerical path; the later batch-row fix must
not be represented as having been part of that earlier benchmark.

### W4 user-only power setting

`pmset -g custom` still reports `lowpowermode 1` for AC power. Changing this requires the user's
administrator password and remains intentionally unattempted. After the current controlled A/B,
the user-only action is `sudo pmset -a lowpowermode 0`; the same warm benchmark should then be
rerun if the absolute throughput gain needs to be quantified.

### Live W1 end-to-end A/B (pre-batch-hardening attested source)

- The backend advertised exact revision
  `qwen3-mlx-audio-0.4.1-bf16-b2-icl-v3-runtime-sha256-823785445bef20a93db04daab469d99139b0a4685a99adc71d769af5734bdb0c`.
- After warmup, 20/20 measured old/new route pairs were WAV-byte-identical across three real book
  phrases. The old route was `/generate/exact` plus history polling and original-version download;
  the new route was `/generate/stream/exact` with the mandatory explicit empty effects chain.
- Old-route median was 9.3181037085 seconds; new-route median was 5.9724888745 seconds. The
  ratio of medians is 1.5601709613x and the paired median saving is 2.0683254375 seconds. Means
  were 8.6643513457 versus 6.7448648749 seconds, for a paired mean saving of 1.9194864709 seconds.
- Low Power Mode remained enabled during the benchmark, so these are internally controlled route
  comparisons rather than claims about the machine's eventual unthrottled absolute throughput.
- The old route necessarily persisted benchmark history. Cleanup removed only the 21 benchmark
  generation rows and their 42 version rows/files; post-cleanup queries verified zero matching DB
  records and zero matching files. The benchmark backend was then stopped cleanly to free Metal
  for the final validation pass.

### Final integration audit

- Adversarial review found that a bare shared hard link was not an integrity boundary. The pool now
  uses an atomic per-key slot with `audio.wav` plus SHA-256-attested metadata, validates the slot
  under a lock, refuses symlinked roots/shards, and treats different valid PCM for one key as an
  integrity failure rather than silently selecting a winner.
- Completed local phrase WAVs whose recorded checksum changed are regenerated (or replaced from an
  attested pool). A trusted pool is preferred over an unauthenticated crash-status local WAV.
- The remaining external audit is intentionally open: a structurally valid local WAV written in
  the crash window before its manifest checksum is durable must also be regenerated when no trusted
  pool exists; HTTP 507 must be fatal instead of entering the transient-backend retry loop; and
  shared-pool ownership/cleanup is only partially wired. No first commit is ready until those three
  issues are fixed and the final external validation is rerun.

### Closing the three open issues (2026-08-23)

The previous entry ended "No first commit is ready until those three issues are fixed and the
final external validation is rerun." A 13-agent adversarial audit then re-derived the whole change
set from the code: six refutation lenses, each paired with an independent skeptic told to refute
its findings, plus a completeness critic. 39 raw findings, 10 killed by their skeptics, 28
surviving with an explicit `refuted=false` verdict. O1, O2 and O3 were all confirmed real, and the
audit added one major the earlier list had missed. All four are now fixed.

- **O1 — an unattested phrase WAV is no longer adopted.** `render_phrased.py` used to accept a
  local phrase WAV whenever it merely decoded, on the reasoning that the content-addressed
  filename proves the synthesis inputs. It proves what was *asked for*, not what the bytes are. A
  phrase interrupted after `_atomic_write_wav` but before its checksum reached the manifest has no
  attestation, so anything that damaged those bytes afterwards was laundered into a `completed`
  record carrying a fresh checksum of the damage — and then published to the shared pool, where the
  sibling variant hard-linked it as trusted. Only the pool, whose slots carry their own SHA-256,
  may now rescue a phrase the manifest does not vouch for; everything else is re-synthesised.
- **O2 — the retry classifier no longer treats standing conditions as outages.** `507` matched
  `500 <= code < 600`, so a full disk cost a full inference per attempt for the whole outage budget
  and then died behind a "backend unavailable" banner naming the wrong problem; it is now fatal and
  says so. `http.client.IncompleteRead` is `HTTPException`, not `OSError`, so a body truncated by a
  backend dying mid-response escaped `gen()` uncaught and killed the render on first occurrence; it
  is now retried within the same bounded budget. And deterministic synthesis faults reached the
  client as a bare `500 Internal Server Error` indistinguishable from a recoverable fault: a new
  `DeterministicSynthesisError` marks the faults that are a pure function of (text, seed, frozen
  voice), and the streaming route maps it to HTTP 400 with the real message, restoring the contract
  the queued route always had. Deliberately not a bare `except RuntimeError` — a model load/release
  failure is infrastructure, not a property of the request, and must keep propagating as a 500. The
  existing `test_stream_generation_releases_disk_audio_when_model_context_exit_fails` proved that
  distinction matters: a blanket catch broke it.
- **O3 — the shared pool has an owner and a way to be reclaimed.** `_require_phrase_pool_owner` had
  no call site and `PHRASE_POOL_OWNER_MARKER` was never written, so the destructive `rmtree` ran on
  a derived path with only path-shape checks. Creating a phrased job now stamps its pool, and both
  removal paths verify the stamp. The guard is strict about a marker naming a *different* job and
  tolerant of a missing one — a pool is addressed by `out_dir` plus a 32-hex job id, so refusing an
  unstamped directory would trade a real recurring leak for a collision that cannot happen.
  `remove()` now reclaims the pool too: it preserves partial WAVs on purpose, but the pool is
  derived cache nothing can address once the bundle naming it is gone, and its hard links otherwise
  keep the whole book's phrase audio alive after the visible work directories are deleted. The path
  recorded at creation is now authoritative instead of being re-derived from two different bases.
- **C1's backend half is tested.** The byte-identity claim rests on the backend treating an explicit
  `effects_chain: []` as "no effects" while an *absent* one inherits the profile's chain, and
  nothing tested it — the claim was backed only by an unreproducible sentence in a commit message.
  Two tests now pin both halves of that distinction.

Every fix carries a regression test, and each test was mutation-checked by reverting the fix and
confirming the test fails: 507-fatal, IncompleteRead-retried, IncompleteRead-still-bounded,
unattested-WAV-regenerated, pool-stamped-at-creation, pool-reclaimed-by-remove, and the
empty-effects-chain contract all fail without their fix and pass with it. The one exception is
recorded honestly: `test_unattested_phrase_is_never_published_to_the_shared_pool` passes with and
without the change, because the pool's existing conflict detector already refuses a mismatched
publish; it is a characterisation test for that invariant, not a regression test for this fix.

`backend/routes/generations.py` and `backend/utils/chunked_tts.py` are both in
`MLX_QWEN_TTS_LOCAL_NUMERICAL_SOURCE_PATHS`, so the attestation moved again. The embedded
fingerprint was updated in the same change and verified to match the live AST computation:
`voicebox-mlx = 77d741fbddffe8b5ae4b58d9c754de230e97b15dd6a65eff378b2242455f9675`, revision
`qwen3-mlx-audio-0.4.1-bf16-b2-icl-v3-runtime-sha256-c5f3b9993ee1bf1b5b09f02c4dfd2f9b0ae354073e96a1e5463e82d6ac5d51b8`.
No saved job is stranded by it: the operator agreed to discard `887308ee5984446ebe60242e39040bf6`,
whose two voices had both already failed on the `ch003/000353` crash, and it has been removed along
with its 323 MB of render directories. The progress store is now empty. Those 854 phrases per voice
would not have been reusable in any case — `synthesis_sha` covers `algorithm`, which went
`phrased-v2` to `phrased-v4`.

Validation: Voicebox `760 passed, 4 skipped, 1 failed`; audiobook `317 passed, 1 failed`. Both
failures are the documented pre-existing ones (`test_hf_progress_tracker`, and the staging-sweep
test the audit separately diagnosed as dead — `run_one` returns at the earlier backend-revision
gate and never reaches the sweeper). Both suites gained tests and lost none. The audiobook suite
must be run with the conda interpreter; the `python3` on PATH has no numpy and fails at collection.

Still open and deliberately not done here: the remaining 24 surviving findings are minors and nits
(pool crash debris, silent `os.link` EXDEV fallback, `F_FULLFSYNC` versus `os.fsync` on Darwin, the
untested `effective = valid_samples` branch, the dead staging-sweep test), and macOS Low Power Mode
is still on — `sudo pmset -a lowpowermode 0` needs the operator's password.

### Closing F8 and F4, and putting the audiobook workspace under git (2026-08-23)

The completeness critic that had failed on a session limit was relaunched and completed. It
reported ten findings; each was checked against the code rather than accepted. Its headline "F1
blocker" — that verified-success cleanup now aborts on an unstamped pool, leaving the audiobook
suite at 308/2 — is **false**: it sampled the tree during a mutation-testing window, when those
exact lines had been deliberately reverted to prove the new tests catch regressions.
`_require_phrase_pool_owner` returns on `FileNotFoundError`, that test passes, and the suite is
green. Two of its findings were real regressions introduced while fixing O1/O2/O3, and were fixed
at once: the pool path had lost the containment rule that keeps it inside `out_dir` (the per-voice
work directories have always had one), and its docstring claimed a property the code did not have.

The two findings left open at the previous checkpoint are now closed.

- **F8 — a pool conflict could delete the only attested copy.** On resume the renderer republishes
  phrases `phrase_done` has just verified against their recorded SHA-256. `_publish_to_share`
  answered a conflict by `os.replace`-ing the *source* out of the cache and raising, which for that
  caller meant destroying the one file this render had vouched for while leaving the suspect pool
  slot in place — and `PhraseShareIntegrityError` was caught nowhere in production code, so it
  surfaced as an unhandled traceback with the manifest still claiming the phrase was complete.
  Publication now takes `source_attested`: unattested bytes are quarantined as before so a retry
  can adopt the pool's attested copy, attested bytes are kept and the pool is named as the outlier,
  and the synthesis-branch call records the failure in the manifest before re-raising.
- **F4 — the capacity estimator's third copy of the sharing identity is correct.**
  `audiobook_capacity._phrased_synthesis_group` omits `algorithm`, which the renderer's
  `synthesis_params` includes. That cannot mis-group anything: `RENDER_ALGORITHM_VERSION` is a
  module constant, identical for every voice in one estimate, so it can never split a group; and a
  voice with no `synthesis_fingerprint` falls back to `("unique", index)`, which over-budgets. The
  estimator is exact or conservative, never under-budgeting. The structural risk of two
  hand-maintained copies is real, so it is now pinned by tests rather than merged: one reads
  `render_phrased.py` and fails if the renderer's synthesis identity gains or loses a field, and one
  asserts every field the estimator *can* observe still splits its groups. Merging them would drag
  numpy, soundfile and ffmpeg into a pure-arithmetic module for no benefit.

Both fixes are mutation-verified: reverting each makes its test fail. Audiobook suite `322 passed,
1 failed`; Voicebox `762 passed, 4 skipped, 1 failed`. Both failures remain the documented
pre-existing ones. `ruff check` and `ruff format --check` pass on every touched Voicebox file.

macOS Low Power Mode is now **off for AC power** (`pmset -g live` reports `lowpowermode 0` while
drawing from AC). The 1.5602x route measurement was taken with it on, so it remains a valid
internally-controlled comparison; absolute throughput should be better than that benchmark implied.

The audiobook workspace at `ebook/audiobook/voice-profile/build` is now a git repository. It had
never been under version control — its only safety net was the `<name>.bak-<reason>` convention,
which is why several generations of `.bak-*` snapshots sit beside the sources. Since ~684 MB of the
directory is generated research output, `.gitignore` ignores everything by default and names what
to keep: 356 files, 4.4 MB, all code. The `.bak-*` snapshots stay on disk but out of history.

---

# Fabián fine-tuning run

## Contract

Create one dedicated Spanish narrator model from the user-specified recording
and matching EPUB of *Los engranajes de Occidente*. Prepare accurate, bounded
audio/text examples, train Qwen3-TTS 1.7B, evaluate unseen chapters and new prose,
and expose the successful model through one Voicebox profile. Model weights,
source text, audio and experiment manifests stay in the ignored
`voice-profile/finetuning/` directory. Existing voices and audiobook jobs remain
the baseline until the candidate passes quality and warm generation-time checks.

Completion requires real training, a loadable checkpoint, intelligible sample
generation and a comparison with the current voice. A prepared dataset or an
unverified checkpoint is not a completed voice.

## Initial evidence and decisions

- Repository baseline: `8346274`; the main checkout was clean. The private
  `voice-profile/build` repository has 58 existing changes; do not include them
  in this work.
- Runtime: macOS 26.6.2, Apple M2 Max, 64 GiB unified memory. Metal and PyTorch
  MPS backward execution were verified outside the sandbox. No CUDA GPU.
- Free disk at inspection: approximately 23 GiB. Do not delete existing user
  files. Reuse available source weights and stream/chapter-bound audio reads.
- Audio: 55,253.252 seconds, 44.1 kHz stereo AAC. Chapters 0 and 39 are credits
  by another speaker and are excluded. Narration chapters are 1–38.
- Existing Whisper word timestamps cover all narration chapters. They are
  alignment candidates, not trusted training labels; compare against the actual
  supplied EPUB and reject disagreements and uncertain boundaries.
- Split by whole chapter before training. Keep validation and final test
  material separate from training and reference conditioning.
- Prefer checkpoint selection against held-out data over an arbitrary maximum
  number of epochs. More training can overfit and does not guarantee improvement.
- The official example assumes Flash Attention/CUDA. Its target indexing must
  be checked against the installed Transformers loss and generation contract;
  double label shifts must not enter this run.
- Existing archived BF16 MLX base weights and tokenizer are available locally.
  No paid compute or cloud upload has been authorized or used.

## Verification baseline

`backend/venv/bin/python -m pytest backend/tests/test_mlx_qwen_optimizations.py
backend/tests/test_mlx_qwen_dtype_backport.py backend/tests/test_qwen_download.py
-q`: **83 passed** with native Metal access. The initial sandbox run had five
Metal-access failures; the native rerun passed all of them.

## Current status

Completed and installed as **Fabián — fine-tuned**. The selected update-926
checkpoint passed all final held-out quality/speed gates. The real Voicebox API
generated repeatable, independently verified Spanish speech with this profile.
Existing voices and recordings are unchanged. See the final recovery handoff
below for verification and the remaining subjective/UI-review limitations.

## Continuation after the restart (2026-09-06)

- Recovery commit: `8affd3c`, clean `feat/fabian-finetuning` branch. The recovered
  corpus contains 4,750 candidates and codec caches. Independent clip verification,
  the durable trainer, export, evaluation and profile installation still remain.
- This is the native macOS checkout, not an AppTec appliance/container. AC power,
  64 GiB memory and approximately 83 GiB free disk were confirmed. The old command
  attempted a batch of eight long clips; that does not establish the restart cause.
  Continue with one clip per GPU step, gradient accumulation and an explicit MPS
  allocation cap. Do not run competing large GPU workloads.
- Baseline: native `pytest backend/tests scripts/finetune_qwen -q` passed
  **895 tests, 4 skipped**. The sandbox run aborted at Metal initialization.
  `bun run ci` passed (26 frontend tests, boundary lint, TypeScript, both builds).
  Whole-backend Ruff has 380 existing lint findings; formatting also has existing
  debt. Use the repository's 120-column Python configuration for changed code.
- Reuse the archived BF16 base at commit
  `a6eb4f68e4b056f1215157bb696209bc82a6db48` and local Whisper medium at
  `7fc08c4eac4c316526498f147dfdee6f6303f975`. Both are under the old
  `.cache/_to_delete/huggingface/hub` cache; verify contents and bind checkpoints
  to hashes. Source assets are private and remain ignored.
- Work batches: (1) verify actual exported audio and validate cached codecs,
  (2) implement/test atomic training resume and checkpoint selection, then train,
  (3) merge adapters into a standalone BF16 CustomVoice checkpoint and evaluate
  held-out material/new prose against the unchanged baseline, (4) expose a passing
  model through one dedicated Voicebox profile and exercise real generation.
- Select the fixed speaker reference from verified **training** material, so
  validation/test chapters cannot leak through reference conditioning. Evaluate
  validation throughout training; reserve final test examples for final evaluation.

### Durable training implementation

- Added an audited verified-corpus loader, bounded single-clip gradient
  accumulation, deterministic epoch order, optimizer/RNG resume, validation-based
  selection and early stopping. Resume and best weights each use two alternating
  slots with an atomic hash-checked pointer; no base weights are duplicated in
  optimizer checkpoints. Changed data, codecs, runtime or controlling code reject
  exact resume instead of silently continuing a different experiment.
- Added BF16 adapter merging and standalone CustomVoice export. The fixed speaker
  is stored at codec embedding 3000, following Qwen's checkpoint layout. The export
  must improve validation loss before it is emitted; generation quality is a
  separate required gate.
- The installed MLX CustomVoice path interleaves text/audio, while this trainer
  uses the official non-streaming prefix. Integration must explicitly match that
  prefix on the loaded local model. Using the unmodified MLX CustomVoice path
  would create a train/inference mismatch.
- Focused verification now covers crash-before-pointer publication, corrupted
  checkpoints, exact optimizer continuation, split leakage, stale verification,
  invalid codecs, and BF16 merging: 27 tests passed before integration work.

### Local profile and evaluation integration

- Kept the existing CustomVoice engine/profile schema. Operator-installed
  `finetuned:` identifiers bind immutable, hash-checked local checkpoints; HTTP
  callers cannot supply arbitrary filesystem model paths. Generation, synchronous
  speech and streaming bind the local backend before any built-in model download.
- Patched only the loaded local model's input preparation. An executed test with
  identical tiny Torch/MLX weights confirmed the complete non-streaming prefix
  matches the training tensor numerically. Existing model instances are unaffected.
- The frontend restricts this profile to Spanish/1.7B, omits unsupported delivery
  instructions and bypasses the built-in speaker download prompt. Native model
  inference stays serialized by Voicebox's existing accelerator lifecycle guard.
- Added reproducible paired evaluation: held-out passages and new prose,
  independent generated-audio transcription, frozen base speaker-encoder cosine,
  and warm processing/audio timing. Baseline reference reconstruction is tested
  byte-for-byte against Voicebox's actual normalization and PCM WAV save path.
  Quality/speed thresholds are fixed in advance, not adjusted after seeing scores.
- Installation requires a complete passing final-test report for the exact export,
  recomputes acceptance gates, verifies all artifact hashes and preserves every
  existing profile. Repeated installation returns the same new profile.
- Focused evaluation/installation tests: **26 passed**. Frontend `bun run ci`
  passed again (26 tests, boundary lint, TypeScript and web/desktop builds).
  Existing bundle-size/dynamic-import warnings remain; no checks were weakened.
- The first full integration regression run found two expected stale source-
  fingerprint failures (**936 passed, 4 skipped**). Added the new profile binding
  and local model lifecycle owners to the fingerprint's explicit source coverage,
  then refreshed the computed AST hash. The exact-generation revision changes;
  frozen old jobs are not silently relabeled or allowed to mix runtime revisions.
- After the fingerprint refresh, the complete native Python regression suite
  passed: **938 passed, 4 skipped**. All changed Python files pass repository-
  configured Ruff lint and format checks; changed frontend files pass Biome.
  Real training and generated-audio acceptance are still outstanding at this point.
- Final pre-training review expanded exact-resume code identity to include
  `data.py` and `checkpoint.py`, which also control tensors/state. An added test
  proves that changing the loader changes the experiment identity. No run had
  started under the earlier two-file identity, so no training was invalidated.
- Added a distinct original passage of more than 150 words to both validation
  and final-test generation. This checks completion beyond the 3–18 second
  training-clip range; no generated-audio scores had been seen when adding it.

### Verified corpus and pilot start

- Independent Whisper verification completed all **4,750** exported clips.
  Accepted: **3,704 train / 8.6239 h**, **227 validation / 0.5345 h**,
  **346 test / 0.8581 h**. The 473 transcript disagreements remain excluded.
- Starting `voice-profile/finetuning/run-v1` from the complete audited corpus,
  rank 64, learning rate 0.00002, accumulation 4, one clip in GPU memory, MPS
  fraction 0.45. First pause at one optimizer update to exercise real checkpoint
  recovery, then continue to a 100-update pilot and validation audio evaluation.
  Independent verification has exited; no competing GPU process was started.
- The one-update MPS pilot completed and paused cleanly. Baseline held-out loss
  **2.78724 → 2.75423**; allocated tensors **5.23 GiB**, driver allocation
  **10.31 GiB**. Read-back verified update 1 / position 4, all 462 optimizer
  parameter states and both CPU/MPS RNG states. Resuming that same run to update
  100 for the first real-audio validation comparison.
- The pinned Transformers 4.57.3 emits a generic Mistral-regex warning for this
  local Qwen config. Source inspection shows its non-Mistral skip is mistakenly
  limited to configs at or below 4.57.2; this config is `qwen3_tts`/4.57.3 and uses
  `Qwen2Tokenizer`. Neither training nor MLX inference applies that Mistral patch.
  Do not change Qwen tokenization to silence it. Optional SoX/Flash-Attention
  import warnings are not failures; the exercised local path uses SoundFile/mel
  processing and MPS/SDPA successfully.
- The first real resume exposed a defect that CPU optimizer tests had not
  covered: installing FP32 adapters makes Hugging Face's generic `model.dtype`
  report float32 (its first parameter), while the frozen embeddings remain BF16.
  Resume converted the saved BF16 speaker to that incorrect generic dtype,
  promoting the prefix and triggering a native mixed-dtype Metal assertion.
  The process aborted; the laptop and durable update-1 checkpoint were intact.
- Reproduced the dtype change on the actual base without GPU work. Added a
  BF16/FP32 mixed-model regression, restore against the actual audio-embedding
  weight dtype, and an early descriptive rejection for mismatched prefixes.
  Because controlling code changed, preserve `run-v1` unchanged and restart
  the one-update/recovery pilot in **`run-v2`**, not by editing its contract.
- `run-v2` reproduced both the original baseline and first-update validation
  losses exactly. Its real native resume then passed updates 5 and 10 without
  the Metal assertion, at about 4 seconds/update and 10–11 GiB driver allocation.
  Continuing to update 100 before the first real-audio comparison.
- The resumed pilot reached update **100 / 400 training clips** and paused
  successfully. Validation: total **2.39871**, main **1.06874**, residual
  **4.43324**, versus baseline 2.78724 / 1.36157 / 4.75222. Resume training took
  about 455 seconds plus final validation/checkpoint writes. Proceeding to
  `fabian-pilot` export and validation-only audio evaluation; final test remains
  untouched for model selection.
- The standalone `fabian-pilot` model loaded and generated all 16 validation
  cases, including 57.68 seconds of continuous original prose. Independent
  Whisper evaluation passed all predeclared gates: candidate WER **1.149%**
  versus baseline **0.575%**; worst candidate WER **14.286%** on one held-out passage;
  mean base-encoder speaker cosine **0.98845 vs 0.98495**; median warm processing
  ratio **0.57578 vs 0.61766**; no token-limit hits. These are automated metrics,
  not a human listening review or a claim of improved perceived voice quality.
- Continuing `run-v2` from update 100 with its original three-epoch ceiling and
  validation-patience rule. The 100-update export and comparison are preserved.
  The final test corpus has not been used to tune or choose this candidate.
- Detailed inspection of the pilot's independent transcript found that all six
  candidate word errors came from an omitted ending in `ch25_00009` (42 words).
  Passing the five aggregate gates was therefore insufficient for promotion.
  Added a **stricter sixth gate**, requiring the final three normalized words of
  every prompt in its actual transcript. This catches premature EOS separately
  from a maximum-token truncation. Preserve the original pilot report as evidence;
  it is not installable under the strengthened acceptance rules. Re-evaluate the
  final trained candidate on validation, including this passage, before final test.
- Subsequent validation improved to **2.36212 at update 200** and **2.33527 at
  update 300**. The long worker gradually slowed from about 4–5 seconds/update
  to about 10–14, while allocations stayed bounded and macOS reported no recorded
  thermal/performance warning or competing model worker. About 1.3 GiB swap and
  68 GiB free disk were observed; power remained AC.
- Requested a graceful SIGTERM only for the owned worker. It finished and saved
  **update 387 / position 1548**, then exited cleanly. Restarting the exact same
  `run-v2` command tests whether releasing accumulated native process state
  restores throughput; no hyperparameters, data, weights or contract were edited.
- The update-387 checkpoint resumed successfully; initial throughput improved
  to around 7–8 seconds/update, then varied around 8–10. Validation improved to
  **2.32822 at update 400**. A read-only Darwin scheduling check returned normal
  priority (0), so no process priorities or machine power settings were changed.
- Full native regression after the resume and ending fixes: **943 passed,
  4 skipped**, in 148 seconds while training was active. Skips are two Python
  3.13-only audio compatibility cases (this environment is Python 3.12), a Windows
  ROCm E2E case, and an opt-in heavy ROCm install. All 29 touched Python files
  passed the repository-configured lint and formatting checks.
- Validation improved again to **2.31621 at update 500**. The next full native
  regression, run concurrently with training and frontend builds, completed
  **942 passed / 1 failed / 4 skipped**: the unchanged
  `test_nonpositive_chunk_size_fails_promptly` exceeded its 10-second subprocess
  limit. An isolated pytest retry during training also timed out. A direct timed
  import subsequently took **6.63 seconds**, then the invalid-size call correctly
  raised `ValueError`. No test limit or assertion was changed; rerun the full
  checks without competing training/build work before final acceptance.
- The repeated frontend CI passed all 26 tests, boundary/type checks and both
  builds. Ruff lint/format passed all **28** changed Python files from the
  pre-fine-tuning base; the earlier 29-file note was a counting error. Diff
  whitespace checks and the scan for newly skipped/disabled tests, suppression
  markers and unfinished placeholders are clean.
- Training completed its first full epoch at **update 926**. Selected validation
  loss progressed through **2.30770 (600), 2.30365 (700), 2.29393 (800),
  2.28352 (900), 2.28145 (926)**. The original three-epoch ceiling and patience
  rule remain unchanged. Throughput recovered to roughly 5–7 seconds/update
  without another restart or changes to machine settings; allocations remain
  bounded. Final audio evaluation and installation are still pending.
- The full training run **finished normally at update 1,200**, exit 0. Three
  consecutive validation checks did not improve the selected loss by 0.001:
  **2.28322 (1,000), 2.28552 (1,100), 2.28149 (1,200)**. The immutable selected
  checkpoint is therefore **update 926 / loss 2.28144715**, versus baseline
  2.78724186. No ceiling, patience, hyperparameter or acceptance gate was relaxed.
  Exported this selected checkpoint as private standalone **`fabian-v1`**.
- The selected model's full validation-audio check **failed**; it is not installed.
  Candidate WER **4.215%**, worst **26.190%**, versus baseline **0.575%**. All 22
  candidate word errors were suffix omissions in `ch08_00069` and `ch25_00009`;
  the other 14 passages, including 56.08 seconds of original prose, transcribed
  correctly. Speaker cosine and measured warm speed passed; final test remains
  unused. Preserve `evaluation-trained-validation` unchanged as failed evidence.
- Testing a bounded validation-only decoding probe at temperatures **0.7 and
  0.0**, using the same two failed texts/seeds and long validation prose. Greedy
  decoding (0.0) hit every duration limit and is rejected. The installed MLX
  sampler restores EOS after top-k filtering; a separate observational probe
  will check the actual failed EOS ranks before attributing the defect to that
  behavior. No weights, training contract, acceptance limits or tests were weakened.
- The observational decoder probe **disconfirmed** the EOS-filter explanation:
  EOS was already rank 1 at both failed stops. Temperature 0.7 still omitted an
  ending; temperature 0.0 generated unintelligible duration-limited output.
  Neither decoding alternative is promoted.
- A separate probe reused Voicebox's existing clause-aware splitter at **200
  characters**, original temperature 0.9 and deterministic offset seeds. Both
  failed endings and the long prose completed; the three whole WAVs had WER
  **4.545%, 2.381%, 0.578%**, with no duration-limit hits. Integrated this bounded
  policy only for local fine-tuned profiles through the shared generation path;
  ordinary engines retain their previous limits and callers may request smaller
  chunks. Unbounded direct backend calls now reject before inference.
- New tests reproduced the missing cap before the fix, then passed. Focused
  regression is **53 passed**. The evaluator now exercises the real bounded
  backend and disk-backed chunk assembly, propagates duration-limit errors,
  and reports no fabricated aggregate token count. The installer additionally
  binds reports to evaluator and inference source hashes. Refreshed the actual
  numerical source fingerprint; old frozen jobs are not relabeled.
- With training/inference stopped, two consecutive complete native regression
  passes succeeded: **954 passed / 4 unchanged skips**, in **37.71 s** and
  **30.11 s**. The earlier unchanged import-timeout test passed both times;
  no timeout or assertion was changed. Ruff lint/format pass all **30** Python
  files changed since the pre-fine-tuning base.
- Full bounded validation exposed an MLX worker-thread error at warmup, before
  producing any evaluated candidate audio. The loaded model retained lazy arrays
  on a thread-local stream unavailable to `asyncio.to_thread`. A tiny native
  regression reproduced the same failure, independently of the large model.
  The local loader now creates a cross-thread stream; inference uses that same
  stream under the existing global lifecycle guard and synchronizes before
  releasing ownership, including on failure. This follows the installed MLX
  API's explicit requirement for serialized evaluation; no packages or compiler
  settings were changed. The test also retains lazy state between separate
  executor lifetimes. Focused regression: **35 passed**. Preserve the failed
  `evaluation-bounded-validation` directory and use a new identity-bound
  `evaluation-threaded-validation` run with unchanged weights and gates.
- Stream-fix verification: two consecutive full native passes **955 passed /
  4 unchanged skips**, **30.46 s / 27.14 s**; all touched Python lint/format
  checks pass. Frontend CI again passed **26 tests**, boundary checks, all type
  checks and both builds. Existing bundle-size warnings remain informational.
- The repaired full-model worker reached inference but hit a duration limit on
  `ch08_00024`. A second tiny native test isolated the cause: MLX 0.32's RNG
  state is thread-local, while mlx-lm's compiled categorical sampler captures
  the importing thread's list. In another worker it repeated the same draw
  **32 times** instead of advancing the requested seed. The local model now
  binds a private copy of Qwen's sampling method to a compiled categorical
  function with explicit request-local keys. The key splitting sequence matches
  upstream's valid single-thread implementation exactly. No global method,
  filtering rule, temperature, duration limit, test threshold or weights changed.
  The new regression failed before the fix, then exact token-array parity
  passed across two executor lifetimes; a different seed produces different
  output. Focused regression is **36 passed**. Preserve the failed threaded
  output and re-evaluate under a fresh source identity.
- Seed-fix full regression: **956 passed / 4 unchanged skips** twice, in
  **31.82 s / 29.57 s**. Native sampler tests additionally verify the upstream
  method and its global categorical function were not modified. Touched Python
  lint/format and diff checks pass; frontend sources are unchanged since the
  immediately preceding successful complete frontend CI run.
- Full repaired-path validation **passed all six gates** on 16 paired passages
  and their 32 independent Whisper transcripts. Candidate WER **0.766%** (worst
  **4.545%**) versus baseline **0.575%**; speaker cosine **0.98892 / 0.98495**;
  median warm processing/audio ratio **0.44257 / 0.57958**. No missing endings or
  duration-limit hits. The 59.35-second new-prose sample completed. These are
  automated metrics, not a listening review. Proceed to the previously untouched
  final test selection with exactly these weights, code, seeds and settings.
- The untouched final test comparison **passed all six gates** on chapters 12
  and 30 plus four new prose passages (16 paired cases / 32 WAVs). Candidate
  WER **0.734%**, worst **2.151%**, versus baseline **0.550%**; speaker cosine
  **0.98896 / 0.98401**; median warm processing/audio ratio **0.45146 / 0.59557**
  (about 24% lower). No missing endings or duration-limit hits; the longest new
  prose generated **62.42 s**. No final-test feedback was used to change the
  checkpoint, runtime, settings or acceptance gates.
- Before installation, saved a private SQLite backup and checked `quick_check`
  (`ok`). Recorded canonical row hashes for all **15 existing profiles** and
  **32 sample records**, plus the ordered sample-file hashes, in private
  `before-install.json`. Installation must add one new profile and preserve all
  those original rows and recordings.

### Final recovery handoff

- Installed profile **`ecf402ed-3f23-4326-8651-035d1d5f54dc`**, preset
  **`finetuned:fabian`**, name **Fabián — fine-tuned**, bound to the unchanged
  private **`fabian-v1`** export. The installer independently revalidated the
  complete final report and artifact hashes before registration.
- The database now has **16 profiles / 32 samples**, and `quick_check` is `ok`.
  All 15 original profile rows, all 32 sample rows and all 32 audio files match
  their pre-installation checksums exactly. The private database backup remains
  available. Normal backend startup reclaimed one completed/orphaned temporary
  exact-voice snapshot; original recordings and saved audiobook jobs were not
  changed by this work.
- Restored the source API at **http://127.0.0.1:17493** and the browser interface
  at **http://127.0.0.1:5173**. Health, profile lookup and HTML serving returned
  successful responses. Browser automation could not connect: no browser was
  available, native UI startup failed, and the in-app browser was unavailable.
  Do not claim that the browser controls were clicked or visually reviewed.
- Exercised real `/generate` twice with the installed profile, new 304-character
  Spanish prose, seed **314159**, and normal output processing. Both requests
  completed two bounded chunks, returned **20.99 s / 24 kHz / mono** audio,
  and produced **byte-identical WAVs**. Independent Whisper transcription was
  **55/55 words correct**, including the complete ending. The sample and its
  verification report are private `app-smoke.wav` and
  `app-smoke-verification.json`; both entries remain in app history.
- This is automated speech, identity-proxy, timing and runtime verification,
  not a human listening review or a promise of better subjective voice quality.
  The 200-character inference windows are a deliberate validation-backed
  reliability choice. No cloud compute, uploads, pushes or publication occurred.
- Preserve the completed training checkpoints, immutable export, failed
  diagnostic reports, successful validation/final-test audio and the database
  backup. Only the three disposable sampling-probe scripts created under this
  task's `/tmp` scratch directory are removed; their private result artifacts
  remain available.
- Final two consecutive native regressions: **956 passed / 4 unchanged skips**
  in **29.37 s / 27.33 s**. Final frontend CI: **26 passed**, boundary/type
  checks and web/desktop frontend builds passed; only the previously recorded
  bundle warnings remain. Ruff lint/format pass all **30** changed Python files;
  Biome passes all **4** changed frontend files; full diff whitespace and
  newly-added placeholder/suppression/skip scans are clean. Platform/opt-in skips
  remain the two Python 3.13 checks, Windows ROCm E2E and heavy ROCm install.
- No training, evaluation or ASR worker remains active. The source app remains
  listening only on loopback, API **17493** / UI **5173**, with the new voice
  verified warm. Work is saved in local commits on **`feat/fabian-finetuning`**;
  no push or release was performed. The completion record follows the
  fully-implement workflow's executed validation and durable local-commit gates.

## Sísifo continuation — 2026-09-06

### Contract and batches

- Continue the completed Fabián model using the user's 30 M4B recordings of
  *El despertar de Sísifo (2021–2022)*, without assuming every voice is his.
  Preserve source audio, the installed `fabian-v1`, its checkpoints, existing
  profiles, and the unrelated dirty private `voice-profile/build` repository.
- In scope: source-bound episode preparation, conservative announcer exclusion,
  reference-based speaker filtering with explicit negative controls, independent
  verification of actual cut WAVs, a new parent-checkpoint training experiment,
  and paired evaluation against v1 before any separate profile installation.
  No cloud compute, audio uploads, paid services, pushes or publication.
- Existing large-v3-turbo word timestamps are useful hypotheses, not verified
  labels: they reference an older location and were not bound to these source
  bytes. Match Unicode-normalized filenames, verify durations, hash all inputs,
  and require independent Whisper-medium word agreement. Do not describe this
  corpus as EPUB-aligned; there is no source EPUB for these episodes.
- Reserve whole episodes 6 and 24 for validation, 11 and 28 for final test,
  before speaker reference selection or training. Keep original book splits
  unchanged. Use verified original training material for rehearsal and retain
  the parent speaker embedding. New final prose must not reuse v1's final set.
- Batches: (1) verified speaker-aware corpus preparation, (2) explicit immutable
  parent warm-start and paired local-model evaluation, (3) bounded native
  training, evaluation and conditional separate installation, (4) two complete
  regression passes and durable local commits. An improved validation loss is
  not sufficient for promotion; keep the six existing speech-quality gates.
- Native macOS runtime reconfirmed. Baseline: **956 passed / 4 existing skips**,
  frontend **26 passed**, boundary/type checks and both frontend builds pass;
  only existing bundle warnings. Fine-tuning Ruff lint passes. Disk has 36 GiB
  free; GPU jobs must run sequentially with the existing allocation cap.

### Speaker-aware preparation

- Reused the local frozen Qwen ECAPA encoder without loading the talker or
  adding dependencies. The CPU-only real-audio probe took 15.5 seconds.
  Across the first three episodes, all 12 announcer controls had negative
  target-versus-announcer margins; narrator checks were positive. Existing
  narrator references passed leave-one-recording-out margin checks (minimum
  0.02055). Preserve the private probe report; these are identity proxies.
- Fixed conservative cutoffs **target cosine >=0.97**, **margin >=0.015** on
  both the complete clip and every overlapping 3-second / 1.5-second-stride
  window. The absolute cutoff intentionally rejects some real narrator clips.
  Only training episodes provide references or calibration controls; no final
  test feedback sets thresholds. Constant source-level headroom avoids AAC
  floating-point overshoots; no denoising, compression or pitch changes.
- All 30 current M4B durations exactly match their timestamp metadata. Source
  and transcript hashes now bind those hypotheses to this folder. The new
  `sisifo-corpus-v1` preparation is restartable per candidate/episode, and keeps
  rejected-speaker metadata. First seven episodes retain **247 / 521** bounded
  candidates; low-confidence ASR and quiet-boundary checks have already excluded
  other material before this count. These are not yet independently verified
  training examples.
- Preparation completed in **365.3 seconds** across all 30 episodes: **1,333
  training clips / 2.67904 h**, **74 validation / 0.14990 h**, **73 test /
  0.14187 h**. Another **968** bounded candidates failed speaker screening and
  **2** crossed wrapper boundaries. Exact word verification is running on the
  1,480 exported WAVs; no unverified label has entered training.
- Preparation batch saved as local commit **`bb9beb5`**. Complete native
  regression: **974 passed / 4 unchanged skips**; Ruff lint/format and diff
  whitespace checks pass. No frontend source changes since its green CI run.
- Continuation design: inherit v1's selected update **926** adapters and fixed
  speaker into a **new** experiment, with a fresh AdamW optimizer and new
  held-out baseline. Validate parent base/rank/hash/contract/completion and
  retain the exact original reference. Source-qualified validation sampling
  remains balanced, including when chapter numbers coincide. Compose only
  already-verified clips; original and new held-out groups stay disjoint.
- Warm-start/rehearsal batch saved as **`eff996c`**, with **980 passed / 4
  unchanged skips**. The original reference wins over identically numbered
  new episodes. Source preparation, checkpoint recovery and current v1 files
  remain unchanged.
- Actual read-only baseline resolution exposed a wrong table name in the new
  evaluator (`voice_profiles` versus the real ORM's `profiles`). Replaced the
  hand-written test schema with `VoiceProfile.__table__`; it reproduced the
  failure before fixing the query. This is why unit-green alone is not used as
  completion evidence. No database write was made by the failed lookup.
- Predeclared separate validation/final evaluation configs before training:
  four fresh Spanish passages per split, including **198 / 202 word** long
  narration. Baseline profile is the installed **Fabián — fine-tuned** v1,
  not the older zero-shot clone. Use the identical bounded runtime and seeds
  for both; keep the six original acceptance gates unchanged.
- Fixed evaluator verification: **982 passed / 4 unchanged skips**, including
  the ORM-backed profile lookup regression; Ruff lint/format and diff checks
  pass. A real read-only lookup and full model-file hash validation resolve
  `finetuned:fabian` to the unchanged `fabian-v1` export.
- Real-audio mixed-speaker challenge: a clean narrator clip passed; inserting
  **1.5 seconds** or **3 seconds** of the announcer caused rejection. In the
  1.5-second case the full-clip score alone still passed, but an overlapping
  window's margin fell to **0.00457**, below the fixed 0.015 threshold. This
  supports the window rule; it is not a guarantee of perfect diarization.
- Evaluation changes saved as local commit **`959a811`**. The complete actual
  1,480-WAV audit passed: finite mono 24 kHz, exact durations, source-bound
  text/audio hashes, wrapper exclusions and every speaker window rechecked;
  maximum peak **0.90000004** (floating-point roundoff). All 30 source M4Bs
  and timestamp JSONs remain byte-identical to the preparation inventory.
- Independent Whisper-medium verification finished: **1,169 train / 2.32649 h**,
  **67 validation / 0.13474 h**, **61 test / 0.12056 h**. **183 / 1,480** clips
  were rejected for non-exact normalized word agreement. Their metadata remains
  available; do not promote them by editing labels or acceptance flags.
- Codec preparation is running sequentially after ASR. Next compose
  `sisifo-mixed-v1` with all accepted new clips plus **600** deterministic
  original training clips (and original held-outs), preserving `ch04_00077`.
  Planned new run: `sisifo-run-v1`, original base, `--init-run run-v2`, rank 64,
  learning rate **1e-5**, accumulation 4, at most 3 epochs, validation every
  100 updates / 64 source-balanced clips, patience 3, MPS fraction 0.45.
  No parent optimizer or old validation counters are inherited.
- Codec extraction completed in **about 52 seconds**. Real composition passed the
  complete audio/text/speaker/codec audit: **1,769 train / 3.750275 h**,
  **294 validation / 0.669269 h**, **407 test / 0.978629 h**; corpus identity
  `1588ea6f34333a6c8425e4bf41c4b28c2109fe4f907e93596f853bde8fa3ad1a`.
  All 26 training episodes contribute independently verified clips.
- Started the full native continuation at **12:29 UTC**, shell session **40443**,
  with the planned command and no artificial update limit. Parent update 926
  loaded successfully. Its new mixed-validation baseline is **2.43565202**;
  first updates are finite, approximately 4–7 seconds each, with **~5.0 GiB**
  allocated / **~9.8 GiB** driver memory after warm-up. Current disk free is
  **33 GiB**. The first crash-safe baseline checkpoint is committed on disk.
- Before the first rotated recovery save, compared the new step-0 snapshot
  directly with the selected parent: **all 462 adapter tensors and the speaker
  match exactly**, and AdamW's state is empty. Training subsequently reached
  update **50** with finite gradients/losses and normal checkpoint rotation.
  This is an in-progress run, not a completed or promoted voice.
- After training genuinely completes, export its selected checkpoint to the
  new private `fabian-sisifo-v2` directory, run the predeclared
  `sisifo-evaluation-validation-config.json`, then the final test config only
  after validation passes. Compare to installed v1, do not weaken gates or use
  final-test feedback to select the checkpoint. Only a passing candidate may
  be installed under a separate voice identifier. Finish with real application
  generation and two complete regression passes; keep all prior artifacts.
- The first complete training pass finished at update **443**, covering all
  **1,769** clips. Validation losses: start **2.43565202**, update 100
  **2.39981806**, 200 **2.38838900**, 300 **2.38080665**, 400 **2.37874837**,
  first-pass end **2.37812390**. Update **400** remains selected: the last
  reduction is below the fixed 0.001 selection threshold (one stale check).
  Training continues under the unchanged patience/epoch limits. Selected
  checkpoint hash verification passed; disk free remains approximately 31 GiB.
- Training **completed normally at update 600**, after three checks without
  meaningful improvement (443, 500, 600). Final selected snapshot is update
  **400**, loss **2.37874837** versus inherited-model baseline **2.43565202**
  on the new mixed validation set (about **2.34%** lower). Final update 600
  scored 2.38114139 and was not selected. No gates, seed, learning rate,
  patience or dataset were changed after starting the experiment.
- Exported the selected snapshot successfully to the new standalone private
  **`fabian-sisifo-v2`** BF16 model. Parent `fabian-v1`, original run-v2 and
  all new recovery checkpoints are preserved. Paired validation generation
  against installed v1 is starting; the new candidate is **not yet installed**.
- Update-400 paired validation **failed complete endings**, while the other
  five gates passed. Candidate WER **0.860%**, worst **9.375%**, cosine
  **0.983637**, median RTF **0.406637**; v1 WER **0.491%**, cosine **0.988573**,
  RTF **0.407595**. The 197-character `sisifo_ch06_00084` dropped/replaced its
  final phrase. Both cached Whisper-medium and Whisper-small failed to recover
  the ending on full audio and isolated tails. Preserve this failed export,
  report and diagnostic; no installation or final-test inference occurred.
- Next validation-stage candidate: retained earlier best **update 300**
  (loss **2.38080665**), exported separately as `fabian-sisifo-v2-step300`.
  Added a tested explicit retained-step export option: require a completed run,
  match its contract and selected step, hash the snapshot, never rewrite either
  checkpoint pointer. Reject nonfinite validation metrics. New configs preserve
  every validation/final passage, seed, baseline and gate; only candidate/output
  paths differ. This is audio-based validation selection, not final-test tuning.
- Update-300 greedy-ASR validation initially failed with **15.23%** aggregate
  WER: one 8.56-second clip produced over 100 repeated words and timestamps
  reaching 14.38 seconds (compression ratio **5.912**). Independent small-model
  full/tail decoding and medium-model tail decoding all recover the correct
  ending; full medium decoding without timestamps does too. In contrast, the
  real update-400 missing ending remains wrong in timestamp-free decoding.
- Fixed the measurement failure, not the quality gates: empty or repetitive
  ASR (Whisper's existing **2.4** compression cutoff, including nonfinite
  values) gets exactly one greedy timestamp-free retry on the same full audio,
  without expected-text hints. Preserve both attempts and block unresolved
  failures. Wrong words/endings never trigger retries. Tests were added before
  implementation and first failed on the missing helper. New comparisons use
  `sisifo-step300-asr2-*`; prior failed reports remain unchanged. Model, speech
  inference, passages, seeds, six thresholds and the untouched final set stay
  fixed. Code-bound cache checks require fresh paired generation after this
  evaluator change.
- ASR safeguard batch verified: **994 passed / 4 unchanged skips**, frontend
  **26 passed**, boundary/type checks and web/desktop frontend builds pass;
  Ruff lint/format and diff checks pass. This includes preserved failed-ASR
  evidence, stale policy-cache rejection and unchanged ending rejection.
- Fresh step-300 validation passes **all six unchanged gates**: candidate
  WER **0.2457%**, worst **4.545%**, mean cosine **0.986388**, median RTF
  **0.405481**, no missing endings or token-limit hits. Baseline v1: WER
  **0.4914%**, cosine **0.988573**, RTF **0.406873**. All 28 regenerated
  candidate waveforms are sample-identical to the original step-300 run;
  FLOAT WAV file hashes differ only because libsndfile stamps the PEAK
  chunk's creation time. Starting the predeclared final test now, with this
  checkpoint, inference implementation, ASR policy and all settings frozen.
- Initial step-300 final report: five gates pass, ending gate fails on
  `sisifo_ch11_00025`. Medium ASR adds "de la humanidad" after a 6.4-second
  utterance; aggregate WER **0.4848%** versus v1 **0.7273%**, cosine
  **0.987517** versus **0.988882**, RTF **0.408204** versus **0.408020**.
  No model change, installation or test-driven checkpoint reselection occurred.
- Further measurement evidence: small ASR reads the exact full sentence and
  tail. Medium ASR alternates invented "de la vida"/"de la humanidad" endings;
  its cropped-tail output assigns the extra phrase to seconds **4–8** of only
  **3.9 seconds** of input. Downloaded the 1.61 GB public MLX large-v3-turbo
  verifier locally, pinned revision `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb`;
  it independently reads the exact requested ending. It also reproduces the
  genuinely wrong update-400 ending, supporting the distinction. No audio upload.
- Before running complete additional comparisons, require **both** small and
  large verifiers to pass all six unchanged gates on the entire validation
  **and** final sets. Keep the initial medium reports and diagnostics unchanged;
  do not substitute just one favorable transcript. Added identity-bound
  `crosscheck.py` to rescore all existing paired WAVs using the same scorer,
  preserve the original, match the upstream weights-file selection, and verify
  original/recognizer/code provenance again at installation. Failure of either
  independent full comparison prevents installation. The generation model,
  exact audio, prompts, seeds and six gate definitions remain unchanged.
- Cross-check tooling verification: **998 passed / 4 unchanged skips**;
  frontend 26 tests, boundary/type checks and both frontend builds pass. Ruff
  lint/format and diff checks pass. Starting all four independent comparisons.
- All four cross-checks completed; the added acceptance requirement **did not
  pass**. Small: validation WER **0.4914%**, one ending disagreement; test
  **0.9697%**, two ending disagreements. Large: validation **0.2457%**, all
  gates pass; test **0.3636%**, one ending/25%-worst-WER failure caused by
  joining "Res gestae" as "ResGestae". Small disagreements include
  "coetáneo"/"coetanio", "complots"/"con plots", "inundan"/"inunda".
  Both additional models recover the disputed medium-ASR added-phrase ending.
- **No promotion or installation.** Do not weaken gates, swap favorable
  per-clip transcripts, or keep selecting checkpoints against this consumed
  final set. Training and all local comparisons are complete; activation is
  held for a listening review/explicit decision about these automatic failures.
  Prepared the private `voice-profile/finetuning/sisifo-review.md` with exact
  audio links, discrepancies, metrics and the decision needed. New-profile
  application generation is intentionally not run while installation is held.
  All 16 profiles/32 samples and installed v1 remain unchanged. All 30 source
  M4Bs and timestamp files rehash unchanged; full prepared-audio audit and mixed
  corpus identity recheck also pass. The completed run retains its recovery
  snapshots, both exports, all original reports and the local ASR model.
- Final loop-until-dry verification completed twice with **998 passed / 4
  unchanged skips** (28.54 s and 28.12 s), frontend **26 passed**, boundary and
  type checks, web and desktop frontend builds. Fine-tuning Ruff lint/format,
  diff whitespace and explicit no-weakened-tests/no-stubs audits pass. Existing
  deprecation/bundle warnings and the four documented platform/opt-in skips
  remain; no new skips or disabled checks. No human listening assessment,
  new-profile API smoke, or packaged release build is claimed.
- Final read-only checks: all four cross-check provenance chains validate;
  installed v1's complete registration/model hashes equal the evaluation
  baseline; database quick-check is healthy with **16 profiles / 32 samples /
  zero Sísifo v2 profiles**; Voicebox on port **17493** is healthy. All six
  review-page audio links exist. Removed only the nine task-specific temporary
  diagnostic/setup scripts; private reports, samples, model downloads,
  checkpoints and training corpora are retained. No cloud audio upload, paid
  compute, git push, or publishing occurred. Work remains on local branch
  `feat/fabian-finetuning`; **training complete, activation held for review**.
