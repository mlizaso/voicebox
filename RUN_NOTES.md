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
