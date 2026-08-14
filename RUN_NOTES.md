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

- The interrupted `La balada de Soi Cowboy` render still contains 16 of 33 current-contract WAV
  units, totalling 12,073.696 seconds (3 h 21 min 14 s). Three older WAV records are present but
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
  healthy state, and reported `qwen3-mlx-audio-0.4.1-bf16-speaker-v1`; the test process was stopped
  afterward without touching the normal 17494 data/service.
- Voicebox backend regression set: 169 passed, 4 skipped, 1 known flaky progress test deselected;
  the two pre-existing collection/Metal smoke-test blockers were excluded explicitly.
- Focused exact-generation, MLX backport and clone correctness tests pass, including rejection
  before profile/history/task/queue creation on a runtime mismatch.
- External launcher recovery, persistence, renderer/assembler, real FFmpeg and stub profile-import
  suites pass. Shell syntax and Python 3.9 compilation pass.
