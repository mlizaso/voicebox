# Local Qwen fine-tuning

These offline tools prepare and train one Spanish Qwen3-TTS 1.7B narrator.
Keep recordings, EPUBs, corpora, checkpoints and exports in the ignored
`voice-profile/finetuning/` directory. A saved adapter or lower training loss
alone does not establish voice quality.

Use the native macOS backend environment for training; GPU execution needs
Metal access. The private `voice-profile/.venv` contains the independent
`mlx-whisper` verifier. No cloud compute is used.

1. `python -m scripts.finetune_qwen.corpus CONFIG.json` aligns existing word
   timestamps against the actual EPUB and exports bounded, quiet-cut WAVs.
   The configuration assigns whole chapters to validation and final test.
2. `python -m scripts.finetune_qwen.encode CORPUS LOCAL_BASE` saves reusable
   codec targets per clip. `--limit N` bounds newly processed clips.
3. `python -m scripts.finetune_qwen.verify_corpus CORPUS LOCAL_WHISPER` checks
   the exported audio independently, accepting only normalized word agreements.
   Verification is resumable per clip. Rejections are retained for inspection.
4. `python -m scripts.finetune_qwen.train CORPUS LOCAL_BASE RUN_DIRECTORY`
   trains rank-64 adapters with one clip in GPU memory at a time, accumulation
   over four clips, and a fixed reference chosen only from verified training
   material. Validation selects the best snapshot and stops after three
   evaluations without improvement; the default ceiling is three epochs.
5. `python -m scripts.finetune_qwen.export LOCAL_BASE RUN_DIRECTORY EXPORT`
   merges the selected adapters into BF16 weights, embeds the speaker as
   `fabian`, copies the tokenizer, and publishes a complete CustomVoice model.
   Exports never overwrite another model. A checkpoint whose held-out loss
   did not improve over the base is rejected.
6. `python -m scripts.finetune_qwen.evaluate generate EVALUATION_CONFIG.json`
   generates the candidate and unchanged baseline sequentially, with the same
   passages/seeds. Run `python -m scripts.finetune_qwen.evaluate transcribe
   EVALUATION_OUTPUT LOCAL_WHISPER` in the independent verifier environment.
7. After a final **test** evaluation passes, run
   `python -m scripts.finetune_qwen.install EXPORT EVALUATION_OUTPUT/evaluation.json
   --data-dir VOICEBOX_DATA --slug fabian --name "Fabián — fine-tuned"`.
   The installer adds one local preset profile and never replaces another
   registration or existing voice. Repeating an identical installation is safe.

Prefix long commands with `/usr/bin/caffeinate -i` to prevent idle sleep while
they run. Training limits MPS allocations to 45% of the device's recommended
working set; do not disable the allocator's memory limits. The independent
verifier and trainer should run sequentially.

## Resume and inspect

Run the **same training command** to resume. The run contract binds model and
codec bytes, verified audio/text, split membership, training settings, code,
and package versions. An identity mismatch requires a new experiment directory.
`--max-updates N` pauses at an optimizer update for a pilot evaluation; it can
be increased or omitted when resuming. SIGINT/SIGTERM finish the current update
and save before exiting. A forced termination loses at most the interval since
the last committed checkpoint (25 updates by default).

`status.json` records the current phase, update, validation loss and GPU memory.
`metrics.jsonl` contains the progress log. Two alternating `resume-*.pt` files
retain optimizer/RNG states; two `best-*.pt` files retain selected adapters.
Atomic, hash-checked JSON pointers commit each save. Never hand-edit these
pointers or delete a checkpoint while training is running.

The export uses **non-streaming text conditioning**. In PyTorch, pass
`non_streaming_mode=True` to `generate_custom_voice`. The installed mlx-audio
0.4.1 CustomVoice path otherwise uses interleaved text conditioning, so its
input preparation must match the training prefix before evaluating the export.

Evaluate untouched test chapters and new prose, inspect/listen to real samples,
and compare warm generation speed and speaker similarity against the existing
voice before installing a candidate. Do not promote a failed candidate.

## Evaluation and local profile

The evaluation configuration is a private JSON object with absolute paths:

```json
{
  "corpus": "/path/to/corpus",
  "candidate": "/path/to/export",
  "base": "/path/to/local-base",
  "output": "/path/to/evaluation-output",
  "split": "validation",
  "data_dir": "/path/to/voicebox/data",
  "baseline_profile_id": "existing-cloned-profile-uuid"
}
```

Use validation for pilot decisions and a fresh output directory with `split`
set to `test` only for the final selected checkpoint. Each run selects six
duration-spread passages from each held-out chapter plus four independently
written prose passages. The baseline uses Voicebox's real reference ordering,
normalization, saved WAV format and pinned MLX inference optimizations.

The predeclared automated gates are: word error at most 5% overall/20% for any
sample, no more than two percentage points worse than the baseline, mean speaker
cosine within 0.03 of baseline, median warm processing/audio ratio no more than
10% slower, no duration-limit truncations, and all final three-word endings
present in the independent transcript. The ending check was added after the
pilot exposed an early EOS that passed the aggregate error-rate thresholds;
it is required before any installation. Samples and independent Whisper
transcripts are retained. These are automated checks, not a claim that someone
has listened to the result. Inspect the audio for subjective delivery quality.

ASR uses greedy decoding without reference-text hints. An empty transcript or
Whisper compression ratio above 2.4 triggers one deterministic retry with
timestamp tokens disabled; both raw attempts are retained. An unresolved loop
blocks scoring rather than being trimmed or ignored. No retry is triggered by
word error or an incorrect ending. Changes to the evaluation implementation
require a new comparison directory and invalidate old ASR caches.

When recognizers disagree, `python -m scripts.finetune_qwen.crosscheck
GENERATION_DIR OTHER_LOCAL_WHISPER NEW_REPORT_DIR` scores **every** original
paired WAV with the other recognizer and the same six gates. It never replaces
the original report or splices individual transcripts. Original generation,
primary evaluation, recognizer weights/config and cross-check code are hashed;
installation revalidates that provenance. Both MLX NPZ and safetensors weights
are supported. Declare any multi-recognizer acceptance requirement before
running these complete comparisons; an isolated diagnostic is not a pass.

The installer rechecks the final paired results and all model hashes before
registering the export in `VOICEBOX_DATA/finetuned_voices`. The resulting
`finetuned:fabian` CustomVoice profile loads this local checkpoint, never a
downloaded built-in speaker. It supports Spanish, Qwen 1.7B and fixed delivery;
the generation form does not offer unsupported languages, sizes or delivery
instructions. Older cloned profiles and frozen audiobook jobs are unchanged.

Local fine-tuned profiles cap each model call at **200 characters**, using the
existing sentence/clause-aware chunker and crossfades for longer narration.
Validation found early EOS on long single sentences; lower-temperature and
greedy decoding did not solve it reliably. The bounded path preserves all input
text and deterministic per-chunk seeds, and also applies to streaming/synchronous
API requests. Evaluation uses this same path, and installation requires its
current source hashes to match the evaluated implementation. Other engines keep
their existing chunk sizes. Direct low-level calls exceeding the local limit
are rejected; use `generate_chunked` when integrating the exported narrator.

Use `LocalQwenCustomVoiceBackend.voice_request` for model loading and ownership.
Its loader and inference share a cross-thread MLX stream under the global
lifecycle guard; loading a raw lazy MLX model on one thread and evaluating it
on another is not safe. Offline evaluation uses the same loader and serialized
inference path. The native regression includes lazy buffers surviving separate
executor lifetimes.
The local model also binds a private compiled categorical sampler with explicit
request-local PRNG keys. This preserves the seeded single-thread MLX sequence
without depending on the importing thread's captured random-state list; Qwen's
filtering and other backends are unchanged.

## Mixed-speaker episode preparation

`python -m scripts.finetune_qwen.episodes CONFIG.json` prepares a separately
declared collection of M4B episodes using existing Whisper word timestamps.
Its private config names `audio_directory`, `timestamps`, `output`, `base`,
`expected_episodes`, disjoint `validation_episodes`/`test_episodes`, and training-
only `speaker_references` (objects with `path` and optional `episode`).
`negative_episode`, `control_episodes`, and `speaker_thresholds` declare the
announcer controls and minimum target cosine / target-negative margin.

The preparer matches Unicode-normalized filenames, checks actual durations,
binds source/transcript/model/code hashes, excludes the first 20 and last 25
seconds, and breaks at uncertain ASR words. Whole clips and every overlapping
three-second window must pass frozen local ECAPA speaker comparisons. Explicit
announcer controls must fail. This is conservative screening, not perfect
diarization or a listening review. Run the independent `verify_corpus` step on
the actual exported WAVs; ASR hypotheses are not source-book text.

Episode IDs are source-qualified. Source hashes and declared split ownership,
speaker decisions, independent word agreements and codec hashes are rechecked
before training. All source files and rejected-candidate metadata are retained.

## Continue a completed voice

After independently verifying and encoding the episode corpus, compose it with
original verified training examples for rehearsal:

```sh
python -m scripts.finetune_qwen.combine EPISODES ORIGINAL NEW_CORPUS --count 600 --reference-id ch04_00077
python -m scripts.finetune_qwen.train NEW_CORPUS LOCAL_BASE NEW_RUN --init-run COMPLETED_RUN --lr 1e-5
```

Use the parent contract's actual reference ID, not an arbitrary replacement.
Composition preserves whole-source held-out splits and removes cross-corpus
text leakage. A warm start requires the same base/rank, a completed parent with
an improved selected checkpoint, and its exact training reference. Only selected
adapters and speaker conditioning are inherited: optimizer, update counters,
random state, and the new corpus's baseline/selection history start fresh.
Both source runs and the parent export remain untouched. Resume the new run
with the same full command, including `--init-run`.

For paired evaluation, `baseline_profile_id` may identify an installed local
fine-tuned profile. The evaluator resolves and hashes its registered checkpoint,
uses the same real bounded backend for both voices, and scores both with the
unchanged base speaker encoder. Supply four freshly written passages in the
private evaluation config's `novel` list, including a long-form passage of at
least 150 words. Select these before evaluating the candidate; keep validation
and final prose separate. Original-book test passages are now regression checks,
while the reserved new episodes are the untouched final-test source.

If the lowest-loss snapshot fails the **validation audio** gates, a previously
selected snapshot still retained by the completed run can be exported with
`export ... --step UPDATE` to a fresh model directory. This never changes the
training pointers: it validates the completed run/contract and binds the chosen
file hash into the new export. Keep every failed report, use the identical
validation passages/seeds/gates, and evaluate the final test only after choosing
a passing candidate. Missing overwritten snapshots cannot be reconstructed by
changing a pointer or inventing optimizer state.

## Checks

```sh
backend/venv/bin/python -m pytest scripts/finetune_qwen -q
backend/venv/bin/ruff check --config backend/pyproject.toml scripts/finetune_qwen
backend/venv/bin/ruff format --config backend/pyproject.toml --check scripts/finetune_qwen
```

The workflow follows the single-speaker checkpoint layout described in
[Qwen's fine-tuning guide](https://github.com/QwenLM/Qwen3-TTS/tree/main/finetuning).
Its explicit causal losses are checked against the installed inference code;
the upstream training example's shifted targets are not reused.
