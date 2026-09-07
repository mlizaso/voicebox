# Audiobook operating guide

For the complete project entry point, start with the [main README](../README.md).
For only the commands, use the [audiobook command card](../AUDIOBOOK_COMMANDS.md).
This guide covers the private tools present on this Mac, checked against the
local launcher and renderer. They are not a built-in Voicebox tab or part of a
fresh clone: `voice-profile/` is ignored and `voice-profile/build/` has its own Git
history. Its existing repository has no configured remote.

## Start here

```bash
cd /Users/manexlizaso/Developer/manex/voicebox
python3 voice-profile/build/make_audio.py
```

The launcher needs a Python interpreter with Tk. Its constants currently point
render workers to `voice-profile/.venv/bin/python`, Voicebox to
`backend/venv/bin/python`, and FFmpeg/FFprobe to `/opt/homebrew/bin/`.
Installing the main backend alone does not create the private renderer environment
or supply the narrator samples. If moving the workspace, inspect these paths in
`voice-profile/build/make_audio.py` before launching.

The launcher defaults to `http://127.0.0.1:17494`, but automatically reuses a
compatible local backend serving the same data directory, including port `17493`
or a custom port. The terminal prints the selected URL. It starts a backend when
needed and retries if a competing owner exits during startup. The data directory
defaults to `/Users/manexlizaso/Developer/manex/voicebox/data`; an explicit
`VOICEBOX_URL` disables switching to another address.

| Launcher setting | Default / meaning |
| --- | --- |
| `VOICEBOX_ROOT` | Main checkout path above |
| `VOICEBOX_URL` | `http://127.0.0.1:17494` |
| `VOICEBOX_DATA_DIR` | `<VOICEBOX_ROOT>/data`; passed to backend `--data-dir` |
| `VOICEBOX_BACKEND_LOG` | `/tmp/voicebox-backend.log` |

The shell renderer `voice-profile/build/voice.sh` also accepts `AUDIOBOOK_PYTHON`.
That shell setting is not an override for the GUI's separate Python constant.
Do not run competing generations through the desktop and book backends on the
same Apple GPU. Keep the backend stable, without development reloads, for a book.

## New book

1. Choose **New audiobook** and the narrator variants to evaluate.
2. Choose source and chapters. EPUB, repaired JSON, Markdown, and plain text are
   supported. When an EPUB has a repaired sibling manifest, inspect the selected
   narration text and chapter list before accepting it.
3. Select an output directory and `.m4b`, `.m4a`, `.mp3`, or `.wav`. Review title,
   author, and narrator metadata; these also determine the final filename.
4. Review disk capacity. The wizard checks backend storage and output volumes,
   selected voices, partial audio, assembly intermediates, and headroom.
5. Prefer **Demo rendering (5 min/voice)** when comparing narrators. Each voice
   receives the same representative excerpt and produces a lossless WAV. Five
   minutes is a planning estimate, not a duration guarantee.
6. Listen, select a narrator, and start full rendering. While the setup window
   remains open, the chosen demo voice can be carried back into book setup. After
   reopening an old demo, listen to its outputs and create a new book with the winner.

Voice definitions are in `voice-profile/build/voices/`; original reference audio
and transcripts are in `voice-profile/samples/`. Backend profile IDs are runtime
handles. The renderer's frozen voice content identifies reusable audio.

**Which voices?** lists the local reference-audio presets and installed fine-tuned
narrators from Voicebox, including Fabián v1 and v2. Select either kind for a demo
or a complete audiobook; both use the same saved-progress workflow. Fabián v2
retains its experimental label.

Fine-tuned jobs freeze the checkpoint manifest hash, speaker, rendering settings,
and runtime revision. Long text is split deterministically into model-sized units;
finished phrases are checksum-verified and reused on resume. Keep the installed
checkpoint and its `data/finetuned_voices/` registration: the job stores their
identity, not a second copy of the model weights. Missing profile metadata can be
recreated automatically. Missing or changed model files stop the render with an
explanation, preserving existing audio instead of mixing different models.

## Saved progress and files

| Item | Location / lifetime |
| --- | --- |
| Central saved jobs | `~/Library/Application Support/Fabian Audiobook Maker/progress/` |
| Job contents | Mutable `job.json` plus frozen normalized text, references, voice definitions, and controlling settings |
| Partial audio / phrase pool | Hidden directories in the selected output folder, with their paths recorded in the saved job |
| Backend artifacts | The backend's actual `generations/`, with profile copies in `profiles/` and metadata in `voicebox.db` |
| Final audiobook | Selected output directory, with title/variant/identity in its filename |
| Durable book log | Selected output directory, with `[progress <job-id>]` in its name |
| Backend startup diagnostics | `/tmp/voicebox-backend.log`, unless overridden |

Find the backend's actual storage rather than inferring it from a port (substitute
the URL printed by the launcher if different):

```bash
curl --fail --silent --show-error http://127.0.0.1:17494/health/filesystem \
  | python3 -m json.tool
```

For example, the existing Soi Cowboy audiobook is in
`/Users/manexlizaso/Developer/manex/ebook/ebook/la-balada-de-soi-cowboy/`.
This is an output choice, not a hardcoded location for all books.

## Pause, resume, and recovery

- **Save progress** writes the durable checkpoint. Automatic saves also run on a
  five-minute timer and after meaningful phases; render manifests retain completed units.
- **Save & close** stops the active renderer with progress retained. A hard
  interruption may require repeating the unit that was in flight.
- Reopen the wizard and choose **Resume**. If the renderer is still alive, the
  launcher can attach to its progress instead of creating a competing writer.
- Resume validates frozen input identity and completed WAV checksums. It does
  not adopt a file merely because its name looks like a completed phrase.
- A changed model/runtime fingerprint blocks exact resume. Restore the matching
  implementation or start a new job; never rewrite saved hashes to bypass the check.
- Failed mastering or a partial multi-voice result keeps progress. Successful
  cleanup happens only after every selected final output passes validation.
- **Remove** removes the saved job and its owned reusable pool metadata/cache;
  it keeps partial render WAVs and final books. It is not a disk-space cleanup
  operation for every artifact belonging to a book.

Back up both the central progress store and output work directories while stopped.
Keep the corresponding private voice assets and runtime revision. A final M4B is
a listening copy, not a synthesis checkpoint.

## Quality and time

The current audiobook contract uses Spanish Qwen `1.7B` with the pinned MLX
implementation; the base chunked contract is 1,200 characters, 10 ms crossfade,
and a fixed seed. Individual narrator definitions and phrased jobs freeze their
own additional parameters. Preserve those saved settings when resuming.

The pipeline includes the selected delivery processing and mastering. Do not
apply an old manual pitch or loudness recipe a second time to a wizard output.
M4B/M4A/MP3/WAV share the mastering contract, but their codecs are not all lossless.
Choose the format deliberately and compare listening quality on the demo.

Verified phrase reuse avoids repeating identical synthesis across compatible
variants. The two recent renderer I/O improvements use direct SoundFile reads for
compatible mono 24 kHz PCM16 WAVs and compact one-pass JSON checkpoints. FFmpeg
fallback and checkpoint flush frequency are retained. Byte-preservation tests
passed; these changes do not alter model sampling or mastering.

Autoregressive TTS still synthesizes new phrases sequentially. A long book also
needs final assembly and mastering. Keep a single inference worker, avoid reloads,
and compare narrators on demos. Switching to a smaller/quantized model, changing
reference length, chunking, or processing requires a separate quality comparison.
No full 13-hour generation-time benchmark was performed in the latest audit.

## Implementation map

| File in `voice-profile/build/` | Role |
| --- | --- |
| `make_audio.py` | GUI, backend startup, voice/source/output selection, resume and attachment |
| `audiobook_progress.py` | Durable jobs, frozen input bundles, locks, progress validation |
| `audiobook_sources.py` | Book ingestion and normalized chapters |
| `audiobook_capacity.py` | Storage estimates and capacity checks |
| `render_book.py` | Chunked book rendering |
| `render_phrased.py` | Phrase rendering, verified reuse, and manifests |
| `voice.sh` | Command-line entry point for narrator operations |
| `import_profile.py` | Idempotent import/update of the private narrator profile |

Historical investigations and measured benchmarks are in [RUN_NOTES.md](../RUN_NOTES.md)
and the local `voice-profile/analysis/` directory. Those records describe their
specific revisions; use this guide and the README for current operation.
