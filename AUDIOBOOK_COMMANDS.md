# Audiobook command card

Run these commands in Terminal on this Mac. The first command is the normal
one: it opens the audiobook window. Choose **New audiobook** or **Resume** in
that window.

```bash
cd /Users/manexlizaso/Developer/manex/voicebox
python3 voice-profile/build/make_audio.py
```

If the window cannot start its backend, use a second Terminal window:

```bash
cd /Users/manexlizaso/Developer/manex/voicebox
backend/venv/bin/python -m backend.main --host 127.0.0.1 --port 17494 --data-dir /Users/manexlizaso/Developer/manex/voicebox/data
```

Keep that terminal open, then run the first command again.

Check the audiobook backend and its real storage paths:

```bash
curl --fail --silent --show-error http://127.0.0.1:17494/health | python3 -m json.tool
curl --fail --silent --show-error http://127.0.0.1:17494/health/filesystem | python3 -m json.tool
```

List the private narrator variants and see which one the command-line helper
currently considers active:

```bash
voice-profile/build/voice.sh list
voice-profile/build/voice.sh current
```

See or open saved jobs for **Resume**:

```bash
ls -lt "$HOME/Library/Application Support/Fabian Audiobook Maker/progress"
open "$HOME/Library/Application Support/Fabian Audiobook Maker/progress"
```

Check the local tools before a long render:

```bash
python3 -c 'import tkinter; print("launcher Tk: OK")'
voice-profile/.venv/bin/python -c 'import tkinter; print("renderer Python/Tk: OK")'
test -x /opt/homebrew/bin/ffmpeg && echo "FFmpeg: OK"
test -x /opt/homebrew/bin/ffprobe && echo "FFprobe: OK"
df -h /Users/manexlizaso/Developer/manex/voicebox/data
```

The normal book workflow is the GUI above. This lower-level command is only
for an already prepared phrase document and output directory:

```bash
voice-profile/build/voice.sh render DOCS_JSON OUTPUT_DIRECTORY
```

Do not run `render_book.py` directly for a normal book; the GUI preserves the
saved job, frozen inputs, checksums, and resume state.

These are developer/security checks, not audiobook commands. They do not create
audio and are not required before clicking **New audiobook** or **Resume**:

```bash
bun run ci
bun audit
bun pm untrusted
```

`bun run ci` tests and builds the Voicebox app. `bun audit` checks JavaScript
packages online. `bun pm untrusted` checks package install scripts. The docs
workspace has its own audit; it is unrelated to audiobook rendering:

```bash
(cd docs && bun audit)
```

The root audit passed on 2026-09-06. The docs audit currently reports three
known advisories already documented in `docs/SECURITY_AUDIT_2026-09-05.md` and
`docs/patches/README.md`; do not try to fix those as part of making an audiobook.
