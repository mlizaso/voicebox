# E2E Test Fixtures

Place two files here before running `test_all_models_e2e.py`:

- `reference_voice.wav` — a clean speech sample, mono, 16–24 kHz, ~5–15 seconds.
- `reference_voice.txt` — the **exact** transcription of the WAV (single line, no trailing newline required).

These are used to create a cloned voice profile for every cloning-capable engine (qwen, luxtts, chatterbox, chatterbox_turbo, tada). The two default fixture filenames are ignored by the root `.gitignore`. Keep other private samples outside the repository or explicitly ignored; verify before staging.

You can point the test at different files with:

```
backend/venv/bin/python backend/tests/test_all_models_e2e.py \
  --reference-wav /path/to/your.wav \
  --reference-text "exact transcription here"
```
