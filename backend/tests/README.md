# Backend tests

Run from the repository root with the backend environment:

```bash
backend/venv/bin/python -m pytest backend/tests -q
```

`just test` runs the same directory with verbose output. This is primarily an
automated regression suite, not a collection of manual scripts. It covers HTTP
validation/security, profile and archive operations, generation publication and
replay, model lifecycle/queue behavior, exact-runtime contracts, stories, captures,
and recovery. Most tests use synthetic audio, temporary databases, and mocks.

## Focused checks

```bash
backend/venv/bin/python -m pytest backend/tests/test_generation_replay_settings.py -q
backend/venv/bin/python -m pytest backend/tests/test_archive_import_security.py -q
backend/venv/bin/python -m pytest backend/tests/test_api_security.py -q
backend/venv/bin/ruff check backend/tests
backend/venv/bin/ruff format --check backend/tests
```

New regression tests should fail when the corresponding fix is removed. Do not
weaken assertions or exact-audio/runtime checks to accommodate a failure.
Platform-specific skips and existing lint debt must be stated separately from
the checks that passed. On Apple Silicon, Metal-related checks need the host
runtime; an isolated environment that cannot initialize Metal is not an audio
quality test.

## Real models and frozen builds: opt in

`test_all_models_e2e.py` is a standalone runner. It locates/builds a frozen
server, uses a temporary data directory and port, then exercises selected engines.
It may download large models and perform real inference; it is not part of the
ordinary mocked pytest quality guarantee.

```bash
backend/venv/bin/python backend/tests/test_all_models_e2e.py --help
backend/venv/bin/python backend/tests/test_all_models_e2e.py \
  --only qwen --skip-build --binary /absolute/path/to/voicebox-server \
  --reference-wav /absolute/path/to/reference.wav \
  --reference-text 'The exact words spoken in the sample.'
```

See [fixture instructions](fixtures/README.md) and the
[original design](E2E_MODEL_TEST_DESIGN.md). The runner's current `--help` and
implementation take precedence over the dated design. Keep private samples out
of Git and use an explicit results directory if retaining reports.

## Other suites

- Shared app: `bun run --cwd app test`; the Bun preload installs storage before
  persisted stores load, so test order does not decide the storage implementation.
- Native: commands in [CONTRIBUTING.md](../../CONTRIBUTING.md).
- Private audiobook renderer: separate tests in ignored `voice-profile/build/`;
  these are not included in the backend pytest command.

Current verification evidence is in the [audit report](../../docs/SECURITY_AUDIT_2026-09-05.md).
