# Security policy and deployment boundaries

This fork's current security work is recorded in the
[2026-09-05 audit](docs/SECURITY_AUDIT_2026-09-05.md) and [Unreleased](CHANGELOG.md).
The source still carries the upstream 0.5.0 application version; version numbers
alone do not identify which later fork fixes an installed binary contains.
Do not assume the upstream updater distributes this checkout's changes.

## Report a vulnerability

Report sensitive details privately to the repository owner, including the exact
commit/build, platform, reproducible input, affected boundary, and impact. Do not
put tokens, personal audio, or database contents in a public issue.
For vulnerabilities inherited from upstream, its published contact in this
repository is [security@voicebox.sh](mailto:security@voicebox.sh). This fork does
not make a new response-time or release-support promise on upstream's behalf.

## Local and remote operation

- The backend binds to loopback by default. Credential-free access requires both
  the direct network peer and requested Host to be local. Local processes are
  inside this trust boundary; the API is not a multi-user isolation system.
- Remote access requires a URL-safe 32–512-character `VOICEBOX_REMOTE_API_TOKEN`
  and explicit trusted hosts. Separate browser origins need an explicit CORS
  allowlist. Wildcards are refused; Host/origin checks remain active with auth.
- Use HTTPS or a trusted SSH tunnel. Preserve public Host/HTTPS scheme through
  the proxy and trust forwarding headers only from that proxy. The explicit
  insecure-HTTP compatibility option is not suitable for public deployment.
- The app keeps bearer tokens in memory and uses authenticated, bounded media
  requests with cookies omitted. Switching servers cancels/discards stale data.
- MCP server-local audio paths are available only to direct local callers.
  Remote authenticated callers upload audio; a bearer token does not grant
  arbitrary server-path transcription.

See [remote mode](docs/content/docs/overview/remote-mode.mdx) and
[Docker](docs/content/docs/overview/docker.mdx) for the actual configuration.
The supplied Compose mapping is host-loopback port 17600 and requires a token
because the internal bridge peer is treated as a network client.

## Data and native boundaries

Sensitive POSIX directories/files are private by default. Data roots and managed
top-level directories must be real directories. Archive/upload limits, validated
paths/metadata, and durable ownership records protect imports and file operations.
Model migration refuses to delete existing destination models.

`VOICEBOX_SHARED_GENERATIONS=1` deliberately exposes generated audio to host
readers, while database, profiles, captures, cache, and logs remain private.
Back up the entire stopped data root; do not delete SQLite recovery files as a
lock workaround. See [storage](README.md#where-your-files-are-saved).

Native commands enforce window/connection/origin restrictions. Directory opening
requires the current local server and rejects network/device paths before filesystem
access. Capture/output buffers, clipboard snapshots, focus targets, and remote
media have bounded ownership and lifetime. Cross-platform manual testing is still
required for native changes.

## Models, cloud, and updates

Local synthesis/transcription runs on the selected backend. Remote mode sends
inputs to that backend; explicit cloud login uses the configured cloud service.
Do not describe optional remote/cloud use as data that never leaves the machine.
Cloud backup/sync proposals are not a guarantee of implemented encrypted backups.

Updater verification uses the configured signing key and release channel. Keep
signing material and private voice references out of commits. Do not downgrade
validation or exact-runtime checks to accept an incompatible model/package update.

Known incompatible upstream dependency advisories remain, including parts of the
pinned Transformers and Linux GTK graph. The audit documents reachability and
compatibility limits. A passing test suite or a zero-advisory JavaScript audit is
not a guarantee that the whole application has no vulnerabilities.
