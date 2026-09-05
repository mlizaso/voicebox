# Voicebox landing site

This Next.js site is the public presentation, downloads, blog, and cloud/pricing
surface. It is separate from the voice studio in `app/`, `web/`, and `tauri/`.
For running Voicebox or finding profiles/audio, use the [main README](../README.md).

## Run and validate

Install the shared Bun workspace from the repository root:

```bash
bun install --frozen-lockfile
bun run dev:landing
```

Next.js normally prints `http://localhost:3000`. The documentation site uses the
same default port, so select another port or run them separately.

```bash
bun run --cwd landing test
bun run build:landing
bun run --cwd landing start
```

For an environment that cannot start Turbopack workers, run
`bun run --cwd landing build --webpack`. `start` requires a completed build.

## Structure and configuration

| Path | Responsibility |
| --- | --- |
| `src/app/` | App Router pages, download redirects, cloud/pricing/token surfaces |
| `src/components/` | Landing UI components |
| `src/lib/constants.ts` | Repository/site constants |
| `src/lib/releases.ts` | GitHub release links, star/download totals, five-minute cache |
| `src/lib/blog.ts`, `src/posts/` | Markdown blog loading and posts |
| `src/lib/token-stats.ts` | Token-stat integrations |
| `public/` | Public images, video, and static assets |
| `tests/` | Redirect and release-cache regressions |
| `nixpacks.toml` | Deployment build/start configuration |

Download URLs come from release assets and the configured repository; there are
no `LATEST_VERSION` or `DOWNLOAD_LINKS` constants to update manually. The
configured upstream download channel does not automatically distribute changes
from this fork.

Release/star refreshes share an in-flight request and use a bounded timeout.
Download totals are cached only after all pages succeed; a failed refresh can
serve the previous complete result. Download redirects use the canonical site
origin instead of request forwarding headers.

## Content and deployment

Write posts under `src/posts/` using the metadata expected by `src/lib/blog.ts`.
Keep dated posts as historical articles; operational instructions belong in the
project docs. Inspect the repository's deployment settings and `nixpacks.toml`
before publishing. A successful local build is not a deployment.
