# Voicebox documentation

The [repository README](../README.md) is the one-file operating guide: setup,
profile/audio storage, generation, the private audiobook workspace, and code layout.
This directory also contains the Fumadocs/Next.js documentation site.

## Run and build the site

`docs/` is a separate Bun workspace with its own lockfile and dependency patch.

```bash
cd docs
bun install --frozen-lockfile
bun run dev
```

Open the URL printed by Next.js, normally `http://localhost:3000`. The landing
site also defaults to 3000, so run one at a time or choose another port.

```bash
bun run test
bun run build
```

If the environment prevents Turbopack worker startup, use the supported Next.js
builder override: `bun run build --webpack`. `bun run start` serves a completed
production build. Building docs does not require generating speech.

## Content map

| Path | Purpose |
| --- | --- |
| `AUDIOBOOKS.md` | Current private audiobook operating guide |
| `content/docs/overview/` | User workflows, storage, installation, remote mode |
| `content/docs/developer/` | Setup, architecture, service/engine internals |
| `content/docs/api-reference/` | Generated operation pages |
| `content/docs/index.mdx` | Documentation home page |
| `content/docs/**/meta.json` | Navigation titles and order |
| `openapi.json` | Snapshot of the current backend schema |
| `scripts/generate-openapi.ts` | Generate pages grouped by route family from that snapshot |
| `lib/source.ts`, `source.config.ts` | Fumadocs content loader and schema |
| `app/[[...slug]]/` | Documentation page rendering/layout |
| `app/api/search/`, `app/llms-full.txt/` | Search and machine-readable docs |
| `patches/`, `tests/` | Maintained parser fix and regression tests |
| `plans/` and root `RUN_NOTES.md` | Dated design and implementation records |

## Refresh the API reference

The live backend's `/docs` and `/openapi.json` are authoritative. From the root,
with a backend already running on loopback 17493:

```bash
curl --fail --silent --show-error http://127.0.0.1:17493/openapi.json \
  -o docs/openapi.json
node_modules/.bin/biome format --write docs/openapi.json
```

Then, from `docs/`:

```bash
bun scripts/generate-openapi.ts
bun run build --webpack
```

Commit the schema and generated Markdown/navigation together. Do not hand-edit
generated operation wrappers. Inspect removed/renamed operations so obsolete
pages do not linger. A remote schema request needs a bearer header; never bake
the token into the schema or an example URL.

## Writing and reviewing

Use MDX with `title` and `description` frontmatter for website pages. Internal
site links use routes such as `/overview/quick-start`; ordinary repository docs
use relative file links. Register new pages in the corresponding `meta.json`.
Keep code paths, commands, multipart field names, and defaults tied to the code.

Use the README for the complete everyday workflow and specialized pages for
detail. Label proposals and historical benchmark results with their date/status;
do not present an old roadmap as implemented functionality. Preserve stamped
changelog entries and historical audit evidence.

See [patch maintenance](patches/README.md) before upgrading docs dependencies.
The landing site has a separate [guide](../landing/README.md). This repository's
CI files do not establish an automatic docs deployment contract; publish only
through the deployment configured by the repository owner.
