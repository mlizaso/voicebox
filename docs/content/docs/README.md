---
title: "Documentation source guide"
description: "Where Voicebox documentation lives and how to maintain it"
---

This is the Fumadocs content tree. The repository's `README.md` is the complete
operating guide; `docs/README.md` describes installation, builds, API generation,
navigation, and documentation maintenance.

- `overview/`: end-user workflows and storage.
- `developer/`: architecture, development, and service internals.
- `api-reference/`: generated from `docs/openapi.json`; regenerate rather than
  hand-editing the operation wrappers.
- `index.mdx`: site home page; `meta.json` files control navigation.

From `docs/`, run `bun install --frozen-lockfile`, then `bun run dev`.
Validate with `bun run test` and `bun run build` (or `bun run build --webpack`
when Turbopack cannot start in the environment).
