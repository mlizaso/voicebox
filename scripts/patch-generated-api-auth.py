"""Preserve Voicebox authentication when regenerating the fetch client."""

from __future__ import annotations

import sys
from pathlib import Path

AUTH_IMPORT = "import { authenticatedFetch } from '../authenticatedFetch';\n"
IMPORT_ANCHOR = "import type { OpenAPIConfig } from './OpenAPI';\n"
FETCH_CALL = "return await fetch(url, request);"
AUTHENTICATED_FETCH_CALL = "return await authenticatedFetch(url, request);"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-generated-api-auth.py <core/request.ts>")

    path = Path(sys.argv[1])
    source = path.read_text(encoding="utf-8")
    if AUTH_IMPORT not in source:
        if source.count(IMPORT_ANCHOR) != 1:
            raise RuntimeError("generated request.ts import anchor changed")
        source = source.replace(IMPORT_ANCHOR, f"{IMPORT_ANCHOR}{AUTH_IMPORT}")

    if FETCH_CALL in source:
        if source.count(FETCH_CALL) != 1:
            raise RuntimeError(
                "generated request.ts contains an unexpected number of fetch calls"
            )
        source = source.replace(FETCH_CALL, AUTHENTICATED_FETCH_CALL)
    if source.count(AUTHENTICATED_FETCH_CALL) != 1:
        raise RuntimeError("generated request.ts request call shape changed")

    path.write_text(source, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
