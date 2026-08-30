"""Response helpers for private files that must never outlive a request."""

from collections.abc import Callable
from threading import Lock
from urllib.parse import quote

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.types import Receive, Scope, Send


def safe_content_disposition(disposition_type: str, filename: str) -> str:
    """Build a Content-Disposition header safe for non-ASCII filenames.

    Uses the RFC 5987 ``filename*`` parameter so browsers can decode UTF-8
    filenames while the ``filename`` fallback stays ASCII-only.
    """
    ascii_name = "".join(c for c in filename if c.isascii() and (c.isalnum() or c in " -_.")).strip() or "download"
    utf8_name = quote(filename, safe="")
    return f"{disposition_type}; filename=\"{ascii_name}\"; filename*=UTF-8''{utf8_name}"


class CleanupFileResponse(FileResponse):
    """Run cleanup after success and also when streaming raises/disconnects."""

    def __init__(self, *args, cleanup: Callable[[], None], **kwargs):
        self._cleanup = _RunOnce(cleanup)
        super().__init__(*args, background=BackgroundTask(self._cleanup), **kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette normally invokes BackgroundTask only after the final
            # body send. A disconnect can raise before that point.
            self._cleanup()


class _RunOnce:
    """Make a synchronous cleanup safe across success and failure paths."""

    def __init__(self, cleanup: Callable[[], None]) -> None:
        self._cleanup = cleanup
        self._lock = Lock()
        self._called = False

    def __call__(self) -> None:
        with self._lock:
            if self._called:
                return
            self._called = True
        self._cleanup()
