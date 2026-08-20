"""Response helpers for private files that must never outlive a request."""

from collections.abc import Callable

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.types import Receive, Scope, Send


class CleanupFileResponse(FileResponse):
    """Run cleanup after success and also when streaming raises/disconnects."""

    def __init__(self, *args, cleanup: Callable[[], None], **kwargs):
        self._cleanup = cleanup
        super().__init__(*args, background=BackgroundTask(cleanup), **kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette normally invokes BackgroundTask only after the final
            # body send. A disconnect can raise before that point.
            self._cleanup()
