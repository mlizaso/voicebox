"""Exclusive ownership of downloaded backend installations."""

import threading


class BackendOperationGuard:
    """Reserve one backend's storage until its owning operation fully exits."""

    def __init__(self, backend_label: str, busy_error: type[RuntimeError]) -> None:
        self._backend_label = backend_label
        self._busy_error = busy_error
        self._lock = threading.Lock()
        self._active: tuple[object, str] | None = None

    def reserve(self, operation: str) -> object:
        """Reserve storage synchronously, before any background task starts."""
        token = object()
        with self._lock:
            if self._active is not None:
                raise self._busy_error(f"{self._backend_label} backend {self._active[1]} operation already in progress")
            self._active = (token, operation)
        return token

    def release(self, token: object) -> None:
        """Release only the matching reservation; stale callbacks are harmless."""
        with self._lock:
            if self._active is not None and self._active[0] is token:
                self._active = None

    def active_name(self) -> str | None:
        """Return the current operation for status reporting."""
        with self._lock:
            return self._active[1] if self._active is not None else None
