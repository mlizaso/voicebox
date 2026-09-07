"""Early startup checks and actionable diagnostics for backend entry points."""

import sys

from . import config
from .data_permissions import UnsafeDataPathError
from .data_root_lock import DataRootInUseError, DataRootLockError, acquire_data_root_lock
from .server_discovery import find_local_backend, local_probe_host, server_url


def _report_existing_backend(error: DataRootInUseError, host: str, port: int, *, strict_port: bool) -> int:
    """Reuse the verified local backend while preserving explicit address requirements."""
    local_host = local_probe_host(host)
    existing_url = find_local_backend(
        error.data_dir, host=host, port=port, owner_host=error.host, owner_port=error.port
    )
    if existing_url:
        if host in {"127.0.0.1", "localhost", "::1"} and (
            not strict_port or existing_url == server_url(local_host, port)
        ):
            print(f"Voicebox is already running at {existing_url} using {str(error.data_dir)!r}.")
            if existing_url != server_url(local_host, port):
                print(f"Reusing that server; the requested address {server_url(host, port)} was not started.")
            print("Continue using that backend; no second server was started.")
            return 0
        print(f"{error}\nExisting backend: {existing_url}", file=sys.stderr)
        print(
            f"Requested address {server_url(host, port)} was not started. Connect to {existing_url}, "
            "or stop the existing backend in its terminal/app and retry this command. "
            "Changing the port does not allow two backends to share one data directory.",
            file=sys.stderr,
        )
        return 1
    print(str(error), file=sys.stderr)
    if error.host and error.port:
        print(f"Owner's configured address: {server_url(error.host, error.port)!r}", file=sys.stderr)
    print(
        "The existing backend may still be starting or may require authentication. "
        "Wait for it to become ready, or stop it in its terminal/app and retry. "
        "Do not delete .voicebox.lock; it is released automatically when its process exits.",
        file=sys.stderr,
    )
    return 75  # Temporary contention: launchers may wait for the owner to become ready.


def run_server(
    *,
    host: str,
    port: int,
    data_dir: str | None = None,
    strict_port: bool = False,
) -> int:
    """Reserve the root through startup/shutdown and explain expected failures."""
    try:
        if data_dir is not None:
            config.set_data_dir(data_dir)
        else:
            config.initialize_data_permissions()
        data_root_lock = acquire_data_root_lock(host=host, port=port)
    except DataRootInUseError as exc:
        return _report_existing_backend(exc, host, port, strict_port=strict_port)
    except (OSError, UnsafeDataPathError, DataRootLockError) as exc:
        print(
            f"Voicebox cannot prepare its data directory {data_dir or str(config.get_data_dir())!r}: {exc}\n"
            "Check that --data-dir names a real, writable directory and retry.",
            file=sys.stderr,
        )
        return 2

    try:
        try:
            import uvicorn

            from .app import app  # lazy: heavy import, after ownership is established
        except ImportError as exc:
            print(
                f"Voicebox could not load its backend dependencies: {exc}\n"
                "From the Voicebox repository, run `just setup-python`, then retry this command.",
                file=sys.stderr,
            )
            return 2

        # Keep the same OS lock from preflight through lifespan; releasing and
        # reacquiring here would allow simultaneous launchers to race.
        app.state.data_root_lock = data_root_lock
        uvicorn.run(app, host=host, port=port, reload=False)
        return 0
    finally:
        # Also covers import failures, interrupted startup and port-bind exits.
        if not data_root_lock.managed_by_lifespan:
            data_root_lock.release()
