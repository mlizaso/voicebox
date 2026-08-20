"""
Configuration module for voicebox backend.

Handles data directory configuration for production bundling.
"""

import logging
import os
from pathlib import Path

from .data_permissions import (
    PRIVATE_DIR_MODE,
    apply_managed_file_permissions,
    ensure_data_root,
    ensure_managed_directory,
    generation_dir_mode,
    repair_data_permissions as _repair_data_permissions,
    set_private_umask,
)

logger = logging.getLogger(__name__)

# Allow users to override the HuggingFace model download directory.
# Set VOICEBOX_MODELS_DIR to an absolute path before starting the server.
# This sets HF_HUB_CACHE so all huggingface_hub downloads go to that path.
_custom_models_dir = os.environ.get("VOICEBOX_MODELS_DIR")
if _custom_models_dir:
    os.environ["HF_HUB_CACHE"] = _custom_models_dir
    logger.info("Model download path set to: %s", _custom_models_dir)


def _absolute_lexical_path(path: str | Path) -> Path:
    """Make a path absolute without following its final symbolic link."""
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


# Default data directory (used in development). Keep the lexical path so the
# secure opener can reject a symlink at the configured root itself.
_data_dir = _absolute_lexical_path("data")

_PRIVATE_MANAGED_DIRS = (
    "backends",
    "cache",
    "captures",
    "deletion_journal",
    "exact_voice_snapshots",
    "logs",
    "models",
    "profiles",
)
_TRUE_ENV_VALUES = frozenset({"1", "on", "true", "yes"})


def shared_generations_enabled() -> bool:
    """Return whether generated audio is intentionally host-readable."""
    return os.environ.get("VOICEBOX_SHARED_GENERATIONS", "").strip().lower() in _TRUE_ENV_VALUES


def _ensure_data_root() -> None:
    """Create and harden only the configured data root."""
    shared_generations = shared_generations_enabled()
    set_private_umask()
    ensure_data_root(
        _data_dir,
        shared_generations=shared_generations,
    )


def _ensure_managed_dir(name: str) -> Path:
    _ensure_data_root()
    mode = generation_dir_mode(shared_generations_enabled()) if name == "generations" else PRIVATE_DIR_MODE
    return ensure_managed_directory(_data_dir, name, mode=mode)


def initialize_data_permissions() -> None:
    """Create managed directories and repair owned sensitive data safely."""
    _ensure_data_root()
    for name in _PRIVATE_MANAGED_DIRS:
        ensure_managed_directory(_data_dir, name, mode=PRIVATE_DIR_MODE)
    ensure_managed_directory(
        _data_dir,
        "generations",
        mode=generation_dir_mode(shared_generations_enabled()),
    )
    report = repair_data_permissions()
    if report.repaired or report.skipped:
        logger.info(
            "Voicebox data permission repair: %d updated, %d safely skipped",
            report.repaired,
            report.skipped,
        )


def repair_data_permissions():
    """Repair existing managed paths beneath the configured root."""
    return _repair_data_permissions(
        _data_dir,
        shared_generations=shared_generations_enabled(),
    )


def _path_relative_to_any_data_dir(path: Path) -> Path | None:
    """Extract the path within a data dir from an absolute or relative path."""
    parts = path.parts
    for idx, part in enumerate(parts):
        if part != "data":
            continue

        tail = parts[idx + 1 :]
        if tail:
            return Path(*tail)
        return Path()

    return None


def managed_storage_relative_path(path: str | Path | None) -> Path | None:
    """Return the lexical path a stored value resolves to below the data root.

    This mirrors :func:`resolve_storage_path`, including the historical
    absolute ``.../data/...`` rebasing rule, but never resolves filesystem
    links.  Destructive callers can therefore compare database ownership
    without accidentally following a path outside the managed root.
    """
    if path is None:
        return None
    stored_path = Path(path)
    if not stored_path.parts:
        return None

    root = _absolute_lexical_path(_data_dir)
    candidate = stored_path
    if stored_path.is_absolute():
        absolute = _absolute_lexical_path(stored_path)
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            legacy_relative = _path_relative_to_any_data_dir(stored_path)
            if legacy_relative is None:
                return None
            rebased = _absolute_lexical_path(root / legacy_relative)
            try:
                relative = rebased.relative_to(root)
            except ValueError:
                return None
            # Match resolve_storage_path(): prefer a live current-root copy,
            # or rebase a path whose former installation no longer exists.
            try:
                if not rebased.exists() and stored_path.exists():
                    return None
            except OSError:
                return None
    else:
        if candidate.parts and candidate.parts[0] == "data":
            candidate = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path()
        absolute = _absolute_lexical_path(root / candidate)
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            return None

    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        return None
    return relative


def set_data_dir(path: str | Path):
    """
    Set the data directory path.

    Args:
        path: Path to the data directory
    """
    global _data_dir
    previous_data_dir = _data_dir
    _data_dir = _absolute_lexical_path(path)
    try:
        initialize_data_permissions()
    except BaseException:
        _data_dir = previous_data_dir
        raise
    logger.info("Data directory set to: %s", _data_dir)


def get_data_dir() -> Path:
    """
    Get the data directory path.

    Returns:
        Path to the data directory
    """
    return _data_dir


def get_deletion_journal_dir() -> Path:
    """Return the private directory used to recover interrupted deletions."""
    return _ensure_managed_dir("deletion_journal")


def to_storage_path(path: str | Path) -> str:
    """Convert a filesystem path to a DB-safe path relative to the data dir."""
    apply_managed_file_permissions(
        _data_dir,
        Path(path),
        shared_generations=shared_generations_enabled(),
    )
    resolved_path = Path(path).resolve()

    # Prefer the exact configured root.  A custom root may itself live below a
    # directory named ``data`` (for example ``/srv/data/voicebox``); applying
    # the legacy rebasing rule first would store ``voicebox/profiles/...`` and
    # later resolve it as ``<root>/voicebox/profiles/...``.
    try:
        return str(resolved_path.relative_to(_data_dir))
    except ValueError:
        pass

    relative_to_any_data_dir = _path_relative_to_any_data_dir(resolved_path)
    if relative_to_any_data_dir is not None:
        return str(relative_to_any_data_dir)

    return str(resolved_path)


def resolve_storage_path(path: str | Path | None) -> Path | None:
    """Resolve a DB-stored path against the configured data dir."""
    if path is None:
        return None

    stored_path = Path(path)
    # Empty paths (e.g. failed generations) must not resolve to the data
    # dir itself, which exists and would defeat the callers' 404 guards.
    # Path("") is truthy, so check parts rather than the raw value.
    if not stored_path.parts:
        return None
    if stored_path.is_absolute():
        resolved_path = stored_path.resolve()
        try:
            resolved_path.relative_to(_data_dir)
        except ValueError:
            pass
        else:
            return resolved_path

        rebased_path = _path_relative_to_any_data_dir(stored_path)
        if rebased_path is not None:
            candidate = (_data_dir / rebased_path).resolve()
            if candidate.exists() or not stored_path.exists():
                return candidate

        return stored_path

    # 0.3.0 records sometimes stored relative paths with the data-dir name
    # baked in (e.g. "data/profiles/..."). Joining those directly with
    # _data_dir produces a spurious "<data_dir>/data/profiles/..." nest.
    if stored_path.parts and stored_path.parts[0] == "data":
        stored_path = Path(*stored_path.parts[1:]) if len(stored_path.parts) > 1 else Path()

    return (_data_dir / stored_path).resolve()


def get_db_path() -> Path:
    """Get database file path."""
    _ensure_data_root()
    return _data_dir / "voicebox.db"


def get_profiles_dir() -> Path:
    """Get profiles directory path."""
    return _ensure_managed_dir("profiles")


def get_generations_dir() -> Path:
    """Get generations directory path."""
    return _ensure_managed_dir("generations")


def get_captures_dir() -> Path:
    """Get captures directory path."""
    return _ensure_managed_dir("captures")


def get_cache_dir() -> Path:
    """Get cache directory path."""
    return _ensure_managed_dir("cache")


def get_logs_dir() -> Path:
    """Get private application log directory path."""
    return _ensure_managed_dir("logs")


def get_models_dir() -> Path:
    """Get models directory path."""
    return _ensure_managed_dir("models")


# Voicebox Cloud (backup & sync). Two hosts: the web app owns auth + device
# pairing (voicebox.sh), the API owns sync + account endpoints
# (api.voicebox.sh). Override both for local development, e.g.
# VOICEBOX_CLOUD_URL=http://localhost:17592 VOICEBOX_CLOUD_API_URL=http://localhost:17593
def get_cloud_web_url() -> str:
    """Base URL of the Voicebox Cloud web app (auth + /connect + exchange)."""
    return os.environ.get("VOICEBOX_CLOUD_URL", "https://voicebox.sh").rstrip("/")


def get_cloud_api_url() -> str:
    """Base URL of the Voicebox Cloud API (bearer-authenticated sync/account)."""
    return os.environ.get("VOICEBOX_CLOUD_API_URL", "https://api.voicebox.sh").rstrip("/")
