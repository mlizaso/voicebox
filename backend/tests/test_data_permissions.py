"""Model-free tests for Voicebox local-data permission policy."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from backend import config, data_permissions as permissions
from backend.database import session as database_session


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture(autouse=True)
def _restore_process_umask():
    if os.name != "posix":
        yield
        return
    original = os.umask(0)
    os.umask(original)
    try:
        yield
    finally:
        os.umask(original)


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_default_database_startup_hardens_new_data_and_sqlite_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "voicebox-data"
    root.mkdir(mode=0o755)
    os.chmod(root, 0o755)
    monkeypatch.delenv("VOICEBOX_SHARED_GENERATIONS", raising=False)
    monkeypatch.setattr(config, "_data_dir", root)

    config.initialize_data_permissions()
    database_session.init_db()

    profile_dir = config.get_profiles_dir() / "profile-id"
    profile_dir.mkdir(mode=0o755)
    sample = profile_dir / "sample.wav"
    sample.write_bytes(b"private voice sample")
    os.chmod(profile_dir, 0o755)
    os.chmod(sample, 0o644)
    assert config.to_storage_path(sample) == "profiles/profile-id/sample.wav"

    generation = config.get_generations_dir() / "book.wav"
    generation.write_bytes(b"generated audio")
    os.chmod(generation, 0o644)
    assert config.to_storage_path(generation) == "generations/book.wav"

    assert _mode(root) == 0o700
    assert _mode(config.get_db_path()) == 0o600
    assert _mode(profile_dir) == 0o700
    assert _mode(sample) == 0o600
    assert _mode(config.get_generations_dir()) == 0o700
    assert _mode(generation) == 0o600
    for name in ("cache", "captures", "deletion_journal", "exact_voice_snapshots", "logs"):
        assert _mode(root / name) == 0o700

    database_session.engine.dispose()


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_startup_repairs_existing_owned_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "existing"
    sample = root / "profiles" / "profile-id" / "sample.wav"
    generation = root / "generations" / "old.wav"
    cache_file = root / "cache" / "prompt.bin"
    capture = root / "captures" / "recording.wav"
    log_file = root / "logs" / "watchdog.log"
    for path in (sample, generation, cache_file, capture, log_file):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing")
    database = root / "voicebox.db"
    database.write_bytes(b"sqlite")

    for directory in (
        root,
        sample.parent.parent,
        sample.parent,
        generation.parent,
        cache_file.parent,
        capture.parent,
        log_file.parent,
    ):
        os.chmod(directory, 0o755)
    for path in (sample, generation, cache_file, capture, log_file, database):
        os.chmod(path, 0o644)

    monkeypatch.delenv("VOICEBOX_SHARED_GENERATIONS", raising=False)
    monkeypatch.setattr(config, "_data_dir", root)
    config.initialize_data_permissions()

    assert _mode(root) == 0o700
    assert _mode(sample.parent) == 0o700
    assert _mode(sample) == 0o600
    assert _mode(generation.parent) == 0o700
    assert _mode(generation) == 0o600
    assert _mode(cache_file) == 0o600
    assert _mode(capture) == 0o600
    assert _mode(log_file) == 0o600
    assert _mode(database) == 0o600


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_shared_generation_mode_does_not_expose_other_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "shared-exports"
    monkeypatch.setenv("VOICEBOX_SHARED_GENERATIONS", "yes")
    monkeypatch.setattr(config, "_data_dir", root)
    config.initialize_data_permissions()

    generation_dir = config.get_generations_dir() / "book"
    generation_dir.mkdir()
    generation = generation_dir / "chapter.wav"
    generation.write_bytes(b"generated audio")
    config.to_storage_path(generation)

    profile_dir = config.get_profiles_dir() / "profile-id"
    profile_dir.mkdir()
    sample = profile_dir / "sample.wav"
    sample.write_bytes(b"voice sample")
    config.to_storage_path(sample)
    database = config.get_db_path()
    database.write_bytes(b"sqlite")

    assert _mode(root) == 0o711
    assert _mode(config.get_generations_dir()) == 0o755
    assert _mode(generation_dir) == 0o755
    assert _mode(generation) == 0o644
    assert _mode(config.get_profiles_dir()) == 0o700
    assert _mode(profile_dir) == 0o700
    assert _mode(sample) == 0o600
    assert _mode(database) == 0o600
    for name in ("cache", "captures", "deletion_journal", "exact_voice_snapshots", "logs"):
        assert _mode(root / name) == 0o700


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_permission_repair_skips_symlinks_and_paths_outside_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "data"
    profiles = root / "profiles"
    profiles.mkdir(parents=True)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    os.chmod(outside, 0o644)
    linked_sample = profiles / "linked.wav"
    linked_sample.symlink_to(outside)
    hardlinked_sample = profiles / "hardlinked.wav"
    os.link(outside, hardlinked_sample)

    monkeypatch.setattr(config, "_data_dir", root)
    report = config.repair_data_permissions()

    assert report.skipped >= 1
    assert _mode(outside) == 0o644
    assert not permissions.apply_managed_file_permissions(
        root,
        linked_sample,
        shared_generations=False,
    )
    assert not permissions.apply_managed_file_permissions(
        root,
        outside,
        shared_generations=False,
    )
    assert not permissions.apply_managed_file_permissions(
        root,
        hardlinked_sample,
        shared_generations=False,
    )
    assert _mode(outside) == 0o644


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_startup_refuses_managed_directory_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "data"
    outside = tmp_path / "outside-profiles"
    root.mkdir()
    outside.mkdir()
    (root / "profiles").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(config, "_data_dir", root)

    with pytest.raises(permissions.UnsafeDataPathError):
        config.initialize_data_permissions()

    assert not any(outside.iterdir())


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_set_data_dir_refuses_root_symlink_without_changing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    target = tmp_path / "outside-target"
    target.mkdir()
    marker = target / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    os.chmod(target, 0o755)
    os.chmod(marker, 0o644)
    linked_root = tmp_path / "voicebox-data"
    linked_root.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(config, "_data_dir", tmp_path / "placeholder")

    with pytest.raises(permissions.UnsafeDataPathError):
        config.set_data_dir(linked_root)

    assert linked_root.is_symlink()
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert _mode(target) == 0o755
    assert _mode(marker) == 0o644


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_set_data_dir_refuses_ancestor_symlink_without_changing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker.txt"
    marker.write_text("unchanged", encoding="utf-8")
    os.chmod(outside, 0o755)
    os.chmod(marker, 0o644)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(config, "_data_dir", tmp_path / "placeholder")

    with pytest.raises(permissions.UnsafeDataPathError):
        config.set_data_dir(linked_parent / "voicebox-data")

    assert linked_parent.is_symlink()
    assert not (outside / "voicebox-data").exists()
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert _mode(outside) == 0o755
    assert _mode(marker) == 0o644


@pytest.mark.skipif(
    not permissions._supports_posix_permissions(),
    reason="secure POSIX descriptor operations are unavailable",
)
def test_set_data_dir_refuses_filesystem_root(monkeypatch: pytest.MonkeyPatch):
    original = config.get_data_dir()
    monkeypatch.setattr(config, "_data_dir", original)

    with pytest.raises(permissions.UnsafeDataPathError):
        config.set_data_dir(Path(Path.cwd().anchor))

    assert config.get_data_dir() == original


def test_non_posix_data_root_validation_still_rejects_drive_or_share_root(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(permissions, "_supports_posix_permissions", lambda: False)

    with pytest.raises(permissions.UnsafeDataPathError):
        permissions.ensure_data_root(
            Path(Path.cwd().anchor),
            shared_generations=False,
        )


def test_non_posix_permission_support_is_a_safe_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "data"
    sample = root / "profiles" / "sample.wav"
    sample.parent.mkdir(parents=True)
    sample.write_bytes(b"sample")
    os.chmod(root, 0o755)
    os.chmod(sample.parent, 0o755)
    os.chmod(sample, 0o644)

    monkeypatch.setattr(permissions, "_supports_posix_permissions", lambda: False)
    monkeypatch.setattr(config, "_data_dir", root)
    config.initialize_data_permissions()

    assert _mode(root) == 0o755
    assert _mode(sample.parent) == 0o755
    assert _mode(sample) == 0o644


def test_non_posix_fallback_refuses_linked_ancestor_without_outside_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(permissions, "_supports_posix_permissions", lambda: False)

    with pytest.raises(permissions.UnsafeDataPathError, match="unsafe component"):
        permissions.ensure_data_root(
            linked_parent / "voicebox-data",
            shared_generations=False,
        )

    assert not (outside / "voicebox-data").exists()
