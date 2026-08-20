"""Exclusive lifetime ownership tests for one Voicebox data directory."""

import os
import stat

import pytest

from backend import config, data_permissions
from backend.data_root_lock import DataRootInUseError, acquire_data_root_lock


def test_data_root_lock_excludes_second_backend_and_keeps_stable_inode(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()

    first = acquire_data_root_lock()
    lock_path = data_dir / ".voicebox.lock"
    first_stat = lock_path.stat()
    try:
        with pytest.raises(DataRootInUseError, match="already owns"):
            acquire_data_root_lock()
        if data_permissions._supports_posix_permissions():
            assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        assert f"pid={os.getpid()}" in lock_path.read_text(encoding="utf-8")
    finally:
        first.release()

    second = acquire_data_root_lock()
    try:
        assert lock_path.stat().st_ino == first_stat.st_ino
    finally:
        second.release()

    assert lock_path.is_file()


@pytest.mark.skipif(os.name != "posix", reason="creating test symlinks is not portable")
def test_data_root_lock_refuses_link_entry(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", data_dir)
    config.initialize_data_permissions()
    outside = tmp_path / "outside.lock"
    outside.write_text("untouched", encoding="utf-8")
    lock_path = data_dir / ".voicebox.lock"
    lock_path.symlink_to(outside)

    with pytest.raises(RuntimeError, match="not a regular file"):
        acquire_data_root_lock()

    assert outside.read_text(encoding="utf-8") == "untouched"
