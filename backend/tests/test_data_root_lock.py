"""Exclusive lifetime ownership tests for one Voicebox data directory."""

import errno
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


def test_data_root_lock_reports_owner_and_recovers_stale_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "_data_dir", tmp_path)
    (tmp_path / ".voicebox.lock").write_text("pid=999999999\nhost=localhost\nport=8000\n", encoding="utf-8")
    first = acquire_data_root_lock(host="127.0.0.1", port=17493)
    try:
        with pytest.raises(DataRootInUseError) as error:
            acquire_data_root_lock(host="127.0.0.1", port=17494)
        assert error.value.pid == os.getpid()
        assert error.value.host == "127.0.0.1"
        assert error.value.port == 17493
        assert error.value.data_dir == tmp_path
    finally:
        first.release()
    first.release()  # CLI cleanup and lifespan shutdown may both release.


@pytest.mark.skipif(os.name != "posix", reason="flock error behavior is POSIX-specific")
def test_data_root_lock_does_not_mislabel_io_failures_as_contention(tmp_path, monkeypatch):
    import fcntl

    monkeypatch.setattr(config, "_data_dir", tmp_path)

    def fail(_fd, _operation):
        raise OSError(errno.EIO, "disk failure")

    monkeypatch.setattr(fcntl, "flock", fail)
    with pytest.raises(OSError, match="disk failure"):
        acquire_data_root_lock()
