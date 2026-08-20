"""Power-loss recovery for CUDA/ROCm directory-swap transactions."""

import json
import os
from pathlib import Path

import pytest

from backend.utils.backend_archive import (
    BackendInstallRecoveryError,
    commit_backend_install,
    delete_backend_install,
    recover_backend_install,
)

_BACKENDS = [
    ("cuda", "voicebox-server-cuda.exe"),
    ("rocm", "voicebox-server-rocm.exe"),
]
_CRASH_PHASES = ["journal", "backup", "live", "cleanup"]


def _write_install(directory: Path, executable_name: str, payload: bytes) -> None:
    directory.mkdir(parents=True)
    (directory / executable_name).write_bytes(payload)
    (directory / "runtime.dll").write_bytes(payload + b"-runtime")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/os._exit crash injection")
@pytest.mark.parametrize(("backend_name", "executable_name"), _BACKENDS, ids=["cuda", "rocm"])
@pytest.mark.parametrize("crash_phase", _CRASH_PHASES)
@pytest.mark.parametrize("has_existing_install", [False, True], ids=["initial", "update"])
def test_swap_recovers_every_hard_crash_boundary(
    backend_name: str,
    executable_name: str,
    crash_phase: str,
    has_existing_install: bool,
    tmp_path: Path,
):
    root = tmp_path / "backends"
    root.mkdir()
    live = root / backend_name
    staging = root / f"{backend_name}-staging"
    backup = root / f"{backend_name}-backup"
    journal = root / f".{backend_name}-install-swap-v1.json"
    if has_existing_install:
        _write_install(live, executable_name, b"old")
    _write_install(staging, executable_name, b"new")

    child = os.fork()
    if child == 0:

        def crash_at(phase: str) -> None:
            if phase == crash_phase:
                os._exit(73)

        commit_backend_install(
            root,
            backend_name,
            executable_name,
            transition_hook=crash_at,
        )
        os._exit(0)

    _pid, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 73

    recover_backend_install(root, backend_name, executable_name)

    expected = b"old" if has_existing_install and crash_phase == "journal" else b"new"
    assert (live / executable_name).read_bytes() == expected
    assert not staging.exists()
    assert not backup.exists()
    assert not journal.exists()
    assert not list(root.glob(f".{backend_name}-install-swap-*.tmp"))


@pytest.mark.parametrize(("backend_name", "executable_name"), _BACKENDS, ids=["cuda", "rocm"])
def test_legacy_backup_is_restored_before_unjournaled_staging(
    backend_name: str,
    executable_name: str,
    tmp_path: Path,
):
    root = tmp_path / "backends"
    root.mkdir()
    (root / backend_name).mkdir()  # The legacy eager accessor recreated an empty live directory.
    _write_install(root / f"{backend_name}-backup", executable_name, b"old")
    _write_install(root / f"{backend_name}-staging", executable_name, b"unproven")

    recover_backend_install(root, backend_name, executable_name)

    assert (root / backend_name / executable_name).read_bytes() == b"old"
    assert not (root / f"{backend_name}-backup").exists()
    assert not (root / f"{backend_name}-staging").exists()


@pytest.mark.parametrize(("backend_name", "executable_name"), _BACKENDS, ids=["cuda", "rocm"])
def test_unjournaled_fresh_staging_is_discarded_even_with_executable(
    backend_name: str,
    executable_name: str,
    tmp_path: Path,
):
    root = tmp_path / "backends"
    root.mkdir()
    staging = root / f"{backend_name}-staging"
    _write_install(staging, executable_name, b"partial")

    recover_backend_install(root, backend_name, executable_name)

    assert not (root / backend_name).exists()
    assert not staging.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork/os._exit crash injection")
@pytest.mark.parametrize(("backend_name", "executable_name"), _BACKENDS, ids=["cuda", "rocm"])
def test_fresh_install_crash_after_executable_write_does_not_publish_staging(
    backend_name: str,
    executable_name: str,
    tmp_path: Path,
):
    root = tmp_path / "backends"
    root.mkdir()
    staging = root / f"{backend_name}-staging"

    child = os.fork()
    if child == 0:
        staging.mkdir()
        (staging / executable_name).write_bytes(b"partial")
        os._exit(74)

    _pid, status = os.waitpid(child, 0)
    assert os.WIFEXITED(status)
    assert os.WEXITSTATUS(status) == 74

    recover_backend_install(root, backend_name, executable_name)

    assert not (root / backend_name).exists()
    assert not staging.exists()


@pytest.mark.parametrize(("backend_name", "executable_name"), _BACKENDS, ids=["cuda", "rocm"])
def test_corrupt_swap_journal_fails_closed_without_deleting_candidates(
    backend_name: str,
    executable_name: str,
    tmp_path: Path,
):
    root = tmp_path / "backends"
    root.mkdir()
    staging = root / f"{backend_name}-staging"
    _write_install(staging, executable_name, b"new")
    journal = root / f".{backend_name}-install-swap-v1.json"
    journal.write_text(json.dumps({"backend": "wrong", "version": 1}))

    with pytest.raises(BackendInstallRecoveryError, match="does not match"):
        recover_backend_install(root, backend_name, executable_name)

    assert (staging / executable_name).read_bytes() == b"new"
    assert journal.exists()


@pytest.mark.parametrize(("backend_name", "executable_name"), _BACKENDS, ids=["cuda", "rocm"])
def test_delete_removes_live_and_interrupted_install_artifacts(
    backend_name: str,
    executable_name: str,
    tmp_path: Path,
):
    root = tmp_path / "backends"
    root.mkdir()
    for suffix in ("", "-staging", "-backup"):
        _write_install(root / f"{backend_name}{suffix}", executable_name, suffix.encode() or b"live")
    journal = root / f".{backend_name}-install-swap-v1.json"
    journal.write_text("interrupted")

    assert delete_backend_install(root, backend_name, executable_name) is True
    assert not any(root.iterdir())
