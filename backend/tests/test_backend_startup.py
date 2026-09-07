"""Startup failures must stay actionable without weakening lifetime ownership."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend import config, main, startup
from backend.data_root_lock import DataRootInUseError, acquire_data_root_lock


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setattr(config, "_data_dir", root)
    config.initialize_data_permissions()
    return root


def test_help_works_without_installed_dependencies():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "backend.main", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "--data-dir" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("port", ["0", "-1", "65536"])
def test_invalid_port_fails_before_preparing_data(port, monkeypatch, capsys):
    run = Mock()
    monkeypatch.setattr(main, "run_server", run)
    with pytest.raises(SystemExit) as error:
        main.main(["--port", port])
    assert error.value.code == 2
    assert "--port must be between" in capsys.readouterr().err
    run.assert_not_called()


@pytest.mark.parametrize(
    ("existing_url", "strict_port", "expected_code"),
    [
        ("http://127.0.0.1:17494", False, 0),
        ("http://127.0.0.1:17493", False, 0),
        ("http://127.0.0.1:17493", True, 1),
        (None, False, 75),
    ],
)
def test_duplicate_launch_reports_owner_without_loading_dependencies(
    data_root, monkeypatch, capsys, existing_url, strict_port, expected_code
):
    first = acquire_data_root_lock(host="127.0.0.1", port=17493)
    inode = (data_root / ".voicebox.lock").stat().st_ino
    monkeypatch.setattr(startup, "find_local_backend", lambda *_args, **_kwargs: existing_url)
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    try:
        assert startup.run_server(host="127.0.0.1", port=17494, strict_port=strict_port) == expected_code
        with pytest.raises(DataRootInUseError):
            acquire_data_root_lock()
        assert (data_root / ".voicebox.lock").stat().st_ino == inode
        assert not (data_root / "voicebox.db").exists()
    finally:
        first.release()
    output = capsys.readouterr()
    assert "Traceback" not in output.err
    assert "dependencies" not in output.err
    if expected_code == 0:
        assert "already running" in output.out
        assert existing_url in output.out
        assert not output.err
        if existing_url.endswith(":17493"):
            assert "requested address http://127.0.0.1:17494 was not started" in output.out
    else:
        assert "PID" in output.err
        assert str(data_root) in output.err
        if existing_url:
            assert existing_url in output.err
        else:
            assert "may still be starting" in output.err


def test_cli_allows_callers_to_require_their_original_port(monkeypatch):
    run = Mock(return_value=0)
    monkeypatch.setattr(main, "run_server", run)
    assert main.main(["--port", "17494", "--strict-port"]) == 0
    assert run.call_args.kwargs["strict_port"] is True


def test_preflight_holds_lock_until_server_exits(data_root, monkeypatch):
    app = SimpleNamespace(state=SimpleNamespace())

    def run(application, **_kwargs):
        assert application is app
        assert not app.state.data_root_lock.released
        with pytest.raises(DataRootInUseError):
            acquire_data_root_lock()

    monkeypatch.setitem(sys.modules, "backend.app", SimpleNamespace(app=app))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    assert startup.run_server(host="127.0.0.1", port=17494) == 0
    assert app.state.data_root_lock.released
    second = acquire_data_root_lock()
    second.release()


def test_loopback_owner_does_not_satisfy_a_request_to_listen_on_all_interfaces(data_root, monkeypatch):
    monkeypatch.setattr(startup, "find_local_backend", lambda *_args, **_kwargs: "http://127.0.0.1:17494")
    first = acquire_data_root_lock(host="127.0.0.1", port=17494)
    try:
        assert startup.run_server(host="0.0.0.0", port=17494) == 1
    finally:
        first.release()


def test_missing_dependencies_release_lock_and_explain_setup(data_root, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert startup.run_server(host="127.0.0.1", port=17494) == 2
    assert "just setup-python" in capsys.readouterr().err
    retry = acquire_data_root_lock()
    retry.release()


def test_invalid_data_path_is_reported_without_traceback(data_root, tmp_path, capsys):
    invalid = tmp_path / "a-file"
    invalid.write_text("keep this", encoding="utf-8")
    assert startup.run_server(host="127.0.0.1", port=17494, data_dir=str(invalid)) == 2
    error = capsys.readouterr().err
    assert "real, writable directory" in error
    assert "Traceback" not in error
    assert invalid.read_text(encoding="utf-8") == "keep this"
    assert config.get_data_dir() == data_root


@pytest.mark.parametrize("error", [SystemExit(3), RuntimeError("unexpected server failure")])
def test_server_exit_before_lifespan_releases_lock(data_root, monkeypatch, error):
    app = SimpleNamespace(state=SimpleNamespace())
    monkeypatch.setitem(sys.modules, "backend.app", SimpleNamespace(app=app))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=Mock(side_effect=error)))
    with pytest.raises(type(error)):
        startup.run_server(host="127.0.0.1", port=17494)
    retry = acquire_data_root_lock()
    retry.release()


def test_failed_lifespan_drain_retains_lock_for_process_teardown(data_root, monkeypatch):
    app = SimpleNamespace(state=SimpleNamespace())

    def run(application, **_kwargs):
        application.state.data_root_lock.managed_by_lifespan = True
        raise RuntimeError("writer did not drain")

    monkeypatch.setitem(sys.modules, "backend.app", SimpleNamespace(app=app))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    try:
        with pytest.raises(RuntimeError, match="writer did not drain"):
            startup.run_server(host="127.0.0.1", port=17494)
        with pytest.raises(DataRootInUseError):
            acquire_data_root_lock()
    finally:
        app.state.data_root_lock.release()
