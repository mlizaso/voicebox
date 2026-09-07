"""Discovery must not confuse another service or another data root with ours."""

import json
import os
from unittest.mock import Mock

import pytest

from backend import server_discovery as discovery


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"healthy": False},
        {"healthy": True, "directories": None},
        {"healthy": True, "directories": [{"name": "data", "path": "/different-root"}]},
        {"healthy": True, "directories": [None, "invalid"]},
    ],
)
def test_discovery_rejects_wrong_or_incomplete_storage_health(tmp_path, monkeypatch, payload):
    response = Mock(status=200)
    response.read.return_value = json.dumps(payload).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(discovery.http.client, "HTTPConnection", Mock(return_value=connection))
    assert not discovery.uses_data_dir("127.0.0.1", 17493, tmp_path)
    connection.close.assert_called_once()


@pytest.mark.parametrize("status", [301, 401, 500])
def test_discovery_does_not_follow_redirects_or_ignore_http_errors(tmp_path, monkeypatch, status):
    connection = Mock()
    connection.getresponse.return_value = Mock(status=status)
    monkeypatch.setattr(discovery.http.client, "HTTPConnection", Mock(return_value=connection))
    assert not discovery.uses_data_dir("127.0.0.1", 17493, tmp_path)
    connection.getresponse.return_value.read.assert_not_called()


@pytest.mark.parametrize("body", [b"x" * 65537, b"not json", b"\xff"])
def test_discovery_bounds_and_validates_response(tmp_path, monkeypatch, body):
    response = Mock(status=200)
    response.read.return_value = body
    connection = Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(discovery.http.client, "HTTPConnection", Mock(return_value=connection))
    assert not discovery.uses_data_dir("127.0.0.1", 17493, tmp_path)
    response.read.assert_called_once_with(65537)


def test_discovery_accepts_exact_root(tmp_path, monkeypatch):
    response = Mock(status=200)
    response.read.return_value = json.dumps(
        {"healthy": True, "directories": [{"name": "data", "path": str(tmp_path.resolve())}]}
    ).encode()
    connection = Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(discovery.http.client, "HTTPConnection", Mock(return_value=connection))
    assert discovery.uses_data_dir("127.0.0.1", 17493, tmp_path)


def test_discovery_probes_only_loopback_and_honors_owner_port(tmp_path, monkeypatch):
    probe = Mock(side_effect=[False, True])
    monkeypatch.setattr(discovery, "uses_data_dir", probe)
    assert (
        discovery.find_local_backend(tmp_path, host="localhost", port=17494, owner_host="::", owner_port=19000)
        == "http://[::1]:19000"
    )
    assert [call.args[:2] for call in probe.call_args_list] == [("127.0.0.1", 17494), ("::1", 19000)]


def test_legacy_owner_discovery_uses_documented_ports(tmp_path, monkeypatch):
    probe = Mock(side_effect=[False, True])
    monkeypatch.setattr(discovery, "uses_data_dir", probe)
    assert discovery.find_local_backend(tmp_path, host="127.0.0.1", port=17494) == "http://127.0.0.1:17493"
    assert [call.args[:2] for call in probe.call_args_list] == [("127.0.0.1", 17494), ("127.0.0.1", 17493)]


def test_discovery_never_requests_remote_hosts(tmp_path, monkeypatch):
    connect = Mock()
    monkeypatch.setattr(discovery.http.client, "HTTPConnection", connect)
    assert not discovery.uses_data_dir("example.com", 8000, tmp_path)
    connect.assert_not_called()


def test_discovery_reads_recorded_nonstandard_port_without_changing_lock(tmp_path, monkeypatch):
    lock = tmp_path / ".voicebox.lock"
    payload = "pid=123\nhost=localhost\nport=19234\n"
    lock.write_text(payload, encoding="ascii")
    before = lock.stat()
    probe = Mock(side_effect=[False, True])
    monkeypatch.setattr(discovery, "uses_data_dir", probe)

    assert discovery.find_local_backend(tmp_path, host="127.0.0.1", port=17494) == "http://127.0.0.1:19234"
    assert [call.args[:2] for call in probe.call_args_list] == [("127.0.0.1", 17494), ("127.0.0.1", 19234)]
    assert lock.read_text(encoding="ascii") == payload
    assert lock.stat().st_ino == before.st_ino
    assert lock.stat().st_mtime_ns == before.st_mtime_ns


def test_legacy_ipv6_owner_is_discovered(tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "uses_data_dir", lambda host, port, _root: (host, port) == ("::1", 17493))
    assert discovery.find_local_backend(tmp_path, host="127.0.0.1", port=17494) == "http://[::1]:17493"


@pytest.mark.parametrize(
    "payload",
    [b"invalid metadata", b"\xff", b"host=example.com\nport=19234\n", b"host=localhost\nport=99999\n"],
)
def test_malformed_or_remote_owner_metadata_cannot_redirect_discovery(tmp_path, monkeypatch, payload):
    (tmp_path / ".voicebox.lock").write_bytes(payload)
    probe = Mock(return_value=False)
    monkeypatch.setattr(discovery, "uses_data_dir", probe)
    assert discovery.find_local_backend(tmp_path, host="127.0.0.1", port=17494) is None
    assert all(call.args[0] in {"127.0.0.1", "::1"} for call in probe.call_args_list)
    assert all(call.args[1] in {8000, 17493, 17494} for call in probe.call_args_list)


@pytest.mark.skipif(os.name != "posix", reason="FIFO and symlink creation requires POSIX")
@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_discovery_does_not_open_unsafe_lock_entries(tmp_path, monkeypatch, kind):
    lock = tmp_path / ".voicebox.lock"
    if kind == "fifo":
        os.mkfifo(lock)
    else:
        outside = tmp_path / "outside"
        outside.write_text("host=localhost\nport=19234\n", encoding="ascii")
        lock.symlink_to(outside)
    opener = Mock(side_effect=AssertionError("unsafe lock must not be opened"))
    monkeypatch.setattr(discovery.os, "open", opener)
    monkeypatch.setattr(discovery, "uses_data_dir", Mock(return_value=False))
    assert discovery.find_local_backend(tmp_path, host="127.0.0.1", port=17494) is None
    opener.assert_not_called()
