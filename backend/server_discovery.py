"""Standard-library-only discovery shared with the private audiobook launcher."""

import http.client
import json
import os
import stat
from pathlib import Path


def read_owner_details(fd: int) -> dict:
    """Read bounded diagnostic metadata; the OS lock alone decides ownership."""
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        payload = os.read(fd, 1024).decode("ascii")
    except (OSError, UnicodeError):
        return {}
    fields = dict(line.split("=", 1) for line in payload.splitlines() if "=" in line)
    details = {}
    for name, maximum in (("pid", 2**31 - 1), ("port", 65535)):
        value = fields.get(name, "")
        if value.isdecimal() and 0 < int(value) <= maximum:
            details[name] = int(value)
    if fields.get("host"):
        details["host"] = fields["host"]
    return details


def _recorded_owner(data_dir: Path) -> dict:
    """Read an existing regular lock file without creating or modifying it."""
    path = data_dir / ".voicebox.lock"
    fd = None
    try:
        entry = path.lstat()
        if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
            return {}
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            return {}
        if (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino):
            return {}
        return read_owner_details(fd)
    except OSError:
        return {}
    finally:
        if fd is not None:
            os.close(fd)


def local_probe_host(host: str | None) -> str | None:
    """Map a local bind address to a reachable loopback address."""
    if host in {"localhost", "127.0.0.1", "0.0.0.0"}:
        return "127.0.0.1"
    if host in {"::", "::1"}:
        return "::1"
    return None


def server_url(host: str, port: int) -> str:
    """Format an HTTP address, including IPv6 literals."""
    return f"http://{'[' + host + ']' if ':' in host else host}:{port}"


def uses_data_dir(host: str, port: int, data_dir: Path) -> bool:
    """Verify local storage health without proxies, redirects, or large reads."""
    if local_probe_host(host) is None:
        return False
    connection = http.client.HTTPConnection(host, port, timeout=1)
    try:
        connection.request("GET", "/health/filesystem")
        response = connection.getresponse()
        if response.status != 200:
            return False
        body = response.read(65537)
        if len(body) > 65536:
            return False
        payload = json.loads(body)
        if not isinstance(payload, dict) or payload.get("healthy") is not True:
            return False
        directories = payload.get("directories")
        if not isinstance(directories, list):
            return False
        roots = [entry for entry in directories if isinstance(entry, dict) and entry.get("name") == "data"]
        return len(roots) == 1 and roots[0].get("path") == str(data_dir.resolve())
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        connection.close()


def find_local_backend(
    data_dir: Path,
    *,
    host: str,
    port: int,
    owner_host: str | None = None,
    owner_port: int | None = None,
) -> str | None:
    """Find a healthy local backend serving exactly the requested data root."""
    candidates = []
    local_host = local_probe_host(host)
    if local_host:
        candidates.append((local_host, port))
    if owner_host is None or owner_port is None:
        recorded = _recorded_owner(data_dir)
        owner_host = recorded.get("host")
        owner_port = recorded.get("port")
    local_owner = local_probe_host(owner_host)
    if local_owner and owner_port:
        candidates.append((local_owner, owner_port))
    # Older backends recorded only a PID. Probe the documented ports and verify
    # the data root rather than assuming any open port belongs to this server.
    candidates.extend(("127.0.0.1", fallback) for fallback in (17493, 17494, 8000))
    candidates.extend(("::1", fallback) for fallback in (17493, 17494, 8000))
    for candidate_host, candidate_port in dict.fromkeys(candidates):
        if uses_data_dir(candidate_host, candidate_port, data_dir):
            return server_url(candidate_host, candidate_port)
    return None
