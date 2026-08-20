"""Transport-security regressions for the stdio MCP shim."""

import pytest

from backend.mcp_shim import __main__ as mcp_shim


@pytest.fixture(autouse=True)
def _clear_shim_environment(monkeypatch):
    for name in (
        "VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP",
        "VOICEBOX_HOST",
        "VOICEBOX_PORT",
        "VOICEBOX_REMOTE_API_TOKEN",
        "VOICEBOX_SCHEME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_loopback_shim_keeps_http_default():
    assert mcp_shim._base_url() == (
        "http://127.0.0.1:17493/mcp/",
        "http://127.0.0.1:17493/health",
    )


def test_remote_shim_refuses_to_send_bearer_over_plain_http(monkeypatch):
    monkeypatch.setenv("VOICEBOX_HOST", "voicebox.example")
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "x" * 32)

    with pytest.raises(ValueError, match="refusing to send"):
        mcp_shim._base_url()


def test_remote_shim_accepts_https_and_formats_ipv6(monkeypatch):
    monkeypatch.setenv("VOICEBOX_HOST", "::1")
    monkeypatch.setenv("VOICEBOX_SCHEME", "https")
    monkeypatch.setenv("VOICEBOX_PORT", "443")
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "x" * 32)

    assert mcp_shim._base_url() == (
        "https://[::1]:443/mcp/",
        "https://[::1]:443/health",
    )


def test_remote_shim_plain_http_requires_explicit_insecure_opt_in(monkeypatch):
    monkeypatch.setenv("VOICEBOX_HOST", "voicebox.lan")
    monkeypatch.setenv("VOICEBOX_REMOTE_API_TOKEN", "x" * 32)
    monkeypatch.setenv("VOICEBOX_ALLOW_INSECURE_REMOTE_HTTP", "1")

    assert mcp_shim._base_url()[0] == "http://voicebox.lan:17493/mcp/"
