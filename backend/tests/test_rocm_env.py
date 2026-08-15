from backend.utils.rocm_env import should_probe_rocminfo


def test_macos_does_not_probe_linux_rocm_tooling() -> None:
    assert should_probe_rocminfo("darwin") is False


def test_linux_and_windows_keep_rocm_discovery() -> None:
    assert should_probe_rocminfo("linux") is True
    assert should_probe_rocminfo("win32") is True
