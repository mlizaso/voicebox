"""Windows sidecars must preserve the pipes used for readiness and diagnostics."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from backend.build_binary import build_server


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm"])
def test_windows_sidecars_preserve_standard_streams(backend):
    def probe(command, **_kwargs):
        # Build argument tests must never run package installers or real probes.
        assert command[1] == "-c"
        stdout = "7.2.1" if backend == "rocm" and "torch.version.hip" in command[2] else ""
        return CompletedProcess(command, 0, stdout=stdout)

    with (
        patch("backend.build_binary.PyInstaller.__main__.run") as run,
        patch("backend.build_binary.platform.system", return_value="Windows"),
        patch("backend.build_binary.os.chdir"),
        patch("subprocess.run", side_effect=probe),
    ):
        build_server(cuda=backend == "cuda", rocm=backend == "rocm")
    arguments = run.call_args.args[0]
    assert "server.py" in arguments
    assert "--noconsole" not in arguments
    assert "--windowed" not in arguments
