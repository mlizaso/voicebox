"""Asset conversion must finish its batch even when the first count is zero."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "convert-assets.sh"
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="The asset-conversion tooling uses a POSIX Bash environment",
)


@pytest.mark.parametrize("extension", ["png", "PNG", "mov", "MOV"])
@pytest.mark.parametrize("conversion_succeeds", [True, False])
def test_asset_conversion_processes_every_file_and_reports_totals(tmp_path, extension, conversion_succeeds):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / SCRIPT.name
    shutil.copyfile(SCRIPT, script)
    tools = tmp_path / "bin"
    tools.mkdir()
    converter = (
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "source = args[args.index('-i') + 1] if '-i' in args else args[2]\n"
        "if pathlib.Path(source).read_bytes() == b'bad input':\n"
        "    sys.exit(1)\n"
        "output = args[args.index('-o') + 1] if '-o' in args else args[-2]\n"
        "pathlib.Path(output).write_bytes(b'converted')\n"
    )
    for name in ("cwebp", "ffmpeg"):
        executable = tools / name
        executable.write_text(converter)
        executable.chmod(0o700)
    sources = []
    for directory in ("landing/public", "docs/public"):
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        for name in ("first asset", "second asset"):
            source = folder / f"{name}.{extension}"
            source.write_bytes(b"source audio or image" if conversion_succeeds else b"bad input")
            sources.append(source)

    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env={**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Conversion complete!" in result.stdout
    assert ("4 converted, 0 failed" if conversion_succeeds else "0 converted, 4 failed") in result.stdout
    for source in sources:
        if conversion_succeeds:
            assert not source.exists()
        else:
            assert source.read_bytes() == b"bad input"
    outputs = list(tmp_path.rglob("*.webp")) + list(tmp_path.rglob("*.webm"))
    assert len(outputs) == (4 if conversion_succeeds else 0)
    assert all(output.read_bytes() == b"converted" for output in outputs)
