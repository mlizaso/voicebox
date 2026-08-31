"""The OpenAPI generator is directory-independent and cleans helper processes."""

import os
import shutil
import signal
import subprocess
from contextlib import suppress
from pathlib import Path

SOURCE_SCRIPT = Path(__file__).parents[2] / "scripts" / "generate-api.sh"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def _script_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    caller = tmp_path / "caller"
    for directory in (
        repo / "scripts",
        repo / "backend" / "venv" / "bin",
        repo / "app" / "src" / "lib" / "api" / "core",
        fake_bin,
        caller,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    script = repo / "scripts" / "generate-api.sh"
    shutil.copy2(SOURCE_SCRIPT, script)
    (repo / "backend" / "venv" / "bin" / "activate").write_text("")
    (repo / "app" / "src" / "lib" / "api" / "core" / "request.ts").write_text("export {};\n")
    (repo / "scripts" / "patch-generated-api-auth.py").write_text("")

    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
count=0
if [ -f "$FAKE_CURL_STATE" ]; then count=$(cat "$FAKE_CURL_STATE"); fi
count=$((count + 1))
printf '%s' "$count" > "$FAKE_CURL_STATE"
if [ "$count" -eq 1 ]; then exit 1; fi
printf '%s' '{"openapi":"3.0.0"}'
""",
    )
    _write_executable(
        fake_bin / "uvicorn",
        """#!/bin/sh
printf '%s' "$PWD" > "$FAKE_UVICORN_CWD"
printf '%s' "$*" > "$FAKE_UVICORN_ARGS"
printf '%s' "$$" > "$FAKE_UVICORN_PID"
trap 'printf stopped > "$FAKE_UVICORN_STOPPED"; exit 0' TERM INT
while true; do sleep 1; done
""",
    )
    _write_executable(
        fake_bin / "bunx",
        """#!/bin/sh
case " $* " in
  *" --version "*) printf '1.0.0\n'; exit 0 ;;
esac
printf '%s' "$PWD" > "$FAKE_BUNX_CWD"
if [ "${FAKE_BUNX_FAIL:-0}" = "1" ]; then exit 23; fi
exit 0
""",
    )
    _write_executable(fake_bin / "bun", "#!/bin/sh\nexit 99\n")
    _write_executable(fake_bin / "python", "#!/bin/sh\nexit 0\n")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_CURL_STATE": str(tmp_path / "curl-state"),
            "FAKE_UVICORN_CWD": str(tmp_path / "uvicorn-cwd"),
            "FAKE_UVICORN_ARGS": str(tmp_path / "uvicorn-args"),
            "FAKE_UVICORN_PID": str(tmp_path / "uvicorn-pid"),
            "FAKE_UVICORN_STOPPED": str(tmp_path / "uvicorn-stopped"),
            "FAKE_BUNX_CWD": str(tmp_path / "bunx-cwd"),
        }
    )
    return repo, caller, env


def _stop_leaked_backend(env: dict[str, str]) -> None:
    pid_path = Path(env["FAKE_UVICORN_PID"])
    stopped_path = Path(env["FAKE_UVICORN_STOPPED"])
    if stopped_path.exists() or not pid_path.exists():
        return
    with suppress(ProcessLookupError):
        os.kill(int(pid_path.read_text()), signal.SIGTERM)


def _run_script(script: Path, *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script)],
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )


def test_generator_resolves_paths_from_repository_and_stops_helper(tmp_path: Path) -> None:
    repo, caller, env = _script_fixture(tmp_path)
    try:
        result = _run_script(repo / "scripts" / "generate-api.sh", cwd=caller, env=env)

        assert result.returncode == 0
        assert (repo / "app" / "openapi.json").read_text() == '{"openapi":"3.0.0"}'
        assert Path(env["FAKE_UVICORN_CWD"]).read_text() == str(repo)
        assert Path(env["FAKE_UVICORN_ARGS"]).read_text().startswith("backend.main:app ")
        assert Path(env["FAKE_BUNX_CWD"]).read_text() == str(repo / "app")
        assert Path(env["FAKE_UVICORN_STOPPED"]).read_text() == "stopped"
    finally:
        _stop_leaked_backend(env)


def test_generator_stops_helper_when_generation_fails(tmp_path: Path) -> None:
    repo, _caller, env = _script_fixture(tmp_path)
    env["FAKE_BUNX_FAIL"] = "1"
    try:
        result = _run_script(repo / "scripts" / "generate-api.sh", cwd=repo, env=env)

        assert result.returncode == 23
        assert Path(env["FAKE_UVICORN_STOPPED"]).read_text() == "stopped"
    finally:
        _stop_leaked_backend(env)
