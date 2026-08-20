"""Frozen-build coverage for the exact MLX runtime metadata contract."""

from unittest.mock import patch

from backend.backends.mlx_runtime import MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS
from backend.build_binary import MLX_RUNTIME_METADATA_DISTRIBUTIONS, build_server


def _copied_metadata(args: list[str]) -> list[str]:
    return [args[index + 1] for index, argument in enumerate(args[:-1]) if argument == "--copy-metadata"]


def test_apple_binary_copies_every_exact_mlx_runtime_distribution_once():
    with (
        patch("backend.build_binary.PyInstaller.__main__.run") as pyinstaller_run,
        patch("backend.build_binary.is_apple_silicon", return_value=True),
        patch("backend.build_binary.platform.system", return_value="Darwin"),
        patch("backend.build_binary.os.chdir"),
    ):
        build_server()

    args = pyinstaller_run.call_args.args[0]
    copied_metadata = _copied_metadata(args)
    expected = set(MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS)

    assert set(MLX_RUNTIME_METADATA_DISTRIBUTIONS) == expected
    assert expected <= set(copied_metadata)
    assert all(copied_metadata.count(distribution) == 1 for distribution in expected)
