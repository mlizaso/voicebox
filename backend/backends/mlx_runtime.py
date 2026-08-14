"""Lightweight identity for Voicebox's guarded MLX Qwen TTS runtime."""

from importlib import metadata

MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION = "0.4.1"
MLX_QWEN_TTS_IMPLEMENTATION_REVISION = f"qwen3-mlx-audio-{MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION}-bf16-speaker-v1"


def get_installed_mlx_audio_version() -> str | None:
    """Return the installed distribution version without importing mlx-audio."""
    try:
        return metadata.version("mlx-audio")
    except metadata.PackageNotFoundError:
        return None


def get_mlx_qwen_tts_implementation_revision() -> str | None:
    """Return the exact guarded numerical revision, or fail closed."""
    if get_installed_mlx_audio_version() != MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION:
        return None
    return MLX_QWEN_TTS_IMPLEMENTATION_REVISION
