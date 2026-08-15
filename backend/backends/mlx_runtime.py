"""Lightweight identity for Voicebox's guarded MLX Qwen TTS runtime."""

from collections.abc import Mapping
from hashlib import sha256
from importlib import metadata
from types import MappingProxyType

MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS: Mapping[str, str] = MappingProxyType(
    {
        "mlx-audio": "0.4.1",
        "mlx": "0.32.0",
        "mlx-lm": "0.31.1",
        # mlx-metal supplies the actual Darwin runtime imported as ``mlx``.
        "mlx-metal": "0.32.0",
    }
)
MLX_AUDIO_QWEN_DTYPE_BACKPORT_VERSION = MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS["mlx-audio"]
MLX_QWEN_DTYPE_PATCH_REVISION = "bf16-speaker-v1"
MLX_QWEN_TTS_PINNED_MODELS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "1.7B": (
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "a6eb4f68e4b056f1215157bb696209bc82a6db48",
        ),
        "0.6B": (
            "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            "1eccf1cb2519b5a4e8a95b5f0544f3303568164f",
        ),
    }
)


def build_mlx_qwen_tts_implementation_revision(
    *,
    runtime_packages: Mapping[str, str] = MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS,
    patch_revision: str = MLX_QWEN_DTYPE_PATCH_REVISION,
    model_revisions: Mapping[str, tuple[str, str]] = MLX_QWEN_TTS_PINNED_MODELS,
) -> str:
    """Build a compact identity covering code, patch, repositories, and weights."""
    mlx_audio_version = runtime_packages.get("mlx-audio")
    if not mlx_audio_version:
        raise ValueError("MLX Qwen TTS runtime pins must include mlx-audio")
    canonical_specs = "\n".join(
        (
            f"patch={patch_revision}",
            *(f"runtime:{package}={version}" for package, version in sorted(runtime_packages.items())),
            *(f"{size}={repo}@{revision}" for size, (repo, revision) in sorted(model_revisions.items())),
        )
    )
    model_fingerprint = sha256(canonical_specs.encode("utf-8")).hexdigest()
    identity = f"qwen3-mlx-audio-{mlx_audio_version}-{patch_revision}-runtime-sha256-{model_fingerprint}"
    if len(identity) > 128:
        raise ValueError("MLX Qwen TTS implementation identity exceeds the API's 128-character limit")
    return identity


MLX_QWEN_TTS_IMPLEMENTATION_REVISION = build_mlx_qwen_tts_implementation_revision()


def get_mlx_qwen_tts_model_spec(model_size: str) -> tuple[str, str]:
    """Return the immutable Hugging Face repository and commit for a model size."""
    try:
        return MLX_QWEN_TTS_PINNED_MODELS[model_size]
    except KeyError as exc:
        raise ValueError(f"Unsupported pinned MLX Qwen TTS model size: {model_size}") from exc


def get_installed_mlx_audio_version() -> str | None:
    """Return the installed distribution version without importing mlx-audio."""
    try:
        return metadata.version("mlx-audio")
    except metadata.PackageNotFoundError:
        return None


def get_mlx_qwen_tts_implementation_revision() -> str | None:
    """Return the exact guarded numerical revision, or fail closed."""
    for package, expected_version in MLX_QWEN_TTS_RUNTIME_PACKAGE_PINS.items():
        try:
            installed_version = metadata.version(package)
        except metadata.PackageNotFoundError:
            return None
        if installed_version != expected_version:
            return None
    return MLX_QWEN_TTS_IMPLEMENTATION_REVISION
